"""Merge the three Kaggle datasets into one clean, de-duplicated, split dataset.

Pipeline (each step is recorded per image in data/manifest.csv):
  1. scan      - read every image straight from the zips, map its folder to the 10-class taxonomy
  2. validate  - drop unreadable, tiny or blank images
  3. dedupe    - exact (SHA-1) and near (perceptual hash + thumbnail check) duplicates, across all sources
  4. conflicts - drop duplicate groups whose copies carry different labels
  5. cap       - cap over-represented classes
  6. split     - stratified 70/15/15 train/val/test
  7. build     - write resized 224x224 JPEGs to data/processed/<split>/<class>/
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from . import config as C

IMAGE_EXT = (".jpg", ".jpeg", ".png")


def source_zip(key: str) -> Path:
    """Locate a source zip under RAW_DIR by its Kaggle name or the browser download name."""
    src = C.SOURCES[key]
    for name in (src["zip_name"], src["local_name"]):
        p = C.RAW_DIR / name
        if p.exists():
            if p.stat().st_size != src["size"]:
                raise ValueError(f"{p} is {p.stat().st_size} bytes, expected {src['size']} - different dataset version")
            return p
    raise FileNotFoundError(f"{src['zip_name']} not found in {C.RAW_DIR}")


def list_members() -> pd.DataFrame:
    rows = []
    for key, src in C.SOURCES.items():
        with zipfile.ZipFile(source_zip(key)) as z:
            for m in z.namelist():
                if not m.startswith(src["prefix"]) or not m.lower().endswith(IMAGE_EXT):
                    continue
                folder = m[len(src["prefix"]):].split("/")[0]
                rows.append({"source": key, "member": m, "orig_label": folder,
                             "label": src["labels"][folder]})
    return pd.DataFrame(rows)


def _phash(gray32: np.ndarray) -> int:
    """64-bit DCT perceptual hash of a 32x32 grayscale image (same algorithm as imagehash.phash)."""
    from scipy.fft import dctn
    d = dctn(gray32.astype(np.float64), norm="ortho")[:8, :8]
    bits = (d > np.median(d)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


@lru_cache(maxsize=4)
def _open_zip(path: str) -> zipfile.ZipFile:
    """One open handle per zip per worker process (re-reading a 36k-entry index per image is slow)."""
    return zipfile.ZipFile(path)


def _features(args) -> dict:
    """Decode one image and compute validation stats and hashes. Runs in a worker process."""
    zip_path, member = args
    raw = _open_zip(zip_path).read(member)
    out = {"member": member, "sha1": hashlib.sha1(raw).hexdigest(), "bytes": len(raw)}
    try:
        img = Image.open(io.BytesIO(raw))
        out["format"], out["mode"] = img.format, img.mode
        out["width"], out["height"] = img.size
        img.draft("RGB", (256, 256))
        img = ImageOps.exif_transpose(img).convert("RGB")
        small = np.asarray(img.resize((64, 64), Image.BILINEAR), dtype=np.float32)
    except Exception as e:  # noqa: BLE001 - any decode failure means the file is unusable
        out["error"] = f"{type(e).__name__}: {e}"[:200]
        return out
    out["pixel_std"] = float(small.std())
    out["mean_r"], out["mean_g"], out["mean_b"] = (float(v) for v in small.mean(axis=(0, 1)))
    gray = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS))
    out["phash"] = _phash(gray)
    out["thumb"] = np.asarray(img.resize((32, 32), Image.BILINEAR), dtype=np.uint8).tobytes()
    return out


def compute_features(members: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
    jobs = [(str(source_zip(s)), m) for s, m in zip(members["source"], members["member"])]
    with ProcessPoolExecutor(workers) as ex:
        feats = list(ex.map(_features, jobs, chunksize=64))
    f = pd.DataFrame(feats).drop(columns="member")
    return pd.concat([members.reset_index(drop=True), f], axis=1)


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def near_duplicate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """All pairs within PHASH_MAX_DIST, with their thumbnail correlation (brightness-invariant)."""
    idx = df.index.to_numpy()
    h = np.array([np.uint64(v) for v in df["phash"]], dtype=np.uint64)
    thumbs = np.stack([np.frombuffer(t, np.uint8) for t in df["thumb"]]).astype(np.float32)
    thumbs = (thumbs - thumbs.mean(1, keepdims=True)) / (thumbs.std(1, keepdims=True) + 1e-6)
    pairs = []
    for start in range(0, len(h), 512):
        block = h[start:start + 512, None] ^ h[None, :]
        dist = np.bitwise_count(block)
        i, j = np.nonzero(dist <= C.PHASH_MAX_DIST)
        i = i + start
        keep = i < j
        for a, b in zip(i[keep], j[keep]):
            pairs.append((idx[a], idx[b], int(dist[a - start, b]),
                          float((thumbs[a] * thumbs[b]).mean())))
    return pd.DataFrame(pairs, columns=["a", "b", "phash_dist", "thumb_ncc"])


def build_manifest(feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = feat.copy()
    if "error" not in df:
        df["error"] = np.nan
    df["status"] = "kept"
    df["dup_of"] = ""

    bad = df["error"].notna()
    df.loc[bad, "status"] = "invalid_unreadable"
    small = ~bad & (df[["width", "height"]].min(axis=1) < C.MIN_SIDE)
    df.loc[small, "status"] = "invalid_too_small"
    blank = ~bad & ~small & (df["pixel_std"] < C.MIN_PIXEL_STD)
    df.loc[blank, "status"] = "invalid_blank"

    ok = df[df["status"] == "kept"]
    uf = _UnionFind(len(df))
    # exact byte duplicates
    for _, grp in ok.groupby("sha1"):
        first = grp.index[0]
        for other in grp.index[1:]:
            uf.union(first, other)
    # near duplicates (resized / re-encoded copies)
    pairs = near_duplicate_pairs(ok)
    pairs["confirmed"] = pairs["thumb_ncc"] >= C.THUMB_MIN_NCC
    for a, b in pairs.loc[pairs["confirmed"], ["a", "b"]].itertuples(index=False):
        uf.union(a, b)

    df["cluster"] = [uf.find(i) for i in range(len(df))]
    prio = {s: k for k, s in enumerate(C.SOURCE_PRIORITY)}
    df["_prio"] = df["source"].map(prio)
    df["_area"] = df["width"].fillna(0) * df["height"].fillna(0)
    kept = df["status"] == "kept"
    for cid, grp in df[kept].groupby("cluster"):
        if len(grp) == 1:
            continue
        if grp["label"].nunique() > 1:  # same photo, different labels -> untrustworthy
            df.loc[grp.index, "status"] = "label_conflict"
            continue
        best = grp.sort_values(["_prio", "_area"], ascending=[True, False]).index[0]
        rest = grp.index.drop(best)
        df.loc[rest, "status"] = "duplicate"
        df.loc[rest, "dup_of"] = df.at[best, "source"] + ":" + df.at[best, "member"]

    rng = np.random.default_rng(C.SEED)
    for label, grp in df[df["status"] == "kept"].groupby("label"):
        if len(grp) > C.CLASS_CAP:
            drop = rng.choice(grp.index.to_numpy(), len(grp) - C.CLASS_CAP, replace=False)
            df.loc[drop, "status"] = "capped"

    df["split"] = ""
    for label, grp in df[df["status"] == "kept"].groupby("label"):
        order = rng.permutation(grp.index.to_numpy())
        n = len(order)
        n_train = round(n * C.SPLIT_FRACTIONS["train"])
        n_val = round(n * C.SPLIT_FRACTIONS["val"])
        df.loc[order[:n_train], "split"] = "train"
        df.loc[order[n_train:n_train + n_val], "split"] = "val"
        df.loc[order[n_train + n_val:], "split"] = "test"

    df["id"] = df["source"] + "_" + df["sha1"].str[:12]
    df["phash"] = df["phash"].map(lambda v: f"{int(v):016x}" if pd.notna(v) else "")
    cols = ["id", "source", "member", "orig_label", "label", "status", "split", "dup_of", "cluster",
            "width", "height", "format", "mode", "bytes", "sha1", "phash", "pixel_std",
            "mean_r", "mean_g", "mean_b", "error"]
    return df[cols], pairs


def _write_one(args) -> None:
    zip_path, member, dest = args
    if Path(dest).exists():
        return
    img = Image.open(io.BytesIO(_open_zip(zip_path).read(member)))
    img.draft("RGB", (C.IMG_SIZE * 2, C.IMG_SIZE * 2))
    img = ImageOps.exif_transpose(img).convert("RGB").resize((C.IMG_SIZE, C.IMG_SIZE), Image.LANCZOS)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, quality=95)


def build_processed(manifest: pd.DataFrame, workers: int = 8) -> pd.Series:
    """Resize every kept image to 224x224 into data/processed/<split>/<class>/<id>.jpg."""
    kept = manifest[manifest["status"] == "kept"]
    jobs = [(str(source_zip(r.source)), r.member, str(C.PROCESSED_DIR / r.split / r.label / f"{r.id}.jpg"))
            for r in kept.itertuples()]
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_write_one, jobs, chunksize=64))
    return kept.groupby(["split", "label"]).size()
