"""Step 1: scan, validate, de-duplicate, cap and split the three source datasets -> data/manifest.csv.

Usage:  python scripts/prepare_data.py            (zips in data/raw/, or set RECYCLEVISION_RAW)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from recyclevision import config as C, data  # noqa: E402


def main() -> None:
    cache = C.DATA_DIR / "features.pkl"
    if cache.exists():
        feat = pd.read_pickle(cache)
    else:
        members = data.list_members()
        print(f"scanning {len(members):,} images from {members['source'].nunique()} sources ...")
        feat = data.compute_features(members)
        feat.to_pickle(cache)
    manifest, pairs = data.build_manifest(feat)
    manifest.to_csv(C.MANIFEST_PATH, index=False)
    pairs.to_csv(C.DATA_DIR / "near_duplicate_pairs.csv", index=False)
    print(pd.crosstab(manifest["source"], manifest["status"], margins=True))
    kept = manifest[manifest["status"] == "kept"]
    print(pd.crosstab(kept["label"], kept["split"], margins=True))


if __name__ == "__main__":
    main()
