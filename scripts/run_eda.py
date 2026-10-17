"""Exploratory data analysis of the merged dataset -> reports/figures/eda_*.png + reports/metrics/eda_summary.json

Covers the brief (images per class, sample images per class, pixel/colour distribution) plus what the
merge itself needs explaining: where images came from, how many duplicates were removed, image sizes.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from recyclevision import config as C  # noqa: E402

PALETTE = dict(zip(C.CLASSES, plt.get_cmap("tab10").colors))
STATUS_ORDER = ["kept", "duplicate", "capped", "label_conflict", "invalid_blank", "invalid_too_small"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / name, dpi=130)
    plt.close(fig)


def main():
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(C.MANIFEST_PATH)
    kept = m[m["status"] == "kept"]
    summary = {"raw_images": len(m), "final_images": len(kept)}

    # 1. pipeline funnel: raw -> fate of every image, per source
    fate = pd.crosstab(m["source"], m["status"]).reindex(columns=STATUS_ORDER, fill_value=0)
    fate.index = [C.SOURCES[s]["title"].split(" - ")[0] + f" [{s}]" for s in fate.index]
    ax = fate.plot.barh(stacked=True, figsize=(10, 3.6), colormap="Set2")
    ax.set(xlabel="images", title="What happened to each raw image, by source dataset")
    ax.legend(ncol=3, fontsize=8, loc="lower right")
    save(ax.figure, "eda_pipeline_funnel.png")
    summary["status_by_source"] = pd.crosstab(m["source"], m["status"]).to_dict()

    # 2. images per class: before (raw, label-mapped) vs after cleaning, and per split
    raw = m.groupby("label").size().reindex(C.CLASSES)
    fin = kept.groupby("label").size().reindex(C.CLASSES)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    pd.DataFrame({"raw (all sources)": raw, "final (clean, unique)": fin}).plot.bar(ax=axes[0], rot=35)
    axes[0].set(title="Images per class: raw vs final", ylabel="images")
    split = pd.crosstab(kept["label"], kept["split"]).reindex(index=C.CLASSES)[["train", "val", "test"]]
    split.plot.bar(stacked=True, ax=axes[1], rot=35, colormap="viridis")
    axes[1].set(title="Final dataset: stratified 70/15/15 split", ylabel="images")
    save(fig, "eda_class_distribution.png")
    summary["final_per_class"] = fin.to_dict()
    summary["imbalance_ratio_max_min"] = round(float(fin.max() / fin.min()), 2)
    summary["split_sizes"] = kept["split"].value_counts().to_dict()

    # 3. sample images per class
    rng = np.random.default_rng(C.SEED)
    n = 6
    fig, axes = plt.subplots(len(C.CLASSES), n, figsize=(n * 1.7, len(C.CLASSES) * 1.75))
    for r, label in enumerate(C.CLASSES):
        ids = kept[kept["label"] == label]
        pick = ids.iloc[rng.choice(len(ids), n, replace=False)]
        for c, row in enumerate(pick.itertuples()):
            ax = axes[r, c]
            ax.imshow(Image.open(C.PROCESSED_DIR / row.split / label / f"{row.id}.jpg"))
            ax.set_xticks([]), ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=11, rotation=0, ha="right", va="center")
    fig.suptitle("Random samples per class (after resizing to 224x224)")
    save(fig, "eda_samples.png")

    # 4. original image sizes (why resizing is needed)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in C.CLASSES:
        d = kept[kept["label"] == label]
        ax.scatter(d["width"], d["height"], s=4, alpha=0.35, color=PALETTE[label], label=label)
    ax.set(xscale="log", yscale="log", xlabel="width (px)", ylabel="height (px)",
           title="Original image sizes vary ~50x -> all resized to 224x224")
    ax.legend(markerscale=4, fontsize=8, ncol=2)
    save(fig, "eda_image_sizes.png")
    summary["original_width_px"] = kept["width"].describe()[["min", "50%", "max"]].to_dict()
    summary["original_height_px"] = kept["height"].describe()[["min", "50%", "max"]].to_dict()

    # 5. colour / pixel-intensity distribution per class (from the processed 224x224 images)
    hist = {}
    bins = np.linspace(0, 256, 33)
    for label in C.CLASSES:
        d = kept[kept["label"] == label].sample(min(150, (kept["label"] == label).sum()), random_state=C.SEED)
        px = np.concatenate([np.asarray(Image.open(C.PROCESSED_DIR / r.split / label / f"{r.id}.jpg")).reshape(-1, 3)[::7]
                             for r in d.itertuples()])
        hist[label] = [np.histogram(px[:, k], bins=bins, density=True)[0] for k in range(3)]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey=True)
    for ax, label in zip(axes.ravel(), C.CLASSES):
        for k, col in enumerate("rgb"):
            ax.plot(bins[:-1] + 4, hist[label][k], color=col, lw=1.4)
        ax.set_title(label)
    fig.suptitle("Pixel-intensity distribution per colour channel (150 images/class)")
    save(fig, "eda_color_histograms.png")

    means = kept.groupby("label")[["mean_r", "mean_g", "mean_b"]].mean().reindex(C.CLASSES)
    bright = kept.assign(brightness=kept[["mean_r", "mean_g", "mean_b"]].mean(axis=1))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    means.plot.bar(color=["#d62728", "#2ca02c", "#1f77b4"], ax=axes[0], rot=35)
    axes[0].set(title="Mean R/G/B value per class", ylabel="mean pixel value (0-255)", ylim=(0, 260))
    axes[0].legend(["R", "G", "B"], ncol=3, loc="upper center", fontsize=8)
    bright.boxplot(column="brightness", by="label", ax=axes[1], rot=35, grid=False)
    axes[1].set(title="Image brightness by class", xlabel="", ylabel="mean pixel value")
    fig.suptitle("")
    save(fig, "eda_color_means.png")
    summary["mean_rgb_per_class"] = means.round(1).to_dict(orient="index")

    # 6. duplicates removed across sources
    d = m[m["status"] == "duplicate"].copy()
    d["kept_copy_from"] = d["dup_of"].str.split(":").str[0]
    summary["duplicates_removed"] = int(len(d))
    summary["duplicates_by_pair"] = {f"{a}->{b}": int(v) for (a, b), v in d.groupby(["source", "kept_copy_from"]).size().items()}
    (C.METRICS_DIR).mkdir(parents=True, exist_ok=True)
    (C.METRICS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str)[:2500])


if __name__ == "__main__":
    main()
