"""Single source of truth for paths, classes, data sources and training settings."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("RECYCLEVISION_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = Path(os.environ.get("RECYCLEVISION_RAW", DATA_DIR / "raw"))  # holds the three Kaggle zips
PROCESSED_DIR = DATA_DIR / "processed"                                  # train/val/test/<class>/*.jpg
MANIFEST_PATH = DATA_DIR / "manifest.csv"                               # every raw image + its fate
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

SEED = 42
IMG_SIZE = 224
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Final 10-class taxonomy. "organic" is what the brief calls it; the datasets call it "biological".
CLASSES = [
    "battery", "cardboard", "clothes", "glass", "metal",
    "organic", "paper", "plastic", "shoes", "trash",
]

# Recyclable vs. non-recyclable stream, used by the app to say which bin an item belongs in.
BIN_GUIDE = {
    "battery": ("Hazardous / e-waste", "Never in household bins - take to an e-waste or battery drop-off."),
    "cardboard": ("Dry recyclable", "Flatten and keep dry."),
    "clothes": ("Textile reuse", "Donate or use a textile collection point."),
    "glass": ("Dry recyclable", "Rinse; keep separate from ceramics."),
    "metal": ("Dry recyclable", "Rinse cans and tins."),
    "organic": ("Wet / compostable", "Compost or wet-waste bin."),
    "paper": ("Dry recyclable", "Keep dry and free of food."),
    "plastic": ("Dry recyclable", "Rinse; check the resin code."),
    "shoes": ("Textile reuse", "Donate if wearable, else textile collection."),
    "trash": ("Reject / landfill", "Non-recyclable mixed waste."),
}

# The three datasets linked in the brief. `zip_name` is how Kaggle names the download;
# `size` lets every machine (laptop or Colab) verify it has the exact same version.
SOURCES = {
    "gc12": {
        "title": "Garbage Classification (12 classes) - mostafaabla",
        "kaggle": "mostafaabla/garbage-classification",
        "zip_name": "garbage-classification-12.zip",
        "local_name": "archive.zip",
        "size": 250641573,
        "prefix": "garbage_classification/",
        "labels": {
            "battery": "battery", "biological": "organic", "brown-glass": "glass",
            "green-glass": "glass", "white-glass": "glass", "cardboard": "cardboard",
            "clothes": "clothes", "metal": "metal", "paper": "paper", "plastic": "plastic",
            "shoes": "shoes", "trash": "trash",
        },
    },
    "gd10": {
        "title": "Garbage Classification V2 (10 classes) - sumn2u",
        "kaggle": "sumn2u/garbage-classification-v2",
        "zip_name": "garbage-classification-v2.zip",
        "local_name": "archive (1).zip",
        "size": 1147864104,
        "prefix": "original/",  # the zip also holds standardized_256/ and _384/ copies - ignored
        "labels": {
            "battery": "battery", "biological": "organic", "cardboard": "cardboard",
            "clothes": "clothes", "glass": "glass", "metal": "metal", "paper": "paper",
            "plastic": "plastic", "shoes": "shoes", "trash": "trash",
        },
    },
    "tn6": {
        "title": "Garbage Classification (6 classes, TrashNet) - asdasdasasdas",
        "kaggle": "asdasdasasdas/garbage-classification",
        "zip_name": "garbage-classification-6.zip",
        "local_name": "archive (2).zip",
        "size": 85969666,
        "prefix": "Garbage classification/Garbage classification/",  # second, lower-case copy ignored
        "labels": {c: c for c in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]},
    },
}
# When two sources hold the same photo, keep the copy from the earliest source in this list.
SOURCE_PRIORITY = ["tn6", "gc12", "gd10"]

# Cleaning thresholds
MIN_SIDE = 64          # smaller images carry too little detail
MIN_PIXEL_STD = 6.0    # near-blank images
PHASH_MAX_DIST = 8     # perceptual-hash Hamming distance for a duplicate candidate
THUMB_MIN_NCC = 0.95   # ...confirmed only if 32x32 colour thumbnails correlate this strongly.
                       # Calibrated by eye: >=0.95 is the same photo (re-encoded, re-cropped, re-lit);
                       # below that are different objects on similar white backgrounds.

# Class cap: clothes alone is ~1/3 of the merged data; capping keeps accuracy honest.
CLASS_CAP = 2500

# Training
BATCH_SIZE = 32
SMOKE_LIMIT = int(os.environ.get("RECYCLEVISION_SMOKE", "0"))  # >0: use N images per class (tests only)
MODEL_NAMES = ["baseline_cnn", "mobilenetv2", "efficientnetb0", "resnet50"]
