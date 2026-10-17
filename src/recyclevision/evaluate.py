"""Model evaluation: metrics, model selection, statistical comparison, speed and size."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, top_k_accuracy_score)

from . import config as C


def predictions(model, split: str) -> tuple[np.ndarray, np.ndarray]:
    """(probabilities, true labels) for a split, cached in reports/metrics/preds_<model>_<split>.npz."""
    from .train import make_dataset, split_files

    cache = C.METRICS_DIR / f"preds_{model.name}_{split}.npz"
    _, y = split_files(split)
    if cache.exists():
        z = np.load(cache)
        if len(z["y"]) == len(y) and (z["y"] == y).all():
            return z["probs"], y
    probs = model.predict(make_dataset(split, batch_size=64), verbose=0)
    np.savez_compressed(cache, probs=probs, y=y)
    return probs, y


def metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    pred = probs.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y, pred, labels=range(len(C.CLASSES)))
    # false-positive rate per class = FP / (FP + TN), averaged
    fp = cm.sum(0) - np.diag(cm)
    tn = cm.sum() - cm.sum(0) - cm.sum(1) + np.diag(cm)
    return {
        "accuracy": accuracy_score(y, pred),
        "top3_accuracy": top_k_accuracy_score(y, probs, k=3, labels=range(len(C.CLASSES))),
        "precision_macro": p, "recall_macro": r, "f1_macro": f,
        "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": fw,
        "false_positive_rate_macro": float((fp / (fp + tn)).mean()),
    }


def per_class(probs: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    rep = classification_report(y, probs.argmax(1), labels=range(len(C.CLASSES)),
                                target_names=C.CLASSES, output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T.loc[C.CLASSES]


def bootstrap_f1_ci(probs: np.ndarray, y: np.ndarray, n: int = 1000, seed: int = C.SEED) -> tuple[float, float]:
    """95% bootstrap confidence interval for macro-F1 (resampling test images)."""
    rng = np.random.default_rng(seed)
    pred = probs.argmax(1)
    scores = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        scores.append(precision_recall_fscore_support(y[i], pred[i], average="macro", zero_division=0)[2])
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray) -> dict:
    """Exact McNemar test: do two models make significantly different errors on the same images?"""
    a_ok, b_ok = pred_a == y, pred_b == y
    only_a, only_b = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": float(p)}


def latency_ms(model, runs: int = 30) -> dict:
    """Single-image latency (what one app upload costs) and batch-32 throughput on this machine."""
    x1 = np.random.default_rng(0).uniform(0, 255, (1, C.IMG_SIZE, C.IMG_SIZE, 3)).astype("float32")
    x32 = np.repeat(x1, 32, axis=0)
    for _ in range(3):
        model(x1, training=False)
        model(x32, training=False)
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        model(x1, training=False)
        t.append((time.perf_counter() - s) * 1000)
    s = time.perf_counter()
    for _ in range(5):
        model(x32, training=False)
    per_img = (time.perf_counter() - s) * 1000 / (5 * 32)
    return {"latency_ms_single": float(np.median(t)), "latency_ms_per_image_batch32": float(per_img)}


def save_json(obj, name: str) -> None:
    (C.METRICS_DIR / name).write_text(json.dumps(obj, indent=2, default=float))
