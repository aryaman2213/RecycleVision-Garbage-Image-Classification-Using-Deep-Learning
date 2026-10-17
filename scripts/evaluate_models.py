"""Step 4: evaluate every trained model, pick the best by validation macro-F1, write tables and figures.

Usage:  python scripts/evaluate_models.py [--skip-latency]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402

from recyclevision import config as C, evaluate as E  # noqa: E402
from recyclevision.inference import load_model  # noqa: E402
from recyclevision.train import split_files  # noqa: E402

LABELS = {"baseline_cnn": "Baseline CNN", "mobilenetv2": "MobileNetV2",
          "efficientnetb0": "EfficientNetB0", "resnet50": "ResNet50"}


def confusion_figure(probs, y, name):
    cm = pd.crosstab(pd.Series(y).map(dict(enumerate(C.CLASSES))), pd.Series(probs.argmax(1)).map(dict(enumerate(C.CLASSES))))
    cm = cm.reindex(index=C.CLASSES, columns=C.CLASSES, fill_value=0)
    cm.to_csv(C.METRICS_DIR / f"confusion_{name}.csv")
    norm = cm.div(cm.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(norm, annot=cm.values, fmt="d", cmap="Greens", vmin=0, vmax=1, cbar_kws={"label": "row share (recall)"}, ax=ax)
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"{LABELS[name]} - test confusion matrix (counts; colour = row %)")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / f"confusion_{name}.png", dpi=130)
    plt.close(fig)


def curves_figure(names):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name in names:
        h = pd.read_csv(C.METRICS_DIR / f"history_{name}.csv")
        axes[0].plot(h["epoch"] + 1, h["val_accuracy"], label=LABELS[name])
        axes[1].plot(h["epoch"] + 1, h["val_loss"], label=LABELS[name])
        info = json.loads((C.METRICS_DIR / f"train_{name}.json").read_text())
        if "frozen_epochs" in info:
            axes[0].axvline(info["frozen_epochs"] + 0.5, color="grey", lw=0.6, ls=":")
    axes[0].set(title="Validation accuracy (dotted = fine-tuning starts)", xlabel="epoch")
    axes[1].set(title="Validation loss", xlabel="epoch")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "training_curves.png", dpi=130)
    plt.close(fig)


def comparison_figure(table):
    t = table.set_index("model")[["accuracy", "precision_macro", "recall_macro", "f1_macro"]]
    t.index = [LABELS[n] for n in t.index]
    ax = t.plot.bar(figsize=(10, 4.5), rot=0, ylim=(max(0, t.values.min() - 0.1), 1.0), width=0.8)
    ax.axhline(0.85, color="red", ls="--", lw=1, label="85% target")
    ax.legend(loc="lower right", ncol=5, fontsize=8)
    ax.set_title("Test-set performance by model")
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "model_comparison.png", dpi=130)
    plt.close()


def misclassified_figure(probs, y, name, n=16):
    paths, _ = split_files("test")
    pred = probs.argmax(1)
    wrong = np.where(pred != y)[0]
    wrong = wrong[np.argsort(-probs[wrong, pred[wrong]])][:n]  # most confident mistakes first
    if not len(wrong):
        return
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(len(wrong) / cols)), cols, figsize=(11, 2.9 * np.ceil(len(wrong) / cols)))
    for ax, i in zip(axes.ravel(), wrong):
        ax.imshow(Image.open(paths[i]))
        ax.set_title(f"true: {C.CLASSES[y[i]]}\npred: {C.CLASSES[pred[i]]} ({probs[i, pred[i]]:.0%})", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{LABELS[name]} - most confident test mistakes")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / f"misclassified_{name}.png", dpi=110)
    plt.close(fig)


def main(skip_latency: bool):
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    names = [n for n in C.MODEL_NAMES if (C.MODELS_DIR / f"{n}.keras").exists()]
    rows, test_probs = [], {}
    for name in names:
        model = load_model(C.MODELS_DIR / f"{name}.keras")
        pv, yv = E.predictions(model, "val")
        pt, yt = E.predictions(model, "test")
        test_probs[name] = pt
        info = json.loads((C.METRICS_DIR / f"train_{name}.json").read_text())
        row = {"model": name, "val_f1_macro": E.metrics(pv, yv)["f1_macro"], **E.metrics(pt, yt)}
        row["f1_macro_ci95"] = E.bootstrap_f1_ci(pt, yt)
        row["params_m"] = model.count_params() / 1e6
        row["size_mb"] = (C.MODELS_DIR / f"{name}.keras").stat().st_size / 1e6
        row["train_minutes"] = info["train_seconds"] / 60
        lat_file = C.METRICS_DIR / f"latency_{name}.json"
        if not skip_latency:
            lat = E.latency_ms(model) | {"device": tf.config.list_physical_devices("GPU") and "GPU" or "CPU"}
            E.save_json(lat, lat_file.name)
        if lat_file.exists():
            row.update(json.loads(lat_file.read_text()))
        rows.append(row)
        E.per_class(pt, yt).to_csv(C.METRICS_DIR / f"per_class_{name}.csv")
        confusion_figure(pt, yt, name)
        print(f"{name}: test acc {row['accuracy']:.4f}  macro-F1 {row['f1_macro']:.4f}  val macro-F1 {row['val_f1_macro']:.4f}")
    table = pd.DataFrame(rows).sort_values("val_f1_macro", ascending=False)
    best = table.iloc[0]["model"]
    _, yt = split_files("test")
    table["mcnemar_p_vs_best"] = [
        np.nan if n == best else E.mcnemar(test_probs[best].argmax(1), test_probs[n].argmax(1), yt)["p_value"]
        for n in table["model"]]
    table.to_csv(C.METRICS_DIR / "model_comparison.csv", index=False)
    meta = {"model": best, "file": f"{best}.keras", "selected_by": "highest validation macro-F1",
            "classes": C.CLASSES, "img_size": C.IMG_SIZE,
            "test_accuracy": float(table.iloc[0]["accuracy"]), "test_f1_macro": float(table.iloc[0]["f1_macro"])}
    (C.MODELS_DIR / "best_model.json").write_text(json.dumps(meta, indent=2))
    comparison_figure(table)
    curves_figure(names)
    misclassified_figure(test_probs[best], yt, best)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    print("best:", meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-latency", action="store_true")
    main(ap.parse_args().skip_latency)
