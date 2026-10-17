"""Step 5: Grad-CAM gallery for the best model + one held-out example image per class for the app.

Usage:  python scripts/make_gradcam_gallery.py
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from recyclevision import config as C, inference as I  # noqa: E402
from recyclevision.train import split_files  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "app" / "examples"


def main():
    model, meta = I.load_best()
    z = np.load(C.METRICS_DIR / f"preds_{meta['model']}_test.npz")
    probs, y = z["probs"], z["y"]
    paths, _ = split_files("test")
    pred = probs.argmax(1)
    rng = np.random.default_rng(C.SEED)

    # one correctly-classified, confident test image per class -> gallery + app examples
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    for old in EXAMPLES.glob("*.jpg"):
        old.unlink()
    fig, axes = plt.subplots(4, 5, figsize=(13, 11))  # rows: image, Grad-CAM, image, Grad-CAM
    chosen = {}
    for k, label in enumerate(C.CLASSES):
        ok = np.where((y == k) & (pred == k) & (probs[:, k] > 0.8))[0]
        i = int(rng.choice(ok))
        chosen[label] = Path(paths[i]).name
        img = Image.open(paths[i])
        shutil.copy(paths[i], EXAMPLES / f"{label}.jpg")
        r, c = 2 * (k // 5), k % 5
        axes[r, c].imshow(img)
        axes[r, c].set_title(f"{label} ({probs[i, k]:.0%})", fontsize=10)
        axes[r + 1, c].imshow(I.overlay(img, I.gradcam(model, img)))
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{meta['model']}: where the model looks (held-out test images)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(C.FIGURES_DIR / "gradcam_gallery.png", dpi=110)
    plt.close(fig)

    # Grad-CAM on the most confident mistakes: explains *why* errors happen
    wrong = np.where(pred != y)[0]
    wrong = wrong[np.argsort(-probs[wrong, pred[wrong]])][:6]
    fig, axes = plt.subplots(2, len(wrong), figsize=(2.4 * len(wrong), 5))
    for c, i in enumerate(wrong):
        img = Image.open(paths[i])
        axes[0, c].imshow(img)
        axes[0, c].set_title(f"true {C.CLASSES[y[i]]}\npred {C.CLASSES[pred[i]]} {probs[i, pred[i]]:.0%}", fontsize=8)
        axes[1, c].imshow(I.overlay(img, I.gradcam(model, img)))
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Grad-CAM on the most confident mistakes", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(C.FIGURES_DIR / "gradcam_mistakes.png", dpi=110)
    plt.close(fig)
    (C.METRICS_DIR / "app_examples.json").write_text(json.dumps(chosen, indent=2))
    print("examples:", chosen)


if __name__ == "__main__":
    main()
