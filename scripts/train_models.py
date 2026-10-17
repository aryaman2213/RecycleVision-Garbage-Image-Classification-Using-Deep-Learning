"""Step 3: train the baseline CNN and the transfer-learning models (GPU recommended - see notebooks/).

Usage:  python scripts/train_models.py [--models baseline_cnn mobilenetv2 ...] [--retrain]
Models already saved in models/ are skipped, so an interrupted Colab run can simply be restarted.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

from recyclevision import config as C  # noqa: E402
from recyclevision.train import train_model  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=C.MODEL_NAMES, choices=C.MODEL_NAMES)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--frozen-epochs", type=int, default=8)
    ap.add_argument("--finetune-epochs", type=int, default=15)
    ap.add_argument("--baseline-epochs", type=int, default=40)
    a = ap.parse_args()
    if tf.config.list_physical_devices("GPU"):
        keras.mixed_precision.set_global_policy("mixed_float16")  # ~2x faster on T4 tensor cores
    print("TensorFlow", tf.__version__, "| Keras", keras.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))
    for name in a.models:
        if (C.MODELS_DIR / f"{name}.keras").exists() and not a.retrain:
            print(f"skip {name}: already trained")
            continue
        print(f"\n===== {name} =====")
        print(train_model(name, a.frozen_epochs, a.finetune_epochs, a.baseline_epochs))
