"""Inference helpers shared by the Streamlit app and the evaluation report: load, predict, Grad-CAM."""
from __future__ import annotations

import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from . import config as C
from .models import GRADCAM_LAYER


def load_model(path: str | Path) -> keras.Model:
    return keras.models.load_model(path, compile=False)


def load_best() -> tuple[keras.Model, dict]:
    """The model chosen by scripts/evaluate_models.py (highest validation macro-F1)."""
    meta = json.loads((C.MODELS_DIR / "best_model.json").read_text())
    return load_model(C.MODELS_DIR / meta["file"]), meta


def preprocess(image: Image.Image) -> np.ndarray:
    """Same steps as the training data: honour EXIF rotation, RGB, resize to 224x224, pixels 0-255."""
    image = ImageOps.exif_transpose(image).convert("RGB").resize((C.IMG_SIZE, C.IMG_SIZE), Image.LANCZOS)
    return np.asarray(image, dtype=np.float32)


def predict(model: keras.Model, image: Image.Image, top_k: int = 3) -> list[tuple[str, float]]:
    probs = model(preprocess(image)[None], training=False).numpy()[0]
    order = np.argsort(probs)[::-1][:top_k]
    return [(C.CLASSES[i], float(probs[i])) for i in order]


def gradcam(model: keras.Model, image: Image.Image, class_index: int | None = None) -> np.ndarray:
    """Grad-CAM heatmap (224x224, 0-1) showing which regions drove the prediction.

    Uses the class logit (pre-softmax score): at ~100% confidence the softmax saturates and its
    gradients vanish, which would give an empty heatmap exactly for the most confident predictions.
    """
    layer = model.get_layer(GRADCAM_LAYER[model.name])
    head = model.get_layer("predictions")
    grad_model = keras.Model(model.inputs, [layer.output, head.input, model.output])
    x = tf.convert_to_tensor(preprocess(image)[None])
    with tf.GradientTape() as tape:
        conv, features, probs = grad_model(x, training=False)
        idx = int(tf.argmax(probs[0])) if class_index is None else class_index
        logit = tf.matmul(tf.cast(features, tf.float32), head.kernel[:, idx:idx + 1])[:, 0] + head.bias[idx]
    grads = tape.gradient(logit, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))                   # importance of each feature map
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights[0], axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    return np.asarray(Image.fromarray(np.uint8(cam * 255)).resize((C.IMG_SIZE, C.IMG_SIZE), Image.BILINEAR)) / 255.0


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    import matplotlib

    base = preprocess(image) / 255.0
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return Image.fromarray(np.uint8(np.clip((1 - alpha) * base + alpha * heat, 0, 1) * 255))
