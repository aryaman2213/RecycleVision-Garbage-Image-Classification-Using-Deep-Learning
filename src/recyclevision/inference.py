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


def load_best(warmup: bool = True) -> tuple[keras.Model, dict]:
    """The model chosen by scripts/evaluate_models.py (highest validation macro-F1).

    warmup runs one dummy prediction so the first real request does not pay the graph-tracing cost.
    """
    meta = json.loads((C.MODELS_DIR / "best_model.json").read_text())
    model = load_model(C.MODELS_DIR / meta["file"])
    if warmup:
        model.predict_on_batch(np.zeros((1, C.IMG_SIZE, C.IMG_SIZE, 3), "float32"))
    return model, meta


def to_rgb(image: Image.Image) -> Image.Image:
    """Upright RGB image: applies EXIF rotation (phone photos) and puts transparent images on white."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        return canvas
    return image.convert("RGB")


def preprocess(image: Image.Image) -> np.ndarray:
    """Same steps as the training data: upright RGB, resize to 224x224, pixels 0-255 (models normalise)."""
    return np.asarray(to_rgb(image).resize((C.IMG_SIZE, C.IMG_SIZE), Image.LANCZOS), dtype=np.float32)


def predict(model: keras.Model, image: Image.Image, top_k: int = 3) -> list[tuple[str, float]]:
    probs = np.asarray(model.predict_on_batch(preprocess(image)[None]))[0]  # compiled graph: ~5x faster than eager
    order = np.argsort(probs)[::-1][:top_k]
    return [(C.CLASSES[i], float(probs[i])) for i in order]


_GRADCAM_FN: dict[int, tf.types.experimental.GenericFunction] = {}


def _gradcam_fn(model: keras.Model):
    """Build (once per model) a compiled function returning the Grad-CAM map and predicted class."""
    key = id(model)
    if key not in _GRADCAM_FN:
        layer = model.get_layer(GRADCAM_LAYER[model.name])
        head = model.get_layer("predictions")
        grad_model = keras.Model(model.inputs, [layer.output, head.input, model.output])

        @tf.function(reduce_retracing=True)
        def fn(x, class_index):
            with tf.GradientTape() as tape:
                conv, features, probs = grad_model(x, training=False)
                idx = tf.where(class_index < 0, tf.argmax(probs[0], output_type=tf.int32), class_index)
                kernel = tf.gather(head.kernel, idx, axis=1)
                logit = tf.reduce_sum(tf.cast(features[0], tf.float32) * kernel) + tf.gather(head.bias, idx)
            grads = tape.gradient(logit, conv)
            weights = tf.reduce_mean(grads, axis=(1, 2))         # importance of each feature map
            return tf.nn.relu(tf.reduce_sum(conv[0] * weights[0], axis=-1))

        _GRADCAM_FN[key] = fn
    return _GRADCAM_FN[key]


def gradcam(model: keras.Model, image: Image.Image, class_index: int | None = None) -> np.ndarray:
    """Grad-CAM heatmap (224x224, 0-1) showing which regions drove the prediction.

    Uses the class logit (pre-softmax score): at ~100% confidence the softmax saturates and its
    gradients vanish, which would give an empty heatmap exactly for the most confident predictions.
    """
    x = tf.convert_to_tensor(preprocess(image)[None])
    cam = _gradcam_fn(model)(x, tf.constant(-1 if class_index is None else class_index, tf.int32)).numpy()
    cam = cam / (cam.max() + 1e-8)
    return np.asarray(Image.fromarray(np.uint8(cam * 255)).resize((C.IMG_SIZE, C.IMG_SIZE), Image.BILINEAR)) / 255.0


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    import matplotlib

    base = preprocess(image) / 255.0
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return Image.fromarray(np.uint8(np.clip((1 - alpha) * base + alpha * heat, 0, 1) * 255))
