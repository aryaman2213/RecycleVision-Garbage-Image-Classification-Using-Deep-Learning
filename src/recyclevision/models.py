"""Model builders: a from-scratch baseline CNN and three ImageNet transfer-learning models.

Every model takes raw RGB pixels in [0, 255] at 224x224, so the app never has to know which
normalisation a backbone expects - augmentation (training only) and normalisation live inside
the model. Backbones are built "flat" (input_tensor=...) so Grad-CAM can reach their last conv layer.
Keras then counts the augmentation/normalise layers as part of the backbone, so those must stay
weight-free or the ImageNet weight file no longer lines up.
"""
from __future__ import annotations

import keras
from keras import layers, ops

from . import config as C

# backbone name -> (constructor, last conv layer for Grad-CAM, fraction of backbone layers to fine-tune)
BACKBONES = {
    "mobilenetv2": (keras.applications.MobileNetV2, "out_relu", 0.45),
    "efficientnetb0": (keras.applications.EfficientNetB0, "top_activation", 0.45),
    "resnet50": (keras.applications.ResNet50, "conv5_block3_out", 0.35),
}
GRADCAM_LAYER = {name: spec[1] for name, spec in BACKBONES.items()} | {"baseline_cnn": "conv_last"}


def augmentation() -> keras.Sequential:
    """Random flips, rotation, zoom, shift, contrast and brightness. Inactive at inference."""
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2, fill_mode="reflect"),
        layers.RandomZoom((-0.2, 0.2), fill_mode="reflect"),
        layers.RandomTranslation(0.1, 0.1, fill_mode="reflect"),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
    ], name="augmentation")


@keras.saving.register_keras_serializable(package="recyclevision")
class CaffePreprocess(layers.Layer):
    """ResNet50's ImageNet preprocessing: RGB->BGR and subtract the channel means. No weights."""

    MEAN_BGR = (103.939, 116.779, 123.68)

    def call(self, x):
        return ops.flip(x, axis=-1) - ops.convert_to_tensor(self.MEAN_BGR, dtype=x.dtype)


def _normalise(x, name: str):
    """Each backbone's own ImageNet normalisation, built from plain layers (no Lambda -> safe to load)."""
    if name == "mobilenetv2":                                     # [-1, 1]
        return layers.Rescaling(1 / 127.5, offset=-1, name="normalise")(x)
    if name == "efficientnetb0":                                  # model rescales internally
        return x
    if name == "resnet50":                                        # "caffe": RGB->BGR, subtract mean
        return CaffePreprocess(name="normalise")(x)
    if name == "baseline_cnn":                                    # [0, 1]
        return layers.Rescaling(1 / 255, name="normalise")(x)
    raise ValueError(name)


def _head(x, n_classes: int, dropout: float = 0.3):
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="drop1")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(dropout, name="drop2")(x)
    return layers.Dense(n_classes, activation="softmax", dtype="float32", name="predictions")(x)


def build_baseline_cnn(n_classes: int = len(C.CLASSES)) -> keras.Model:
    inputs = keras.Input((C.IMG_SIZE, C.IMG_SIZE, 3), name="image")
    x = augmentation()(inputs)
    x = _normalise(x, "baseline_cnn")
    for i, filters in enumerate([32, 64, 128, 256]):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu", name="conv_last" if i == 3 else None)(x)
        if i < 3:
            x = layers.MaxPooling2D()(x)
    return keras.Model(inputs, _head(x, n_classes), name="baseline_cnn")


def build_transfer(name: str, n_classes: int = len(C.CLASSES), weights: str | None = "imagenet"):
    """Returns (model, backbone). The backbone starts frozen; see unfreeze_top()."""
    ctor = BACKBONES[name][0]
    inputs = keras.Input((C.IMG_SIZE, C.IMG_SIZE, 3), name="image")
    x = augmentation()(inputs)
    x = _normalise(x, name)
    backbone = ctor(include_top=False, weights=weights, input_tensor=x)
    backbone.trainable = False
    outputs = _head(backbone.output, n_classes)
    return keras.Model(inputs, outputs, name=name), backbone


def unfreeze_top(backbone: keras.Model, fraction: float) -> int:
    """Unfreeze the top `fraction` of backbone layers, keeping BatchNorm frozen (standard practice)."""
    backbone.trainable = True
    n = len(backbone.layers)
    cut = int(n * (1 - fraction))
    trainable = 0
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= cut and not isinstance(layer, layers.BatchNormalization)
        trainable += layer.trainable
    return trainable


def build(name: str, weights: str | None = "imagenet"):
    if name == "baseline_cnn":
        return build_baseline_cnn(), None
    return build_transfer(name, weights=weights)
