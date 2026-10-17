"""tf.data input pipeline and the training procedure shared by all four models."""
from __future__ import annotations

import json
import time

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from . import config as C
from .models import BACKBONES, build, unfreeze_top


def split_files(split: str) -> tuple[list[str], np.ndarray]:
    """Image paths and integer labels for a split, in a fixed (sorted) order."""
    m = pd.read_csv(C.MANIFEST_PATH, usecols=["id", "label", "status", "split"])
    m = m[(m["status"] == "kept") & (m["split"] == split)].sort_values("id")
    if C.SMOKE_LIMIT:  # tiny stratified subset for quick end-to-end tests
        m = m.groupby("label", group_keys=False).head(C.SMOKE_LIMIT)
    paths = [str(C.PROCESSED_DIR / split / lab / f"{i}.jpg") for i, lab in zip(m["id"], m["label"])]
    labels = m["label"].map({c: k for k, c in enumerate(C.CLASSES)}).to_numpy()
    return paths, labels


def _load(path, label):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (C.IMG_SIZE, C.IMG_SIZE))  # no-op for processed images; guards others
    return tf.cast(img, tf.float32), label


def make_dataset(split: str, shuffle: bool = False, batch_size: int = C.BATCH_SIZE) -> tf.data.Dataset:
    paths, labels = split_files(split)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.uint8), y)).cache()  # ~2 GB for train as uint8
    if shuffle:
        ds = ds.shuffle(4096, seed=C.SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def class_weights() -> dict[int, float]:
    """'Balanced' weights: n_samples / (n_classes * n_class_samples)."""
    _, y = split_files("train")
    counts = np.bincount(y, minlength=len(C.CLASSES))
    return {k: float(len(y) / (len(C.CLASSES) * c)) for k, c in enumerate(counts)}


def _compile(model, lr):
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")],
    )


def _fit(model, train, val, epochs, cw, log_path, patience, initial_epoch=0, reduce_lr=True):
    cbs = [
        keras.callbacks.EarlyStopping("val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.CSVLogger(str(log_path), append=initial_epoch > 0),
    ]
    if reduce_lr:  # not combinable with a LearningRateSchedule
        cbs.append(keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.3, patience=max(2, patience // 2), min_lr=1e-6))
    h = model.fit(train, validation_data=val, epochs=initial_epoch + epochs, initial_epoch=initial_epoch,
                  class_weight=cw, callbacks=cbs, verbose=2)
    return len(h.history["loss"])


def export_float32(trained: keras.Model, name: str, path) -> None:
    """Training may use mixed precision; the deployed copy is rebuilt in float32 so it runs well on CPU."""
    policy = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy("float32")
    try:
        clean, _ = build(name)  # same weights arg as training: EfficientNet adds a layer for "imagenet"
        assert len(clean.layers) == len(trained.layers)
        for dst, src in zip(clean.layers, trained.layers):  # same construction order; auto-names differ
            w = src.get_weights()
            assert [a.shape for a in w] == [a.shape for a in dst.get_weights()], (src.name, dst.name)
            dst.set_weights(w)
        x = np.random.default_rng(0).uniform(0, 255, (4, C.IMG_SIZE, C.IMG_SIZE, 3)).astype("float32")
        diff = float(np.abs(clean(x, training=False).numpy() - trained(x, training=False).numpy()).max())
        if diff > 0.02:  # warn, never lose a trained model over it
            print(f"WARNING: float32 export differs from the trained model by {diff:.4f}")
        clean.save(path)
        return diff
    finally:
        keras.mixed_precision.set_global_policy(policy)


def train_model(name: str, frozen_epochs: int = 8, finetune_epochs: int = 15, baseline_epochs: int = 25) -> dict:
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(C.SEED)
    train, val = make_dataset("train", shuffle=True), make_dataset("val")
    cw = class_weights()
    log = C.METRICS_DIR / f"history_{name}.csv"
    model, backbone = build(name)
    t0 = time.time()
    if backbone is None:
        _compile(model, 1e-3)
        epochs = _fit(model, train, val, baseline_epochs, cw, log, patience=7)
        phases = {"scratch_epochs": epochs}
    else:
        _compile(model, 1e-3)
        e1 = _fit(model, train, val, frozen_epochs, cw, log, patience=3)
        n_trainable = unfreeze_top(backbone, BACKBONES[name][2])
        steps = finetune_epochs * len(train)
        _compile(model, keras.optimizers.schedules.CosineDecay(1e-4, steps, alpha=0.05))
        e2 = _fit(model, train, val, finetune_epochs, cw, log, patience=4, initial_epoch=e1, reduce_lr=False)
        phases = {"frozen_epochs": e1, "finetune_epochs": e2, "finetuned_backbone_layers": n_trainable,
                  "backbone_layers": len(backbone.layers)}
    info = {"model": name, "train_seconds": round(time.time() - t0, 1), **phases,
            "params": int(model.count_params()),
            "device": tf.config.list_physical_devices("GPU")[0].name if tf.config.list_physical_devices("GPU") else "CPU",
            "tensorflow": tf.__version__, "keras": keras.__version__}
    info["export_max_prob_diff"] = export_float32(model, name, C.MODELS_DIR / f"{name}.keras")
    (C.METRICS_DIR / f"train_{name}.json").write_text(json.dumps(info, indent=2))
    return info
