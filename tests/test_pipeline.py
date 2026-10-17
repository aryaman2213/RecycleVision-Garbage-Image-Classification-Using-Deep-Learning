"""Fast checks on the data contract, the model builders and inference (no GPU or training needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recyclevision import config as C  # noqa: E402

MANIFEST = pd.read_csv(C.MANIFEST_PATH)
KEPT = MANIFEST[MANIFEST["status"] == "kept"]


def test_manifest_accounts_for_every_raw_image():
    assert len(MANIFEST) == 30301
    assert not MANIFEST[["source", "member"]].duplicated().any()


def test_kept_images_are_unique_and_labelled():
    assert KEPT["sha1"].is_unique, "exact duplicates survived"
    assert set(KEPT["label"]) == set(C.CLASSES)
    assert KEPT["id"].is_unique


def test_no_duplicate_cluster_spans_two_splits():
    """The leakage guard: every duplicate group contributes at most one image, so no split overlap."""
    assert KEPT["cluster"].is_unique


def test_split_is_stratified_70_15_15():
    share = KEPT["split"].value_counts(normalize=True)
    assert abs(share["train"] - 0.70) < 0.01 and abs(share["val"] - 0.15) < 0.01
    per_class = pd.crosstab(KEPT["label"], KEPT["split"], normalize="index")
    assert (per_class["train"].between(0.68, 0.72)).all()


def test_class_cap_respected():
    assert KEPT["label"].value_counts().max() <= C.CLASS_CAP


@pytest.mark.parametrize("name", C.MODEL_NAMES)
def test_models_build_with_expected_io(name):
    from recyclevision.models import GRADCAM_LAYER, build

    model, _ = build(name, weights=None)  # architecture only; no download
    assert model.input_shape == (None, C.IMG_SIZE, C.IMG_SIZE, 3)
    assert model.output_shape == (None, len(C.CLASSES))
    model.get_layer(GRADCAM_LAYER[name])  # Grad-CAM target exists
    probs = model(np.zeros((1, C.IMG_SIZE, C.IMG_SIZE, 3), "float32"), training=False).numpy()
    assert np.isclose(probs.sum(), 1, atol=1e-4)


def test_preprocess_handles_any_image_mode():
    from PIL import Image

    from recyclevision.inference import preprocess

    for mode in ["RGB", "RGBA", "L", "P", "CMYK"]:
        arr = preprocess(Image.new(mode, (500, 300)))
        assert arr.shape == (C.IMG_SIZE, C.IMG_SIZE, 3) and arr.dtype == np.float32


@pytest.mark.skipif(not (C.MODELS_DIR / "best_model.json").exists(), reason="no trained model yet")
def test_best_model_predicts_and_explains():
    from PIL import Image

    from recyclevision import inference as I

    model, meta = I.load_best()
    img = Image.open(next((C.PROCESSED_DIR / "test" / "glass").glob("*.jpg"))) if C.PROCESSED_DIR.exists() \
        else Image.new("RGB", (224, 224), "white")
    top = I.predict(model, img)
    assert len(top) == 3 and top[0][1] >= top[1][1] >= top[2][1]
    cam = I.gradcam(model, img)
    assert cam.shape == (C.IMG_SIZE, C.IMG_SIZE) and np.isfinite(cam).all()
