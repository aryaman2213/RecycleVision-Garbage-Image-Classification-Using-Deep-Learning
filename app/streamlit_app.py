"""RecycleVision - upload a photo of waste, get its category, confidence, top-3 and a Grad-CAM explanation.

Run locally:  streamlit run app/streamlit_app.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recyclevision import config as C  # noqa: E402
from recyclevision import inference as I  # noqa: E402

st.set_page_config(page_title="RecycleVision", page_icon="♻️", layout="wide")

EXAMPLES_DIR = ROOT / "app" / "examples"
MODEL_LABELS = {"baseline_cnn": "Baseline CNN", "mobilenetv2": "MobileNetV2",
                "efficientnetb0": "EfficientNetB0", "resnet50": "ResNet50"}


@st.cache_resource(show_spinner="Loading the model (first visit only) ...")
def get_model():
    model, meta = I.load_best()
    I.gradcam(model, Image.new("RGB", (C.IMG_SIZE, C.IMG_SIZE)))  # compile Grad-CAM once, up front
    return model, meta


@st.cache_data
def comparison_table() -> pd.DataFrame | None:
    p = C.METRICS_DIR / "model_comparison.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p)
    out = pd.DataFrame({
        "Model": t["model"].map(MODEL_LABELS).fillna(t["model"]),
        "Test accuracy": t["accuracy"] * 100,
        "Precision": t["precision_macro"],
        "Recall": t["recall_macro"],
        "Macro-F1": t["f1_macro"],
        "Top-3 accuracy": t["top3_accuracy"] * 100,
        "Size (MB)": t["size_mb"],
        "Parameters (M)": t["params_m"],
    })
    if "latency_ms_single" in t:
        out["CPU ms / image"] = t["latency_ms_single"]
    return out


def load_upload(upload) -> Image.Image | None:
    try:
        image = Image.open(upload)
        image.load()
        return I.to_rgb(image)  # upright (EXIF) and RGB, exactly what the model sees
    except Exception:  # noqa: BLE001 - corrupt, truncated, oversized or unsupported files
        st.error("That file could not be read as an image. Please upload a JPG, PNG, WEBP or BMP photo.")
        return None


def classify(image: Image.Image, show_cam: bool) -> None:
    model, meta = get_model()
    t0 = time.perf_counter()
    top = I.predict(model, image, top_k=3)
    ms = (time.perf_counter() - t0) * 1000
    label, conf = top[0]
    stream, tip = C.BIN_GUIDE[label]

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.image(image, caption="Your image", width="stretch")
        if show_cam:
            cam = I.gradcam(model, image)
            st.image(I.overlay(image, cam), width="stretch",
                     caption="Grad-CAM: red = regions that drove the prediction")
    with right:
        if conf >= C.CONFIDENCE_THRESHOLD:
            st.success(f"### {label.capitalize()} · {conf:.1%} confident")
            st.markdown(f"**Disposal stream:** {stream}  \n{tip}")
        else:
            st.warning(f"### Not sure - maybe {label} ({conf:.1%})\n\n"
                       "Try a closer photo of a single item on a plain background.")
            st.markdown(f"**If it is {label}:** {stream} - {tip}")
        if label == "battery" and conf >= C.CONFIDENCE_THRESHOLD:
            st.error("Hazardous waste: batteries can start fires in collection trucks. Never put them in household bins.")
        st.markdown("#### Top-3 predictions")
        for name, p in top:
            st.progress(float(np.clip(p, 0.0, 1.0)), text=f"{name} - {p:.1%}")
        st.caption(f"Model: {MODEL_LABELS.get(meta['model'], meta['model'])} · prediction took {ms:.0f} ms")


def main() -> None:
    st.title("♻️ RecycleVision")
    st.write("Upload a photo of a single waste item. The model sorts it into one of 10 categories: "
             + ", ".join(C.CLASSES) + ".")

    try:
        _, meta = get_model()
    except FileNotFoundError:
        st.error("No trained model found. Run `scripts/train_models.py` and `scripts/evaluate_models.py` first.")
        st.stop()

    with st.sidebar:
        st.header("About the model")
        st.markdown(f"**{MODEL_LABELS.get(meta['model'], meta['model'])}**, selected by {meta['selected_by']}.  \n"
                    f"Test accuracy **{meta['test_accuracy']:.1%}**, macro-F1 **{meta['test_f1_macro']:.3f}** "
                    f"on {meta['n_test_images']:,} held-out images.")
        show_cam = st.toggle("Show Grad-CAM explanation", value=True,
                             help="Heatmap of the image regions the model relied on.")
        st.divider()
        st.markdown(f"Trained on {meta['n_dataset_images']:,} unique images merged from three Kaggle datasets "
                    "(duplicates removed across sources).")
        st.caption("Works best on one item per photo. Low-confidence results are flagged as unsure.")

    tab_classify, tab_models = st.tabs(["Classify", "Model comparison"])
    with tab_classify:
        upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
        examples = sorted(p.stem for p in EXAMPLES_DIR.glob("*.jpg")) if EXAMPLES_DIR.exists() else []
        image = None
        if upload is not None:
            image = load_upload(upload)
        elif examples:
            choice = st.selectbox("...or try an example from the held-out test set", examples,
                                  index=None, placeholder="Choose an example")
            if choice:
                image = I.to_rgb(Image.open(EXAMPLES_DIR / f"{choice}.jpg"))
        if image is not None:
            classify(image, show_cam)

    with tab_models:
        table = comparison_table()
        if table is None:
            st.info("Run scripts/evaluate_models.py to generate the comparison.")
        else:
            st.markdown(f"All models evaluated on the same {meta['n_test_images']:,} held-out test images. "
                        f"**{MODEL_LABELS.get(meta['model'], meta['model'])}** has the highest validation macro-F1 "
                        "and is the one deployed here.")
            st.dataframe(
                table, hide_index=True, width="stretch",
                column_config={
                    "Test accuracy": st.column_config.NumberColumn(format="%.1f%%"),
                    "Top-3 accuracy": st.column_config.NumberColumn(format="%.1f%%"),
                    "Precision": st.column_config.NumberColumn(format="%.3f"),
                    "Recall": st.column_config.NumberColumn(format="%.3f"),
                    "Macro-F1": st.column_config.NumberColumn(format="%.3f"),
                    "Size (MB)": st.column_config.NumberColumn(format="%.0f"),
                    "Parameters (M)": st.column_config.NumberColumn(format="%.1f"),
                    "CPU ms / image": st.column_config.NumberColumn(format="%.0f"),
                })
            for fig, caption in [("model_comparison.png", "Test-set metrics per model (dashed line: 85% target)"),
                                 (f"confusion_{meta['model']}.png", "Confusion matrix of the deployed model")]:
                if (C.FIGURES_DIR / fig).exists():
                    st.image(str(C.FIGURES_DIR / fig), caption=caption, width="stretch")


main()
