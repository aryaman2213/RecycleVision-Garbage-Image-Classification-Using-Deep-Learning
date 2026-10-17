"""RecycleVision - upload a photo of waste, get its category, confidence, top-3 and a Grad-CAM explanation.

Run locally:  streamlit run app/streamlit_app.py
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recyclevision import config as C  # noqa: E402
from recyclevision import inference as I  # noqa: E402

st.set_page_config(page_title="RecycleVision", page_icon="♻️", layout="wide")

EXAMPLES_DIR = ROOT / "app" / "examples"
LOW_CONFIDENCE = 0.55  # below this the app says it is unsure instead of asserting a class


@st.cache_resource(show_spinner="Loading model ...")
def get_model():
    return I.load_best()


@st.cache_data
def comparison_table():
    p = C.METRICS_DIR / "model_comparison.csv"
    return pd.read_csv(p) if p.exists() else None


def classify(image: Image.Image, show_cam: bool):
    model, meta = get_model()
    t0 = time.perf_counter()
    top = I.predict(model, image, top_k=3)
    ms = (time.perf_counter() - t0) * 1000
    label, conf = top[0]
    stream, tip = C.BIN_GUIDE[label]

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.image(image, caption="Your image", use_container_width=True)
        if show_cam:
            with st.spinner("Computing Grad-CAM ..."):
                cam = I.gradcam(model, image)
            st.image(I.overlay(image, cam), caption="Grad-CAM: red = regions that drove the prediction",
                     use_container_width=True)
    with right:
        if conf >= LOW_CONFIDENCE:
            st.success(f"### {label.capitalize()}  ·  {conf:.1%} confident")
        else:
            st.warning(f"### Not sure - maybe {label} ({conf:.1%})\n"
                       "Try a closer photo of a single item on a plain background.")
        st.markdown(f"**Disposal stream:** {stream}  \n{tip}")
        st.markdown("#### Top-3 predictions")
        for name, p in top:
            st.progress(min(max(p, 0.0), 1.0), text=f"{name} - {p:.1%}")
        st.caption(f"Model: {meta['model']} · inference {ms:.0f} ms on this server")


def main():
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
        st.markdown(f"**{meta['model']}** - selected by {meta['selected_by']}.  \n"
                    f"Test accuracy **{meta['test_accuracy']:.1%}**, macro-F1 **{meta['test_f1_macro']:.3f}** "
                    "on 2,423 held-out images.")
        show_cam = st.toggle("Show Grad-CAM explanation", value=True)
        st.divider()
        st.markdown("Trained on 16,153 unique images merged from three Kaggle datasets "
                    "(duplicates removed across sources).")

    tab_classify, tab_models = st.tabs(["Classify", "Model comparison"])
    with tab_classify:
        upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
        examples = sorted(EXAMPLES_DIR.glob("*.jpg")) if EXAMPLES_DIR.exists() else []
        choice = None
        if examples and upload is None:
            choice = st.selectbox("...or try an example", ["-"] + [p.stem for p in examples])
        image = None
        if upload is not None:
            try:
                image = Image.open(upload)
                image.load()
            except (UnidentifiedImageError, OSError):
                st.error("That file is not a readable image.")
        elif choice and choice != "-":
            image = Image.open(EXAMPLES_DIR / f"{choice}.jpg")
        if image is not None:
            classify(image, show_cam)

    with tab_models:
        table = comparison_table()
        if table is None:
            st.info("Run scripts/evaluate_models.py to generate the comparison.")
        else:
            cols = ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "top3_accuracy",
                    "size_mb", "params_m", "latency_ms_single"]
            st.dataframe(table[[c for c in cols if c in table]].round(4), hide_index=True, use_container_width=True)
            for fig in ["model_comparison.png", f"confusion_{meta['model']}.png"]:
                if (C.FIGURES_DIR / fig).exists():
                    st.image(str(C.FIGURES_DIR / fig), use_container_width=True)


main()
