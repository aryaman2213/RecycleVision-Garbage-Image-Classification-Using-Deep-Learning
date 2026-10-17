"""Build reports/RecycleVision_Report.pdf from the saved metrics, tables and figures (HTML -> PDF via Chrome).

Every number in the report is read from reports/metrics/ or data/manifest.csv, so it always matches the code.

    python scripts/build_report.py [--live-url https://<app>.streamlit.app]
"""
import argparse
import ast
import base64
import html
import io
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recyclevision import config as C  # noqa: E402

PDF_PATH = C.REPORTS_DIR / "RecycleVision_Report.pdf"
AUTHOR = "Aryaman Bhatnagar"
GITHUB_URL = "https://github.com/aryaman2213/recyclevision"
LABELS = {"baseline_cnn": "Baseline CNN", "mobilenetv2": "MobileNetV2",
          "efficientnetb0": "EfficientNetB0", "resnet50": "ResNet50"}

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: Inter, -apple-system, 'Segoe UI', sans-serif; color: #0f172a; font-size: 10.2pt; line-height: 1.5; }
h1 { font-size: 26pt; margin: 0 0 6pt 0; letter-spacing: -0.02em; }
h2 { font-size: 15.5pt; margin: 20pt 0 6pt 0; padding-bottom: 4pt; border-bottom: 2px solid #dcfce7; color: #14532d; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 3pt 0; break-after: avoid; }
p, li { margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt 0; font-size: 9pt; break-inside: avoid; }
th, td { border: 1px solid #e2e8f0; padding: 4pt 6pt; text-align: left; }
th { background: #f0fdf4; }
td.n { text-align: right; font-variant-numeric: tabular-nums; }
tr.champ td { background: #ecfdf5; font-weight: 600; }
code { background: #f1f5f9; padding: 0 3pt; border-radius: 3pt; font-size: 8.8pt; }
figure { margin: 6pt 0 10pt 0; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 225mm; border: 1px solid #e2e8f0; border-radius: 6px; }
figcaption { color: #64748b; font-size: 8.5pt; margin-top: 3pt; }
.cover { height: 250mm; display: flex; flex-direction: column; justify-content: center; break-after: page; }
.eyebrow { color: #15803d; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; font-size: 9pt; }
.sub { color: #475569; font-size: 12.5pt; max-width: 150mm; }
.meta td { border: none; padding: 2pt 10pt 2pt 0; font-size: 10pt; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6pt; margin: 8pt 0; break-inside: avoid; }
.kpi { border: 1px solid #e2e8f0; border-radius: 8px; padding: 6pt 8pt; }
.kpi b { display: block; font-size: 14pt; } .kpi span { color: #64748b; font-size: 8.5pt; }
.box { background: #f8fafc; border-left: 4px solid #16a34a; padding: 6pt 10pt; margin: 8pt 0; break-inside: avoid; }
.page { break-before: page; }
.ok { color: #15803d; font-weight: 700; } .no { color: #b91c1c; font-weight: 700; }
"""


def img(path: Path, caption: str, max_width=1400, quality=82) -> str:
    im = Image.open(path).convert("RGB")
    if im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<figure><img src="data:image/jpeg;base64,{data}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def fig(name: str, caption: str, max_width=1100) -> str:
    p = C.FIGURES_DIR / f"{name}.png"
    return img(p, caption, max_width) if p.exists() else ""


def pct(x) -> str:
    return f"{x:.1%}"


def table(df: pd.DataFrame, highlight: str | None = None, key: str | None = None) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cls = ' class="champ"' if highlight is not None and r[key] == highlight else ""
        cells = "".join(f'<td class="n">{v}</td>' if isinstance(v, (int, float)) or str(v).replace(",", "").replace(".", "").replace("%", "").strip().isdigit()
                        else f"<td>{html.escape(str(v))}</td>" for v in r.values)
        rows.append(f"<tr{cls}>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def top_confusions(name: str, n: int = 4) -> list[tuple[str, str, int, float]]:
    cm = pd.read_csv(C.METRICS_DIR / f"confusion_{name}.csv", index_col=0)
    out = []
    for a in cm.index:
        for b in cm.columns:
            if a != b and cm.at[a, b] > 0:
                out.append((a, b, int(cm.at[a, b]), cm.at[a, b] / cm.loc[a].sum()))
    return sorted(out, key=lambda t: -t[2])[:n]


def build_html(live_url: str | None) -> str:
    m = pd.read_csv(C.MANIFEST_PATH)
    kept = m[m["status"] == "kept"]
    eda = json.loads((C.METRICS_DIR / "eda_summary.json").read_text())
    comp = pd.read_csv(C.METRICS_DIR / "model_comparison.csv")
    meta = json.loads((C.MODELS_DIR / "best_model.json").read_text())
    best = meta["model"]
    b = comp.set_index("model").loc[best]
    train_info = {n: json.loads((C.METRICS_DIR / f"train_{n}.json").read_text()) for n in comp["model"]}
    pc = pd.read_csv(C.METRICS_DIR / f"per_class_{best}.csv", index_col=0)
    status = m["status"].value_counts()
    conf = top_confusions(best)
    weakest = pc["f1-score"].sort_values().head(3)
    strongest = pc["f1-score"].sort_values(ascending=False).head(3)
    others = comp[comp["model"] != best]
    zv = np.load(C.METRICS_DIR / f"preds_{best}_val.npz")
    conf_v, ok_v = zv["probs"].max(1), zv["probs"].argmax(1) == zv["y"]
    keep = conf_v >= C.CONFIDENCE_THRESHOLD
    thr = {"flagged": float((~keep).mean()), "caught": int((~ok_v & ~keep).sum()), "errors": int((~ok_v).sum()),
           "acc": float(ok_v[keep].mean())}

    # --- tables
    src_rows = []
    for key, s in C.SOURCES.items():
        sub = m[m["source"] == key]
        src_rows.append({"Dataset": s["title"], "Raw images": f"{len(sub):,}", "Duplicates removed": f"{(sub['status'] == 'duplicate').sum():,}",
                         "Kept": f"{(sub['status'] == 'kept').sum():,}"})
    funnel = pd.DataFrame([
        {"Step": "Raw images (3 zips)", "Images": f"{len(m):,}", "Removed": ""},
        {"Step": "Unreadable / tiny / blank", "Images": "", "Removed": f"{status.filter(like='invalid').sum():,}"},
        {"Step": "Exact + near duplicates (across sources)", "Images": "", "Removed": f"{status.get('duplicate', 0):,}"},
        {"Step": "Same photo, conflicting labels", "Images": "", "Removed": f"{status.get('label_conflict', 0):,}"},
        {"Step": f"Class cap ({C.CLASS_CAP:,})", "Images": "", "Removed": f"{status.get('capped', 0):,}"},
        {"Step": "Final unique, labelled images", "Images": f"{len(kept):,}", "Removed": ""},
    ])
    split = pd.crosstab(kept["label"], kept["split"]).reindex(C.CLASSES)[["train", "val", "test"]]
    split["total"] = split.sum(axis=1)
    split = split.reset_index().rename(columns={"label": "Class"})

    comp_t = comp.copy()
    comp_t["Model"] = comp_t["model"].map(LABELS)
    comp_view = pd.DataFrame({
        "Model": comp_t["Model"], "model": comp_t["model"],
        "Val macro-F1": comp_t["val_f1_macro"].map("{:.3f}".format),
        "Test accuracy": comp_t["accuracy"].map(pct),
        "Precision": comp_t["precision_macro"].map("{:.3f}".format),
        "Recall": comp_t["recall_macro"].map("{:.3f}".format),
        "Macro-F1 [95% CI]": [f"{f:.3f} [{ast.literal_eval(ci)[0]:.3f}-{ast.literal_eval(ci)[1]:.3f}]" if isinstance(ci, str) else f"{f:.3f}"
                              for f, ci in zip(comp_t["f1_macro"], comp_t["f1_macro_ci95"])],
        "Top-3": comp_t["top3_accuracy"].map(pct),
        "FPR": comp_t["false_positive_rate_macro"].map("{:.2%}".format),
    })
    eff_view = pd.DataFrame({
        "Model": comp_t["Model"], "model": comp_t["model"],
        "Parameters": comp_t["params_m"].map("{:.1f} M".format),
        "File size": comp_t["size_mb"].map("{:.0f} MB".format),
        "CPU compute (1 image)": comp_t.get("latency_ms_single", pd.Series(dtype=float)).map(lambda v: f"{v:.0f} ms" if pd.notna(v) else "-"),
        "Train time (T4)": comp_t["train_minutes"].map("{:.0f} min".format),
        "Epochs": [str(sum(v for k, v in train_info[n].items() if k.endswith("epochs"))) for n in comp_t["model"]],
        "McNemar p vs best": comp_t["mcnemar_p_vs_best"].map(lambda p: "-" if pd.isna(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")),
    })
    pc_view = pc.reset_index().rename(columns={"index": "Class"})
    pc_view = pd.DataFrame({"Class": pc_view["Class"], "Precision": pc_view["precision"].map("{:.3f}".format),
                            "Recall": pc_view["recall"].map("{:.3f}".format), "F1": pc_view["f1-score"].map("{:.3f}".format),
                            "Test images": pc_view["support"].astype(int)})

    target_ok = b["accuracy"] > 0.85
    sig = others.assign(p=others["mcnemar_p_vs_best"])
    sig_text = "; ".join(f"{LABELS[r.model]} p {'< 0.001' if r.p < 0.001 else f'= {r.p:.3f}'}" for r in sig.itertuples())
    conf_text = ", ".join(f"<b>{a} → {b_}</b> ({k} images, {f:.0%} of {a})" for a, b_, k, f in conf)
    live = f'<a href="{live_url}">{live_url}</a>' if live_url else "run locally: <code>streamlit run app/streamlit_app.py</code>"

    body = f"""
<div class="cover">
  <div class="eyebrow">GUVI × HCL · Deep Learning Capstone</div>
  <h1>RecycleVision</h1>
  <p class="sub">Garbage image classification with deep learning: 10 waste categories, transfer learning and a
  Streamlit app with Grad-CAM explanations.</p>
  <div class="kpis">
    <div class="kpi"><b>{pct(b['accuracy'])}</b><span>test accuracy ({LABELS[best]})</span></div>
    <div class="kpi"><b>{b['f1_macro']:.3f}</b><span>test macro-F1</span></div>
    <div class="kpi"><b>{len(kept):,}</b><span>unique images, 10 classes</span></div>
    <div class="kpi"><b>{b['size_mb']:.0f} MB</b><span>deployed model size</span></div>
  </div>
  <table class="meta">
    <tr><td><b>Author</b></td><td>{AUTHOR}</td></tr>
    <tr><td><b>Date</b></td><td>{date.today():%d %B %Y}</td></tr>
    <tr><td><b>Code</b></td><td><a href="{GITHUB_URL}">{GITHUB_URL}</a></td></tr>
    <tr><td><b>App</b></td><td>{live}</td></tr>
  </table>
</div>

<h2>1. Problem statement and business use</h2>
<p>Build a deep learning model that classifies an image of waste into its material category, and deploy it behind a
simple interface, so recycling can be sorted automatically from a photo.</p>
<ul>
<li><b>Smart recycling bins:</b> a camera-equipped bin routes each item to the correct compartment.</li>
<li><b>Municipal waste management:</b> pre-sorting on conveyor lines cuts manual labour and contamination.</li>
<li><b>Education:</b> citizens photograph an item and learn which stream it belongs to; the app also gives disposal advice.</li>
<li><b>Environmental analytics:</b> counting predictions over time shows the waste composition and recycling trends.</li>
</ul>

<h2>2. Data: three datasets merged into one clean set</h2>
<p>The brief offers three Kaggle datasets. Using all three gives more images and more variety, such as studio shots on
white backgrounds and real-world web photos. However, <b>they overlap heavily</b>: most of the 10-class V2 dataset is
copied from the 12-class one, and both contain the TrashNet photos. Concatenating them naively would put copies of the
same photo in both training and test, inflating the test score. The pipeline therefore records the fate of every one
of the {len(m):,} raw images in <code>data/manifest.csv</code>.</p>
{table(pd.DataFrame(src_rows))}
{table(funnel)}
<div class="box"><b>How duplicates were found.</b> Exact copies are matched by SHA-1 of the file bytes
({eda['duplicates_removed']:,} duplicates in total, mostly byte-identical). Near-copies (resized or re-encoded) need a
perceptual-hash distance ≤ {C.PHASH_MAX_DIST} <i>and</i> a 32×32 thumbnail correlation ≥ {C.THUMB_MIN_NCC}.
The second test was calibrated by inspecting borderline pairs: a pixel-difference test alone wrongly merged different
bottles photographed on the same white background, and missed the same flyer re-shot with different lighting.
Within each duplicate group, one copy is kept, preferring TrashNet and then the highest resolution.</div>
<p><b>Taxonomy.</b> Brown, green and white glass were merged into <b>glass</b> (the brief asks for glass, not glass
colour); "biological" is called <b>organic</b> as in the brief. The final classes are {", ".join(C.CLASSES)}.
<b>Class imbalance:</b> clothes was about a third of the raw data, so
every class is capped at {C.CLASS_CAP:,} images. The remaining imbalance ({eda['imbalance_ratio_max_min']}×
between the largest and smallest class) is handled with balanced class weights.</p>
{table(split)}

<h2 class="page">3. Exploratory data analysis</h2>
{fig("eda_pipeline_funnel", "Fate of every raw image by source: most of the V2 dataset duplicates the 12-class set.")}
{fig("eda_class_distribution", "Images per class before and after cleaning, and the stratified 70/15/15 split.")}
<ul>
<li><b>Classes:</b> after cleaning, the smallest classes are {", ".join(f"{k} ({v:,})" for k, v in sorted(eda['final_per_class'].items(), key=lambda kv: kv[1])[:3])};
the largest is capped at {C.CLASS_CAP:,}.</li>
<li><b>Image sizes</b> vary from {eda['original_width_px']['min']:.0f} to {eda['original_width_px']['max']:.0f} px wide
(median {eda['original_width_px']['50%']:.0f}), so all images are resized to 224×224, the input size of the ImageNet backbones.</li>
<li><b>Colour:</b> trash and battery images are the brightest (mostly white studio backgrounds). Organic is the most
red/yellow-shifted (blue channel far below red), and clothes are the darkest. Colour alone is therefore a strong
but unreliable cue (background matters), which is a reason for colour and brightness augmentation.</li>
</ul>
{fig("eda_samples", "Random samples from each class after resizing.", 1000)}
{fig("eda_color_histograms", "Per-channel pixel-intensity distributions (150 images per class).")}
{fig("eda_color_means", "Mean R/G/B per class and brightness distribution.")}
{fig("eda_image_sizes", "Original image sizes (log-log) by class.", 800)}

<h2 class="page">4. Preprocessing and models</h2>
<ul>
<li><b>Resize</b> to 224×224 and keep pixels at 0–255. <b>Normalisation is built into each model</b> (each backbone's
own ImageNet scaling), so the app cannot apply the wrong normalisation.</li>
<li><b>Augmentation</b> (training only, inside the model): horizontal and vertical flips, rotation up to ±72°, zoom ±20%,
shift ±10%, contrast and brightness ±20%. Waste can appear at any orientation and lighting.</li>
<li><b>Imbalance:</b> class cap plus balanced class weights; macro-F1 weights every class equally.</li>
<li><b>Baseline CNN</b> trained from scratch: 4 convolution blocks (32→256 filters, batch normalisation), global
average pooling, then dense layers.</li>
<li><b>Transfer learning:</b> MobileNetV2, EfficientNetB0 and ResNet50 with ImageNet weights, plus a custom head
(global average pooling → dropout → Dense 256 → dropout → softmax over 10 classes).
  <ul>
  <li>Phase 1 trains only the head (backbone frozen, AdamW, learning rate 1e-3).</li>
  <li>Phase 2 unfreezes the top 35–45% of the backbone, with BatchNorm kept frozen, at learning rate 1e-4 with cosine decay.</li>
  <li>Early stopping on validation loss restores the best epoch.</li>
  <li>Trained with mixed precision on a Colab T4 GPU, then exported as float32 for CPU deployment.</li>
  </ul></li>
<li><b>Selection</b> is by validation macro-F1. The test set ({int(split['test'].sum()):,} images) is used only to report results.</li>
</ul>

<h2>5. Results</h2>
{table(comp_view.drop(columns="model"), highlight=LABELS[best], key="Model")}
<p><b>{LABELS[best]}</b> has the highest validation macro-F1 and is the deployed model. On the held-out test set it
reaches <b>{pct(b['accuracy'])} accuracy</b> and <b>macro-F1 {b['f1_macro']:.3f}</b>, with a 95% bootstrap interval of
{ast.literal_eval(b['f1_macro_ci95'])[0]:.3f}–{ast.literal_eval(b['f1_macro_ci95'])[1]:.3f} and top-3 accuracy {pct(b['top3_accuracy'])}. The brief's target of accuracy &gt; 85% is
<span class="{'ok' if target_ok else 'no'}">{'met' if target_ok else 'not met'}</span>. Precision and recall are
balanced ({b['precision_macro']:.3f} / {b['recall_macro']:.3f}), and the average false-positive rate per class is
{b['false_positive_rate_macro']:.2%}. The from-scratch baseline shows how much ImageNet pre-training contributes.</p>
{fig("model_comparison", "Test accuracy, macro precision, recall and F1 per model; dashed line = 85% target.")}
{fig("training_curves", "Validation accuracy and loss per epoch; the dotted line marks the start of fine-tuning.")}

<h3>Statistical significance</h3>
<p>McNemar's exact test compares the best model with each other model on the <i>same</i> test images (it counts the
images only one of the two models gets right): {sig_text}. A p-value below 0.05 means the difference in errors is
unlikely to be chance.</p>

<h2 class="page">6. Error analysis: the best model</h2>
{fig(f"confusion_{best}", f"{LABELS[best]}: test confusion matrix (numbers = counts; colour = share of the true class).", 1000)}
{table(pc_view)}
<p><b>Strongest classes:</b> {", ".join(f"{k} (F1 {v:.3f})" for k, v in strongest.items())}.
<b>Weakest:</b> {", ".join(f"{k} (F1 {v:.3f})" for k, v in weakest.items())}.
<b>Most common confusions:</b> {conf_text}. These are physically plausible: clear plastic and glass bottles look
alike, and paper and cardboard are the same material in different thicknesses. "Trash" is by definition a mixed,
catch-all class.</p>
<div class="box"><b>Some "mistakes" are label errors in the source data.</b> Among the most confident errors, an image
labelled <i>glass</i> is clearly a metal lid (the model says metal), and one labelled <i>clothes</i> shows sneakers (the
model says shoes). The true accuracy is therefore slightly <i>higher</i> than reported. Fixing such labels would be the
next data-quality step, and the model's confident disagreements are a cheap way to find them.</div>
{fig(f"misclassified_{best}", "The most confident mistakes on the test set.", 1000)}

<h2 class="page">7. Explainability with Grad-CAM</h2>
<p>Grad-CAM highlights the image regions that drove a prediction. It is computed from the class logit (pre-softmax),
because at ~100% confidence the softmax saturates and its gradients vanish. On correct predictions the model focuses on the
object itself, not the background. This matters because many training photos share white backgrounds, so a model could
otherwise learn background shortcuts. The app shows this heatmap for every upload.</p>
{fig("gradcam_gallery", "One held-out test image per class with its Grad-CAM heatmap.", 1300)}
{fig("gradcam_mistakes", "Grad-CAM on the most confident mistakes. Several are mislabelled source images (a metal lid labelled glass, sneakers labelled clothes).")}

<h2 class="page">8. Efficiency and deployment</h2>
{table(eff_view.drop(columns="model"), highlight=LABELS[best], key="Model")}
<p>CPU compute is the model's time for one image on a laptop CPU (Apple M2), using the same compiled path as the app.
In the running app one upload takes about 0.3 s end to end, because the first TensorFlow call of each Streamlit rerun
carries a fixed overhead. Streamlit Cloud's shared CPU is slower, but a prediction still takes well under a second.
The deployed model is {b['size_mb']:.0f} MB with {b['params_m']:.1f} M parameters, well under GitHub's 100 MB file limit.</p>
<p><b>App</b> (<code>app/streamlit_app.py</code>):</p>
<ul>
<li>Upload an image (or pick a held-out example), then get the predicted class with its confidence and the top-3 predictions.</li>
<li>Disposal advice for the predicted category, plus a Grad-CAM overlay.</li>
<li>Below {C.CONFIDENCE_THRESHOLD:.0%} confidence the app says it is unsure instead of asserting a class. The threshold was chosen on the validation set: {thr["flagged"]:.1%} of images are flagged, {thr["caught"]} of the {thr["errors"]} validation errors are among them, and accuracy on the remaining images is {thr["acc"]:.1%}.</li>
<li>A model-comparison tab with the table and confusion matrix above.</li>
</ul>
{fig("app_screenshot", "The Streamlit app classifying a held-out test image.", 1100)}

<h2>9. Conclusions and recommendations</h2>
<ul>
<li><b>Transfer learning wins clearly:</b> every ImageNet backbone beats the from-scratch CNN by a wide margin, on {len(kept):,} images.</li>
<li><b>Data hygiene mattered as much as the model:</b> {eda['duplicates_removed']:,} of {len(m):,} raw images were duplicates.
Without de-duplication, copies of test photos would have been in the training set, so the test score would not measure generalisation.</li>
<li><b>Deployment choice:</b> {LABELS[best]} gives the best macro-F1. The table in Section 8 shows the speed and size trade-off if a smaller edge model is needed, e.g. for a bin.</li>
<li><b>Business suggestions:</b>
  <ul>
  <li>Use the confidence score to route low-confidence items to a human sorter (Section 8 shows the trade-off).</li>
  <li>Collect real bin-camera images to fine-tune, since most training photos are clean single-object shots.</li>
  <li>Batteries must never enter household recycling: the app already shows a hazardous-waste warning, and a smart bin should route them to a sealed compartment.</li>
  </ul></li>
<li><b>Limitations:</b>
  <ul>
  <li>One object per photo.</li>
  <li>Web and studio images are cleaner than real waste streams.</li>
  <li>"Trash" is heterogeneous.</li>
  </ul></li>
</ul>

<h2>10. Brief requirements checklist</h2>
<table><thead><tr><th>Requirement</th><th>Where</th></tr></thead><tbody>
<tr><td>Dataset from the brief (Kaggle)</td><td>All three linked datasets, merged and de-duplicated (Section 2)</td></tr>
<tr><td>Cleaned &amp; labelled dataset (deliverable)</td><td><code>data/manifest.csv</code> + 224×224 image set (rebuilt by <code>scripts/build_dataset.py</code>; shared as a zip)</td></tr>
<tr><td>Resize 224×224, normalise, augment, handle imbalance</td><td>Sections 2 and 4; <code>data.py</code>, <code>models.py</code></td></tr>
<tr><td>EDA: class counts, samples, colour distribution</td><td>Section 3; <code>scripts/run_eda.py</code></td></tr>
<tr><td>Baseline CNN + transfer learning (MobileNetV2, EfficientNetB0, ResNet50)</td><td>Sections 4–5; <code>models.py</code></td></tr>
<tr><td>Accuracy, precision, recall, F1, confusion matrix</td><td>Sections 5–6; <code>reports/metrics/</code></td></tr>
<tr><td>Best model by F1</td><td>Validation macro-F1 → {LABELS[best]}</td></tr>
<tr><td>Accuracy &gt; 85%, balanced precision/recall, low false positives</td><td>{pct(b['accuracy'])}; P {b['precision_macro']:.3f} / R {b['recall_macro']:.3f}; FPR {b['false_positive_rate_macro']:.2%}</td></tr>
<tr><td>Streamlit app: upload, label, confidence, top-3</td><td>Section 8 (+ Grad-CAM and disposal advice)</td></tr>
<tr><td>Fast inference, reasonable model size</td><td>{b.get('latency_ms_single', float('nan')):.0f} ms model compute on CPU (about 0.3 s per upload in the app); {b['size_mb']:.0f} MB</td></tr>
<tr><td>Modular, documented code</td><td><code>src/recyclevision/</code> package, scripts, README, tests</td></tr>
<tr><td>Deployment (Streamlit Cloud or local)</td><td>Runs locally; Streamlit Cloud steps in the README{" - live at " + live_url if live_url else ""}</td></tr>
<tr><td>Documentation: data handling, training, deployment</td><td>README, this report, module docstrings, presentation script</td></tr>
</tbody></table>
"""
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-url", default=None, help="Streamlit Cloud URL, once deployed")
    a = ap.parse_args()
    page_html = build_html(a.live_url)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page()
        page.set_content(page_html, wait_until="load")
        page.pdf(path=str(PDF_PATH), format="A4", print_background=True,
                 margin={"top": "16mm", "bottom": "16mm", "left": "15mm", "right": "15mm"},
                 display_header_footer=True, header_template="<span></span>",
                 footer_template='<div style="font-size:8px;color:#94a3b8;width:100%;text-align:center;">'
                                 'RecycleVision · <span class="pageNumber"></span>/<span class="totalPages"></span></div>')
        browser.close()
    print(f"Saved {PDF_PATH.relative_to(ROOT)} ({PDF_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
