"""Generate the Colab training notebook.

  python scripts/make_colab_notebook.py              -> notebooks/train_colab.ipynb (clones the GitHub repo)
  python scripts/make_colab_notebook.py --bundle X   -> X with the code + manifest embedded (no GitHub needed)
"""
import argparse
import base64
import io
import sys
import tarfile
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recyclevision import config as C  # noqa: E402

REPO = "https://github.com/aryaman2213/recyclevision.git"


def bundle_b64() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted([*ROOT.glob("src/recyclevision/*.py"), *ROOT.glob("scripts/*.py"), C.MANIFEST_PATH]):
            tar.add(p, arcname=str(p.relative_to(ROOT)))
    return base64.b64encode(buf.getvalue()).decode()


def notebook(bundle: bool) -> nbf.NotebookNode:
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    downloads = "\n".join(
        f'!curl -sSL -o /content/raw/{s["zip_name"]} https://www.kaggle.com/api/v1/datasets/download/{s["kaggle"]}'
        for s in C.SOURCES.values())
    if bundle:
        get_code = code(f'import base64, io, tarfile, os\nos.makedirs("/content/recyclevision", exist_ok=True)\n'
                        f'B = "{bundle_b64()}"\n'
                        'tarfile.open(fileobj=io.BytesIO(base64.b64decode(B))).extractall("/content/recyclevision")\n'
                        '!ls -R /content/recyclevision | head -40')
    else:
        get_code = code(f"!git clone -q {REPO} /content/recyclevision 2>/dev/null || git -C /content/recyclevision pull -q\n"
                        "!ls /content/recyclevision")
    cells = [
        md("# RecycleVision - GPU training on Google Colab\n"
           "Runtime -> Change runtime type -> **T4 GPU**, then Runtime -> Run all.\n\n"
           "Steps: get the code -> download the 3 Kaggle datasets -> rebuild the exact de-duplicated "
           "dataset from `data/manifest.csv` -> train 4 models -> evaluate -> zip results. "
           "Trained models are written straight to Google Drive, so after a disconnect just Run all again: "
           "finished models are skipped.\n\n"
           "**Note:** the repository already contains the trained models and results from the original run. "
           "Running this notebook retrains everything from scratch (about 1.5 h on a free T4). Afterwards, run "
           "`scripts/evaluate_models.py` once more on a CPU machine to measure CPU latency (Colab measures GPU)."),
        code("!nvidia-smi -L\nimport tensorflow as tf, keras, sys\n"
             "print(sys.version.split()[0], 'TF', tf.__version__, 'Keras', keras.__version__, tf.config.list_physical_devices('GPU'))"),
        md("## 1. Code and manifest"),
        get_code,
        md("## 2. Datasets (public Kaggle downloads, size-checked against the manifest's versions)"),
        code(f"!mkdir -p /content/raw\n{downloads}\n!ls -l /content/raw"),
        code("%cd /content/recyclevision\n!RECYCLEVISION_RAW=/content/raw python scripts/build_dataset.py"),
        md("## 3. Persist outputs to Google Drive (approve the Drive pop-up)"),
        code("from google.colab import drive\ndrive.mount('/content/drive')\n"
             "OUT = '/content/drive/MyDrive/recyclevision_outputs'\n"
             "!mkdir -p {OUT}/models {OUT}/reports && rm -rf models reports && ln -s {OUT}/models models && ln -s {OUT}/reports reports\n"
             "!ls -l models/ reports/"),
        md("## 4. Train (one cell per model, so progress is visible and resumable)"),
        *[code(f"!python scripts/train_models.py --models {n}") for n in C.TRAIN_ORDER],
        md("## 5. Evaluate on the held-out test set"),
        code("!python scripts/evaluate_models.py"),
        md("## 6. Download everything"),
        code("!cd /content && rm -f recyclevision_results.zip && zip -qr recyclevision_results.zip "
             "recyclevision/models/ recyclevision/reports/ && ls -lh recyclevision_results.zip\n"
             "from google.colab import files\nfiles.download('/content/recyclevision_results.zip')"),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update({"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                        "kernelspec": {"name": "python3", "display_name": "Python 3"}})
    return nb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, help="write a self-contained notebook to this path")
    a = ap.parse_args()
    out = a.bundle or ROOT / "notebooks" / "train_colab.ipynb"
    nbf.write(notebook(bool(a.bundle)), out)
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")
