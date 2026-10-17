"""Step 2: write data/processed/<split>/<class>/<id>.jpg (224x224) for every image the manifest keeps."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from recyclevision import config as C, data  # noqa: E402

if __name__ == "__main__":
    manifest = pd.read_csv(C.MANIFEST_PATH)
    counts = data.build_processed(manifest)
    n_files = sum(1 for _ in C.PROCESSED_DIR.rglob("*.jpg"))
    assert n_files == counts.sum(), f"{n_files} files on disk, manifest keeps {counts.sum()}"
    print(counts.unstack(0))
    print(f"OK: {n_files:,} images in {C.PROCESSED_DIR}")
