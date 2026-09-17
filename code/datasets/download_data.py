"""Download the public Music Notes dataset without storing credentials in code."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DATASET_HANDLE = "kishanj/music-notes-datasets"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw"


def download(output: Path = DEFAULT_OUTPUT) -> Path:
    """Download the Kaggle dataset with kagglehub (or explain the requirement)."""
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed. Run `pip install kagglehub` or install "
            "the project's requirements, then retry."
        ) from exc

    output.mkdir(parents=True, exist_ok=True)
    try:
        cache_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    except Exception as exc:
        raise RuntimeError(
            "Kaggle download failed. Authenticate outside this repository (for example, "
            "set KAGGLE_API_TOKEN in your shell or configure Kaggle credentials), then run "
            "`python code/datasets/download_data.py`. Credentials must never be committed."
        ) from exc

    # Copy rather than use a cache path so every downstream stage has a stable input.
    for source in cache_path.rglob("*"):
        if source.is_file():
            target = output / source.relative_to(cache_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = download(args.output)
    print(f"Dataset is available in {result}")
