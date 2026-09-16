"""Clean, standardize, and split the Music Notes image dataset.

The `preprocess_image` function is the single image transformation used by both
training and inference. Raw source files are never modified; invalid, unlabelled,
duplicate, and justified outlier files are excluded from the processed manifests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
IMAGE_SIZE = 64
RANDOM_STATE = 42
CLASS_NAMES = (
    "whole_note", "half_note", "quarter_note", "eighth_note", "sixteenth_note",
)
CLASS_DISPLAY_NAMES = {
    "whole_note": "Whole Note", "half_note": "Half Note",
    "quarter_note": "Quarter Note", "eighth_note": "Eighth Note",
    "sixteenth_note": "Sixteenth Note",
}
CLASS_ALIASES = {
    "whole_note": {"whole note", "whole", "whole_note"},
    "half_note": {"half note", "half", "half_note"},
    "quarter_note": {"quarter note", "quarter", "quarter_note"},
    "eighth_note": {"eighth note", "eighth", "eight", "eigth", "eighth_note"},
    "sixteenth_note": {"sixteenth note", "sixteenth", "sixteenth_note"},
}


def _normalise_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def infer_label(path: Path, raw_dir: Path) -> str | None:
    """Infer a canonical label from any parent directory or filename."""
    candidates = [part for part in path.relative_to(raw_dir).parts[:-1]] + [path.stem]
    normalised = " ".join(_normalise_text(part) for part in candidates)
    for label, aliases in CLASS_ALIASES.items():
        if any(alias.replace("_", " ") in normalised for alias in aliases):
            return label
    return None


def _content_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Find non-white content while preserving antialiased dark pixels."""
    array = np.asarray(image, dtype=np.uint8)
    mask = array < 245
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def preprocess_image(image_or_path: Image.Image | str | Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Return a cropped, padded 64x64 grayscale image normalized to [0, 1]."""
    close_after = False
    if isinstance(image_or_path, (str, Path)):
        image = Image.open(image_or_path)
        close_after = True
    else:
        image = image_or_path.copy()
    try:
        image = ImageOps.exif_transpose(image).convert("L")
        bbox = _content_bbox(image)
        if bbox is None:
            raise ValueError("image is blank after grayscale conversion")
        image = image.crop(bbox)
        width, height = image.size
        side = max(width, height)
        # 8% border avoids touching an edge after resizing.
        padding = max(1, round(side * 0.08))
        canvas = Image.new("L", (side + 2 * padding, side + 2 * padding), color=255)
        canvas.paste(image, ((canvas.width - width) // 2, (canvas.height - height) // 2))
        resized = canvas.resize((image_size, image_size), Image.Resampling.LANCZOS)
        return np.asarray(resized, dtype=np.float32) / 255.0
    finally:
        image.close()
        if close_after:
            # PIL image opened above is already safely closed; retained for clarity.
            pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_outlier(records: list[dict[str, Any]]) -> set[Path]:
    """Remove only blank/nearly-full images and extreme robust-MAD shape/content outliers."""
    rejected: set[Path] = set()
    for label in CLASS_NAMES:
        group = [row for row in records if row["label"] == label]
        if len(group) < 10:
            continue
        for field in ("width", "height", "ink_fraction"):
            values = np.array([row[field] for row in group], dtype=float)
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            # Content fractions near zero/full are objectively unusable regardless of MAD.
            if field == "ink_fraction":
                for row in group:
                    if row[field] < 0.001 or row[field] > 0.98:
                        rejected.add(row["path"])
            if mad <= 0:
                continue
            robust_z = 0.6745 * np.abs(values - median) / mad
            for row, z_score in zip(group, robust_z):
                if z_score > 6.0:
                    rejected.add(row["path"])
    return rejected


def validate_raw_data(raw_dir: Path = RAW_DIR) -> dict[str, int]:
    """Fail early unless all five labels and at least one readable image are present.

    This is intentionally read-only: cleaning decisions are recorded in the
    preprocessing report rather than destroying the original downloaded corpus.
    """
    raw_dir = raw_dir.resolve()
    files = [path for path in raw_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    by_class: Counter[str] = Counter()
    readable = 0
    for path in files:
        label = infer_label(path, raw_dir)
        if label is None:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            readable += 1
            by_class[label] += 1
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
            continue
    missing = set(CLASS_NAMES) - set(by_class)
    if not files:
        raise RuntimeError(f"No image files found under {raw_dir}")
    if missing:
        raise RuntimeError(f"Raw data validation failed; labels not found: {sorted(missing)}")
    if readable < 10:
        raise RuntimeError("Raw data validation failed; fewer than ten readable labelled images")
    summary = {"found": len(files), "readable_labelled": readable, **dict(by_class)}
    logging.info("Raw data validation passed: %s", summary)
    return summary


def run_preprocessing(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> dict[str, Any]:
    """Create processed PNGs and stratified manifests from raw image files."""
    raw_dir, processed_dir = raw_dir.resolve(), processed_dir.resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_dir}")
    files = sorted(path for path in raw_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise RuntimeError(f"No PNG/JPG/JPEG files found under {raw_dir}")

    log_dir = processed_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"preprocess_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.log"
    logging.basicConfig(filename=log_path, level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Found %d candidate image files", len(files))
    report: Counter[str] = Counter(found=len(files))
    records: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for path in files:
        label = infer_label(path, raw_dir)
        if label is None:
            report["unlabelled"] += 1
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                grayscale = ImageOps.exif_transpose(image).convert("L")
                bbox = _content_bbox(grayscale)
                if bbox is None:
                    report["blank"] += 1
                    continue
                width, height = grayscale.size
                ink_fraction = float((np.asarray(grayscale, dtype=np.uint8) < 245).mean())
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
            report["corrupt"] += 1
            continue
        file_hash = _sha256(path)
        if file_hash in hashes:
            report["duplicates"] += 1
            continue
        hashes.add(file_hash)
        records.append({"path": path, "source_path": str(path.relative_to(REPO_ROOT)), "label": label,
                        "sha256": file_hash, "width": width, "height": height, "ink_fraction": ink_fraction})

    outliers = _is_outlier(records)
    records = [row for row in records if row["path"] not in outliers]
    report["outliers"] = len(outliers)
    if not records:
        raise RuntimeError("No usable images remained after validation")
    class_counts = Counter(row["label"] for row in records)
    missing = set(CLASS_NAMES) - set(class_counts)
    if missing:
        raise RuntimeError(f"Missing required classes after cleaning: {sorted(missing)}")
    if min(class_counts.values()) < 2:
        raise RuntimeError("Each class needs at least two valid unique images for stratified split")

    images_dir = processed_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)
    for index, row in enumerate(records):
        array = preprocess_image(row["path"])
        target = images_dir / row["label"] / f"{index:06d}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.round(array * 255).astype(np.uint8), mode="L").save(target)
        row["image_path"] = str(target.relative_to(REPO_ROOT)).replace("\\", "/")
        row.pop("path")

    labels = [row["label"] for row in records]
    train_rows, test_rows = train_test_split(records, test_size=0.2, random_state=RANDOM_STATE,
                                             stratify=labels)
    fields = ["image_path", "label", "sha256", "source_path", "width", "height", "ink_fraction"]
    for name, rows in (("train.csv", train_rows), ("test.csv", test_rows)):
        with (processed_dir / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    report["saved"] = len(records)
    report["train"] = len(train_rows)
    report["test"] = len(test_rows)
    report["per_class"] = dict(sorted(class_counts.items()))
    report["random_state"] = RANDOM_STATE
    report["preprocessing"] = {"image_size": IMAGE_SIZE, "grayscale": True, "crop_threshold": 245,
                                "normalization": "pixel / 255.0"}
    with (processed_dir / "preprocessing_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    logging.info("Preprocessing complete: %s", dict(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return dict(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    run_preprocessing(args.raw_dir, args.processed_dir)
