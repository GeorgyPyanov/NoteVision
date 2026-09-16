"""Train and evaluate a CPU-friendly HOG + multinomial Logistic Regression model."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# `python code/models/train.py` otherwise resolves the stdlib `code` module
# before this repository package because the script directory is first on path.
_PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_IMPORTS))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score)

from code.datasets.preprocess import (CLASS_DISPLAY_NAMES, CLASS_NAMES, IMAGE_SIZE,
                                      PROCESSED_DIR, RANDOM_STATE)
from code.models.features import DEFAULT_HOG_PARAMS, extract_batch, extract_hog_features

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
MLRUNS_DIR = REPO_ROOT / "mlruns"


def _absolute_paths(frame: pd.DataFrame) -> list[Path]:
    return [REPO_ROOT / value for value in frame["image_path"].tolist()]


def _validate_manifests(train: pd.DataFrame, test: pd.DataFrame) -> None:
    required = {"image_path", "label", "sha256"}
    if not required.issubset(train.columns) or not required.issubset(test.columns):
        raise ValueError(f"Manifests need columns {sorted(required)}")
    if set(train["label"]) - set(CLASS_NAMES) or set(test["label"]) - set(CLASS_NAMES):
        raise ValueError("Manifest contains an unsupported label")
    overlap = set(train["sha256"]) & set(test["sha256"])
    if overlap:
        raise ValueError(f"Duplicate image hashes cross the train/test boundary: {len(overlap)}")
    missing = [path for path in _absolute_paths(pd.concat([train, test])) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Processed image is missing: {missing[0]}")


def train_and_evaluate(processed_dir: Path = PROCESSED_DIR, models_dir: Path = MODELS_DIR) -> dict[str, Any]:
    """Fit, evaluate, and package the model; log the same run to local MLflow."""
    processed_dir, models_dir = processed_dir.resolve(), models_dir.resolve()
    train = pd.read_csv(processed_dir / "train.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    _validate_manifests(train, test)
    models_dir.mkdir(parents=True, exist_ok=True)
    log_dir = models_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(filename=log_dir / f"train_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.log",
                        level=logging.INFO, force=True, format="%(asctime)s %(levelname)s %(message)s")

    X_train = extract_batch(_absolute_paths(train), DEFAULT_HOG_PARAMS)
    X_test = extract_batch(_absolute_paths(test), DEFAULT_HOG_PARAMS)
    y_train, y_test = train["label"].to_numpy(), test["label"].to_numpy()
    counts = train["label"].value_counts()
    class_weight = "balanced" if counts.max() / counts.min() > 1.2 else None
    model = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=1000,
        class_weight=class_weight, random_state=RANDOM_STATE, n_jobs=None,
    )
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start
    predictions = model.predict(X_test)
    one_start = time.perf_counter()
    model.predict_proba(X_test[:1])
    inference_ms = (time.perf_counter() - one_start) * 1000
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "training_seconds": training_seconds,
        "single_image_inference_ms": inference_ms,
        "train_samples": int(len(train)), "test_samples": int(len(test)),
    }
    config = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "class_names": list(CLASS_NAMES), "class_display_names": CLASS_DISPLAY_NAMES,
        "image_size": IMAGE_SIZE, "normalization": "pixel / 255.0",
        "hog_params": DEFAULT_HOG_PARAMS, "random_state": RANDOM_STATE,
        "class_weight": class_weight,
    }
    bundle = {"model": model, "config": config}
    model_path = models_dir / "model.joblib"
    joblib.dump(bundle, model_path)
    with (models_dir / "model_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    with (models_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    figure, axis = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=list(CLASS_NAMES),
                                            xticks_rotation=35, colorbar=False, ax=axis)
    figure.tight_layout()
    matrix_path = models_dir / "confusion_matrix.png"
    figure.savefig(matrix_path, dpi=160)
    plt.close(figure)

    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("notevision")
    with mlflow.start_run(run_name=f"hog-logreg-{config['model_version']}"):
        mlflow.log_params({"classifier": "LogisticRegression", "class_weight": str(class_weight),
                           "random_state": RANDOM_STATE, "image_size": IMAGE_SIZE,
                           **{f"hog_{key}": str(value) for key, value in DEFAULT_HOG_PARAMS.items()}})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(matrix_path), artifact_path="evaluation")
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(models_dir / "model_config.json"), artifact_path="model")
    logging.info("Training complete: %s", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()
    train_and_evaluate(args.processed_dir, args.models_dir)
