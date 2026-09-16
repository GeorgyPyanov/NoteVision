from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from code.datasets.preprocess import IMAGE_SIZE, infer_label, preprocess_image

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("directory", "expected"), [
    ("Whole", "whole_note"), ("Half", "half_note"), ("Quarter", "quarter_note"),
    ("Eight", "eighth_note"), ("Sixteenth", "sixteenth_note"),
])
def test_all_note_labels_are_detected(directory, expected, tmp_path):
    raw_dir = tmp_path / "raw"
    image_path = raw_dir / "Notes" / directory / "example.jpg"
    image_path.parent.mkdir(parents=True)
    assert infer_label(image_path, raw_dir) == expected


def test_preprocess_image_has_canonical_size():
    image = Image.new("L", (20, 40), color=255)
    image.putpixel((10, 20), 0)
    processed = preprocess_image(image)
    assert processed.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert processed.dtype == np.float32
    assert 0 <= processed.min() <= processed.max() <= 1


def test_preprocess_rejects_blank_image():
    with pytest.raises(ValueError, match="blank"):
        preprocess_image(Image.new("L", (32, 32), color=255))


def test_train_and_test_manifests_have_no_shared_hashes():
    train = pd.read_csv(REPO_ROOT / "data" / "processed" / "train.csv")
    test = pd.read_csv(REPO_ROOT / "data" / "processed" / "test.csv")
    assert not (set(train["sha256"]) & set(test["sha256"]))


def _load_dag_module():
    pytest.importorskip("airflow")
    spec = importlib.util.spec_from_file_location(
        "notevision_pipeline_test", REPO_ROOT / "services" / "airflow" / "dags" / "notevision_pipeline.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dag_import_and_regression_contract(monkeypatch):
    monkeypatch.setenv("NOTEVISION_ROOT", str(REPO_ROOT))
    module = _load_dag_module()
    dag = module.dag
    assert callable(module.run_training)
    assert module.train_model_task.python_callable is module.train_task
    assert "train_and_evaluate" not in module.__dict__  # operator must never shadow imported callable
    assert dag.schedule_interval == "*/15 * * * *"
    assert dag.catchup is False
    assert dag.max_active_runs == 1
    expected = ["validate_raw_data", "preprocess_images", "train_and_evaluate", "deploy_services", "check_api_health"]
    assert list(dag.task_ids) == expected
    for upstream, downstream in zip(expected, expected[1:]):
        assert dag.get_task(downstream).upstream_task_ids == {upstream}


def test_train_task_calls_unshadowed_training_callable(monkeypatch):
    monkeypatch.setenv("NOTEVISION_ROOT", str(REPO_ROOT))
    module = _load_dag_module()
    calls = []
    monkeypatch.setattr(module, "run_training", lambda *args: calls.append(args) or {"accuracy": 1.0})
    module.train_task()
    assert len(calls) == 1
