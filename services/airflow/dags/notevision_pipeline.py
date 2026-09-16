"""Airflow DAG that executes the real NoteVision pipeline every five minutes."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(os.environ.get("NOTEVISION_ROOT", Path(__file__).resolve().parents[3])).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.datasets.preprocess import run_preprocessing, validate_raw_data  # noqa: E402
from code.models.train import train_and_evaluate  # noqa: E402

COMPOSE_FILE = PROJECT_ROOT / "code" / "deployment" / "docker-compose.yml"


def _timestamped(message: str) -> None:
    logging.info("[%s] %s", datetime.now().astimezone().isoformat(), message)


def validate_task() -> None:
    _timestamped("Starting raw data validation")
    _timestamped(f"Validation result: {validate_raw_data(PROJECT_ROOT / 'data' / 'raw')}")


def preprocess_task() -> None:
    _timestamped("Starting image preprocessing")
    _timestamped(f"Preprocessing result: {run_preprocessing(PROJECT_ROOT / 'data' / 'raw', PROJECT_ROOT / 'data' / 'processed')}")


def train_task() -> None:
    _timestamped("Starting model training and evaluation")
    _timestamped(f"Training result: {train_and_evaluate(PROJECT_ROOT / 'data' / 'processed', PROJECT_ROOT / 'models')}")


def deploy_services_task() -> None:
    _timestamped("Building/restarting FastAPI and Streamlit through Docker Compose")
    subprocess.run(["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--build", "--remove-orphans"],
                   cwd=PROJECT_ROOT, check=True)


def check_api_health_task() -> None:
    api_url = os.environ.get("NOTE_VISION_HEALTH_URL", "http://localhost:8000/health")
    _timestamped(f"Checking API health at {api_url}")
    last_error: Exception | None = None
    for _ in range(12):
        try:
            response = requests.get(api_url, timeout=5)
            response.raise_for_status()
            if response.json().get("status") == "ok":
                _timestamped("API healthcheck passed")
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(5)
    raise RuntimeError(f"API healthcheck did not pass: {last_error}")


with DAG(
    dag_id="notevision_pipeline",
    description="Clean notes, train HOG classifier, deploy and verify API",
    start_date=datetime(2025, 1, 1),
    schedule="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "notevision", "retries": 0},
    tags=["notevision", "mlops"],
) as dag:
    validate_raw_data_task = PythonOperator(task_id="validate_raw_data", python_callable=validate_task)
    preprocess_images = PythonOperator(task_id="preprocess_images", python_callable=preprocess_task)
    train_and_evaluate = PythonOperator(task_id="train_and_evaluate", python_callable=train_task)
    deploy_services = PythonOperator(task_id="deploy_services", python_callable=deploy_services_task)
    check_api_health = PythonOperator(task_id="check_api_health", python_callable=check_api_health_task)
    validate_raw_data_task >> preprocess_images >> train_and_evaluate >> deploy_services >> check_api_health
