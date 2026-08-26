"""Regression checks for the queued one-click RVC training job."""
from __future__ import annotations

from pathlib import Path

from supabase_worker import WORKER_JOB_TYPES, default_rvc_trainer


def test_worker_claims_rvc_training_jobs() -> None:
    assert "local_worker_rvc_train" in WORKER_JOB_TYPES


def test_worker_defaults_to_checked_in_applio_adapter() -> None:
    assert default_rvc_trainer().endswith("scripts/applio_rvc_train.sh")


def test_applio_adapter_contains_full_training_pipeline() -> None:
    adapter = Path(__file__).parent / "scripts" / "applio_rvc_train.sh"
    text = adapter.read_text(encoding="utf-8")
    for command in ("preprocess", "extract", "train", "index"):
        assert f"run_applio {command}" in text


if __name__ == "__main__":
    test_worker_claims_rvc_training_jobs()
    test_worker_defaults_to_checked_in_applio_adapter()
    test_applio_adapter_contains_full_training_pipeline()
    print("RVC training job routing test passed")
