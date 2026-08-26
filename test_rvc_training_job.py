"""Regression checks for the queued one-click RVC training job."""
from __future__ import annotations

from supabase_worker import WORKER_JOB_TYPES


def test_worker_claims_rvc_training_jobs() -> None:
    assert "local_worker_rvc_train" in WORKER_JOB_TYPES


if __name__ == "__main__":
    test_worker_claims_rvc_training_jobs()
    print("RVC training job routing test passed")
