"""Static checks for the free Kaggle GPU worker entrypoint."""
from __future__ import annotations

from pathlib import Path


def test_kaggle_worker_claims_and_uploads_models() -> None:
    source = (Path(__file__).parent / "cloud" / "kaggle_rvc_worker.py").read_text(encoding="utf-8")
    for required in ("local_worker_rvc_train", "voiceover_jobs", "rvc_model.pth", "rvc_model.index"):
        assert required in source


if __name__ == "__main__":
    test_kaggle_worker_claims_and_uploads_models()
    print("Kaggle worker checks passed")
