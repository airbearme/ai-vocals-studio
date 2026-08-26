#!/usr/bin/env python3
"""Run one queued RVC training job on a Kaggle GPU notebook.

Configure Kaggle Secrets named SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, and
optionally SUPABASE_STORAGE_BUCKET. The notebook runs this file once per
session; the job remains in Supabase if the session is interrupted.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import requests


def _secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient

        return str(UserSecretsClient().get_secret(name) or "").strip()
    except Exception:
        return ""


SUPABASE_URL = _secret("SUPABASE_URL").rstrip("/")
SUPABASE_KEY = _secret("SUPABASE_SERVICE_ROLE_KEY")
BUCKET = _secret("SUPABASE_STORAGE_BUCKET") or "voiceovers"
HEADERS = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}


def request(method: str, path: str, **kwargs) -> requests.Response:
    response = requests.request(method, f"{SUPABASE_URL}{path}", headers=HEADERS, timeout=180, **kwargs)
    response.raise_for_status()
    return response


def claim_job() -> dict | None:
    rows = request(
        "GET",
        "/rest/v1/voiceover_jobs?select=*&status=eq.queued&job_type=eq.local_worker_rvc_train&order=created_at.asc&limit=1",
    ).json()
    if not rows:
        return None
    job = rows[0]
    updated = request(
        "PATCH",
        f"/rest/v1/voiceover_jobs?id=eq.{quote(str(job['id']))}",
        headers={**HEADERS, "Content-Type": "application/json", "Prefer": "return=representation"},
        data=json.dumps({"status": "running", "engine": "Kaggle GPU + Applio"}),
    ).json()
    return updated[0] if updated else job


def update_job(job_id: str, patch: dict) -> None:
    request(
        "PATCH",
        f"/rest/v1/voiceover_jobs?id=eq.{quote(job_id)}",
        headers={**HEADERS, "Content-Type": "application/json", "Prefer": "return=representation"},
        data=json.dumps(patch),
    )


def download_object(object_path: str, target: Path) -> None:
    response = request("GET", f"/storage/v1/object/{quote(BUCKET)}/{quote(object_path, safe='/')}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)


def upload_object(object_path: str, source: Path, content_type: str) -> str:
    request(
        "POST",
        f"/storage/v1/object/{quote(BUCKET)}/{quote(object_path, safe='/')}",
        headers={**HEADERS, "Content-Type": content_type, "x-upsert": "true"},
        data=source.read_bytes(),
    )
    return f"{SUPABASE_URL}/storage/v1/object/public/{quote(BUCKET)}/{quote(object_path, safe='/')}"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "authorized_voice"


def run() -> int:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Add SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to Kaggle Secrets.")
    job = claim_job()
    if not job:
        print("No queued local_worker_rvc_train job found.")
        return 0

    job_id = str(job["id"])
    name = safe_name(str(job.get("voice_name") or f"voice_{job_id[:8]}"))
    root = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
    dataset = root / "rvc_dataset" / name
    output = root / "rvc_output" / name
    dataset.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    try:
        for index, item in enumerate(job.get("input_files") or [], start=1):
            object_path = item.get("path")
            if object_path:
                suffix = Path(str(item.get("name") or f"source_{index}.wav")).suffix or ".wav"
                download_object(str(object_path), dataset / f"source_{index}{suffix}")

        app_root = Path(os.environ.get("APP_ROOT", "/kaggle/working/ai-vocals-studio"))
        sys.path.insert(0, str(app_root))
        from rvc_training import validate_training_dataset

        report = validate_training_dataset(dataset, min_files=3, min_duration_s=90.0)
        update_job(job_id, {"report": {"stage": "dataset_validated", "training": report}})
        if not report["ready"]:
            raise RuntimeError("Dataset failed the 3-clip/90-second RVC readiness gate.")

        applio_root = Path(os.environ.get("APPLIO_ROOT", "/kaggle/working/Applio"))
        python = Path(os.environ.get("APPLIO_PYTHON", str(applio_root / ".venv/bin/python")))
        core = applio_root / "core.py"
        if not python.exists() or not core.exists():
            raise RuntimeError("Applio is not installed. Run the notebook setup cell first.")
        gpu = os.environ.get("RVC_GPU", "0")
        model_name = f"ai_vocals_{name}"
        commands = [
            ["preprocess", "--model_name", model_name, "--dataset_path", str(dataset), "--sample_rate", "40000", "--cut_preprocess", "Automatic", "--chunk_len", "3.0", "--overlap_len", "0.3", "--normalization_mode", "pre"],
            ["extract", "--model_name", model_name, "--sample_rate", "40000", "--f0_method", "rmvpe", "--embedder_model", "contentvec", "--gpu", gpu],
            ["train", "--model_name", model_name, "--sample_rate", "40000", "--total_epoch", os.environ.get("RVC_TRAIN_EPOCHS", "300"), "--save_every_epoch", "50", "--batch_size", os.environ.get("RVC_BATCH_SIZE", "8"), "--gpu", gpu],
            ["index", "--model_name", model_name],
        ]
        for command in commands:
            update_job(job_id, {"engine": "Kaggle GPU + Applio", "report": {"stage": command[0], "training": report}})
            subprocess.run([str(python), str(core), *command], cwd=applio_root, check=True)

        model_dir = applio_root / "logs" / model_name
        models = sorted(model_dir.rglob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
        indices = list(model_dir.rglob("*.index"))
        if not models:
            raise RuntimeError("Applio completed without a model checkpoint.")
        model_path = models[0]
        model_object = f"models/{name}/rvc_model.pth"
        model_url = upload_object(model_object, model_path, "application/octet-stream")
        index_url = None
        if indices:
            index_object = f"models/{name}/rvc_model.index"
            index_url = upload_object(index_object, indices[0], "application/octet-stream")
        update_job(job_id, {
            "status": "complete",
            "engine": "Kaggle GPU + Applio RVC",
            "report": {
                "stage": "registered",
                "training": report,
                "model_object": model_object,
                "model_url": model_url,
                "index_url": index_url,
            },
        })
        print(f"Completed RVC training job {job_id}: {model_url}")
        return 0
    except Exception as exc:
        update_job(job_id, {"status": "failed", "engine": "Kaggle GPU + Applio", "error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(run())
