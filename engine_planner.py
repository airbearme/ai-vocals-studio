"""Engine availability and automatic route planning for voice-over jobs."""
from __future__ import annotations

import os
import shutil
import importlib.util
from pathlib import Path
from typing import Any


def _importable(module_name: str) -> tuple[bool, str]:
    if importlib.util.find_spec(module_name) is not None:
        return True, "available"
    return False, "not installed"


def _has_api_key() -> bool:
    if os.environ.get("ELEVENLABS_API_KEY", "").strip():
        return True
    cfg = Path(".cloud_config")
    if cfg.exists():
        for line in cfg.read_text(errors="ignore").splitlines():
            if line.startswith("ELEVENLABS_API_KEY=") and line.split("=", 1)[1].strip():
                return True
    return False


def _rvc_model_present(profile: dict[str, Any]) -> bool:
    name = str(profile.get("name") or "")
    voice_dir = Path(profile.get("voice_dir") or Path("models") / "voices" / name)
    candidates = [
        voice_dir / "rvc_model.pth",
        voice_dir / "model.pth",
        voice_dir / f"{name}.pth",
    ]
    return any(path.exists() and path.stat().st_size > 10_000 for path in candidates)


def get_engine_status(profile: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return installed/configured status for every supported backend."""
    qwen_ok, qwen_note = _importable("qwen_tts")
    pyworld_ok, pyworld_note = _importable("pyworld")
    demucs_ok, demucs_note = _importable("demucs")
    eleven_ok, eleven_note = _importable("elevenlabs")
    xtts_ok, xtts_note = _importable("TTS")
    rvc_ok, rvc_note = _importable("rvc_python")

    has_rvc_model = _rvc_model_present(profile or {}) if profile else False
    eleven_configured = eleven_ok and _has_api_key()

    return [
        {
            "name": "RVC",
            "available": bool(rvc_ok and has_rvc_model),
            "installed": rvc_ok,
            "configured": has_rvc_model,
            "score": 98,
            "best_for": "audio-to-audio vocal replacement",
            "note": "trained RVC model found" if has_rvc_model else "requires a trained rvc_model.pth",
            "detail": rvc_note,
        },
        {
            "name": "ElevenLabs",
            "available": bool(eleven_configured),
            "installed": eleven_ok,
            "configured": eleven_configured,
            "score": 95,
            "best_for": "high-quality voice-over TTS",
            "note": "API key configured" if eleven_configured else "install package and set ELEVENLABS_API_KEY",
            "detail": eleven_note,
        },
        {
            "name": "Qwen3-TTS",
            "available": qwen_ok,
            "installed": qwen_ok,
            "configured": qwen_ok,
            "score": 90,
            "best_for": "local zero-shot voice-over TTS",
            "note": "local neural cloning engine",
            "detail": qwen_note,
        },
        {
            "name": "XTTS v2",
            "available": xtts_ok,
            "installed": xtts_ok,
            "configured": xtts_ok,
            "score": 86,
            "best_for": "local zero-shot voice-over TTS",
            "note": "Coqui TTS is not Python 3.12 compatible in this venv" if not xtts_ok else "available",
            "detail": xtts_note,
        },
        {
            "name": "WORLD/DSP",
            "available": pyworld_ok,
            "installed": pyworld_ok,
            "configured": pyworld_ok,
            "score": 72,
            "best_for": "fallback pitch/timbre conversion",
            "note": "always-on fallback when pyworld/librosa are installed",
            "detail": pyworld_note,
        },
        {
            "name": "Demucs",
            "available": demucs_ok,
            "installed": demucs_ok,
            "configured": demucs_ok,
            "score": 84,
            "best_for": "song vocal separation",
            "note": "neural song separation" if demucs_ok else "falls back to center-channel separation",
            "detail": demucs_note,
        },
        {
            "name": "SoX",
            "available": shutil.which("sox") is not None,
            "installed": shutil.which("sox") is not None,
            "configured": shutil.which("sox") is not None,
            "score": 60,
            "best_for": "extra audio format support",
            "note": "optional; app also uses soundfile/librosa",
            "detail": "available" if shutil.which("sox") else "not found on PATH",
        },
    ]


def estimate_reference_confidence(profile: dict[str, Any]) -> float:
    """Estimate confidence contribution from reference duration and profile quality."""
    ap = profile.get("audio_profile", {}) or {}
    duration = float(ap.get("reference_duration_s", ap.get("duration_s", 0.0)) or 0.0)
    source_count = float(ap.get("source_count", 1.0) or 1.0)
    duration_score = min(1.0, duration / 60.0)
    source_score = min(1.0, source_count / 4.0)
    pitch_ok = 1.0 if float(ap.get("median_f0_hz", 0.0) or 0.0) > 40 else 0.35
    return 100.0 * (0.55 * duration_score + 0.25 * source_score + 0.20 * pitch_ok)


def choose_best_plan(
    *,
    profile: dict[str, Any],
    mode: str,
    target_has_voice: bool = True,
) -> dict[str, Any]:
    """Pick the best available backend for the requested task."""
    statuses = get_engine_status(profile)
    by_name = {item["name"]: item for item in statuses}
    reference_conf = estimate_reference_confidence(profile)

    if mode in {"song", "clip"} and target_has_voice:
        order = ["RVC", "WORLD/DSP"]
    elif mode == "bed":
        order = ["ElevenLabs", "Qwen3-TTS", "XTTS v2", "WORLD/DSP"]
    else:
        order = ["Qwen3-TTS", "ElevenLabs", "XTTS v2", "WORLD/DSP"]

    chosen = next((by_name[name] for name in order if by_name.get(name, {}).get("available")), by_name["WORLD/DSP"])
    backend_score = float(chosen.get("score", 50))
    target_score = 100.0 if (mode == "bed" or target_has_voice) else 45.0
    confidence = 0.50 * backend_score + 0.35 * reference_conf + 0.15 * target_score
    confidence = max(0.0, min(99.0, confidence))

    return {
        "engine": chosen["name"],
        "confidence": round(confidence, 1),
        "reference_confidence": round(reference_conf, 1),
        "backend_score": backend_score,
        "target_score": target_score,
        "mode": mode,
        "reason": chosen["note"],
        "engines": statuses,
    }
