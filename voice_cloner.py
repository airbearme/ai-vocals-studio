"""
voice_cloner.py — Clone a voice from ANY source: a song, speech, or text.

What this does:
  1. Accepts any audio (a full song, a speech clip, or TTS-generated speech)
     as the voice reference.
  2. For songs, the vocals are extracted first (demucs or center-channel) so
     the clone is built from the singer's clean voice, not the music bed.
  3. Analyzes the voice into an "audio profile" (pitch, timbre envelope, RMS)
     and stores everything in models/voices/<name>/:
        voice_profile.json   — the profile itself
        reference.wav        — clean mono 22.05 kHz reference clip
        source.<orig>        — your original upload
        rvc_model.pth/index  — optional trained RVC model for neural conversion

That profile can then be used to:
   - change any song's vocals with the cloned voice (song_converter.py)
   - speak any text in the cloned voice (Qwen3-TTS neural cloning)

Only use this with voices you own or have explicit permission to clone.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:                                   # pragma: no cover
    HAS_LIBROSA = False

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:                                   # pragma: no cover
    HAS_SF = False

from song_converter import separate_vocals, _load_mono, band_energy_profile
from voice_safety import VoiceSafetyError, validate_voice_clone_request

_ProgressCB = Callable[[str, int], None]

DEFAULT_VOICES_DIR = Path(os.environ.get("VOICES_DIR", "models/voices"))
SAMPLE_RATE = 22050
REFERENCE_DURATION = 12.0    # seconds of clean voice kept as reference
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".aiff"}


def _noop(msg: str, pct: int) -> None:
    pass


def _safe_name(name: str) -> str:
    import re
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe.strip("._") or "cloned_voice"


def collect_audio_sources(paths: Iterable[str | Path]) -> list[Path]:
    """Expand files/folders into a stable list of supported audio files."""
    out: list[Path] = []
    for item in paths:
        path = Path(item).expanduser()
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in AUDIO_EXTENSIONS:
                    out.append(child)
        elif path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            out.append(path)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in out:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def list_voices(voices_dir: str | Path = DEFAULT_VOICES_DIR) -> list[dict]:
    """Return metadata for every saved voice clone, newest first."""
    voices_dir = Path(voices_dir)
    out = []
    if not voices_dir.is_dir():
        return out
    for d in sorted(voices_dir.iterdir(), key=lambda p: p.stat().st_mtime,
                    reverse=True):
        prof_file = d / "voice_profile.json"
        if not d.is_dir() or not prof_file.exists():
            continue
        try:
            with open(prof_file) as f:
                prof = json.load(f)
            out.append({
                "name": prof.get("name", d.name),
                "source_type": prof.get("source_type", "speech"),
                "description": prof.get("description", ""),
                "created": prof.get("created", ""),
                "duration_s": round(float(prof.get("audio_profile", {})
                                          .get("duration_s", 0)), 1),
                "median_f0_hz": round(float(prof.get("audio_profile", {})
                                             .get("median_f0_hz", 0)), 1),
                "has_rvc": bool(prof.get("rvc_available", False)),
                "reference": str(prof_file.parent / "reference.wav"),
                "voice_dir": str(d),
            })
        except Exception:
            continue
    return out


def load_voice_profile(name: str,
                       voices_dir: str | Path = DEFAULT_VOICES_DIR) -> Optional[dict]:
    """Load a voice profile by name. Returns dict or None."""
    prof_file = Path(voices_dir) / _safe_name(name) / "voice_profile.json"
    if not prof_file.exists():
        return None
    try:
        with open(prof_file) as f:
            prof = json.load(f)
        prof["voice_dir"] = str(prof_file.parent)
        return prof
    except Exception:
        return None


def _analyze(audio: np.ndarray, sr: int) -> dict:
    """Produce the audio_profile statistics used for cloning + conversion."""
    from song_converter import estimate_pitch

    median_f0, pitch_std = estimate_pitch(audio, sr)
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    centroid = 0.0
    if HAS_LIBROSA and audio.size:
        try:
            centroid = float(np.mean(
                librosa.feature.spectral_centroid(y=audio, sr=sr)))
        except Exception:
            centroid = 0.0
    envelope = band_energy_profile(audio, sr, n_bands=16)
    return {
        "median_f0_hz": round(median_f0, 2),
        "pitch_std_hz": round(pitch_std, 2),
        "rms": round(rms, 5),
        "spectral_centroid": round(centroid, 1),
        "spectral_envelope": [round(float(v), 2) for v in envelope],
        "duration_s": round(float(len(audio) / sr), 2),
        "sample_rate": sr,
    }
def _trim_silence(audio: np.ndarray, sr: int, top_db: float = 28.0
                  ) -> np.ndarray:
    if not HAS_LIBROSA or audio.size < sr:
        return audio
    try:
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        if trimmed.size > 512:
            return trimmed
    except Exception:
        pass
    return audio


def _pick_loudest_segment(audio: np.ndarray, sr: int,
                          seg_dur: float = REFERENCE_DURATION) -> np.ndarray:
    """
    Return the loudest contiguous segment of `seg_dur` seconds — best for
    songs, where the biggest energy window is almost always a sung section.
    """
    if len(audio) <= int(seg_dur * sr):
        return audio
    hop = sr // 4  # 250 ms
    win = int(seg_dur * sr)
    rms_frames = librosa.feature.rms(y=audio, hop_length=hop)[0]
    frame_t = np.arange(len(rms_frames)) * hop
    best_start = 0
    best_rms = -1.0
    cum = np.concatenate([[0.0], np.cumsum(rms_frames)])
    for s in range(0, len(rms_frames)):
        e = int(s + win / hop)
        if e >= len(cum):
            break
        r = (cum[e] - cum[s]) / max(1, e - s)
        if r > best_rms:
            best_rms = r
            best_start = int(frame_t[s])
    return audio[best_start: best_start + win]


def extract_reference_audio(
    source_path: str | Path,
    source_type: str,
    work_dir: str | Path,
    progress_cb: Optional[_ProgressCB] = None,
) -> tuple[str, dict]:
    """
    Get a clean mono 22.05 kHz reference clip from any source.

    source_type:
      "song"   -> vocals are separated out first, then the loudest ~12 s kept
      "speech" -> silence-trimmed, kept mostly as-is (capped to ~25 s)
      "text"   -> pass the TTS-generated speech straight through

    Returns (reference_wav_path, source_meta).
    """
    cb = progress_cb or _noop
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    source = _load_mono(source_path, SAMPLE_RATE)
    src_meta = {"duration_s": float(len(source) / SAMPLE_RATE)}

    if source_type == "song":
        cb("Extracting the singer's vocals from the song...", 25)
        temp_dir = work_dir / "stems"
        vocals, _, method = separate_vocals(source_path, temp_dir,
                                            method="auto", progress_cb=cb)
        cb(f"Vocals extracted ({method}). Picking best vocal clip...", 60)
        if vocals:
            source = _trim_silence(_load_mono(vocals, SAMPLE_RATE),
                                   SAMPLE_RATE)
        src_meta["separation"] = method
    else:
        cb("Cleaning the voice reference...", 40)
        source = _trim_silence(source, SAMPLE_RATE)

    if source_type == "song":
        ref = _pick_loudest_segment(source, SAMPLE_RATE)
    else:
        cap = int(min(25.0 * SAMPLE_RATE, len(source)))
        ref = source[:cap]

    if len(ref) < SAMPLE_RATE:
        pad = np.zeros(SAMPLE_RATE - len(ref), dtype=np.float32)
        ref = np.concatenate([ref, pad])

    peak = np.max(np.abs(ref)) or 1.0
    ref = ref / peak * 0.95

    ref_path = work_dir / "reference.wav"
    sf.write(str(ref_path), ref.astype(np.float32), SAMPLE_RATE)
    src_meta["reference_duration_s"] = round(float(len(ref) / SAMPLE_RATE), 2)
    return str(ref_path), src_meta

def build_voice_profile(
    name: str,
    source_path: str | Path,
    source_type: str = "speech",
    description: str = "",
    voices_dir: str | Path = DEFAULT_VOICES_DIR,
    progress_cb: Optional[_ProgressCB] = None,
    has_permission: bool = False,
) -> Optional[dict]:
    """
    Create a persisted voice clone from any audio source.

    Steps: extract reference -> analyze -> save json + audio.
    Returns the profile dict (also stored on disk), or None on failure.
    """
    cb = progress_cb or _noop
    try:
        validate_voice_clone_request(
            has_permission=has_permission,
            speaker_name=name,
            description=description,
            source_path=source_path,
        )
    except VoiceSafetyError as e:
        cb(str(e), 0)
        return None

    safe = _safe_name(name)
    voice_dir = Path(voices_dir) / safe
    voice_dir.mkdir(parents=True, exist_ok=True)

    try:
        cb(f"Cloning voice '{name}' from {source_type}...", 10)
        ref_wav, src_meta = extract_reference_audio(
            source_path, source_type, voice_dir, cb)

        cb("Analyzing voice characteristics...", 75)
        ref_audio = _load_mono(ref_wav, SAMPLE_RATE)
        audio_profile = _analyze(ref_audio, SAMPLE_RATE)
        audio_profile.update(src_meta)

        # copy original for provenance
        orig_suffix = Path(source_path).suffix or ".wav"
        orig_copy = voice_dir / f"source{orig_suffix}"
        try:
            shutil.copy2(source_path, orig_copy)
        except Exception:
            pass

        # optional RVC model already dropped into the folder
        rvc_available = (voice_dir / f"{safe}.pth").exists() or bool(
            list(voice_dir.glob("rvc_model.pth")))

        profile = {
            "name": safe,
            "source_type": source_type,
            "description": description,
            "created": datetime.now().isoformat(timespec="seconds"),
            "audio_profile": audio_profile,
            "reference": str(Path(ref_wav).parent / "reference.wav"),
            "rvc_available": bool(rvc_available),
            "voice_dir": str(voice_dir),
            "methods": {
                "dsp": True,
                "qwen_neural": False,
                "rvc": bool(rvc_available),
            },
        }

        with open(voice_dir / "voice_profile.json", "w") as f:
            json.dump(profile, f, indent=2)

        cb("Voice cloned!", 100)
        return profile
    except Exception as e:
        cb(f"Voice cloning failed: {str(e)}", 0)
        return None


def build_voice_profile_from_sources(
    name: str,
    source_paths: Iterable[str | Path],
    source_type: str = "speech",
    description: str = "",
    voices_dir: str | Path = DEFAULT_VOICES_DIR,
    progress_cb: Optional[_ProgressCB] = None,
    has_permission: bool = False,
) -> Optional[dict]:
    """
    Build one voice profile from multiple clips or a training-data folder.

    Each source is cleaned first. Songs are source-separated before analysis;
    speech clips are silence-trimmed. The final profile uses the combined
    reference audio, capped to keep cloning fast and stable.
    """
    cb = progress_cb or _noop
    sources = collect_audio_sources(source_paths)
    if not sources:
        cb("No supported audio sources found.", 0)
        return None

    try:
        validate_voice_clone_request(
            has_permission=has_permission,
            speaker_name=name,
            description=description,
            source_path=" ".join(str(p) for p in sources[:5]),
        )
    except VoiceSafetyError as e:
        cb(str(e), 0)
        return None

    if len(sources) == 1:
        return build_voice_profile(
            name,
            sources[0],
            source_type=source_type,
            description=description,
            voices_dir=voices_dir,
            progress_cb=progress_cb,
            has_permission=has_permission,
        )

    safe = _safe_name(name)
    voice_dir = Path(voices_dir) / safe
    voice_dir.mkdir(parents=True, exist_ok=True)

    refs: list[np.ndarray] = []
    source_meta: list[dict] = []
    try:
        total = len(sources)
        for idx, source in enumerate(sources, start=1):
            pct = 5 + int(60 * (idx - 1) / max(1, total))
            cb(f"Preparing reference {idx}/{total}: {source.name}", pct)
            ref_dir = voice_dir / "references" / f"{idx:04d}"
            ref_wav, meta = extract_reference_audio(
                source,
                source_type,
                ref_dir,
                progress_cb=cb,
            )
            ref_audio = _load_mono(ref_wav, SAMPLE_RATE)
            if ref_audio.size:
                refs.append(ref_audio)
                meta.update({"source": str(source), "reference": str(ref_wav)})
                source_meta.append(meta)

        if not refs:
            cb("No usable voiced audio found in sources.", 0)
            return None

        cb("Combining references...", 68)
        silence = np.zeros(int(0.15 * SAMPLE_RATE), dtype=np.float32)
        combined_parts: list[np.ndarray] = []
        max_total = int(90.0 * SAMPLE_RATE)
        total_len = 0
        for ref in refs:
            remaining = max_total - total_len
            if remaining <= 0:
                break
            clip = ref[:remaining]
            combined_parts.extend([clip, silence])
            total_len += len(clip) + len(silence)
        combined = np.concatenate(combined_parts).astype(np.float32)
        peak = np.max(np.abs(combined)) or 1.0
        combined = combined / peak * 0.95

        ref_path = voice_dir / "reference.wav"
        sf.write(str(ref_path), combined, SAMPLE_RATE)

        cb("Analyzing combined voice characteristics...", 78)
        audio_profile = _analyze(combined, SAMPLE_RATE)
        audio_profile.update({
            "source_count": len(source_meta),
            "reference_duration_s": round(float(len(combined) / SAMPLE_RATE), 2),
        })

        rvc_available = (voice_dir / f"{safe}.pth").exists() or bool(
            list(voice_dir.glob("rvc_model.pth")))

        profile = {
            "name": safe,
            "source_type": source_type,
            "description": description,
            "created": datetime.now().isoformat(timespec="seconds"),
            "audio_profile": audio_profile,
            "reference": str(ref_path),
            "sources": source_meta,
            "rvc_available": bool(rvc_available),
            "voice_dir": str(voice_dir),
            "methods": {
                "dsp": True,
                "qwen_neural": False,
                "rvc": bool(rvc_available),
            },
        }

        with open(voice_dir / "voice_profile.json", "w") as f:
            json.dump(profile, f, indent=2)

        cb("Voice profile built from all sources.", 100)
        return profile
    except Exception as e:
        cb(f"Voice profile build failed: {str(e)}", 0)
        return None


def build_voice_profile_from_text(
    name: str,
    tts_speech_path: str | Path,
    description: str = "Generated from text",
    voices_dir: str | Path = DEFAULT_VOICES_DIR,
    progress_cb: Optional[_ProgressCB] = None,
    has_permission: bool = False,
) -> Optional[dict]:
    """
    Build a voice profile from a text-to-speech sample, so you can clone the
    text's spoken voice directly as your reference.
    """
    return build_voice_profile(
        name, tts_speech_path, "text", description,
        voices_dir=voices_dir, progress_cb=progress_cb,
        has_permission=has_permission)
