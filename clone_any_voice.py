#!/usr/bin/env python3
"""Unified authorized voice cloning, conversion, and voice-over CLI."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np


def _progress(msg: str, pct: int) -> None:
    pct = max(0, min(100, int(pct)))
    print(f"[{pct:3d}%] {msg}")


def _load_profile(profile_or_name: str) -> dict:
    from voice_cloner import DEFAULT_VOICES_DIR, load_voice_profile

    path = Path(profile_or_name).expanduser()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            profile = json.load(f)
        profile.setdefault("voice_dir", str(path.parent))
        return profile

    profile = load_voice_profile(profile_or_name, DEFAULT_VOICES_DIR)
    if profile:
        return profile
    raise FileNotFoundError(f"Voice profile not found: {profile_or_name}")


def build_profile(args: argparse.Namespace) -> dict:
    if args.profile:
        return _load_profile(args.profile)

    if not args.voice_source:
        raise ValueError("Provide --voice-source or --profile.")

    from voice_cloner import build_voice_profile_from_sources

    profile = build_voice_profile_from_sources(
        name=args.name,
        source_paths=args.voice_source,
        source_type=args.voice_source_type,
        description=args.description,
        progress_cb=_progress,
        has_permission=args.i_have_permission,
    )
    if not profile:
        raise RuntimeError("Voice profile build failed.")
    return profile


def convert_target_audio(
    profile: dict,
    target_audio: str,
    target_type: str,
    output_dir: Path,
    vocals_gain_db: float,
) -> Optional[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if target_type == "song":
        from song_converter import change_song

        out, steps = change_song(
            target_audio,
            profile,
            output_dir,
            progress_cb=_progress,
            separation="auto",
            vocals_gain_db=vocals_gain_db,
        )
        print(f"[ok] song conversion engine: {steps.get('conversion')}")
        score_path = output_dir / "converted_vocals.wav"
        if score_path.exists():
            print_accuracy_score(profile, score_path)
        return out

    from song_converter import convert_vocals

    stem = Path(target_audio).stem.replace(" ", "_")
    out_path = output_dir / f"{stem}_in_{profile.get('name', 'voice')}.wav"
    ok, msg = convert_vocals(
        target_audio,
        profile,
        out_path,
        output_dir,
        progress_cb=_progress,
    )
    if not ok:
        raise RuntimeError(msg)
    print(f"[ok] audio conversion engine: {msg}")
    print_accuracy_score(profile, out_path)
    return str(out_path)


def score_voice_accuracy(profile: dict, audio_path: str | Path) -> dict:
    """Estimate how closely an output matches the cloned voice profile."""
    from song_converter import _load_mono, band_energy_profile, estimate_pitch

    target = profile.get("audio_profile", {})
    y = _load_mono(audio_path, 22050)
    if y.size == 0:
        return {"score": 0.0, "pitch": 0.0, "timbre": 0.0, "energy": 0.0}

    ref_f0 = float(target.get("median_f0_hz", 0.0) or 0.0)
    out_f0, _ = estimate_pitch(y, 22050)
    if ref_f0 > 40 and out_f0 > 40:
        pitch_score = 1.0 - min(abs(np.log2(out_f0 / ref_f0)) / 1.0, 1.0)
    else:
        pitch_score = 0.35

    ref_env = np.asarray(target.get("spectral_envelope") or [], dtype=np.float64)
    out_env = np.asarray(band_energy_profile(y, 22050, n_bands=16), dtype=np.float64)
    if ref_env.size == out_env.size and ref_env.size:
        ref_env = ref_env - np.mean(ref_env)
        out_env = out_env - np.mean(out_env)
        denom = (np.linalg.norm(ref_env) * np.linalg.norm(out_env)) + 1e-9
        timbre_score = float((np.dot(ref_env, out_env) / denom + 1.0) / 2.0)
    else:
        timbre_score = 0.35

    ref_rms = float(target.get("rms", 0.0) or 0.0)
    out_rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    if ref_rms > 1e-6 and out_rms > 1e-6:
        energy_score = 1.0 - min(abs(np.log10(out_rms / ref_rms)) / 1.5, 1.0)
    else:
        energy_score = 0.35

    score = 100.0 * (
        0.45 * max(0.0, pitch_score)
        + 0.45 * max(0.0, timbre_score)
        + 0.10 * max(0.0, energy_score)
    )
    return {
        "score": round(float(np.clip(score, 0.0, 100.0)), 1),
        "pitch": round(float(np.clip(pitch_score * 100.0, 0.0, 100.0)), 1),
        "timbre": round(float(np.clip(timbre_score * 100.0, 0.0, 100.0)), 1),
        "energy": round(float(np.clip(energy_score * 100.0, 0.0, 100.0)), 1),
    }


def print_accuracy_score(profile: dict, audio_path: str | Path) -> dict:
    score = score_voice_accuracy(profile, audio_path)
    print(
        "[ok] estimated voice accuracy: "
        f"{score['score']}% "
        f"(pitch={score['pitch']}%, timbre={score['timbre']}%, energy={score['energy']}%)"
    )
    return score


def synthesize_voiceover(
    profile: dict,
    text: str,
    output_dir: Path,
    mood: str = "default",
) -> Optional[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    name = str(profile.get("name") or "cloned_voice")
    reference = profile.get("reference")
    failures: list[str] = []

    try:
        from elevenlabs_engine import ElevenLabsEngine

        el = ElevenLabsEngine(Path.cwd())
        dataset_dir = Path("dataset") / name
        if el.can_synthesize() and dataset_dir.is_dir():
            out = el.synthesize(text, name, mood=mood, progress_cb=_progress)
            final = output_dir / f"{name}_voiceover_elevenlabs.wav"
            shutil.copy2(out, final)
            print("[ok] voice-over engine: ElevenLabs")
            print_accuracy_score(profile, final)
            return str(final)
    except Exception as e:
        failures.append(f"ElevenLabs: {e}")

    if reference and Path(reference).exists():
        try:
            from qwen3_tts_engine import Qwen3TTSEngine

            qe = Qwen3TTSEngine()
            if qe.can_clone():
                out = output_dir / f"{name}_voiceover_qwen.wav"
                result = qe.clone_voice(
                    ref_audio=reference,
                    ref_text=None,
                    target_text=text,
                    speaker_name=name,
                    output_path=out,
                    progress_cb=_progress,
                    has_permission=True,
                )
                if result:
                    print("[ok] voice-over engine: Qwen3-TTS")
                    print_accuracy_score(profile, result)
                    return str(result)
        except Exception as e:
            failures.append(f"Qwen3-TTS: {e}")

        try:
            from xtts_engine import XttsEngine

            xtts = XttsEngine(Path.cwd())
            if xtts.can_synthesize():
                out = xtts.synthesize(
                    text,
                    name,
                    mood=mood,
                    custom_ref=reference,
                    progress_cb=_progress,
                )
                final = output_dir / f"{name}_voiceover_xtts.wav"
                shutil.copy2(out, final)
                print("[ok] voice-over engine: XTTS v2")
                print_accuracy_score(profile, final)
                return str(final)
        except Exception as e:
            failures.append(f"XTTS: {e}")

    try:
        from gtts import gTTS
        from song_converter import convert_vocals

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts_path = Path(tmp.name)
        gTTS(text=text, lang="en").save(str(tts_path))
        out = output_dir / f"{name}_voiceover_dsp.wav"
        convert_vocals(tts_path, profile, out, output_dir, progress_cb=_progress)
        print("[ok] voice-over engine: gTTS + DSP timbre mapping")
        print_accuracy_score(profile, out)
        return str(out)
    except Exception as e:
        failures.append(f"gTTS+DSP: {e}")

    raise RuntimeError("No voice-over backend succeeded: " + " | ".join(failures))


def mix_voiceover_over_bed(
    voiceover_path: str | Path,
    bed_path: str | Path,
    output_path: str | Path,
    voice_gain_db: float = 0.0,
    bed_gain_db: float = -3.0,
) -> str:
    """Mix a generated cloned voice-over over a beat/instrumental bed."""
    import librosa
    import soundfile as sf

    bed, bed_sr = sf.read(str(bed_path), dtype="float32", always_2d=True)
    voice, _ = librosa.load(str(voiceover_path), sr=bed_sr, mono=True)
    voice = voice.astype("float32")
    if voice.ndim == 1:
        voice = np.stack([voice, voice], axis=1)

    n = max(len(bed), len(voice))
    if len(bed) < n:
        bed = np.pad(bed, ((0, n - len(bed)), (0, 0)))
    if len(voice) < n:
        voice = np.pad(voice, ((0, n - len(voice)), (0, 0)))

    voice_gain = 10.0 ** (voice_gain_db / 20.0)
    bed_gain = 10.0 ** (bed_gain_db / 20.0)
    mixed = bed[:n] * bed_gain + voice[:n] * voice_gain
    peak = np.max(np.abs(mixed)) or 1.0
    if peak > 0.98:
        mixed = mixed / peak * 0.98

    sf.write(str(output_path), mixed.astype("float32"), bed_sr)
    return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone an authorized voice from audio/training data, then convert audio or generate voice-over."
    )
    parser.add_argument("--voice-source", action="append", default=[], help="voice source file/folder; repeat for multiple")
    parser.add_argument("--voice-source-type", choices=["speech", "song", "text"], default="speech")
    parser.add_argument("--profile", help="existing profile JSON path or saved voice name")
    parser.add_argument("--name", default=f"cloned_voice_{int(time.time())}")
    parser.add_argument("--description", default="")
    parser.add_argument("--target-audio", help="audio/song to make sound like the cloned voice")
    parser.add_argument("--target-type", choices=["clip", "song", "bed"], default="clip")
    parser.add_argument("--voiceover-text", help="text to synthesize in the cloned voice")
    parser.add_argument("--mood", default="default")
    parser.add_argument("--output-dir", default="outputs/clone_any_voice")
    parser.add_argument("--vocals-gain-db", type=float, default=0.0)
    parser.add_argument(
        "--i-have-permission",
        action="store_true",
        help="confirm you own the voice or have explicit written permission/license to use it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.i_have_permission:
        print("[error] pass --i-have-permission to confirm authorization for this voice.")
        return 1
    if not args.target_audio and not args.voiceover_text:
        print("[error] provide --target-audio and/or --voiceover-text.")
        return 1

    try:
        output_dir = Path(args.output_dir)
        profile = build_profile(args)
        print(f"[ok] voice profile: {profile.get('name')} ({profile.get('reference')})")

        voiceover_out: Optional[str] = None
        if args.voiceover_text:
            voiceover_out = synthesize_voiceover(profile, args.voiceover_text, output_dir, args.mood)
            print(f"[ok] voice-over: {voiceover_out}")

        if args.target_audio and args.target_type == "bed":
            if not voiceover_out:
                raise ValueError("--target-type bed requires --voiceover-text.")
            mixed = output_dir / f"{profile.get('name', 'voice')}_voiceover_mix.wav"
            out = mix_voiceover_over_bed(
                voiceover_out,
                args.target_audio,
                mixed,
                voice_gain_db=args.vocals_gain_db,
            )
            print(f"[ok] voice-over mixed over bed: {out}")

        elif args.target_audio:
            out = convert_target_audio(
                profile,
                args.target_audio,
                args.target_type,
                output_dir,
                args.vocals_gain_db,
            )
            print(f"[ok] converted audio: {out}")

        return 0
    except Exception as e:
        print(f"[error] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
