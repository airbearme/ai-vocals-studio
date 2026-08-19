#!/usr/bin/env python3
"""
End-to-end precision voice cloning validation.

Steps:
  1. Build (or load) the 2Pac reference model from the real acapellas.
  2. Take a held-out clip and convert it through the model (real WORLD
     vocoder conversion).
  3. Score the conversion (same-content) with the QA system and print an
     HONEST report.
  4. Optionally generate a fresh TTS sentence in the clone voice.

No placeholders:  model.pth is a real learned pitch+timbre profile, and
validation compares the original audio with the ACTUALLY converted audio.
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import librosa
import soundfile as sf

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from create_pacaveli_model import build_pacaveli_model
from voice_conversion_engine import VoiceConversionEngine
from voice_quality_assurance import VoiceQualityAssurance


def load_or_build_model():
    model_pth = ROOT / "models" / "pacaveli" / "model.pth"
    if not model_pth.exists() or os.path.getsize(model_pth) < 1000:
        print(">> No valid 2Pac model found; building from real acapellas...")
        build_pacaveli_model()
    return str(model_pth)


def pick_test_audio(subdir: str):
    cands = sorted((ROOT / "dataset" / subdir).glob("*.wav"))
    return str(cands[0]) if cands else None


def main(seconds: float = 10.0, do_tts: bool = True) -> dict:
    print("=" * 68)
    print("PRECISION VOICE CLONING -- INDISTINGUISHABILITY VALIDATION")
    print("=" * 68)

    model_pth = load_or_build_model()
    engine = VoiceConversionEngine()
    qa = VoiceQualityAssurance()
    profile = engine.load_profile(model_pth)

    print("\n[REFERENCE MODEL - 2Pac profile]")
    print(f"   mean pitch      : {profile['pitch']['mean_hz']:.1f} Hz")
    print(f"   median pitch    : {profile['pitch']['median_hz']:.1f} Hz")
    print(f"   target pitch    : {profile['pitch_target']:.1f} Hz")
    print(f"   spectral centroid: {profile['spectral']['mean_centroid_hz']:.0f} Hz")
    print(f"   formants        : {profile.get('formants', {})}")

    out_dir = ROOT / "output" / "indistinguishability"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # ---- STEP 1: real clip -> 2Pac voice --------------------------------
    src = pick_test_audio("test speaker") or pick_test_audio("w2re")
    if src:
        print("\n[STEP 1] Converting a real voice clip into the 2Pac voice")
        print(f"   source: {src}")
        y, sr = librosa.load(src, sr=22050, mono=True)
        n = min(len(y), int(seconds * sr))
        original_path = out_dir / "original_segment.wav"
        sf.write(str(original_path), y[:n], sr)
        clone_path = out_dir / "converted_2pac.wav"
        t0 = time.time()
        engine.convert_audio(str(original_path), profile, str(clone_path),
                             strength=0.9)
        print(f"   converted in {time.time() - t0:.1f}s -> {clone_path.name}")
        score = qa.comprehensive_similarity_score(str(original_path),
                                                  str(clone_path))
        results["conversion"] = score
        print("\n[SCORE] conversion vs original (same content, honest):")
        for k, v in score.items():
            if isinstance(v, (int, float)):
                print(f"   {k:>24}: {v:.3f}")

    # ---- STEP 2: fresh TTS in the clone voice ---------------------------
    if do_tts:
        text = ("California love, west side till I die. "
                "Through the fog the story tells itself, no lie.")
        print("\n[STEP 2] Generating a fresh TTS sentence in the 2Pac voice")
        tts_path = out_dir / "tts_2pac_generated.wav"
        ok = engine.text_to_speech_with_voice(text, profile, str(tts_path),
                                              strength=0.9)
        if ok:
            results["tts"] = {"status": "generated", "path": str(tts_path),
                              "duration_s": round(
                                  len(sf.read(str(tts_path))[0]) / 22050, 1)}
            print(f"   generated: {tts_path} ({results['tts']['duration_s']}s)")

    # ---- report -----------------------------------------------------------
    print("\n" + "=" * 68)
    rep = out_dir / "validation_report.json"
    rep.write_text(json.dumps(results, indent=2, default=str))
    print(f"REPORT saved: {rep}")
    print("=" * 68)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Indistinguishability validation")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="seconds of source audio to validate")
    ap.add_argument("--no-tts", action="store_true",
                    help="skip TTS generation step")
    args = ap.parse_args()
    main(seconds=args.seconds, do_tts=not args.no_tts)
