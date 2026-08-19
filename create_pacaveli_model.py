#!/usr/bin/env python3
"""
Build the Pacaveli (2Pac) voice model from the REAL acapella clips.

The reference clips (dataset/Pacaveli_training/*) are analysed with the
WORLD vocoder.  The resulting pitch + timbre profile (stored in
``models/pacaveli/model.pth``) is what the VoiceConversionEngine uses to
convert any audio/text into the 2Pac-style voice.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from voice_conversion_engine import VoiceConversionEngine


def build_pacaveli_model(input_dir="dataset/Pacaveli_training",
                         processed_dir="dataset/Pacaveli_processed",
                         output_dir="models/pacaveli",
                         pitch_target=150.0):
    """
    Build the Pacaveli (2Pac) model.

    ``pitch_target`` tunes the conversion pitch.  2Pac's documented deep
    baritone / rap register is ~110-150 Hz; the raw acapella rips contain
    some higher-pitched backing/ad-lib content, so we pin the *conversion*
    target at a realistic deep-rap value while still preserving the true
    extracted spectral (timbre) envelope from the data.
    """
    print("Building Pacaveli (2Pac) voice model from real acapellas...")

    sources = []
    for d in (processed_dir, input_dir):
        candidates = sorted(Path(d).glob("*.wav")) + sorted(Path(d).glob("*.mp3"))
        if candidates:
            sources = [str(c) for c in candidates]
            break
    if not sources:
        print("No 2Pac acapella audio found under dataset/Pacaveli_training/")
        return False

    def cb(msg, pct):
        print(f"\r  {msg}  ({pct:>3.0f}%)", end="", flush=True)

    engine = VoiceConversionEngine(max_seconds=90)

    print(f"Analyzing {len(sources)} reference clips...")
    profile = engine.extract_reference_profile(sources, progress_cb=cb)
    print()

    # Pin the conversion pitch target to the realistic deep-rap register.
    profile["pitch_target"] = float(pitch_target)
    profile["pitch"]["conversion_target_hz"] = float(pitch_target)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_pth = out / "model.pth"
    engine.save_profile(profile, str(model_pth))

    voice_profile = {
        "speaker": "Pacaveli",
        "model_name": "pacaveli",
        "type": "voice_clone",
        "engine": "world_vocoder",
        "total_files": len(sources),
        "audio_files": [Path(s).name for s in sources],
        "characteristics": {
            "voice_type": "deep_male_rapper",
            "pitch_mean_hz": profile["pitch"]["mean_hz"],
            "pitch_median_hz": profile["pitch"]["median_hz"],
            "pitch_dominant_hz": profile["pitch"]["dominant_hz"],
            "male_band_median_hz": profile["pitch"]["male_band_median_hz"],
            "pitch_conversion_target_hz": profile["pitch_target"],
            "spectral_centroid_hz": profile["spectral"]["mean_centroid_hz"],
            "formants": profile.get("formants", {}),
        },
        "persona": {
            "style": "west_coast_rap",
            "delivery": "aggressive_confident",
            "emotion": "intense",
        },
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out / "voice_profile.json", "w") as f:
        json.dump(voice_profile, f, indent=2)

    with open(out / "config.json", "w") as f:
        json.dump({"spk": {"Pacaveli": 0}, "version": "4.0"}, f, indent=2)

    print(f"Pacaveli voice model saved to {output_dir}/")
    print(f"  Mean pitch       : {profile['pitch']['mean_hz']:.1f} Hz")
    print(f"  Median pitch     : {profile['pitch']['median_hz']:.1f} Hz")
    print(f"  Dominant band    : {profile['pitch']['dominant_hz']:.1f} Hz")
    print(f"  Male-band median : {profile['pitch']['male_band_median_hz']:.1f} Hz")
    print(f"  Conversion target: {profile['pitch_target']:.1f} Hz")
    print(f"  Spectral centroid: {profile['spectral']['mean_centroid_hz']:.0f} Hz")
    print(f"  Formants         : {profile.get('formants', {})}")
    return True


if __name__ == "__main__":
    build_pacaveli_model()
