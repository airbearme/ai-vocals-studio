#!/usr/bin/env python3
"""
clone_voice_cli.py — Clone a voice from ANY audio clip and generate new speech
in that voice with Qwen3-TTS (real neural cloning, runs locally, no API key).

Usage:
    python clone_voice_cli.py --ref /path/to/ref.wav \
                              --text "Say this in the cloned voice" \
                              --name my_voice \
                              [--ref-text "optional transcript"] \
                              [--song]                  # ref is a full song: vocals get separated first
                              [--model <hf-repo>]       # default: 0.6B on CPU, 1.7B on GPU
                              [--out outputs/my_clone.wav]
                              [--low-mem]

First run downloads the model (~1-3 GB) into the HuggingFace cache.
Clones are also saved to models/voices/<name>/ so the web app can reuse them.

Only use this with voices you own or have explicit permission to clone.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _progress(msg: str, pct: int) -> None:
    bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
    sys.stdout.write(f"\r[{bar}] {pct:3d}%  {msg:<60s}")
    sys.stdout.flush()
    if pct >= 100:
        sys.stdout.write("\n")


def prepare_reference(ref_path: str | Path, is_song: bool) -> str:
    """Turn any source into a clean mono 22.05 kHz reference clip (reuses the
    project's voice_cloner pipeline: vocal separation for songs, silence
    trimming, loudest-segment picking)."""
    from voice_cloner import extract_reference_audio

    work = Path(os.environ.get("VOICES_DIR", "models/voices")) / "_prep"
    work.mkdir(parents=True, exist_ok=True)
    ref_wav, meta = extract_reference_audio(
        ref_path, "song" if is_song else "speech", work, _progress)
    if not ref_wav:
        raise RuntimeError("Could not extract a clean reference clip.")
    dur = meta.get("reference_duration_s", 0)
    print(f"[prep] clean reference: {dur:.1f}s at {meta.get('sample_rate', 22050)} Hz")
    return ref_wav


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3-TTS voice cloning CLI")
    ap.add_argument("--ref", required=True, help="reference audio file (speech or song)")
    ap.add_argument("--text", required=True, help="text to speak in the cloned voice")
    ap.add_argument("--name", default="cloned_voice", help="label saved under models/voices/")
    ap.add_argument("--ref-text", default=None, help="optional transcript of the reference")
    ap.add_argument("--song", action="store_true", help="reference is a full song (separate vocals)")
    ap.add_argument("--model", default=None, help="HuggingFace repo id to use")
    ap.add_argument("--out", default=None, help="output wav path")
    ap.add_argument("--low-mem", action="store_true", help="use float16 on CPU to save RAM")
    args = ap.parse_args()

    ref = Path(args.ref)
    if not ref.exists():
        print(f"[error] reference not found: {ref}")
        return 1

    if args.low_mem:
        os.environ["QWEN_LOW_MEM"] = "1"
    if args.model:
        os.environ["QWEN_TTS_MODEL"] = args.model

    from qwen3_tts_engine import Qwen3TTSEngine

    print("[1/4] Checking Qwen3-TTS engine...")
    engine = Qwen3TTSEngine()
    if not engine.can_clone():
        print("[error] qwen-tts is not importable. Run:  pip install qwen-tts")
        return 1
    print(f"[ok] engine available — device={engine.device}")

    print("[2/4] Preparing clean reference clip...")
    ref_wav = prepare_reference(ref, args.song)

    print("[3/4] Loading model (first run downloads it)...")
    if not engine.load_model(model_name=args.model, progress_cb=_progress):
        print("[error] model load failed")
        return 1

    print("[4/4] Cloning voice and generating speech...")
    out = engine.clone_voice(
        ref_audio=ref_wav,
        ref_text=args.ref_text,          # None = x-vector-only clone, no transcript
        target_text=args.text,
        speaker_name=args.name,
        output_path=args.out,
        progress_cb=_progress,
    )
    if not out:
        print("[error] cloning failed (see message above)")
        return 1

    # persist for reuse in the web app
    engine.save_clone(args.name, ref_wav, description="Cloned via CLI")
    print(f"\n[ok] done! output : {out}")
    print(f"[ok] saved voice: models/voices/{args.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
