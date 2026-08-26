#!/usr/bin/env python3
"""Prepare or register an authorized voice's real RVC model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rvc_training import (
    RvcTrainingError,
    build_training_command,
    import_rvc_model,
    run_training_command,
    validate_training_dataset,
)


def _find_model(output_dir: Path) -> Path:
    candidates = sorted(
        (path for path in output_dir.rglob("*.pth") if path.stat().st_size >= 10_000),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RvcTrainingError(f"Trainer completed without a valid .pth model in {output_dir}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-dir", required=True, help="models/voices/<authorized_voice>")
    parser.add_argument("--model", help="Existing trained RVC .pth to import")
    parser.add_argument("--index", help="Optional trained RVC .index to import")
    parser.add_argument("--dataset", help="Clean authorized dataset used for training")
    parser.add_argument("--output-dir", help="Trainer output directory")
    parser.add_argument("--trainer", default="rvc-train", help="Installed RVC trainer executable")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--i-have-permission", action="store_true", required=True)
    args = parser.parse_args()

    try:
        voice_dir = Path(args.voice_dir).expanduser()
        if args.model:
            profile = import_rvc_model(args.model, voice_dir, index_path=args.index)
            print(json.dumps({"ok": True, "mode": "import", "profile": profile}, indent=2))
            return 0

        if not args.dataset:
            raise RvcTrainingError("Provide --model or --dataset.")
        report = validate_training_dataset(args.dataset)
        report_path = voice_dir / "rvc_training_report.json"
        voice_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if not report["ready"]:
            raise RvcTrainingError(f"Dataset does not meet RVC readiness requirements; see {report_path}")

        output_dir = Path(args.output_dir or voice_dir / "rvc_training_output").expanduser()
        command = build_training_command(args.dataset, output_dir, trainer=args.trainer, epochs=args.epochs)
        print("[rvc] running:", " ".join(command))
        result = run_training_command(command)
        if result.returncode != 0:
            raise RvcTrainingError(result.stderr.strip() or "RVC trainer failed")
        model = _find_model(output_dir)
        index = next(output_dir.rglob("*.index"), None)
        profile = import_rvc_model(model, voice_dir, index_path=index)
        print(json.dumps({"ok": True, "mode": "train", "profile": profile}, indent=2))
        return 0
    except RvcTrainingError as exc:
        print(f"[rvc] error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
