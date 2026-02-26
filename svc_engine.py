"""
svc_engine.py — SO-VITS-SVC-Fork v4 inference + training engine

Inference:  uses Svc.infer_silence() — handles long audio, silence skipping
Training:   drives the svc CLI pipeline (pre-resample → pre-config → pre-hubert → train)
Fallback:   if model is a placeholder (0-byte .pth or no config.json),
            the caller should use DSP effects instead.
"""

from __future__ import annotations
import json, shutil, subprocess, threading, time
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from so_vits_svc_fork.inference.core import Svc
    import torch
    HAS_SOVITS = True
except Exception:
    HAS_SOVITS = False

# ─────────────────────────────────────────────────────────────────
#  Model status constants
# ─────────────────────────────────────────────────────────────────
STATUS_READY       = 'ready'    # real model + config present → SO-VITS
STATUS_PLACEHOLDER = 'placeholder'  # model dir exists but .pth is 0-byte
STATUS_MISSING     = 'missing'  # no .pth file at all
STATUS_NO_CONFIG   = 'no_config'  # real .pth but missing config.json

_ProgressCB = Callable[[str, int], None]   # (message, percent)


# ─────────────────────────────────────────────────────────────────
#  Helper: load audio as float32 mono at target_sr
# ─────────────────────────────────────────────────────────────────
def _load_audio(path: str | Path, target_sr: int = 44100) -> np.ndarray:
    if HAS_LIBROSA:
        audio, _ = librosa.load(str(path), sr=target_sr, mono=True)
        return audio.astype(np.float32)
    # fallback: soundfile (no resampling)
    audio, sr = sf.read(str(path), dtype='float32', always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


# ─────────────────────────────────────────────────────────────────
#  SoVitsEngine
# ─────────────────────────────────────────────────────────────────
class SoVitsEngine:
    """
    Manages SO-VITS-SVC-Fork loading, inference, and training.

    Usage:
        engine = SoVitsEngine(models_dir='models', dataset_dir='dataset')

        # Check if model is fully trained
        status = engine.model_status('Pacaveli')
        if status == STATUS_READY:
            engine.load('Pacaveli')
            engine.convert('Pacaveli', 'input.wav', 'output.wav', pitch_shift=-3)

        # Train a new model
        engine.train('Pacaveli', progress_cb=my_callback)
    """

    def __init__(self,
                 models_dir: str | Path = 'models',
                 dataset_dir: str | Path = 'dataset',
                 work_dir: str | Path = '.'):
        self.MODELS  = Path(models_dir)
        self.DATASET = Path(dataset_dir)
        self.WORK    = Path(work_dir).resolve()   # svc CLI runs here
        self._svc: Svc | None = None
        self._loaded_name: str | None = None
        self._loaded_device: str | None = None
        self._lock = threading.Lock()

    # ── Model discovery ──────────────────────────────────────────

    def model_status(self, model_name: str) -> str:
        """
        Returns STATUS_READY / STATUS_NO_CONFIG / STATUS_PLACEHOLDER / STATUS_MISSING
        """
        model_dir = self.MODELS / model_name
        if not model_dir.is_dir():
            # Also check for a top-level .pth file
            pth = self.MODELS / f'{model_name}.pth'
            if pth.exists() and pth.stat().st_size > 1000:
                return STATUS_NO_CONFIG   # file but no dir/config
            return STATUS_MISSING

        pth = self._find_pth(model_dir)
        if pth is None:
            return STATUS_MISSING
        if pth.stat().st_size == 0:
            return STATUS_PLACEHOLDER

        cfg = model_dir / 'config.json'
        if not cfg.exists():
            return STATUS_NO_CONFIG

        return STATUS_READY

    def _find_pth(self, model_dir: Path) -> Path | None:
        """Return the best .pth file: prefer G_*.pth, then model.pth"""
        g_files = sorted(model_dir.glob('G_*.pth'),
                         key=lambda p: _epoch_from_name(p.stem))
        if g_files:
            return g_files[-1]
        plain = model_dir / 'model.pth'
        if plain.exists():
            return plain
        return None

    def get_speaker(self, model_name: str) -> str:
        """Read speaker name from config.json, fallback to model_name."""
        cfg = self.MODELS / model_name / 'config.json'
        if cfg.exists():
            try:
                with open(cfg) as f:
                    data = json.load(f)
                spk = data.get('spk', {})
                if spk:
                    return list(spk.keys())[0]
            except Exception:
                pass
        return model_name

    # ── Load ─────────────────────────────────────────────────────

    def load(self, model_name: str, device: str | None = None) -> tuple[bool, str]:
        """
        Load SO-VITS model into memory.
        Returns (success, message).
        """
        if not HAS_SOVITS:
            return False, 'so-vits-svc-fork not installed'

        status = self.model_status(model_name)
        if status != STATUS_READY:
            return False, f'Model not ready (status={status})'

        # Skip reload if already loaded
        if self._loaded_name == model_name and self._svc is not None:
            return True, f'Already loaded ({model_name})'

        model_dir = self.MODELS / model_name
        pth = self._find_pth(model_dir)
        cfg = model_dir / 'config.json'

        if device is None:
            device = 'cuda' if (HAS_SOVITS and torch.cuda.is_available()) else 'cpu'

        try:
            with self._lock:
                self._svc = Svc(
                    net_g_path=str(pth),
                    config_path=str(cfg),
                    device=device,
                )
                self._loaded_name   = model_name
                self._loaded_device = device
            return True, f'Loaded {pth.name} on {device}'
        except Exception as e:
            self._svc = None
            self._loaded_name = None
            return False, str(e)

    def unload(self):
        with self._lock:
            self._svc = None
            self._loaded_name = None

    # ── Inference ─────────────────────────────────────────────────

    def convert(self,
                model_name: str,
                input_wav: str | Path,
                output_wav: str | Path,
                pitch_shift: int = 0,
                f0_method: str = 'dio',
                noise_scale: float = 0.4,
                auto_predict_f0: bool = False,
                progress_cb: _ProgressCB | None = None) -> tuple[bool, str]:
        """
        Full pipeline: load model (if needed) → infer → save.
        Returns (success, error_or_empty_string).
        """
        cb = progress_cb or _noop

        # 1. Load model
        cb('Loading SO-VITS model…', 10)
        ok, msg = self.load(model_name)
        if not ok:
            return False, msg

        # 2. Load input audio at model sample rate
        cb('Reading input audio…', 20)
        try:
            with self._lock:
                target_sr = self._svc.hps.data.sampling_rate
        except Exception:
            target_sr = 44100

        try:
            audio = _load_audio(input_wav, target_sr)
        except Exception as e:
            return False, f'Could not load audio: {e}'

        # 3. Run inference
        cb('Running SO-VITS voice conversion…', 40)
        speaker = self.get_speaker(model_name)
        try:
            with self._lock:
                result = self._svc.infer_silence(
                    audio,
                    speaker=speaker,
                    transpose=int(pitch_shift),
                    auto_predict_f0=auto_predict_f0,
                    cluster_infer_ratio=0,
                    noise_scale=noise_scale,
                    f0_method=f0_method,
                    db_thresh=-40,
                    pad_seconds=0.5,
                    chunk_seconds=0.5,
                    absolute_thresh=False,
                    max_chunk_seconds=40,
                )
        except Exception as e:
            return False, f'Inference error: {e}'

        # 4. Save output
        cb('Saving output…', 90)
        try:
            result_np = result.cpu().numpy() if hasattr(result, 'cpu') else np.array(result)
            sf.write(str(output_wav), result_np, target_sr)
        except Exception as e:
            return False, f'Could not save output: {e}'

        cb('Done!', 100)
        return True, ''

    # ── CLI-based inference (no Python API) ──────────────────────

    def convert_cli(self,
                    model_name: str,
                    input_wav: str | Path,
                    output_wav: str | Path,
                    pitch_shift: int = 0,
                    f0_method: str = 'dio',
                    progress_cb: _ProgressCB | None = None) -> tuple[bool, str]:
        """
        Use the `svc infer` CLI instead of the Python API.
        More robust to API version changes.
        """
        cb = progress_cb or _noop
        status = self.model_status(model_name)
        if status != STATUS_READY:
            return False, f'Model not ready: {status}'

        model_dir = self.MODELS / model_name
        pth = self._find_pth(model_dir)
        cfg = model_dir / 'config.json'
        speaker = self.get_speaker(model_name)
        out_dir = Path(output_wav).parent

        cb('Running svc infer…', 30)
        cmd = [
            'svc', 'infer',
            str(input_wav),
            '-m', str(pth),
            '-c', str(cfg),
            '-s', speaker,
            '-o', str(out_dir),
            '-t', str(pitch_shift),
            '-fm', f0_method,
            '-d', 'cuda' if (HAS_SOVITS and torch.cuda.is_available()) else 'cpu',
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               cwd=str(self.WORK), timeout=300)
            if r.returncode != 0:
                return False, r.stderr[-600:] or r.stdout[-600:]
        except subprocess.TimeoutExpired:
            return False, 'svc infer timed out (>5 min)'
        except FileNotFoundError:
            return False, 'svc CLI not found in PATH'
        except Exception as e:
            return False, str(e)

        # The CLI saves to <out_dir>/<input_stem>.wav — rename to output_wav
        expected = out_dir / (Path(input_wav).stem + '.wav')
        target   = Path(output_wav)
        if expected.exists() and expected != target:
            shutil.move(str(expected), str(target))
        if not target.exists():
            return False, 'Output file not created by svc infer'

        cb('Done!', 100)
        return True, ''

    # ── Training pipeline ─────────────────────────────────────────

    def train(self,
              model_name: str,
              progress_cb: _ProgressCB | None = None,
              stop_event: threading.Event | None = None) -> tuple[bool, str]:
        """
        Full SO-VITS training pipeline for model_name.
        Uses training audio from dataset/<model_name>/.

        Steps:
          1. Copy audio → dataset_raw/<model_name>/
          2. svc pre-resample
          3. svc pre-config -t so-vits-svc-4
          4. svc pre-hubert
          5. svc train -t so-vits-svc-4
          6. Copy outputs → models/<model_name>/

        progress_cb(message, percent): called throughout.
        stop_event: set() to abort training.
        """
        cb   = progress_cb or _noop
        stop = stop_event or threading.Event()

        src_dir = self.DATASET / model_name
        if not src_dir.is_dir():
            return False, f'No dataset folder: {src_dir}'

        audio_files = [f for ext in ('*.wav','*.mp3','*.flac','*.m4a','*.ogg')
                       for f in src_dir.glob(ext)]
        if not audio_files:
            return False, f'No audio files in {src_dir}'

        # ── Step 1: populate dataset_raw ──
        cb(f'Staging {len(audio_files)} audio files…', 3)
        raw_dir = self.WORK / 'dataset_raw' / model_name
        raw_dir.mkdir(parents=True, exist_ok=True)
        for fp in audio_files:
            dst = raw_dir / fp.name
            if not dst.exists():
                shutil.copy2(fp, dst)

        # ── Steps 2-5: CLI pipeline ──
        steps = [
            ('Resampling audio to 44 100 Hz…',  12,  ['svc', 'pre-resample']),
            ('Generating model config…',         25,  ['svc', 'pre-config', '-t', 'so-vits-svc-4.0v1']),
            ('Extracting HuBERT features…',      45,  ['svc', 'pre-hubert', '-nf']),
            ('Training model — this can take hours on CPU…', 85,
             ['svc', 'train']),
        ]

        for msg, pct, cmd in steps:
            if stop.is_set():
                return False, 'Training cancelled'
            cb(msg, pct)
            ok, err = self._run_step(cmd, pct, cb, stop)
            if not ok:
                return False, f'{cmd[1]} failed:\n{err}'

        # ── Step 6: collect outputs ──
        cb('Saving model to models/ …', 96)
        try:
            self._collect_outputs(model_name)
        except Exception as e:
            return False, f'Could not collect outputs: {e}'

        cb('Training complete!', 100)
        return True, ''

    def _run_step(self, cmd: list[str], base_pct: int,
                  cb: _ProgressCB, stop: threading.Event) -> tuple[bool, str]:
        """Run one CLI step, stream output to cb, return (success, stderr)."""
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(self.WORK),
                bufsize=1, universal_newlines=True,
            )
        except FileNotFoundError:
            return False, f"'{cmd[0]}' not found in PATH"

        lines = []
        for line in proc.stdout:
            if stop.is_set():
                proc.kill()
                return False, 'Cancelled'
            line = line.rstrip()
            if line:
                lines.append(line)
                cb(line[-120:], -1)   # -1 = don't change progress bar

        proc.wait()
        if proc.returncode != 0:
            tail = '\n'.join(lines[-8:])
            return False, tail

        return True, ''

    def _collect_outputs(self, model_name: str):
        """Copy training artifacts to models/<model_name>/."""
        dest = self.MODELS / model_name
        dest.mkdir(exist_ok=True)

        # config.json
        for cfg_candidate in [
            self.WORK / 'configs' / '44k' / 'config.json',
            self.WORK / 'configs' / 'config.json',
        ]:
            if cfg_candidate.exists():
                shutil.copy2(cfg_candidate, dest / 'config.json')
                break

        # latest G (generator) checkpoint
        for logs_candidate in [
            self.WORK / 'logs' / '44k',
            self.WORK / 'logs',
        ]:
            g_files = sorted(logs_candidate.glob('G_*.pth'),
                             key=lambda p: _epoch_from_name(p.stem)) if logs_candidate.exists() else []
            if g_files:
                latest = g_files[-1]
                shutil.copy2(latest, dest / latest.name)
                # also write a canonical link
                shutil.copy2(latest, dest / 'model.pth')
                break

    # ── Quick inference check ─────────────────────────────────────

    def can_infer(self, model_name: str) -> bool:
        return HAS_SOVITS and self.model_status(model_name) == STATUS_READY

    def is_loaded(self, model_name: str) -> bool:
        return self._loaded_name == model_name and self._svc is not None


# ─────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────
def _epoch_from_name(stem: str) -> int:
    """G_10000 → 10000"""
    try:
        return int(stem.split('_')[-1])
    except (ValueError, IndexError):
        return 0

def _noop(msg: str, pct: int): pass

def status_label(status: str) -> str:
    return {
        STATUS_READY:       '🟢 SO-VITS READY',
        STATUS_NO_CONFIG:   '🟡 MISSING config.json',
        STATUS_PLACEHOLDER: '🟡 PLACEHOLDER — needs training',
        STATUS_MISSING:     '🔴 NO MODEL',
    }.get(status, '❓ UNKNOWN')
