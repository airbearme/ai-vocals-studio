"""
xtts_engine.py — XTTS v2 zero-shot voice cloning engine

Zero-shot: give a reference WAV clip → generate speech in that voice with
matching prosody/delivery. No training required.

Fine-tuned: if models/<name>/xtts_finetuned/ exists, loads that instead
of the base XTTS v2 model for even higher consistency.

Usage:
    from xtts_engine import XttsEngine
    xe = XttsEngine()
    if xe.can_synthesize():
        out_path = xe.synthesize("Hello world", "Pacaveli", mood="aggressive")
"""
from __future__ import annotations
import json, random
from pathlib import Path
from typing import Callable, Optional

# ── Lazy imports: TTS is ~1GB wheel, never import at startup ────────
_TTS_cls = None        # set on first can_synthesize() call
_TTS_model = None      # cached loaded TTS instance
_TTS_model_key: str | None = None  # identifies which model is loaded

_ProgressCB = Callable[[str, int], None]

XTTS_MODEL_ID = 'tts_models/multilingual/multi-dataset/xtts_v2'

# Default mood → reference clip mapping (overridden by voice_moods.json)
_DEFAULT_MOODS: dict[str, list[str]] = {
    'default':      ['speaker_0001.wav', 'speaker_0002.wav'],
    'aggressive':   ['speaker_0003.wav', 'speaker_0004.wav'],
    'storytelling': ['speaker_0002.wav', 'speaker_0005.wav'],
    'emotional':    ['speaker_0001.wav', 'speaker_0005.wav'],
}


def _ensure_tts() -> bool:
    """Import TTS library on first call. Returns True if available."""
    global _TTS_cls
    if _TTS_cls is not None:
        return True
    try:
        from TTS.api import TTS as _TTS  # type: ignore
        _TTS_cls = _TTS
        return True
    except Exception:
        return False


class XttsEngine:
    """
    Wraps Coqui TTS XTTS v2 for zero-shot voice cloning.

    Parameters
    ----------
    base_dir : path to the project root (default: current directory)
        Expects dataset/<model>/ and models/<model>/ under it.
    """

    def __init__(self, base_dir: str | Path = '.'):
        self.base    = Path(base_dir).resolve()
        self.MODELS  = self.base / 'models'
        self.DATASET = self.base / 'dataset'

    # ── Mood / reference management ──────────────────────────────

    def _moods_for(self, model_name: str) -> dict[str, list[str]]:
        """Load mood → clip mapping from voice_moods.json, fall back to defaults."""
        cfg = self.MODELS / model_name / 'voice_moods.json'
        if cfg.exists():
            try:
                with open(cfg) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return _DEFAULT_MOODS

    def get_available_moods(self, model_name: str) -> list[str]:
        """Return moods that have at least one existing reference file."""
        moods   = self._moods_for(model_name)
        ds_dir  = self.DATASET / model_name
        result  = []
        for mood, clips in moods.items():
            for clip in clips:
                if (ds_dir / clip).exists():
                    result.append(mood)
                    break
        return result or list(moods.keys())

    def _pick_ref(self, model_name: str, mood: str) -> Optional[Path]:
        """Pick a random reference clip for the given mood."""
        moods  = self._moods_for(model_name)
        clips  = list(moods.get(mood, moods.get('default', [])))
        ds_dir = self.DATASET / model_name
        random.shuffle(clips)
        for clip in clips:
            p = ds_dir / clip
            if p.exists():
                return p
        # Fallback: any audio file in dataset dir
        for ext in ('*.wav', '*.mp3', '*.flac'):
            found = list(ds_dir.glob(ext))
            if found:
                return random.choice(found)
        return None

    # ── Fine-tune detection ───────────────────────────────────────

    def _finetuned_dir(self, model_name: str) -> Optional[Path]:
        """Return fine-tuned model directory if it exists and has a checkpoint."""
        ft = self.MODELS / model_name / 'xtts_finetuned'
        if not ft.is_dir():
            return None
        # Must have at least one checkpoint or best_model file
        has_ckpt = any(ft.glob('*.pth')) or any(ft.glob('best_model*'))
        return ft if has_ckpt else None

    # ── Public API ────────────────────────────────────────────────

    def can_synthesize(self) -> bool:
        """True if the TTS library is installed and importable."""
        return _ensure_tts()

    def synthesize(
        self,
        text: str,
        model_name: str,
        mood: str = 'default',
        custom_ref: Optional[str | Path] = None,
        progress_cb: Optional[_ProgressCB] = None,
    ) -> Path:
        """
        Generate speech in the target voice.

        Parameters
        ----------
        text       : text to synthesize
        model_name : voice model name (maps to dataset/<model_name>/ for refs)
        mood       : one of 'default','aggressive','storytelling','emotional'
        custom_ref : explicit reference audio path (overrides mood selection)
        progress_cb: (message, percent) callback

        Returns
        -------
        Path to a temporary WAV file (caller is responsible for cleanup).

        Raises
        ------
        RuntimeError if synthesis fails.
        """
        global _TTS_model, _TTS_model_key
        cb = progress_cb or (lambda m, p: None)

        if not _ensure_tts():
            raise RuntimeError(
                'TTS library not installed.\n'
                'Run:  pip install TTS  (installs ~1 GB, downloads XTTS v2 on first use)'
            )

        # ── Pick reference audio ──────────────────────────────────
        if custom_ref:
            ref_path = Path(custom_ref)
            if not ref_path.exists():
                raise RuntimeError(f'Custom reference not found: {ref_path}')
        else:
            ref_path = self._pick_ref(model_name, mood)
            if ref_path is None:
                raise RuntimeError(
                    f'No reference audio found for model={model_name}, mood={mood}.\n'
                    f'Make sure dataset/{model_name}/ contains WAV files.'
                )

        # ── Load / cache TTS model ────────────────────────────────
        ft_dir  = self._finetuned_dir(model_name)
        model_key = str(ft_dir) if ft_dir else XTTS_MODEL_ID

        if _TTS_model is None or _TTS_model_key != model_key:
            cb('Loading XTTS v2 model (first use — downloads ~1.8 GB)…', 10)
            try:
                import torch  # type: ignore
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'

            if ft_dir:
                cb(f'Loading fine-tuned XTTS model from {ft_dir.name}…', 12)
                cfg_path = ft_dir / 'config.json'
                _TTS_model = _TTS_cls(
                    model_path=str(ft_dir),
                    config_path=str(cfg_path) if cfg_path.exists() else None,
                )
            else:
                _TTS_model = _TTS_cls(XTTS_MODEL_ID)
            _TTS_model_key = model_key

        # ── Synthesize ────────────────────────────────────────────
        cb(f'Synthesizing: mood={mood}  ref={ref_path.name}…', 40)
        import tempfile
        with tempfile.NamedTemporaryFile(
            delete=False, suffix='.wav', prefix='xtts_out_'
        ) as f:
            out_path = Path(f.name)

        try:
            _TTS_model.tts_to_file(
                text=text,
                speaker_wav=str(ref_path),
                language='en',
                file_path=str(out_path),
            )
        except Exception as e:
            out_path.unlink(missing_ok=True)
            raise RuntimeError(f'XTTS synthesis failed: {e}') from e

        if not out_path.exists() or out_path.stat().st_size < 100:
            raise RuntimeError('XTTS produced an empty output file.')

        cb('XTTS synthesis complete', 80)
        return out_path
