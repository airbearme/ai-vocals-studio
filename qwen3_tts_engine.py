"""
qwen3_tts_engine.py - Qwen3-TTS voice cloning engine

Advanced voice cloning with 3-second rapid cloning, multi-language support,
and natural language voice control. Use this only with voices you own or have
explicit permission to clone.

Requires: pip install qwen-tts
Models auto-download on first use (~1-2GB each)

Usage:
    from qwen3_tts_engine import Qwen3TTSEngine
    engine = Qwen3TTSEngine()
    if engine.can_clone():
        out_path = engine.clone_voice(
            ref_audio="reference_sample.wav",
            ref_text="This is my reference voice sample",
            target_text="This is the generated vocal take",
            speaker_name="Authorized_Voice"
        )
"""
from __future__ import annotations
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Callable, Optional, Union, List, Tuple

_ProgressCB = Callable[[str, int], None]

# Status constants
STATUS_READY = 'ready'
STATUS_MISSING = 'missing'
STATUS_LOADING = 'loading'
STATUS_ERROR = 'error'

# Lazy globals — qwen-tts is heavy, don't import at startup
_qwen_module = None
_qwen_lock = threading.Lock()
_qwen_ready = threading.Event()
HAS_QWEN: bool = False


def _safe_voice_name(name: str) -> str:
    """Return a filesystem-safe voice label."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe.strip("._") or "cloned_voice"


def _ensure_qwen(progress_cb: Optional[_ProgressCB] = None) -> bool:
    """Import qwen-tts on first call. Thread-safe. Returns True if available."""
    global _qwen_module, HAS_QWEN
    if shutil.which("sox") is None:
        if progress_cb:
            progress_cb("SoX is required for Qwen3-TTS audio processing.", 0)
        HAS_QWEN = False
        _qwen_ready.set()
        return False
    if _qwen_ready.is_set():
        return HAS_QWEN
    if not _qwen_lock.acquire(blocking=False):
        _qwen_ready.wait(timeout=60)
        return HAS_QWEN
    try:
        if progress_cb:
            progress_cb('Loading Qwen3-TTS engine...', 5)

        from qwen_tts import Qwen3TTSModel
        import torch
        import soundfile as sf

        _qwen_module = {
            'Qwen3TTSModel': Qwen3TTSModel,
            'torch': torch,
            'soundfile': sf
        }
        HAS_QWEN = True
        _qwen_ready.set()
        return True
    except ImportError:
        HAS_QWEN = False
        _qwen_ready.set()
        return False
    except Exception:
        HAS_QWEN = False
        _qwen_ready.set()
        return False
    finally:
        _qwen_lock.release()


class Qwen3TTSEngine:
    """Qwen3-TTS voice cloning engine with rapid 3-second cloning"""

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.environ.get("MODEL_CACHE_DIR", "models")
        self.models_dir = Path(models_dir)
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if self._check_cuda() else "cpu"
        self.current_model_name = None

    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def can_clone(self) -> bool:
        """Check if Qwen3-TTS is available and can clone voices"""
        return _ensure_qwen()

    def get_supported_models(self) -> List[str]:
        """Get list of available Qwen3-TTS models"""
        return [
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",      # Best for voice cloning
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",      # Lightweight version
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", # Pre-trained voices
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"  # Voice design capabilities
        ]

    def load_model(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                   progress_cb: Optional[_ProgressCB] = None) -> bool:
        """Load Qwen3-TTS model"""
        if not self.can_clone():
            return False

        try:
            if progress_cb:
                progress_cb(f'Loading {model_name}...', 10)

            modules = _qwen_module
            torch = modules['torch']
            Qwen3TTSModel = modules['Qwen3TTSModel']

            # Load model with optimal settings
            self.model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=self.device,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                attn_implementation="flash_attention_2" if self.device == "cuda" else "eager"
            )

            self.current_model_name = model_name

            if progress_cb:
                progress_cb('Model loaded successfully!', 100)

            return True

        except Exception as e:
            if progress_cb:
                progress_cb(f'Error loading model: {str(e)}', 0)
            return False

    def clone_voice(self,
                   ref_audio: Union[str, Path, Tuple[np.ndarray, int]],
                   ref_text: str,
                   target_text: str,
                   speaker_name: str = "cloned_voice",
                   language: str = "Auto",
                   output_path: Optional[Union[str, Path]] = None,
                   progress_cb: Optional[_ProgressCB] = None) -> Optional[str]:
        """
        Clone voice using reference audio

        Args:
            ref_audio: Reference audio file path or (numpy_array, sample_rate) tuple
            ref_text: Transcript of reference audio
            target_text: Text to synthesize in cloned voice
            speaker_name: Name for the cloned voice
            language: Target language (Auto for auto-detection)
            output_path: Output file path (auto-generated if None)
            progress_cb: Progress callback function

        Returns:
            Path to generated audio file or None if failed
        """
        if not self.model:
            if not self.load_model(progress_cb=progress_cb):
                return None

        try:
            if progress_cb:
                progress_cb('Cloning voice...', 30)

            modules = _qwen_module
            soundfile = modules['soundfile']

            # Generate cloned voice
            wavs, sr = self.model.generate_voice_clone(
                ref_audio=ref_audio,
                ref_text=ref_text,
                text=target_text,
                language=language
            )

            if progress_cb:
                progress_cb('Saving audio...', 90)

            # Generate output path if not provided
            if output_path is None:
                output_dir = Path(os.environ.get("OUTPUT_DIR", "outputs"))
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"{_safe_voice_name(speaker_name)}_qwen_clone.wav"

            # Save audio
            soundfile.write(str(output_path), wavs[0], sr)

            if progress_cb:
                progress_cb('Voice clone complete!', 100)

            return str(output_path)

        except Exception as e:
            if progress_cb:
                progress_cb(f'Cloning failed: {str(e)}', 0)
            return None

    def design_voice(self,
                    text: str,
                    voice_description: str,
                    speaker_name: str = "designed_voice",
                    language: str = "Auto",
                    output_path: Optional[Union[str, Path]] = None,
                    progress_cb: Optional[_ProgressCB] = None) -> Optional[str]:
        """
        Design a voice using natural language description

        Args:
            text: Text to synthesize
            voice_description: Natural language description of desired voice
            speaker_name: Name for the designed voice
            language: Target language
            output_path: Output file path
            progress_cb: Progress callback

        Returns:
            Path to generated audio file or None if failed
        """
        if not self.model:
            # Load voice design model
            if not self.load_model("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", progress_cb):
                return None

        try:
            if progress_cb:
                progress_cb('Designing voice...', 30)

            modules = _qwen_module
            soundfile = modules['soundfile']

            # Generate designed voice
            wavs, sr = self.model.generate_voice_design(
                text=text,
                language=language,
                instruct=voice_description
            )

            if progress_cb:
                progress_cb('Saving audio...', 90)

            # Generate output path if not provided
            if output_path is None:
                output_dir = Path(os.environ.get("OUTPUT_DIR", "outputs"))
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"{_safe_voice_name(speaker_name)}_qwen_design.wav"

            # Save audio
            soundfile.write(str(output_path), wavs[0], sr)

            if progress_cb:
                progress_cb('Voice design complete!', 100)

            return str(output_path)

        except Exception as e:
            if progress_cb:
                progress_cb(f'Voice design failed: {str(e)}', 0)
            return None

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        if not self.model:
            return []

        try:
            return self.model.get_supported_languages()
        except Exception:
            return ["Auto", "Chinese", "English", "Japanese", "Korean",
                   "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]

    def get_supported_speakers(self) -> List[str]:
        """Get list of pre-trained speakers (for CustomVoice models)"""
        if not self.model:
            return []

        try:
            return self.model.get_supported_speakers()
        except Exception:
            return []


# Convenience function for quick cloning
def quick_clone_voice(ref_audio_path: str,
                      ref_text: str,
                      target_text: str,
                      output_name: str = "quick_clone") -> Optional[str]:
    """Quick voice cloning without engine setup"""
    engine = Qwen3TTSEngine()
    return engine.clone_voice(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        target_text=target_text,
        speaker_name=output_name
    )
