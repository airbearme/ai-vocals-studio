"""
elevenlabs_engine.py — ElevenLabs Instant Voice Cloning engine

Uses the ElevenLabs Python SDK (pip install elevenlabs) with an API key
stored in .cloud_config (ELEVENLABS_API_KEY=...) or the env var
ELEVENLABS_API_KEY.

Flow:
  1. First call → uploads reference clips from dataset/<model>/ → ElevenLabs
     creates an Instant Voice Clone → stores voice_id in
     models/<model>/elevenlabs_voice_id.json
  2. Subsequent calls → reuses stored voice_id for TTS generation
  3. Mood → voice_settings mapping controls delivery expressiveness

Models used:
  - eleven_v3  (flagship, most expressive — default)
"""
from __future__ import annotations
import json, os, wave, tempfile
from pathlib import Path
from typing import Callable, Optional

_ProgressCB = Callable[[str, int], None]

# Mood → ElevenLabs VoiceSettings params
# stability:         0=very expressive/variable  1=consistent/flat
# similarity_boost:  0=creative               1=locked to source voice
# style:             0=no style exaggeration  1=heavy exaggeration
# use_speaker_boost: True improves similarity
_MOOD_SETTINGS: dict[str, dict] = {
    'default':      {'stability': 0.50, 'similarity_boost': 0.75, 'style': 0.30, 'use_speaker_boost': True},
    'aggressive':   {'stability': 0.25, 'similarity_boost': 0.85, 'style': 0.65, 'use_speaker_boost': True},
    'storytelling': {'stability': 0.60, 'similarity_boost': 0.75, 'style': 0.30, 'use_speaker_boost': True},
    'emotional':    {'stability': 0.20, 'similarity_boost': 0.80, 'style': 0.55, 'use_speaker_boost': True},
}

EL_MODEL = 'eleven_v3'


class ElevenLabsEngine:
    """
    Wraps ElevenLabs TTS + Instant Voice Cloning.

    Parameters
    ----------
    base_dir : path to project root (expects .cloud_config, models/, dataset/)
    """

    def __init__(self, base_dir: str | Path = '.'):
        self.base    = Path(base_dir).resolve()
        self.MODELS  = self.base / 'models'
        self.DATASET = self.base / 'dataset'

    # ── API key ──────────────────────────────────────────────────

    def _api_key(self) -> Optional[str]:
        """Return API key from env var or .cloud_config."""
        key = os.environ.get('ELEVENLABS_API_KEY', '').strip()
        if key:
            return key
        cfg = self.base / '.cloud_config'
        if cfg.exists():
            for line in cfg.read_text().splitlines():
                if line.startswith('ELEVENLABS_API_KEY='):
                    val = line.split('=', 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
        return None

    # ── Availability check ────────────────────────────────────────

    def can_synthesize(self) -> bool:
        """True if elevenlabs library is installed AND an API key is configured."""
        try:
            import elevenlabs  # noqa: F401  type: ignore
        except ImportError:
            return False
        return bool(self._api_key())

    # ── Voice management ─────────────────────────────────────────

    def _voice_id_file(self, model_name: str) -> Path:
        return self.MODELS / model_name / 'elevenlabs_voice_id.json'

    def _load_voice_id(self, model_name: str) -> Optional[str]:
        f = self._voice_id_file(model_name)
        if f.exists():
            try:
                return json.loads(f.read_text()).get('voice_id')
            except Exception:
                pass
        return None

    def _save_voice_id(self, model_name: str, voice_id: str):
        f = self._voice_id_file(model_name)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({'voice_id': voice_id, 'model': model_name}, indent=2))

    def _pick_clips(self, model_name: str, max_clips: int = 5) -> list[Path]:
        """
        Choose the best reference clips for Instant Voice Cloning.
        Prefers WAV over MP3. Targets clips between 20s and 120s.
        Returns up to max_clips paths.
        """
        ds = self.DATASET / model_name
        # Gather all audio files, WAV first
        wavs  = sorted(ds.glob('*.wav'))
        mp3s  = sorted(ds.glob('*.mp3'))
        flacs = sorted(ds.glob('*.flac'))
        candidates = wavs + mp3s + flacs
        if not candidates:
            raise RuntimeError(
                f'No audio files in dataset/{model_name}/.\n'
                'Import voice clips via the Training tab first.'
            )
        return candidates[:max_clips]

    def ensure_voice(
        self,
        model_name: str,
        progress_cb: Optional[_ProgressCB] = None,
        force_recreate: bool = False,
    ) -> str:
        """
        Return the ElevenLabs voice_id for model_name.
        Creates an Instant Voice Clone on first call and caches the ID.
        """
        from elevenlabs.client import ElevenLabs  # type: ignore

        cb = progress_cb or (lambda m, p: None)
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError(
                'ElevenLabs API key not set.\n'
                'Click 🔑 EL Key in the app to add your key, or set ELEVENLABS_API_KEY in env.'
            )

        if not force_recreate:
            cached = self._load_voice_id(model_name)
            if cached:
                return cached

        # Create new IVC
        clips = self._pick_clips(model_name)
        cb(f'Uploading {len(clips)} clips to ElevenLabs for voice cloning…', 25)

        client = ElevenLabs(api_key=api_key)

        # Open files as binary streams
        file_handles = [open(c, 'rb') for c in clips]
        try:
            voice = client.voices.ivc.create(
                name=f'AI Vocals Studio — {model_name}',
                description=f'Instant Voice Clone for {model_name}',
                files=file_handles,
                remove_background_noise=True,   # strips music/beat from clips
            )
        finally:
            for fh in file_handles:
                fh.close()

        voice_id = voice.voice_id
        self._save_voice_id(model_name, voice_id)
        cb(f'Voice clone created: {voice_id[:12]}…', 45)
        return voice_id

    # ── Synthesis ─────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        model_name: str,
        mood: str = 'default',
        progress_cb: Optional[_ProgressCB] = None,
    ) -> Path:
        """
        Generate speech in the cloned voice.

        Returns
        -------
        Path to a temporary WAV file (caller is responsible for cleanup).

        Raises
        ------
        RuntimeError on failure.
        """
        from elevenlabs.client import ElevenLabs  # type: ignore
        from elevenlabs import VoiceSettings      # type: ignore

        cb = progress_cb or (lambda m, p: None)
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError('ElevenLabs API key not configured.')

        cb('Verifying ElevenLabs voice…', 15)
        voice_id = self.ensure_voice(model_name, progress_cb=cb)

        settings = _MOOD_SETTINGS.get(mood, _MOOD_SETTINGS['default'])
        cb(f'ElevenLabs synthesizing (mood={mood})…', 50)

        client = ElevenLabs(api_key=api_key)

        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=EL_MODEL,
            output_format='pcm_44100',   # raw 16-bit mono PCM
            voice_settings=VoiceSettings(
                stability=settings['stability'],
                similarity_boost=settings['similarity_boost'],
                style=settings['style'],
                use_speaker_boost=settings['use_speaker_boost'],
            ),
        )

        # Collect PCM bytes
        pcm_data = b''.join(chunk for chunk in audio_stream if chunk)

        if not pcm_data:
            raise RuntimeError('ElevenLabs returned empty audio.')

        # Write as WAV (16-bit mono 44100 Hz)
        with tempfile.NamedTemporaryFile(
            delete=False, suffix='.wav', prefix='eleven_'
        ) as f:
            out_path = Path(f.name)

        with wave.open(str(out_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)        # 16-bit
            wf.setframerate(44100)
            wf.writeframes(pcm_data)

        cb('ElevenLabs synthesis complete', 80)
        return out_path

    def delete_voice(self, model_name: str) -> bool:
        """Delete the cloned voice from ElevenLabs and clear cached ID."""
        from elevenlabs.client import ElevenLabs  # type: ignore
        voice_id = self._load_voice_id(model_name)
        if not voice_id:
            return False
        try:
            client = ElevenLabs(api_key=self._api_key())
            client.voices.delete(voice_id)
        except Exception:
            pass  # Delete locally even if API call fails
        vf = self._voice_id_file(model_name)
        if vf.exists():
            vf.unlink()
        return True
