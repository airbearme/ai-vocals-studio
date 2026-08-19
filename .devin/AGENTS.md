# AI Vocals Studio - Agent Configuration

This file provides comprehensive guidance for AI agents working on the AI Vocals Studio project - a complete voice cloning and AI vocals generation system.

## Project Overview

AI Vocals Studio is a professional-grade voice cloning and vocal generation system with multiple deployment modes and audio processing engines. The project supports both real-time voice conversion and text-to-speech generation with advanced voice cloning capabilities.

## Core Architecture

### Application Entry Points

- **`app_minimal.py`** - Lightweight production default using gTTS + pydub (no heavy ML dependencies)
- **`app_modern.py`** - Full-featured Tkinter UI with model management and downloads
- **`app_streamlit.py`** - Streamlit web interface
- **`app_studio.py`** - Advanced studio interface with professional features

### Audio Processing Engines

- **`elevenlabs_engine.py`** - ElevenLabs API integration (best quality, requires API key)
- **`qwen3_tts_engine.py`** - Qwen3-TTS rapid reference voice cloning (3-second cloning)
- **`xtts_engine.py`** - XTTS v2 zero-shot voice cloning (local, free)
- **`rvc_engine.py`** - RVC v2 audio-to-audio voice conversion (best for rap/singing)
- **`svc_engine.py`** - SO-VITS-SVC engine for voice conversion
- **`voice_conversion_engine.py`** - Advanced WORLD vocoder-based voice conversion

### Core Processing Modules

- **`voice_trainer.py`** - VoiceTrainer class for processing raw audio and creating training configs
- **`enhanced_voice_cloner.py`** - EnhancedVoiceCloner for voice characteristic analysis
- **`voice_feature_extractor.py`** - Comprehensive voice feature extraction and analysis
- **`voice_quality_assurance.py`** - Quality assessment and validation
- **`advanced_audio_processor.py`** - Advanced audio processing and augmentation
- **`precision_voice_cloning_system.py`** - High-precision voice cloning system

## Development Workflow

### Voice Cloning Pipeline

```
1. Data Collection → dataset/<speaker>/
2. Audio Optimization → optimize_*_dataset.py
3. Feature Extraction → voice_feature_extractor.py
4. Voice Profile Creation → enhanced_voice_cloner.py
5. Model Training → train_*_model.py or voice_trainer.py
6. Model Validation → test_*_cloning.py
7. Deployment → models/<speaker>/
```

### Audio Processing Standards

**Required Audio Specifications:**
- Sample Rate: 22.05kHz (SO-VITS-SVC standard) or 44.1kHz/48kHz for modern engines
- Format: WAV (preferred), MP3, FLAC, OGG, M4A supported
- Duration: 3-30 seconds per clip for optimal training
- Quality: Clean, minimal background noise, consistent vocal performance

**Audio Optimization Steps:**
1. Normalize audio to -1.0 dB
2. Trim silence (threshold: -40dB)
3. Remove background noise
4. Ensure consistent sample rate
5. Split long clips into optimal segments

### Model Training Guidelines

**Training Data Requirements:**
- Minimum 10 minutes of clean vocal audio for basic cloning
- 30+ minutes for professional quality
- 60+ minutes for indistinguishable quality
- Variety of vocal ranges and emotional states

**Training Configuration:**
- Batch size: 8-16 (adjust based on GPU memory)
- Learning rate: 0.0001 (default)
- Epochs: 1000-5000 (monitor validation loss)
- Validation split: 10-15%

## File Structure Conventions

### Directory Layout

```
ai-vocals-studio/
├── dataset/                  # Training audio organized by speaker
│   ├── <speaker>/
│   │   ├── clip_001.wav
│   │   └── clip_002.wav
├── dataset_raw/             # Raw, unprocessed audio
├── models/                  # Trained voice models
│   ├── <speaker>/
│   │   ├── model.pth
│   │   └── voice_profile.json
│   └── pretrained/
├── output/                  # Generated audio output
├── configs/                 # Training configurations
├── filelists/               # Training file lists
└── logs/                    # Training and application logs
```

### Naming Conventions

- **Training datasets**: `dataset/<speaker_name>/` (lowercase, underscores)
- **Model files**: `model.pth` (standard) or `<speaker>_model.pth`
- **Voice profiles**: `voice_profile.json` (JSON format)
- **Output files**: `<timestamp>_<speaker>_<description>.wav`
- **Config files**: `<engine>_<purpose>.json`

## Code Standards

### Python Conventions

- **Python Version**: 3.10+ (3.12 recommended)
- **Style**: PEP 8 compliant
- **Type Hints**: Required for all function signatures
- **Docstrings**: Google style for all modules and functions
- **Error Handling**: Specific exceptions with descriptive messages

### Audio Processing Best Practices

```python
# Always use context managers for audio files
with sf.load(audio_path, sr=target_sr) as audio, sr:
    # Process audio
    pass

# Validate audio before processing
def validate_audio(audio, sr):
    if audio is None or len(audio) == 0:
        raise ValueError("Empty audio signal")
    if sr < 16000:
        raise ValueError(f"Sample rate {sr} too low, minimum 16kHz")
    return audio, sr

# Use librosa for audio analysis
import librosa
y, sr = librosa.load(path, sr=22050, mono=True)
```

### Threading Patterns

```python
# Standard pattern for long-running operations
def run_async_operation(func, *args, **kwargs):
    def worker():
        try:
            result = func(*args, **kwargs)
            callback(result, None)
        except Exception as e:
            callback(None, e)
    
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread
```

## Testing and Validation

### Voice Cloning Tests

- **`test_voice_cloning.py`** - Basic cloning functionality
- **`test_precision_cloning.py`** - Precision cloning validation
- **`test_indistinguishable_clone.py`** - Indistinguishability testing
- **`test_voice_conversion.py`** - Voice conversion quality
- **`validate_precision_system.py`** - System validation

### Quality Metrics

**Objective Metrics:**
- SNR (Signal-to-Noise Ratio): >20dB acceptable, >30dB excellent
- Dynamic Range: >40dB acceptable, >60dB excellent
- Spectral Consistency: >0.85 correlation
- MOS (Mean Opinion Score): >3.5 acceptable, >4.5 excellent

**Subjective Testing:**
- Listening tests with multiple evaluators
- ABX tests for model comparison
- Real-world deployment testing

## Troubleshooting Guidelines

### Common Issues

**CUDA Out of Memory:**
- Reduce batch size in training config
- Use gradient accumulation
- Reduce audio length or sample rate
- Clear GPU cache between operations

**Poor Voice Quality:**
- Check training audio quality
- Verify sufficient training data
- Adjust learning rate and training duration
- Ensure proper audio preprocessing

**Model Loading Failures:**
- Verify model file integrity
- Check architecture compatibility
- Ensure correct tensor dtypes
- Validate voice profile JSON structure

### Debugging Commands

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Test audio loading
python -c "import librosa; y, sr = librosa.load('test.wav'); print(f'SR: {sr}, Shape: {y.shape}')"

# Validate model structure
python -c "import torch; model = torch.load('model.pth'); print(model.keys())"
```

## Integration Points

### External Services

- **ElevenLabs API**: Requires API key configuration in `.env`
- **Kaggle**: For cloud training and dataset access
- **Google Colab**: For GPU-intensive training
- **Streamlit Cloud**: For web deployment

### Environment Variables

```bash
# Required for ElevenLabs
ELEVENLABS_API_KEY=your_api_key_here

# Optional for cloud services
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_kaggle_key
```

## Performance Optimization

### GPU Acceleration

```python
# Check for CUDA availability
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move models to GPU
model.to(device)

# Use mixed precision for faster training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Audio Processing Optimization

- Use librosa's built-in caching
- Process audio in batches when possible
- Pre-compute features when training multiple models
- Use multiprocessing for parallel audio processing

## Security and Privacy

### Data Handling

- Never commit voice samples or model files to version control
- Use `.gitignore` for all audio files and trained models
- Encrypt sensitive voice profiles if storing externally
- Implement proper user consent for voice data

### API Key Management

- Store API keys in `.env` file (never in code)
- Add `.env` to `.gitignore`
- Use environment-specific configurations
- Rotate API keys regularly

## Deployment Guidelines

### Production Checklist

- [ ] All models tested and validated
- [ ] Error handling comprehensive
- [ ] Logging configured and tested
- [ ] Performance benchmarks met
- [ ] Security audit completed
- [ ] User documentation updated
- [ ] Backup procedures in place

### Web Deployment

- Use Streamlit for rapid web deployment
- Configure proper CORS policies
- Implement rate limiting for API endpoints
- Use CDN for static assets
- Enable HTTPS for all connections

## Continuous Improvement

### Monitoring

- Track model performance metrics
- Monitor GPU usage and memory
- Log processing times
- Track user satisfaction scores

### Maintenance

- Regular dependency updates
- Model retraining with new data
- Performance optimization
- Security patching
- Documentation updates

## Agent-Specific Instructions

### When Working on Voice Cloning

1. **Always validate audio quality** before training
2. **Use existing preprocessing pipelines** - don't reinvent audio processing
3. **Test incrementally** - validate each step of the pipeline
4. **Document model parameters** in voice profiles
5. **Backup successful models** before major changes

### When Working on UI Development

1. **Maintain responsive UI** - use threading for long operations
2. **Provide progress feedback** - users need to know processing status
3. **Handle errors gracefully** - show user-friendly error messages
4. **Test with different screen sizes** - ensure responsive design
5. **Follow existing UI patterns** - maintain consistency

### When Working on Audio Processing

1. **Preserve audio quality** - avoid unnecessary conversions
2. **Use appropriate sample rates** - match target engine requirements
3. **Handle edge cases** - empty files, corrupt audio, etc.
4. **Optimize for memory** - process large files in chunks
5. **Validate outputs** - ensure audio integrity after processing

## Verification Commands

### Quick Health Check

```bash
# Check Python environment
python --version
pip list | grep -E "(torch|librosa|soundfile|pydub)"

# Test audio processing
python -c "import librosa, soundfile; print('Audio processing OK')"

# Test GPU (if available)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Verify project structure
ls -la dataset/ models/ output/
```

### Run Test Suite

```bash
# Basic functionality tests
python test_voice_cloning.py

# Precision cloning tests
python test_precision_cloning.py

# System validation
python validate_precision_system.py
```

## Common Tasks

### Adding a New Voice Model

1. Create dataset directory: `dataset/<speaker>/`
2. Add optimized audio clips (10+ minutes total)
3. Run feature extraction: `python voice_feature_extractor.py`
4. Create voice profile: `python enhanced_voice_cloner.py`
5. Train model: `python train_<speaker>_model.py`
6. Validate quality: `python test_indistinguishable_clone.py`
7. Deploy to models directory

### Optimizing Existing Dataset

1. Run optimization script: `python optimize_<speaker>_dataset.py`
2. Review quality metrics
3. Remove low-quality clips
4. Augment if needed: `python data_augmentation.py`
5. Retrain model with optimized data

### Debugging Model Quality Issues

1. Check training audio quality
2. Verify feature extraction results
3. Review training logs for anomalies
4. Test with known-good audio
5. Compare with baseline model performance

## Project-Specific Tools

### Dataset Optimization Scripts

- `optimize_2pac_dataset.py` - Optimize 2Pac voice dataset
- `optimize_eminem_dataset.py` - Optimize Eminem voice dataset
- `celebrity_voice_assessment.py` - Celebrity voice quality assessment

### Model Creation Scripts

- `create_2pac_model.py` - Create 2Pac voice model
- `create_pacaveli_model.py` - Create Pacaveli voice model
- `create_voice_model.py` - Generic model creation

### Training Scripts

- `train_model.py` - Basic model training
- `train_pacaveli_model.py` - Pacaveli-specific training
- `advanced_trainer.py` - Advanced training with augmentation

## Success Criteria

A task is considered complete when:

1. **Code Changes**: All changes tested and pass validation
2. **Voice Quality**: MOS > 4.0 for new models
3. **Performance**: Processing time within acceptable limits
4. **Documentation**: Relevant documentation updated
5. **Testing**: Test suite passes with no regressions
6. **Deployment**: Ready for production deployment

## Emergency Procedures

### Model Corruption Recovery

1. Restore from backup: `models/<speaker>/backup/`
2. Verify backup integrity
3. Re-train if backup unavailable
4. Update voice profiles
5. Re-test functionality

### Audio Pipeline Failure

1. Identify failure point in pipeline
2. Isolate problematic component
3. Test with known-good audio
4. Fix or replace component
5. Re-validate entire pipeline

### Production Outage

1. Check service status
2. Review error logs
3. Implement workaround if available
4. Communicate with users
5. Root cause analysis
6. Implement permanent fix