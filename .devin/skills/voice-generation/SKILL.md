# Voice Generation Skill

Generates high-quality synthetic speech using trained voice models and various TTS engines.

## When to Use

Use this skill when you need to:
- Generate speech from text using a cloned voice
- Convert audio from one voice to another
- Create vocal tracks for music production
- Generate voiceovers for content
- Test voice model quality with new content

## What It Does

This skill provides comprehensive voice generation including:
- Text-to-speech synthesis with cloned voices
- Audio-to-audio voice conversion
- Batch processing of multiple texts/audio files
- Quality optimization and enhancement
- Format conversion and post-processing
- Real-time generation capabilities

## How to Use

### Basic Usage

```
Generate speech from text "Hello world" using the 2pac voice model.
```

### Advanced Usage

```
Generate speech with the following parameters:
- Text: "The quick brown fox jumps over the lazy dog"
- Voice model: models/2pac/model.pth
- Engine: elevenlabs (highest quality)
- Output format: WAV
- Sample rate: 48000
- Output path: output/generated/2pac_test.wav
- Quality: maximum
```

## Generation Engines

### ElevenLabs Engine (Best Quality)
- **Quality**: Professional, near-human
- **Speed**: Moderate (API-based)
- **Cost**: Requires API key ($5/mo minimum)
- **Best For**: Professional voiceovers, high-end applications

### Qwen3-TTS Engine (Best Free)
- **Quality**: Very good with 3-second reference
- **Speed**: Fast (local processing)
- **Cost**: Free
- **Best For**: Rapid prototyping, cost-effective solutions

### XTTS v2 Engine (Good Free)
- **Quality**: Good zero-shot cloning
- **Speed**: Moderate (local processing)
- **Cost**: Free (1.8GB model download)
- **Best For**: General purpose, offline usage

### RVC v2 Engine (Best for Audio Conversion)
- **Quality**: Excellent for rap/singing
- **Speed**: Fast (local processing)
- **Cost**: Free after model training
- **Best For**: Music production, audio-to-audio conversion

### SO-VITS-SVC Engine (Singing Focus)
- **Quality**: Excellent for singing voice
- **Speed**: Moderate
- **Cost**: Free
- **Best For**: Vocal tracks, music production

## Implementation Notes

The skill uses existing generation infrastructure:
- `elevenlabs_engine.py` - ElevenLabs API integration
- `qwen3_tts_engine.py` - Qwen3-TTS local generation
- `xtts_engine.py` - XTTS v2 zero-shot generation
- `rvc_engine.py` - RVC v2 voice conversion
- `svc_engine.py` - SO-VITS-SVC generation
- `voice_conversion_engine.py` - Advanced voice conversion

## Generation Types

### Text-to-Speech (TTS)
Converts written text to synthetic speech using cloned voice characteristics.

**Parameters:**
- `text`: Input text to synthesize
- `voice_model`: Target voice model
- `engine`: TTS engine to use
- `output_format`: WAV, MP3, FLAC
- `sample_rate`: 22050, 44100, 48000
- `speed`: Speech rate adjustment (0.5-2.0)
- `pitch_shift`: Pitch adjustment in semitones

### Audio-to-Audio Conversion
Transforms existing audio from one voice to another while preserving content.

**Parameters:**
- `source_audio`: Input audio file
- `voice_model`: Target voice model
- `engine`: Conversion engine
- `pitch_preservation`: Maintain original pitch
- `timing_preservation`: Maintain original timing
- `output_format`: WAV, MP3, FLAC

### Batch Generation
Processes multiple texts or audio files in sequence.

**Parameters:**
- `input_list`: List of texts or audio files
- `voice_model`: Target voice model
- `output_directory`: Output folder
- `naming_pattern`: File naming convention
- `parallel_processing`: Enable parallel processing

## Quality Optimization

### Pre-Generation
- **Text Preprocessing**: Clean and normalize text
- **Audio Preprocessing**: Optimize source audio quality
- **Model Selection**: Choose appropriate model for content type
- **Parameter Tuning**: Optimize generation parameters

### Post-Generation
- **Audio Enhancement**: Remove artifacts, improve clarity
- **Format Optimization**: Choose appropriate format/bitrate
- **Quality Validation**: Assess output quality
- **Iterative Refinement**: Adjust parameters based on results

## Output

The skill generates:
- **Generated Audio**: High-quality synthetic speech
- **Generation Metadata**: Processing parameters and timing
- **Quality Metrics**: Objective quality assessment
- **Comparison Data**: Side-by-side with reference if provided
- **Processing Logs**: Detailed generation information

## Error Handling

Common issues and resolutions:
- **API Rate Limits**: Implement rate limiting, use caching
- **Model Loading Errors**: Verify model integrity, check compatibility
- **Poor Quality Output**: Check input quality, adjust parameters
- **Generation Failures**: Verify engine availability, check dependencies
- **Memory Issues**: Process in batches, reduce quality settings

## Best Practices

1. **Test with short samples** before batch processing
2. **Use appropriate engine** for your use case and budget
3. **Optimize input quality** for best output quality
4. **Monitor generation progress** for long processes
5. **Validate output quality** before deployment
6. **Keep generation logs** for debugging and optimization

## Performance Considerations

### Processing Time
- **ElevenLabs**: 1-3 seconds per sentence (API latency)
- **Qwen3-TTS**: 0.5-2 seconds per sentence (local)
- **XTTS v2**: 2-5 seconds per sentence (local)
- **RVC v2**: Real-time to 2x real-time (local)
- **SO-VITS-SVC**: 1-3x real-time (local)

### Resource Requirements
- **CPU**: Modern multi-core processor recommended
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional but recommended for local engines
- **Storage**: 500MB for models, 10MB per minute of audio

## Advanced Features

### Prosody Control
- **Speech Rate**: Adjust speaking speed
- **Pitch Adjustment**: Modify pitch characteristics
- **Emotion Injection**: Add emotional coloring
- **Style Transfer**: Apply speaking styles

### Voice Blending
- **Mixed Voices**: Blend multiple voice models
- **Voice Interpolation**: Create intermediate voices
- **Dynamic Voice Switching**: Change voices within content

### Real-Time Generation
- **Streaming**: Generate audio in real-time
- **Low Latency**: Minimize generation delay
- **Live Processing**: Process live audio input

## Quality Assessment

### Generation Quality Checklist
- [ ] Natural speech rhythm and timing
- [ ] Appropriate pitch and intonation
- [ ] Clear articulation and pronunciation
- [ ] Minimal artifacts or distortions
- [ ] Consistent voice characteristics
- [ ] Suitable emotional expression
- [ ] Intelligible content

### Use Case Guidelines

**Professional Voiceover**:
- Use ElevenLabs engine
- Maximum quality settings
- Post-processing enhancement
- Human review required

**Music Production**:
- Use RVC v2 or SO-VITS-SVC
- Preserve musical timing
- High sample rate (48kHz)
- Integration with DAW

**Content Creation**:
- Use Qwen3-TTS or XTTS v2
- Balance quality and speed
- Batch processing for efficiency
- Automated quality checks

**Prototyping/Testing**:
- Use any available engine
- Default quality settings
- Fast iteration
- Quick validation

## Integration Examples

### Python API

```python
from voice_generation import generate_speech

# Basic generation
audio = generate_speech(
    text="Hello world",
    voice_model="models/2pac/model.pth",
    engine="elevenlabs"
)

# Advanced generation
audio = generate_speech(
    text="The quick brown fox",
    voice_model="models/2pac/model.pth",
    engine="qwen3_tts",
    output_format="wav",
    sample_rate=48000,
    speed=1.1,
    pitch_shift=2
)
```

### Command Line

```bash
# Basic generation
python -m voice_generation --text "Hello world" --model models/2pac/model.pth

# Advanced generation
python -m voice_generation \
    --text "The quick brown fox" \
    --model models/2pac/model.pth \
    --engine elevenlabs \
    --output output/generated.wav \
    --sample-rate 48000
```

## Troubleshooting

### Common Issues

**Unnatural Speech**:
- Check text preprocessing
- Verify model training quality
- Adjust generation parameters
- Try different engine

**Audio Artifacts**:
- Check input audio quality
- Reduce generation speed
- Enable post-processing
- Update audio drivers

**Slow Generation**:
- Check system resources
- Enable GPU acceleration
- Reduce quality settings
- Use faster engine

**Engine Failures**:
- Verify API keys (for cloud engines)
- Check model file integrity
- Update dependencies
- Review error logs