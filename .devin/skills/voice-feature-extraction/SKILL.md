# Voice Feature Extraction Skill

Extracts comprehensive voice characteristics and features from audio files for voice cloning and analysis.

## When to Use

Use this skill when you need to:
- Analyze voice characteristics for cloning
- Create voice profiles for training
- Compare different voice samples
- Extract features for machine learning
- Validate voice model inputs
- Research voice characteristics

## What It Does

This skill provides detailed voice analysis including:
- Pitch and fundamental frequency analysis
- Spectral envelope extraction
- Timbre and voice quality characteristics
- Prosody and rhythm analysis
- Energy and dynamics measurement
- Comprehensive voice profiling

## How to Use

### Basic Usage

```
Extract voice features from the audio file dataset/2pac/sample_001.wav.
```

### Advanced Usage

```
Extract comprehensive voice features with the following parameters:
- Audio file: dataset/2pac/sample_001.wav
- Analysis type: full (pitch, spectral, timbre, prosody)
- Output format: JSON
- Output path: models/2pac/voice_profile.json
- Visualizations: enabled
- Quality assessment: enabled
```

## Feature Categories

### Pitch Features
- **Fundamental Frequency (F0)**: Base pitch in Hz
- **Pitch Range**: Min/max pitch values
- **Pitch Variance**: Statistical pitch variation
- **Pitch Contour**: Pitch over time trajectory
- **Voiced/Unvoiced Ratio**: Proportion of voiced speech

### Spectral Features
- **Spectral Envelope**: Frequency response characteristics
- **Formant Frequencies**: Resonant frequencies (F1, F2, F3)
- **Spectral Centroid**: Frequency center of mass
- **Spectral Bandwidth**: Frequency spread
- **Spectral Roll-off**: High-frequency cutoff
- **MFCCs**: Mel-frequency cepstral coefficients

### Timbre Features
- **Harmonics-to-Noise Ratio (HNR)**: Voice quality measure
- **Jitter**: Pitch perturbation (frequency stability)
- **Shimmer**: Amplitude perturbation (amplitude stability)
- **Voice Quality**: Overall timbre characteristics
- **Breathiness**: Air flow in voice
- **Roughness**: Perceived roughness

### Prosody Features
- **Speech Rate**: Syllables per second
- **Pause Duration**: Timing and rhythm patterns
- **Intensity Contour**: Volume variation over time
- **Rhythm Patterns**: Timing regularity
- **Stress Patterns**: Emphasis and accent patterns

### Energy Features
- **RMS Energy**: Overall signal power
- **Dynamic Range**: Amplitude variation
- **Energy Distribution**: Frequency-band energy
- **Attack/Decay**: Note onset/offset characteristics

## Implementation Notes

The skill uses existing feature extraction infrastructure:
- `voice_feature_extractor.py` - Core feature extraction
- `enhanced_voice_cloner.py` - Voice profiling
- `voice_conversion_engine.py` - Advanced analysis
- `voice_quality_assurance.py` - Quality metrics

## Extraction Process

1. **Audio Preprocessing**
   - Load and validate audio
   - Normalize audio levels
   - Remove silence segments
   - Resample to target rate

2. **Feature Computation**
   - Extract pitch characteristics
   - Compute spectral features
   - Analyze timbre qualities
   - Measure prosody patterns
   - Calculate energy metrics

3. **Feature Aggregation**
   - Statistical summarization (mean, std, min, max)
   - Temporal segmentation analysis
   - Feature correlation analysis
   - Quality assessment

4. **Output Generation**
   - Generate feature vectors
   - Create voice profiles
   - Produce visualizations
   - Generate analysis reports

## Output Formats

### JSON Voice Profile
```json
{
  "speaker": "2pac",
  "extraction_date": "2024-08-19",
  "audio_quality": {
    "snr_db": 28.5,
    "dynamic_range_db": 55.2,
    "duration_seconds": 12.3
  },
  "pitch": {
    "mean_hz": 145.2,
    "std_hz": 23.1,
    "min_hz": 98.5,
    "max_hz": 198.3,
    "range_hz": 99.8
  },
  "spectral": {
    "formants": [650, 1450, 2450],
    "centroid_hz": 1250.5,
    "bandwidth_hz": 890.2,
    "mfcc": [...]
  },
  "timbre": {
    "hnr_db": 15.2,
    "jitter_percent": 0.8,
    "shimmer_percent": 1.2,
    "voice_quality": "breathy"
  },
  "prosody": {
    "speech_rate_syllables_sec": 4.5,
    "pause_ratio": 0.15,
    "rhythm_regularity": 0.78
  }
}
```

### Feature Vectors
- NumPy arrays for machine learning
- Normalized feature vectors
- Time-series feature data
- Statistical summaries

### Visualizations
- Pitch contour plots
- Spectral spectrograms
- Formant trajectories
- Feature distribution charts
- Comparison visualizations

## Quality Assessment

### Audio Quality Metrics
- **SNR**: Signal-to-noise ratio (>20dB acceptable)
- **Dynamic Range**: Amplitude variation (>40dB acceptable)
- **Clipping**: Distortion detection
- **Background Noise**: Noise level assessment

### Feature Quality Metrics
- **Feature Validity**: Range and consistency checks
- **Extraction Confidence**: Reliability of extracted features
- **Artifact Detection**: Identification of extraction errors
- **Completeness**: All required features extracted

## Error Handling

Common issues and resolutions:
- **Silent Audio**: Check audio levels, verify file integrity
- **Poor Pitch Detection**: Improve audio quality, adjust F0 range
- **Feature Extraction Failures**: Verify dependencies, check audio format
- **Inconsistent Results**: Ensure consistent preprocessing
- **Memory Issues**: Process in segments, reduce resolution

## Best Practices

1. **Use high-quality audio** for reliable feature extraction
2. **Normalize audio** before processing
3. **Validate audio quality** before extraction
4. **Use appropriate sample rates** for target analysis
5. **Check feature ranges** for consistency
6. **Document extraction parameters** for reproducibility

## Advanced Features

### Comparative Analysis
- **Voice Similarity**: Compare different voice samples
- **Feature Clustering**: Group similar voice characteristics
- **Dimensionality Reduction**: PCA, t-SNE visualization
- **Statistical Testing**: Significance testing between groups

### Temporal Analysis
- **Segment-wise Features**: Features over time windows
- **Trajectory Analysis**: How features change over time
- **Event Detection**: Detect voice events and boundaries
- **Pattern Recognition**: Identify recurring patterns

### Machine Learning Integration
- **Feature Selection**: Identify most discriminative features
- **Feature Engineering**: Create derived features
- **Normalization**: Standardize feature ranges
- **Dimensionality Reduction**: Reduce feature complexity

## Applications

### Voice Cloning
- Create comprehensive voice profiles
- Identify unique voice characteristics
- Guide model training parameters
- Validate cloned voice quality

### Voice Analysis
- Speaker identification
- Emotion recognition
- Voice disorder detection
- Forensic voice analysis

### Music Production
- Vocal characteristic analysis
- Style transfer preparation
- Mixing and mastering guidance
- Vocal effect design

### Research
- Voice characteristic studies
- Accent and dialect analysis
- Singing voice research
- Voice development tracking

## Performance Considerations

### Processing Time
- **Basic Analysis**: 0.5-2 seconds per minute of audio
- **Full Analysis**: 2-5 seconds per minute of audio
- **Batch Processing**: Near-linear scaling
- **GPU Acceleration**: Available for some features

### Resource Requirements
- **CPU**: Modern multi-core processor recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10MB per minute of audio for features
- **GPU**: Optional for acceleration

## Integration Examples

### Python API

```python
from voice_feature_extractor import extract_features

# Basic extraction
features = extract_features(
    audio_path="dataset/2pac/sample.wav",
    analysis_type="basic"
)

# Advanced extraction
features = extract_features(
    audio_path="dataset/2pac/sample.wav",
    analysis_type="full",
    output_format="json",
    visualizations=True,
    quality_assessment=True
)
```

### Command Line

```bash
# Basic extraction
python -m voice_feature_extractor --audio dataset/2pac/sample.wav

# Advanced extraction
python -m voice_feature_extractor \
    --audio dataset/2pac/sample.wav \
    --analysis full \
    --output models/2pac/voice_profile.json \
    --visualizations \
    --quality-check
```

## Troubleshooting

### Common Issues

**Poor Pitch Detection**:
- Improve audio quality
- Adjust F0 range parameters
- Check for music or background noise
- Use alternative pitch detection method

**Inconsistent Features**:
- Ensure consistent preprocessing
- Check audio sample rate
- Verify normalization parameters
- Review extraction settings

**Memory Errors**:
- Process audio in segments
- Reduce feature resolution
- Close other applications
- Increase system memory

**Extraction Failures**:
- Verify audio file format
- Check dependency versions
- Review error logs
- Test with known-good audio

## Quality Guidelines

### Acceptable Feature Quality
- **Pitch Detection**: >90% voiced frames detected
- **Spectral Quality**: Clear formant structure
- **Timbre Analysis**: Consistent measurements
- **Prosody Features**: Detectable speech patterns

### Excellent Feature Quality
- **Pitch Detection**: >95% voiced frames detected
- **Spectral Quality**: Well-defined formants
- **Timbre Analysis**: Low variance in repeated measures
- **Prosody Features**: Clear rhythmic patterns

## Documentation Standards

### Feature Reports
- Include extraction parameters
- Document audio quality metrics
- Provide feature statistics
- Note any quality issues
- Include visualization references

### Metadata Standards
- Speaker identification
- Recording conditions
- Audio preprocessing steps
- Extraction software version
- Quality assessment results