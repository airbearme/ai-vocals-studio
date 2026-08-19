# Voice Dataset Optimization Skill

Optimizes voice datasets for maximum voice cloning accuracy by processing raw audio files to meet professional standards.

## When to Use

Use this skill when you need to:
- Prepare raw audio files for voice cloning training
- Improve quality of existing voice datasets
- Standardize audio formats and specifications
- Remove noise and artifacts from voice recordings
- Validate dataset quality before training

## What It Does

This skill provides automated audio optimization including:
- Audio normalization to professional standards
- Silence trimming and cleanup
- Background noise reduction
- Sample rate conversion and standardization
- Quality assessment and validation
- Dataset statistics and reporting

## How to Use

### Basic Usage

```
Optimize the dataset for speaker "2pac" to prepare for model training.
```

### Advanced Usage

```
Optimize the dataset in dataset/2pac/ with the following parameters:
- Target sample rate: 22050Hz
- Normalization level: -1.0dB
- Silence threshold: -40dB
- Remove background noise: enabled
- Output directory: dataset/2pac_optimized/
```

## Implementation Notes

The skill uses the existing optimization scripts:
- `optimize_2pac_dataset.py` - For 2Pac specific optimization
- `optimize_eminem_dataset.py` - For Eminem specific optimization
- `advanced_audio_processor.py` - General audio processing
- `voice_quality_assurance.py` - Quality validation

## Quality Standards

- **Sample Rate**: 22.05kHz (SO-VITS-SVC) or 44.1kHz/48kHz (modern engines)
- **Bit Depth**: 16-bit or 24-bit
- **Format**: WAV (preferred), FLAC acceptable
- **SNR**: >20dB minimum, >30dB excellent
- **Duration**: 3-30 seconds per clip optimal
- **Consistency**: Uniform volume and quality across dataset

## Output

The skill generates:
- Optimized audio files in target directory
- Quality assessment report
- Dataset statistics (clip count, total duration, quality metrics)
- Recommendations for improvements

## Error Handling

Common issues and resolutions:
- **Corrupt audio files**: Skipped with logging
- **Incompatible formats**: Converted to WAV
- **Poor quality clips**: Flagged for review
- **Insufficient data**: Warning if <10 minutes total