# Voice Model Training Skill

Trains high-quality voice cloning models from optimized datasets using various AI engines and training configurations.

## When to Use

Use this skill when you need to:
- Train a new voice model from scratch
- Re-train existing models with improved data
- Fine-tune pre-trained models for specific speakers
- Compare different training configurations
- Optimize model performance for specific use cases

## What It Does

This skill provides comprehensive model training including:
- Configuration of training parameters
- Data preparation and validation
- Model architecture selection
- Training execution with monitoring
- Validation and quality assessment
- Model optimization and export

## How to Use

### Basic Usage

```
Train a voice model for speaker "Pacaveli" using the dataset in dataset/Pacaveli/.
```

### Advanced Usage

```
Train a voice model with the following configuration:
- Speaker: Pacaveli
- Dataset: dataset/Pacaveli/ (optimized)
- Engine: SO-VITS-SVC
- Batch size: 12
- Learning rate: 0.0001
- Epochs: 2000
- Validation split: 15%
- GPU acceleration: enabled
- Output: models/Pacaveli/model.pth
```

## Training Engines

### SO-VITS-SVC (Recommended)
- Best for singing voice conversion
- Requires 22.05kHz audio
- Training time: 2-8 hours (GPU)
- Quality: Professional

### RVC v2
- Best for rap and speech
- Faster training (1-4 hours)
- Lower computational requirements
- Quality: Very Good

### XTTS v2
- Zero-shot voice cloning
- No training required for basic use
- Fine-tuning available for quality improvement
- Quality: Good to Excellent

## Implementation Notes

The skill uses existing training infrastructure:
- `voice_trainer.py` - Core training functionality
- `train_model.py` - Basic training script
- `train_pacaveli_model.py` - Speaker-specific training
- `advanced_trainer.py` - Advanced training with augmentation
- `precision_voice_cloning_system.py` - High-precision training

## Training Requirements

### Minimum Requirements
- **Audio Duration**: 10 minutes (basic), 30+ minutes (professional)
- **Audio Quality**: SNR >20dB, clean vocals
- **Hardware**: GPU with 4GB+ VRAM recommended
- **Time**: 2-8 hours depending on dataset size

### Optimal Requirements
- **Audio Duration**: 60+ minutes with variety
- **Audio Quality**: SNR >30dB, professional recording
- **Hardware**: GPU with 8GB+ VRAM
- **Time**: 4-12 hours for maximum quality

## Configuration Parameters

### Core Parameters
- `batch_size`: 8-16 (adjust based on GPU memory)
- `learning_rate`: 0.0001 (default), 0.00001 (fine-tuning)
- `epochs`: 1000-5000 (monitor validation loss)
- `validation_split`: 0.10-0.15
- `sample_rate`: 22050 (SO-VITS-SVC), 48000 (modern)

### Advanced Parameters
- `gradient_accumulation`: 1-4 (for larger effective batch sizes)
- `mixed_precision`: True (faster training, lower memory)
- `data_augmentation`: True (improves robustness)
- `early_stopping`: True (prevents overfitting)

## Output

The skill generates:
- Trained model file (`model.pth`)
- Voice profile JSON with speaker characteristics
- Training logs and metrics
- Validation results and quality scores
- Model performance report

## Quality Assessment

### Objective Metrics
- **Validation Loss**: Should decrease and stabilize
- **Spectral Similarity**: >0.85 correlation
- **MOS Score**: >3.5 (acceptable), >4.5 (excellent)

### Subjective Testing
- Listening tests with original vs generated
- ABX tests for naturalness
- Real-world scenario testing

## Error Handling

Common issues and resolutions:
- **CUDA Out of Memory**: Reduce batch size, use gradient accumulation
- **Poor Convergence**: Adjust learning rate, check data quality
- **Overfitting**: Enable early stopping, add data augmentation
- **Slow Training**: Enable mixed precision, increase batch size

## Best Practices

1. **Always validate data quality** before training
2. **Start with default parameters**, then fine-tune
3. **Monitor training progress** closely
4. **Save checkpoints** regularly
5. **Test incrementally** - don't wait for final model
6. **Keep training logs** for analysis and debugging

## Post-Training Steps

1. **Validate model quality** using test scripts
2. **Run indistinguishability tests** for critical applications
3. **Optimize model** for deployment (quantization, pruning)
4. **Create backup** of successful models
5. **Document model parameters** and performance
6. **Deploy to production** after validation