# AI Vocals Studio - Configuration Guide

This directory contains comprehensive configuration for the AI Vocals Studio project, optimized for voice cloning and AI vocals generation workflows.

## Configuration Structure

```
.devin/
├── AGENTS.md                    # Comprehensive agent workflow guidelines
├── config.json                  # Project configuration and preferences
├── environment.yaml             # Environment setup and dependencies
├── mcp-config.json             # MCP server configuration (placeholders)
├── .gitignore                  # Comprehensive gitignore for voice projects
├── skills/                     # Voice cloning specific skills
│   ├── voice-dataset-optimization/
│   ├── voice-model-training/
│   ├── voice-quality-assessment/
│   ├── voice-generation/
│   └── voice-feature-extraction/
└── README.md                   # This file
```

## Key Configuration Files

### AGENTS.md
Comprehensive guidelines for AI agents working on voice cloning tasks including:
- Project architecture and data flow
- Development workflows and best practices
- Audio processing standards and quality metrics
- Testing and validation procedures
- Troubleshooting guides and emergency procedures

### config.json
Project configuration covering:
- Audio processing parameters (sample rates, formats, quality thresholds)
- Training configurations (batch sizes, learning rates, epochs)
- Generation settings (engines, output formats, quality options)
- Quality assessment thresholds and validation rules
- UI preferences and processing options
- Security and integration settings

### environment.yaml
Environment setup specification including:
- Python version requirements (3.10+, recommended 3.12)
- Core dependencies for audio processing and ML
- Optional dependencies for cloud integration
- System requirements (GPU recommendations)
- Installation and verification procedures

### mcp-config.json
MCP (Model Context Protocol) server configuration with:
- Recommended external services for voice workflows
- Cloud storage, model hosting, and API management options
- Placeholder configuration for future integrations
- Service provider recommendations

## Skills Configuration

The project includes 5 specialized skills for voice cloning workflows:

### 1. voice-dataset-optimization
Optimizes raw audio files for voice cloning training:
- Audio normalization and standardization
- Noise reduction and silence trimming
- Quality assessment and validation
- Dataset statistics and reporting

### 2. voice-model-training
Trains high-quality voice cloning models:
- Multiple engine support (SO-VITS-SVC, RVC, XTTS)
- Training parameter configuration
- Quality monitoring and validation
- Model optimization and export

### 3. voice-quality-assessment
Comprehensive quality evaluation:
- Objective metrics (SNR, spectral analysis, dynamic range)
- Subjective testing (MOS, ABX tests)
- Indistinguishability testing
- Production readiness assessment

### 4. voice-generation
Generates synthetic speech using cloned voices:
- Multiple TTS engines (ElevenLabs, Qwen3-TTS, XTTS)
- Text-to-speech and audio-to-audio conversion
- Batch processing capabilities
- Quality optimization and enhancement

### 5. voice-feature-extraction
Extracts comprehensive voice characteristics:
- Pitch, spectral, and timbre analysis
- Prosody and rhythm measurement
- Voice profiling and comparison
- Machine learning feature preparation

## Usage Examples

### Basic Workflow

1. **Optimize Dataset**
   ```
   Use the voice-dataset-optimization skill to prepare audio files
   ```

2. **Extract Features**
   ```
   Use the voice-feature-extraction skill to analyze voice characteristics
   ```

3. **Train Model**
   ```
   Use the voice-model-training skill to create voice model
   ```

4. **Assess Quality**
   ```
   Use the voice-quality-assessment skill to validate model quality
   ```

5. **Generate Voice**
   ```
   Use the voice-generation skill to create synthetic speech
   ```

### Configuration Updates

To modify project settings:

1. Edit `config.json` for runtime parameters
2. Update `environment.yaml` for dependency changes
3. Modify skill SKILL.md files for workflow adjustments
4. Update AGENTS.md for agent behavior changes

## Best Practices

### Security
- Never commit voice samples or trained models
- Keep API keys in environment variables
- Use `.devin/.gitignore` for sensitive files
- Enable API key validation in production

### Quality Standards
- Target SNR >25dB for training data
- Use minimum 30 minutes of audio for professional quality
- Validate models before deployment
- Monitor quality metrics continuously

### Performance
- Enable GPU acceleration when available
- Use mixed precision for faster training
- Process audio in batches for efficiency
- Enable caching for repeated operations

## Integration Points

### External Services
The configuration supports integration with:
- **ElevenLabs API** - High-quality TTS (requires API key)
- **Kaggle** - Cloud training and datasets
- **Google Colab** - GPU-intensive training
- **Streamlit Cloud** - Web deployment

### MCP Services
Ready for integration with:
- Cloud storage (AWS S3, Google Cloud Storage)
- Model hosting (AWS SageMaker, Hugging Face)
- Database services (PostgreSQL, MongoDB)
- API management (AWS API Gateway)

## Verification

To verify the configuration:

```bash
# Check configuration files exist
ls -la .devin/

# Verify environment setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import librosa, soundfile; print('Audio processing OK')"

# Test skill availability
skill list
```

## Troubleshooting

### Configuration Issues
- Ensure all JSON files are valid
- Check Python version compatibility
- Verify dependencies are installed
- Review error logs in `logs/` directory

### Performance Issues
- Enable GPU acceleration in config.json
- Reduce batch sizes for memory constraints
- Enable mixed precision training
- Use appropriate audio sample rates

### Quality Issues
- Verify training audio quality
- Check feature extraction results
- Review training logs for anomalies
- Validate with test scripts

## Maintenance

### Regular Updates
- Keep dependencies updated
- Review and update quality thresholds
- Add new engines as they become available
- Update documentation with new features

### Monitoring
- Track model performance metrics
- Monitor GPU usage and memory
- Log processing times
- Review user feedback

## Support

For issues or questions:
1. Check AGENTS.md for workflow guidance
2. Review skill documentation for specific tasks
3. Consult config.json for parameter reference
4. Check project logs for error details

## Contributing

When adding new features:
1. Update relevant configuration files
2. Add or modify skills as needed
3. Update AGENTS.md with new workflows
4. Test thoroughly before deployment
5. Document changes in commit messages

---

This configuration provides a complete foundation for professional voice cloning workflows with comprehensive agent guidance, specialized skills, and optimal project settings.