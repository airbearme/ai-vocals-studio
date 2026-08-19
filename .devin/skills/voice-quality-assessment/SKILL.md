# Voice Quality Assessment Skill

Comprehensive quality assessment and validation for voice cloning models and generated audio.

## When to Use

Use this skill when you need to:
- Validate voice cloning model quality
- Assess generated audio naturalness
- Compare different voice models
- Debug voice quality issues
- Ensure production readiness of voice models

## What It Does

This skill provides thorough quality analysis including:
- Objective audio quality metrics (SNR, dynamic range, spectral analysis)
- Subjective quality assessment frameworks
- Voice characteristic analysis (pitch, timbre, prosody)
- Indistinguishability testing
- Comparative model evaluation
- Quality reporting and recommendations

## How to Use

### Basic Usage

```
Assess the quality of the voice model for speaker "2pac" in models/2pac/.
```

### Advanced Usage

```
Perform comprehensive quality assessment for the Pacaveli model:
- Model: models/Pacaveli/model.pth
- Test audio: test_samples/pacaveli_test.wav
- Reference audio: reference/original_pacaveli.wav
- Assessment type: full (objective + subjective)
- Output: quality_report/pacaveli_assessment.json
```

## Assessment Types

### Objective Assessment
- **SNR Analysis**: Signal-to-noise ratio measurement
- **Dynamic Range**: Amplitude variation analysis
- **Spectral Analysis**: Frequency domain characteristics
- **Temporal Analysis**: Timing and rhythm assessment
- **Feature Consistency**: Voice characteristic stability

### Subjective Assessment
- **MOS Testing**: Mean Opinion Score evaluation
- **ABX Testing**: Blind comparison tests
- **Naturalness Rating**: Perceptual quality assessment
- **Speaker Similarity**: Voice characteristic matching
- **Intelligibility Testing**: Speech clarity assessment

### Indistinguishability Testing
- **Turing Test**: Can listeners distinguish real from generated?
- **Forensic Analysis**: Statistical detection of synthetic audio
- **Expert Evaluation**: Professional voice analyst assessment
- **Real-World Testing**: Deployment scenario validation

## Implementation Notes

The skill uses existing assessment infrastructure:
- `voice_quality_assurance.py` - Core quality assessment
- `test_indistinguishable_clone.py` - Indistinguishability testing
- `test_precision_cloning.py` - Precision cloning validation
- `validate_precision_system.py` - System validation
- `celebrity_voice_assessment.py` - Celebrity voice evaluation

## Quality Metrics

### Objective Metrics Thresholds

| Metric | Acceptable | Good | Excellent |
|--------|------------|------|-----------|
| SNR | >20dB | >25dB | >30dB |
| Dynamic Range | >40dB | >50dB | >60dB |
| Spectral Correlation | >0.75 | >0.85 | >0.95 |
| Pitch Stability | <15% variance | <10% variance | <5% variance |
| Timbre Consistency | >0.70 | >0.80 | >0.90 |

### Subjective Metrics Thresholds

| Metric | Acceptable | Good | Excellent |
|--------|------------|------|-----------|
| MOS Score | >3.0 | >3.5 | >4.5 |
| Naturalness | >60% | >75% | >90% |
| Speaker Similarity | >70% | >85% | >95% |
| Intelligibility | >80% | >90% | >98% |

## Assessment Workflow

1. **Preparation**
   - Load model and reference audio
   - Prepare test samples
   - Configure assessment parameters

2. **Objective Analysis**
   - Compute audio quality metrics
   - Analyze voice characteristics
   - Generate objective scores

3. **Subjective Testing**
   - Conduct listening tests
   - Perform ABX comparisons
   - Collect subjective ratings

4. **Comprehensive Evaluation**
   - Combine objective and subjective metrics
   - Generate quality report
   - Provide recommendations

## Output

The skill generates:
- **Quality Report**: Comprehensive JSON report with all metrics
- **Audio Analysis**: Detailed spectral and temporal analysis
- **Comparison Charts**: Visual comparison with reference
- **Recommendations**: Specific improvement suggestions
- **Pass/Fail Assessment**: Production readiness determination

## Error Handling

Common issues and resolutions:
- **Insufficient Test Data**: Generate additional test samples
- **Reference Quality Issues**: Use higher quality reference audio
- **Inconsistent Results**: Increase sample size for statistical significance
- **Metric Conflicts**: Prioritize critical metrics for use case

## Interpretation Guide

### Quality Score Interpretation

**90-100 (Excellent)**: Ready for production deployment
- Indistinguishable from original voice
- Suitable for professional applications
- No significant quality issues

**75-89 (Good)**: Ready for most applications
- High quality with minor imperfections
- Suitable for commercial use
- May need minor refinements

**60-74 (Acceptable)**: Suitable for non-critical applications
- Noticeable but not distracting artifacts
- Acceptable for internal/testing use
- Needs improvement for production

**<60 (Poor)**: Not ready for deployment
- Significant quality issues
- Requires retraining or data improvement
- Not suitable for intended use case

## Best Practices

1. **Use multiple assessment types** for comprehensive evaluation
2. **Include diverse test samples** (different speech content, emotions)
3. **Compare against appropriate baselines** (original voice, similar models)
4. **Consider use case requirements** (talking vs singing, casual vs professional)
5. **Document assessment context** (test conditions, evaluator expertise)
6. **Track quality over time** (model degradation, improvement trends)

## Common Issues and Solutions

### Low SNR Scores
- **Cause**: Background noise in training data
- **Solution**: Improve dataset quality, use noise reduction

### Poor Speaker Similarity
- **Cause**: Insufficient training data or poor data quality
- **Solution**: Add more diverse, high-quality samples

### Inconsistent Quality
- **Cause**: Overfitting to specific audio characteristics
- **Solution**: Add data augmentation, increase regularization

### Unnatural Prosody
- **Cause**: Poor pitch/rhythm modeling
- **Solution**: Improve feature extraction, adjust training parameters

## Production Readiness Checklist

- [ ] All objective metrics above acceptable thresholds
- [ ] Subjective testing meets MOS requirements
- [ ] Indistinguishability testing passed
- [ ] Real-world scenario testing successful
- [ ] Performance benchmarks met
- [ ] Error handling validated
- [ ] Documentation complete
- [ ] Backup procedures in place