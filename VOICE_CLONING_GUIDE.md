# AI Vocals Studio - Voice Cloning Guide

## 🎤 Overview

Your AI Vocals Studio is now set up for voice cloning and voice changing in music! Here's what has been configured and how to use it.

## ✅ What's Been Set Up

### 1. ML Dependencies Installed
- **so-vits-svc-fork**: Advanced voice conversion and cloning library
- **demucs**: Music source separation (vocal isolation)
- **librosa**: Audio analysis and processing
- **torch/torchaudio**: Deep learning framework
- **pydub**: Audio manipulation
- **gtts**: Text-to-speech

### 2. Voice Cloning Models Created
We've created voice models from your existing datasets:
- **test speaker_cloned**: Created from 13 audio files in dataset/test speaker/
- **w2re_cloned**: Created from 13 audio files in dataset/w2re/

These models capture voice characteristics and can be used for voice transformation.

### 3. Available Applications
- **app_studio.py**: Full-featured studio app with advanced effects
- **app_modern.py**: Modern UI with model management
- **app_minimal.py**: Lightweight version (current default)
- **simple_voice_clone.py**: Script to create new voice models
- **test_voice_conversion.py**: Script to test voice conversion

## 🚀 How to Use Voice Cloning

### Step 1: Launch the Application
```bash
./run.sh
```
This will activate the virtual environment and launch the main application.

### Step 2: Import Your Voice Samples
1. Go to the **Training Data** tab
2. Click **"Import Single File"** or **"Import Entire Folder"**
3. Select your voice recordings (WAV, MP3, FLAC, M4A, OGG)
4. The files will be copied to the `dataset/` directory

### Step 3: Create a Voice Model
You have two options:

#### Option A: Quick Model Creation (Recommended)
```bash
source venv/bin/activate
python3 simple_voice_clone.py
```
This creates a voice model using DSP effects and voice analysis.

#### Option B: Full ML Training (Advanced)
For professional-quality voice cloning, you can train a full ML model:
```bash
source venv/bin/activate
python3 test_voice_cloning.py
```
Note: This can take several hours depending on your CPU.

### Step 4: Generate/Convert Audio
1. Go to the **Generate** tab
2. Select your voice model from the dropdown
3. Choose input type:
   - **Text**: Enter text to convert to speech
   - **Audio**: Select an audio file to transform
4. Click **"Generate Vocals"**
5. The output will be saved to the `output/` directory

## 🎵 Voice Changing in Music

### Method 1: Using the Studio App
1. Launch `app_studio.py` for the best experience
2. Select a music file with vocals
3. Choose your cloned voice model
4. Adjust effects (pitch, speed, reverb, etc.)
5. Generate the transformed audio

### Method 2: Using Test Script
```bash
source venv/bin/activate
python3 test_voice_conversion.py
```
This will test voice conversion with your available models.

### Method 3: Manual Processing
For more control, you can use the voice conversion directly in your code:

```python
from svc_engine import SoVitsEngine

engine = SoVitsEngine('models', 'dataset')

# Convert audio to cloned voice
ok, err = engine.convert(
    model_name='test speaker_cloned',
    input_wav='input_music.wav',
    output_wav='output_music.wav',
    pitch_shift=-2,  # Adjust pitch
    f0_method='dio'
)
```

## 📁 Directory Structure

```
ai-vocals-studio/
├── dataset/              # Training audio files
│   ├── test speaker/    # Voice samples for "test speaker"
│   ├── w2re/           # Voice samples for "w2re"
│   └── [your voices]/  # Add your voice samples here
├── models/             # Generated voice models
│   ├── test speaker_cloned/
│   ├── w2re_cloned/
│   └── [your models]/
├── output/             # Generated audio files
├── app_studio.py       # Main studio application
├── simple_voice_clone.py   # Quick model creation
└── test_voice_conversion.py # Voice conversion testing
```

## 🎯 Tips for Best Results

### Voice Sample Quality
- Use clear, high-quality recordings (44.1kHz or higher)
- Record in a quiet environment with minimal background noise
- Include various speaking styles and emotions
- Aim for 10-30 minutes of total audio for best results
- Sample rate: 22.05kHz or 44.1kHz
- Format: WAV (uncompressed) preferred

### Model Selection
- Use **app_studio.py** for the most features and effects
- For quick tests, use **app_minimal.py**
- For professional results, train full ML models

### Voice Transformation
- Start with small pitch adjustments (-2 to +2)
- Experiment with reverb and speed settings
- Use the preview feature to hear results before final export
- Different voice models work better with different input audio

## 🔧 Troubleshooting

### Models Not Showing Up
- Ensure model directories have both `model.pth` and `config.json`
- Check that the model name appears in the dropdown
- Try refreshing the models list

### Poor Voice Quality
- Improve your training audio quality
- Use more diverse voice samples
- Try different model settings
- For best results, use full ML training

### Application Won't Launch
- Ensure virtual environment is activated: `source venv/bin/activate`
- Check that all dependencies are installed
- Try running with `python3 app_studio.py` directly

## 🎤 Next Steps

1. **Test the current setup**: Run `python3 test_voice_conversion.py` to hear your cloned voices
2. **Add your own voice samples**: Import your recordings to create personal voice models
3. **Experiment with effects**: Try different pitch, speed, and reverb settings
4. **Create music**: Transform vocals in your favorite songs
5. **Train advanced models**: For professional results, invest time in full ML training

## 📞 Support

For issues or questions:
- Check the main README.md for general guidance
- Review CLAUDE.md for architecture details
- Test with the provided scripts before using the main app

## 🎉 Enjoy Your Voice Cloning Studio!

You now have a fully functional voice cloning and voice transformation system. Start experimenting with different voices and create amazing audio content!