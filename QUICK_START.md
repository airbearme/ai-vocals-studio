# 🚀 Quick Start Guide

## One-Click Installation & Launch

### Method 1: Full Installation (Recommended)
```bash
./install.sh
```
This will:
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Create desktop launcher
- ✅ Set up all folders

### Method 2: Quick Launch (Already Installed)
```bash
./run.sh
```

### Method 3: Manual Launch
```bash
python app_modern.py
```

## 🖥️ Desktop Launcher
After installation, you'll have a desktop shortcut called "AI Vocals Studio" - just double-click it!

## 📁 What You Need to Know

### First Time Setup:
1. **Run the installer**: `./install.sh`
2. **Add training data**: Go to Training Data tab → Import Entire Folder
3. **Download a model**: Go to Models tab → Select and download a pre-trained model
4. **Load the model**: Select your downloaded model and click "Load Selected Model"
5. **Generate vocals**: Go to Generate tab → Enter text or select audio → Generate

### Folder Structure:
```
ai-vocals-studio/
├── 📁 dataset/     # Put your training audio files here
├── 📁 models/      # AI models are stored here
├── 📁 outputs/     # Generated vocals appear here
├── 🚀 run.sh       # Quick launcher
├── 📦 install.sh   # Full installer
└── 🎤 app_modern.py # Main application
```

## 🎵 Supported Audio Formats
- **WAV, MP3, FLAC, OGG, M4A, AIFF, AAC**

## ⚡ Pro Tips
- **Folder Import**: Use "Import Entire Folder" to add hundreds of audio files at once
- **Model Downloads**: Start with pre-trained models for instant results
- **Flow Alignment**: Enable for rap vocals to improve rhythm
- **Output Format**: Use WAV for quality, MP3 for smaller files

## 🔧 Troubleshooting
- **Dependencies not found?** Run `./install.sh` again
- **Permission denied?** Run `chmod +x run.sh install.sh`
- **Python not found?** Install Python 3 from your package manager

---

🎤 **Ready to create amazing AI vocals!**
