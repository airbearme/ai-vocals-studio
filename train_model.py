#!/usr/bin/env python3
"""
🎤 Train Voice Model
Simple training for your custom voice model
"""

import os
import json
import subprocess
from pathlib import Path

from voice_conversion_engine import VoiceConversionEngine

def train_voice_model(speaker_name="2Pac", model_name="2pac_custom_voice",
                      consent_confirmed=False):
    """Build a real reference profile for an authorized speaker."""
    if not consent_confirmed:
        print("❌ Speaker authorization must be confirmed before model creation")
        return False
    
    print("🎤 Training 2Pac Voice Model")
    print("=" * 40)
    
    # Paths
    speaker_dir = Path("dataset") / speaker_name
    model_dir = Path("models") / model_name
    
    # Check if model setup exists
    if not speaker_dir.exists():
        print("❌ Speaker directory not found!")
        return False
    
    audio_files = (list(speaker_dir.glob("*.wav")) +
                   list(speaker_dir.glob("*.mp3")) +
                   list(speaker_dir.glob("*.flac")))
    total_files = len(audio_files)

    if not audio_files:
        print(f"❌ No audio files found in {speaker_dir}")
        return False

    print(f"🧠 Extracting a real voice profile from {total_files} audio files...")
    engine = VoiceConversionEngine()
    profile = engine.extract_reference_profile([str(path) for path in audio_files])
    profile.update({
        "speaker": speaker_name,
        "model_name": model_name,
        "type": "authorized_voice_profile",
        "consent_confirmed": True,
        "audio_files": [path.name for path in audio_files],
    })
    model_file = model_dir / "model.pth"
    engine.save_profile(profile, str(model_file))

    with open(model_dir / "config.json", "w") as f:
        json.dump({"spk": {speaker_name: 0}, "version": "world_profile"}, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"📍 Model saved: {model_file}")
    print(f"🎯 Voice characteristics captured")
    print(f"📊 Training files: {total_files}")
    print(f"🎤 Voice: {speaker_name}")
    
    return True

def update_app_models():
    """Update the app's model list to include your custom model"""
    
    print("\n🔄 Updating AI Vocals Studio...")
    
    # Add model to the app's model list
    app_models_file = Path("models/available_models.json")
    
    models = {
        "2pac_custom_voice": {
            "name": "2Pac Custom Voice",
            "description": "Custom trained model using 2Pac audio files",
            "speaker": "2Pac",
            "type": "custom_trained",
            "path": "models/2pac_custom_voice/model.pth",
            "created": "2026-02-23"
        }
    }
    
    with open(app_models_file, 'w') as f:
        json.dump(models, f, indent=2)
    
    print("✅ Model list updated")
    print("🎤 Your 2Pac voice model is now available in AI Vocals Studio!")

if __name__ == "__main__":
    consent = input("Confirm you have the speaker's permission (yes/no): ").strip().lower()
    if train_voice_model(consent_confirmed=consent in {"yes", "y"}):
        update_app_models()
        print("\n🚀 SUCCESS! Your 2Pac voice model is ready to use!")
        print("📱 Launch AI Vocals Studio to generate vocals in 2Pac's voice!")
    else:
        print("\n❌ Training failed")
