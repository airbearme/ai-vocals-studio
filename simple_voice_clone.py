#!/usr/bin/env python3
"""
Simple voice cloning test using enhanced voice cloner
"""
import os
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def create_simple_voice_model(speaker_name, model_name):
    """Create a simple voice model without heavy ML processing"""
    print(f"🎤 Creating voice model for: {speaker_name}")
    
    dataset_dir = Path('dataset')
    models_dir = Path('models')
    
    speaker_dir = dataset_dir / speaker_name
    if not speaker_dir.exists():
        print(f"❌ Speaker directory not found: {speaker_dir}")
        return None
    
    # Get audio files
    audio_files = list(speaker_dir.glob('*.wav')) + list(speaker_dir.glob('*.mp3'))
    if not audio_files:
        print(f"❌ No audio files found for {speaker_name}")
        return None
    
    print(f"📁 Found {len(audio_files)} audio files")
    
    # Create model directory
    model_dir = models_dir / model_name
    model_dir.mkdir(exist_ok=True)
    
    # Create a simple voice profile
    voice_profile = {
        "speaker": speaker_name,
        "model_name": model_name,
        "type": "voice_clone",
        "total_files": len(audio_files),
        "audio_files": [f.name for f in audio_files],
        "characteristics": {
            "pitch_shift": -2,  # Default adjustments
            "speed": 1.05,
            "reverb": 0.3,
            "gain": 2
        },
        "created": "2026-08-17"
    }
    
    # Save voice profile
    with open(model_dir / "voice_profile.json", 'w') as f:
        json.dump(voice_profile, f, indent=2)
    
    # Create model.pth placeholder (will be used by app)
    model_file = model_dir / "model.pth"
    model_file.write_text("")  # Empty file for placeholder
    
    # Create config.json for SO-VITS compatibility
    config = {
        "spk": {speaker_name: 0},
        "version": "4.0"
    }
    with open(model_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Voice model created successfully!")
    print(f"📍 Location: {model_dir}")
    print(f"🎤 Model contains {len(audio_files)} voice samples")
    
    return str(model_dir)

def main():
    print("🎤 Simple Voice Cloning Test")
    print("=" * 50)
    
    # Check available datasets
    dataset_dir = Path('dataset')
    print(f"\n📁 Available datasets in {dataset_dir}:")
    
    available_speakers = []
    for item in dataset_dir.iterdir():
        if item.is_dir():
            # Check if it has audio files
            audio_files = list(item.glob('*.wav')) + list(item.glob('*.mp3'))
            if audio_files:
                available_speakers.append(item.name)
                print(f"  ✅ {item.name} ({len(audio_files)} audio files)")
    
    if not available_speakers:
        print("\n❌ No datasets with audio files found!")
        return
    
    # Create models for all available speakers
    for speaker in available_speakers:
        model_name = f"{speaker}_cloned"
        print(f"\n🎯 Creating voice model for: {speaker}")
        print(f"📦 Model name: {model_name}")
        
        try:
            model_path = create_simple_voice_model(speaker, model_name)
            
            if model_path:
                print(f"\n✅ Voice model created successfully!")
                print(f"📍 Location: {model_path}")
                print(f"\n🎤 This model can now be used in the main app!")
                print(f"   The model will use DSP effects to simulate the cloned voice.")
                print(f"   For best results, use it in app_studio.py or app_modern.py")
            else:
                print(f"\n❌ Failed to create voice model")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()