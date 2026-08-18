#!/usr/bin/env python3
"""
Test script to train a real voice model for voice cloning
"""
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from svc_engine import SoVitsEngine

def main():
    print("🎤 Voice Cloning Test Script")
    print("=" * 50)
    
    # Initialize engine
    engine = SoVitsEngine('models', 'dataset')
    
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
            else:
                print(f"  ⚠️  {item.name} (no audio files)")
    
    if not available_speakers:
        print("\n❌ No datasets with audio files found!")
        print("Please add audio files to dataset/<speaker_name>/ first.")
        return
    
    # Select speaker
    speaker = available_speakers[0]  # Use first available
    print(f"\n🎯 Selected speaker: {speaker}")
    
    # Check model status
    status = engine.model_status(speaker)
    print(f"📊 Current model status: {status}")
    
    if status == 'ready':
        print(f"✅ Model {speaker} is already trained and ready!")
        print("You can use it for voice conversion now.")
        return
    
    # Ask if user wants to train
    response = input(f"\n🚀 Do you want to train a model for {speaker}? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Training cancelled.")
        return
    
    print(f"\n🔧 Starting training for {speaker}...")
    print("⏳ This may take a while depending on your CPU...")
    
    def progress_callback(message, percent):
        if percent >= 0:
            print(f"[{percent:3d}%] {message}")
        else:
            print(f"     {message}")
    
    try:
        ok, error = engine.train(speaker, progress_cb=progress_callback)
        
        if ok:
            print(f"\n✅ Training completed successfully!")
            print(f"🎉 Model {speaker} is now ready for voice conversion!")
        else:
            print(f"\n❌ Training failed: {error}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")

if __name__ == "__main__":
    main()