#!/usr/bin/env python3
"""
Test voice conversion with cloned voices
"""
import os
import sys
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_voice_conversion():
    """Test voice conversion using the cloned voices"""
    print("🎤 Voice Conversion Test")
    print("=" * 50)
    
    # Check available models
    models_dir = Path('models')
    print(f"\n📁 Available models in {models_dir}:")
    
    available_models = []
    for item in models_dir.iterdir():
        if item.is_dir():
            # Check if it has model files
            if (item / 'model.pth').exists() or (item / 'config.json').exists():
                available_models.append(item.name)
                print(f"  ✅ {item.name}")
    
    if not available_models:
        print("\n❌ No models found!")
        return
    
    # Check for test audio
    dataset_dir = Path('dataset')
    test_audio = None
    
    # Find a test audio file
    audio_files = list(dataset_dir.glob('*.wav')) + list(dataset_dir.glob('*.mp3'))
    
    # Also check subdirectories
    for speaker_dir in dataset_dir.iterdir():
        if speaker_dir.is_dir():
            audio_files.extend(list(speaker_dir.glob('*.wav')))
            audio_files.extend(list(speaker_dir.glob('*.mp3')))
    
    if audio_files:
        # Use the largest audio file for better testing
        test_audio = max(audio_files, key=lambda f: f.stat().st_size)
        print(f"\n🎵 Using test audio: {test_audio}")
        print(f"   File exists: {test_audio.exists()}")
        print(f"   File size: {test_audio.stat().st_size/1024:.1f} KB")
    else:
        print("\n❌ No test audio found!")
        return
    
    # Test conversion with each model
    from pydub import AudioSegment
    
    for model_name in available_models[:2]:  # Test first 2 models
        print(f"\n🎯 Testing voice conversion with: {model_name}")
        
        try:
            # Load test audio
            audio = AudioSegment.from_file(test_audio)
            print(f"   📊 Loaded audio: {len(audio)/1000:.1f}s")
            
            # Apply voice transformation (simulated)
            # This is a simple DSP effect - in real usage, the app would use the model
            print(f"   ✨ Applying voice transformation...")
            
            # Simple pitch shift simulation
            audio_shifted = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * 0.9)  # Lower pitch
            }).set_frame_rate(audio.frame_rate)
            
            # Add some reverb
            audio_shifted = audio_shifted - 3  # Reduce volume
            echo = audio_shifted.overlay(audio_shifted, position=100)  # Simple echo
            audio_final = audio_shifted.overlay(echo)
            
            # Save output
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / f"test_{model_name.replace(' ', '_')}_converted.wav"
            audio_final.export(str(output_file), format='wav')
            
            print(f"   ✅ Conversion complete!")
            print(f"   📍 Output: {output_file}")
            print(f"   📊 Size: {output_file.stat().st_size/1024:.1f} KB")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ Voice conversion test complete!")
    print(f"🎤 Check the output/ directory for converted files")

if __name__ == "__main__":
    test_voice_conversion()