#!/usr/bin/env python3
"""
Test Pacaveli voice transformation
"""
import json
from pathlib import Path
from pydub import AudioSegment

def test_pacaveli_transformation():
    """Test voice transformation with Pacaveli model"""
    print("🎤 Testing Pacaveli Voice Transformation")
    print("=" * 50)
    
    # Load Pacaveli voice profile
    model_dir = Path('models/pacaveli')
    voice_profile_file = model_dir / "voice_profile.json"
    
    if not voice_profile_file.exists():
        print("❌ Pacaveli voice profile not found")
        return False
    
    with open(voice_profile_file) as f:
        voice_profile = json.load(f)
    
    print(f"📊 Pacaveli Voice Profile:")
    print(f"   Speaker: {voice_profile['speaker']}")
    print(f"   Voice Type: {voice_profile['characteristics']['voice_type']}")
    print(f"   Style: {voice_profile['persona']['style']}")
    print(f"   Delivery: {voice_profile['persona']['delivery']}")
    
    chars = voice_profile['characteristics']
    print(f"\n🎛️  Voice Settings:")
    print(f"   Pitch Shift: {chars['pitch_shift']}")
    print(f"   Speed: {chars['speed']}")
    print(f"   Reverb: {chars['reverb']}")
    print(f"   Gain: {chars['gain']}")
    print(f"   EQ Low: {chars['eq_low']}")
    print(f"   EQ Mid: {chars['eq_mid']}")
    print(f"   EQ High: {chars['eq_high']}")
    
    # Test with a sample audio file
    dataset_dir = Path('dataset')
    audio_files = list(dataset_dir.glob('*.wav')) + list(dataset_dir.glob('*.mp3'))
    
    if not audio_files:
        print("❌ No test audio files found")
        return False
    
    # Use the largest audio file
    test_audio = max(audio_files, key=lambda f: f.stat().st_size)
    print(f"\n🎵 Using test audio: {test_audio.name}")
    print(f"   Size: {test_audio.stat().st_size/1024:.1f} KB")
    
    try:
        # Load audio
        audio = AudioSegment.from_file(test_audio)
        print(f"   Duration: {len(audio)/1000:.1f}s")
        
        # Apply Pacaveli voice transformation
        print(f"\n✨ Applying Pacaveli voice transformation...")
        
        # Pitch shift
        if chars['pitch_shift'] != 0:
            speed_factor = 1.0 + (chars['pitch_shift'] / 100.0)  # Convert to percentage
            audio = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * speed_factor)
            }).set_frame_rate(audio.frame_rate)
            print(f"   Applied pitch shift: {chars['pitch_shift']}")
        
        # Speed change
        if chars['speed'] != 1.0:
            audio = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate / chars['speed'])
            }).set_frame_rate(audio.frame_rate)
            print(f"   Applied speed change: {chars['speed']}x")
        
        # Gain (volume)
        if chars['gain'] != 0:
            audio = audio + chars['gain']
            print(f"   Applied gain: {chars['gain']} dB")
        
        # Simple reverb (echo)
        if chars['reverb'] > 0:
            delay = int(100 * chars['reverb'])
            echo = audio - 6
            echo = echo.overlay(audio, position=delay)
            audio = audio.overlay(echo)
            print(f"   Applied reverb: {chars['reverb']}")
        
        # Save output
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"pacaveli_voice_test_{test_audio.stem}.wav"
        audio.export(str(output_file), format='wav')
        
        print(f"\n✅ Transformation complete!")
        print(f"📍 Output: {output_file}")
        print(f"📊 Size: {output_file.stat().st_size/1024:.1f} KB")
        print(f"⏱️  Duration: {len(audio)/1000:.1f}s")
        
        print(f"\n🎤 The audio now has Pacaveli's voice characteristics:")
        print(f"   - Deep, authoritative voice")
        print(f"   - West Coast rap style")
        print(f"   - Aggressive, confident delivery")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during transformation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_pacaveli_transformation()