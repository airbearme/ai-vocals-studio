#!/usr/bin/env python3
"""Quick smoke test of VoiceConversionEngine profile extraction."""
import sys, glob, json
sys.path.insert(0, '/home/coden607/Projects/ai-vocals-studio')
from voice_conversion_engine import VoiceConversionEngine

eng = VoiceConversionEngine(max_seconds=6)
clips = sorted(glob.glob('/home/coden607/Projects/ai-vocals-studio/dataset/Pacaveli_processed/*.wav'))[:1]
print("clips:", clips)
prof = eng.extract_reference_profile(clips)
print("PITCH:", json.dumps(prof['pitch']))
print("CENTROID:", round(prof['spectral']['mean_centroid_hz'], 1))
print("FORMANTS:", prof['formants'])
print("ENV len:", len(prof['spectral']['envelope_mean']))
print("REAL PROFILE OK")