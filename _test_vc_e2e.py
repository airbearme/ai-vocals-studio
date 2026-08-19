#!/usr/bin/env python3
"""End-to-end test: build Pacaveli (2Pac) profile, convert a source clip."""
import sys, glob
sys.path.insert(0, '/home/coden607/Projects/ai-vocals-studio')
from voice_conversion_engine import VoiceConversionEngine

out = '/tmp/vc_test'
import os
os.makedirs(out, exist_ok=True)

eng = VoiceConversionEngine(max_seconds=12)
clips = sorted(glob.glob('/home/coden607/Projects/ai-vocals-studio/dataset/Pacaveli_processed/*.wav'))[:2]
prof = eng.extract_reference_profile(clips)
print('PITCH:', prof['pitch'])
print('pitch_target:', prof['pitch_target'])

# Convert a source clip (another speaker) into 2Pac voice
src = sorted(glob.glob('/home/coden607/Projects/ai-vocals-studio/dataset/test speaker/*.wav'))[0]
dst = f'{out}/conv_2pac_small.wav'
ok = eng.convert_audio(src, prof, dst, strength=0.9)
print('converted:', ok, os.path.getsize(dst), 'bytes')
print('E2E OK')