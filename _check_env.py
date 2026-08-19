#!/usr/bin/env python3
"""Check training environment capabilities."""
import subprocess
import sys

print("=== Environment Check ===\n")

# Check PyTorch
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU - training will be slower")
except ImportError:
    print("PyTorch: NOT installed")

print()

# Check so-vits-svc-fork
try:
    from so_vits_svc_fork import SVC
    print("so-vits-svc-fork: INSTALLED")
    print("ML training: AVAILABLE")
except ImportError as e:
    print(f"so-vits-svc-fork: NOT installed ({e})")
    print("ML training: Will use fallback voice profile")

print()

# Check dataset
import os
pacaveli_dir = "dataset/Pacaveli_training"
if os.path.exists(pacaveli_dir):
    files = [f for f in os.listdir(pacaveli_dir) if f.endswith(('.wav', '.mp3'))]
    total_size = sum(os.path.getsize(os.path.join(pacaveli_dir, f)) for f in files)
    print(f"Training data ready: {len(files)} files in dataset/Pacaveli_training/")
    print(f"Total size: {total_size / (1024*1024):.1f} MB")
    print("This is sufficient for training (recommended: 3-10 clips)")
else:
    print(f"Training dir not found: {pacaveli_dir}")

print()

# Check models
import os
models_dir = "models"
if os.path.exists(models_dir):
    model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    print(f"Existing models: {model_dirs}")
