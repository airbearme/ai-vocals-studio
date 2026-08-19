#!/bin/bash
cd /home/coden607/Projects/ai-vocals-studio
source venv/bin/activate

# Create directory structure for svc
mkdir -p dataset_raw/Pacaveli
mkdir -p dataset/44k/Pacaveli
mkdir -p configs/44k
mkdir -p logs/44k

# Copy processed WAV files
cp dataset/Pacaveli_processed/pacaveli_*.wav dataset_raw/Pacaveli/ 2>/dev/null

echo "=== Files in dataset_raw/Pacaveli ==="
ls -la dataset_raw/Pacaveli/

echo "=== Running pre-resample ==="
svc pre-resample 2>&1 | tail -10

echo "=== Done with preprocessing check ==="