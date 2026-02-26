#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  AI Vocals Studio — 1-Click Cloud Training (Free GPU)       ║
# ║  Uses Google Colab T4 GPU — no cost, ~6-12 hours training   ║
# ╚══════════════════════════════════════════════════════════════╝

set -e
cd "$(dirname "$(readlink -f "$0")")"
source venv/bin/activate 2>/dev/null || true

MODEL="${1:-}"
COLAB_NB_URL="https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb"
GDRIVE_URL="https://drive.google.com/drive/my-drive"

# ── Pick model interactively if not given ────────────────────
if [ -z "$MODEL" ]; then
    echo ""
    echo "╔════════════════════════════════════════╗"
    echo "║   AI VOCALS STUDIO — CLOUD TRAINING    ║"
    echo "╚════════════════════════════════════════╝"
    echo ""
    echo "Available models:"
    i=1
    declare -a MODEL_LIST
    for d in models/*/; do
        name=$(basename "$d")
        if [ -d "dataset/44k/$name" ] || [ -d "dataset/$name" ]; then
            echo "  [$i] $name"
            MODEL_LIST[$i]="$name"
            ((i++))
        fi
    done
    if [ "$i" -eq 1 ]; then
        echo "  No models with training data found."
        echo "  Go to Training tab → create a model → import audio first."
        exit 1
    fi
    echo ""
    read -p "Select model number: " choice
    MODEL="${MODEL_LIST[$choice]}"
    if [ -z "$MODEL" ]; then echo "Invalid choice."; exit 1; fi
fi

echo ""
echo "Selected: $MODEL"
echo ""

# ── Check features exist ─────────────────────────────────────
FEAT_COUNT=$(find "dataset/44k/$MODEL/" -name "*.data.pt" 2>/dev/null | wc -l)
if [ "$FEAT_COUNT" -eq 0 ]; then
    echo "⚠  No preprocessed features found for $MODEL."
    echo "   Run the full training pipeline first (pre-resample + pre-config + pre-hubert)"
    echo "   then re-run this script."
    exit 1
fi
echo "✅ Found $FEAT_COUNT feature files."

# ── Package files ─────────────────────────────────────────────
PKG_DIR="/tmp/cloud_train_${MODEL}"
ZIP_OUT="$HOME/Desktop/${MODEL}_colab_training.zip"
rm -rf "$PKG_DIR" && mkdir -p "$PKG_DIR"

echo "📦 Packaging files..."
mkdir -p "$PKG_DIR/dataset/44k/$MODEL"
mkdir -p "$PKG_DIR/logs/44k"
mkdir -p "$PKG_DIR/configs/44k"
mkdir -p "$PKG_DIR/filelists/44k"

cp dataset/44k/${MODEL}/*.wav   "$PKG_DIR/dataset/44k/$MODEL/" 2>/dev/null || true
cp dataset/44k/${MODEL}/*.data.pt "$PKG_DIR/dataset/44k/$MODEL/"
cp logs/44k/D_0.pth logs/44k/G_0.pth logs/44k/config.json "$PKG_DIR/logs/44k/"
cp configs/44k/config.json "$PKG_DIR/configs/44k/"
cp filelists/44k/train.txt filelists/44k/val.txt filelists/44k/test.txt "$PKG_DIR/filelists/44k/"

# Generate Colab notebook
cat > "$PKG_DIR/${MODEL}_training.ipynb" << 'NOTEBOOK'
{
 "nbformat": 4, "nbformat_minor": 0,
 "metadata": {"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"accelerator":"GPU"},
 "cells": [
  {"cell_type":"markdown","metadata":{},"source":["# Voice Model Training\n","**Step 1**: Runtime → Change runtime type → T4 GPU\n","**Step 2**: Run all cells in order"]},
  {"cell_type":"code","metadata":{},"source":["!pip install -q so-vits-svc-fork==4.2.30 torchcodec"],"execution_count":null,"outputs":[]},
  {"cell_type":"code","metadata":{},"source":["from google.colab import files\nuploaded = files.upload()\nimport zipfile\nzip_name = list(uploaded.keys())[0]\nwith zipfile.ZipFile(zip_name) as z:\n    z.extractall('/content/')\nprint('✅ Extracted')"],"execution_count":null,"outputs":[]},
  {"cell_type":"code","metadata":{},"source":["import torch\nprint('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE — change runtime type!')"],"execution_count":null,"outputs":[]},
  {"cell_type":"code","metadata":{},"source":["import os\nos.chdir('/content')\n!svc train -c configs/44k/config.json -m logs/44k"],"execution_count":null,"outputs":[]},
  {"cell_type":"code","metadata":{},"source":["import glob\nfrom google.colab import files\ncheckpoints = sorted(glob.glob('/content/logs/44k/G_*.pth'))\nprint('Checkpoints:', checkpoints)\nif checkpoints:\n    files.download(checkpoints[-1])\n    files.download('/content/logs/44k/config.json')\n    print('✅ Download started')"],"execution_count":null,"outputs":[]}
 ]
}
NOTEBOOK

echo "🗜  Creating zip (this may take a minute)..."
(cd "$PKG_DIR" && zip -r "$ZIP_OUT" .)

SIZE=$(du -sh "$ZIP_OUT" 2>/dev/null | cut -f1)
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PACKAGE READY                                               ║"
echo "╟──────────────────────────────────────────────────────────────╢"
echo "║  File: $(basename "$ZIP_OUT") ($SIZE)                       "
echo "║  Location: $ZIP_OUT"
echo "╟──────────────────────────────────────────────────────────────╢"
echo "║  NEXT STEPS (takes ~2 minutes):                              ║"
echo "║                                                              ║"
echo "║  1. Browser opens → Google Drive                            ║"
echo "║     → Upload the zip file from your Desktop                 ║"
echo "║                                                              ║"
echo "║  2. Browser opens → Google Colab                            ║"
echo "║     → File → Upload notebook → pick the .ipynb in the zip  ║"
echo "║     → Runtime → Change runtime type → T4 GPU               ║"
echo "║     → Run All                                               ║"
echo "║                                                              ║"
echo "║  3. When done, Colab downloads G_XXXXX.pth + config.json    ║"
echo "║     Run this to install: bash cloud_train.sh --install      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Install mode (after Colab returns files) ──────────────────
if [ "$1" = "--install" ]; then
    echo "Drop the downloaded G_*.pth and config.json here, then press Enter:"
    read -p "Path to G_*.pth: " PTH_FILE
    read -p "Model name to install into: " INSTALL_MODEL
    mkdir -p "models/$INSTALL_MODEL"
    cp "$PTH_FILE" "models/$INSTALL_MODEL/$(basename "$PTH_FILE")"
    cp "$(dirname "$PTH_FILE")/config.json" "models/$INSTALL_MODEL/config.json" 2>/dev/null || true
    echo "✅ Installed to models/$INSTALL_MODEL/"
    exit 0
fi

# ── Open browser ──────────────────────────────────────────────
echo "Opening browser in 3 seconds..."
sleep 3
DISPLAY="${DISPLAY:-:0}" xdg-open "$GDRIVE_URL" 2>/dev/null &
sleep 2
DISPLAY="${DISPLAY:-:0}" xdg-open "https://colab.research.google.com/" 2>/dev/null &
echo ""
echo "✅ Done! Upload the zip to Drive, open Colab, follow the steps above."
