#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  AI Vocals Studio — 1-Click Cloud Training (Free GPU)       ║
# ║  Uses Google Colab T4 GPU — no cost, ~6-12 hours training   ║
# ╚══════════════════════════════════════════════════════════════╝

set -e
cd "$(dirname "$(readlink -f "$0")")"
source venv/bin/activate 2>/dev/null || true

# Hosted Colab notebook — opens directly from GitHub, no upload needed
COLAB_NB_URL="https://colab.research.google.com/github/airbearme/ai-vocals-studio/blob/main/colab/train_voice_model.ipynb"
GDRIVE_URL="https://drive.google.com/drive/my-drive"

# ── Load local account config (never stored in git) ──────────────
CONFIG_FILE="$(dirname "$(readlink -f "$0")")/.cloud_config"
if [ -f "$CONFIG_FILE" ]; then
    GOOGLE_ACCOUNT="$(grep '^GOOGLE_ACCOUNT=' "$CONFIG_FILE" | cut -d= -f2- | tr -d '"' | tr -d "'")"
    if [ -n "$GOOGLE_ACCOUNT" ]; then
        COLAB_NB_URL="${COLAB_NB_URL}?authuser=${GOOGLE_ACCOUNT}"
        GDRIVE_URL="${GDRIVE_URL}?authuser=${GOOGLE_ACCOUNT}"
    fi
fi

# ── Install mode (after Colab returns files) ──────────────────────
if [ "$1" = "--install" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║   INSTALL TRAINED MODEL                                  ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    echo "Drag the downloaded G_*.pth file into this terminal, or type the path:"
    read -p "  Path to G_*.pth: " PTH_FILE
    PTH_FILE="${PTH_FILE//\'/}"   # strip any drag-and-drop quotes
    if [ ! -f "$PTH_FILE" ]; then echo "❌ File not found: $PTH_FILE"; exit 1; fi
    echo ""
    # List available models to install into
    echo "Install into which model?"
    i=1; declare -a MLIST
    for d in models/*/; do
        name=$(basename "$d")
        echo "  [$i] $name"
        MLIST[$i]="$name"; ((i++))
    done
    echo "  [$i] Create new model name"
    MLIST[$i]="__new__"
    read -p "  Choice: " choice
    INSTALL_MODEL="${MLIST[$choice]}"
    if [ "$INSTALL_MODEL" = "__new__" ]; then
        read -p "  New model name: " INSTALL_MODEL
        INSTALL_MODEL="${INSTALL_MODEL// /_}"
    fi
    if [ -z "$INSTALL_MODEL" ]; then echo "❌ No model selected."; exit 1; fi
    mkdir -p "models/$INSTALL_MODEL"
    cp "$PTH_FILE" "models/$INSTALL_MODEL/$(basename "$PTH_FILE")"
    CFG="$(dirname "$PTH_FILE")/config.json"
    [ -f "$CFG" ] && cp "$CFG" "models/$INSTALL_MODEL/config.json" || true
    echo ""
    echo "✅ Installed to models/$INSTALL_MODEL/"
    echo "   Restart the app and the model will appear automatically."
    exit 0
fi

MODEL="${1:-}"

# ── Pick model interactively if not given ────────────────────────
if [ -z "$MODEL" ]; then
    echo ""
    echo "╔════════════════════════════════════════╗"
    echo "║   AI VOCALS STUDIO — CLOUD TRAINING    ║"
    echo "╚════════════════════════════════════════╝"
    echo ""
    echo "Available models with training data:"
    i=1
    declare -a MODEL_LIST
    for d in models/*/; do
        name=$(basename "$d")
        if [ -d "dataset/44k/$name" ] || [ -d "dataset/$name" ]; then
            FEAT=$(find "dataset/44k/$name/" -name "*.data.pt" 2>/dev/null | wc -l)
            echo "  [$i] $name  ($FEAT feature files)"
            MODEL_LIST[$i]="$name"
            ((i++))
        fi
    done
    if [ "$i" -eq 1 ]; then
        echo "  No models with training data found."
        echo "  Training tab → create a model → import audio → TRAIN MODEL first."
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

# ── Check features exist ──────────────────────────────────────────
FEAT_COUNT=$(find "dataset/44k/$MODEL/" -name "*.data.pt" 2>/dev/null | wc -l)
if [ "$FEAT_COUNT" -eq 0 ]; then
    echo "⚠  No preprocessed feature files (.data.pt) found for $MODEL."
    echo "   In the app: Training tab → select model → click TRAIN MODEL"
    echo "   Wait for pre-hubert step to finish, then run cloud_train.sh again."
    exit 1
fi
echo "✅ Found $FEAT_COUNT feature files."

# ── Package dataset + checkpoints into zip ───────────────────────
PKG_DIR="/tmp/cloud_train_${MODEL}"
# Save to Desktop if it exists, otherwise home dir
if [ -d "$HOME/Desktop" ]; then
    ZIP_OUT="$HOME/Desktop/${MODEL}_colab_training.zip"
else
    ZIP_OUT="$HOME/${MODEL}_colab_training.zip"
fi

rm -rf "$PKG_DIR" && mkdir -p \
    "$PKG_DIR/dataset/44k/$MODEL" \
    "$PKG_DIR/logs/44k" \
    "$PKG_DIR/configs/44k" \
    "$PKG_DIR/filelists/44k"

echo "📦 Packaging files..."
cp dataset/44k/${MODEL}/*.wav    "$PKG_DIR/dataset/44k/$MODEL/" 2>/dev/null || true
cp dataset/44k/${MODEL}/*.data.pt "$PKG_DIR/dataset/44k/$MODEL/"
cp logs/44k/D_0.pth logs/44k/G_0.pth logs/44k/config.json "$PKG_DIR/logs/44k/"
cp configs/44k/config.json "$PKG_DIR/configs/44k/"
cp filelists/44k/train.txt filelists/44k/val.txt filelists/44k/test.txt "$PKG_DIR/filelists/44k/"

# Include the Colab notebook as a backup (repo-hosted URL is primary)
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
if [ -f "$SCRIPT_DIR/colab/train_voice_model.ipynb" ]; then
    cp "$SCRIPT_DIR/colab/train_voice_model.ipynb" "$PKG_DIR/train_voice_model.ipynb"
fi

echo "🗜  Creating zip (this may take a minute for large datasets)..."
(cd "$PKG_DIR" && zip -r "$ZIP_OUT" .)

SIZE=$(du -sh "$ZIP_OUT" 2>/dev/null | cut -f1)
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ PACKAGE READY — ${MODEL}"
echo "╟──────────────────────────────────────────────────────────────╢"
echo "║  Zip: $(basename "$ZIP_OUT")  ($SIZE)"
echo "║  At:  $ZIP_OUT"
echo "╟──────────────────────────────────────────────────────────────╢"
echo "║  3 STEPS TO FREE GPU TRAINING:                               ║"
echo "║                                                              ║"
echo "║  1. Upload the zip to Google Drive (browser opens now)       ║"
echo "║                                                              ║"
echo "║  2. Click the Colab link (browser opens now)                 ║"
echo "║     → Runtime → Change runtime type → T4 GPU                ║"
echo "║     → Runtime → Run All                                      ║"
echo "║     (Colab auto-finds your zip from Drive — nothing to type) ║"
echo "║                                                              ║"
echo "║  3. When done (~6-12 hrs), Colab downloads your model.pth   ║"
echo "║     → In app: click INSTALL MODEL  OR                        ║"
echo "║     → Run: bash cloud_train.sh --install                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Open browser tabs ─────────────────────────────────────────────
echo "Opening Google Drive and Colab in 3 seconds..."
sleep 3
DISPLAY="${DISPLAY:-:0}" xdg-open "$GDRIVE_URL" 2>/dev/null &
sleep 2
DISPLAY="${DISPLAY:-:0}" xdg-open "$COLAB_NB_URL" 2>/dev/null &
echo "✅ Browsers opened. Upload zip to Drive, then run the Colab notebook."
