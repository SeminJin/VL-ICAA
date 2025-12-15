#!/bin/bash

# Script to download all required pretrained models
# Usage: bash download_models.sh

set -e  # Exit on error

echo "=========================================="
echo "Downloading Pretrained Models"
echo "=========================================="

# Create directories
mkdir -p pretrained_weights
mkdir -p output/GIT_LARGE_R_TEXTCAPS/snapshot

# Download DAT pretrained weights
echo ""
echo "Downloading DAT Base model..."
if [ ! -f "pretrained_weights/dat_base_checkpoint.pth" ]; then
    wget https://github.com/LeapLabTHU/DAT/releases/download/v1.0/dat_base_in1k_224.pth \
         -O pretrained_weights/dat_base_checkpoint.pth
    echo "✓ DAT Base model downloaded successfully!"
else
    echo "✓ DAT Base model already exists, skipping..."
fi

# Optional: Download other DAT variants
echo ""
read -p "Download DAT-Tiny model? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -f "pretrained_weights/dat_tiny_checkpoint.pth" ]; then
        wget https://github.com/LeapLabTHU/DAT/releases/download/v1.0/dat_tiny_in1k_224.pth \
             -O pretrained_weights/dat_tiny_checkpoint.pth
        echo "✓ DAT Tiny model downloaded!"
    else
        echo "✓ DAT Tiny model already exists, skipping..."
    fi
fi

echo ""
read -p "Download DAT-Small model? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -f "pretrained_weights/dat_small_checkpoint.pth" ]; then
        wget https://github.com/LeapLabTHU/DAT/releases/download/v1.0/dat_small_in1k_224.pth \
             -O pretrained_weights/dat_small_checkpoint.pth
        echo "✓ DAT Small model downloaded!"
    else
        echo "✓ DAT Small model already exists, skipping..."
    fi
fi

# GIT model setup
echo ""
echo "=========================================="
echo "GIT Model Setup"
echo "=========================================="
echo "The GIT model will be automatically downloaded when you first run training."
echo "To manually download, please ensure azfuse is installed:"
echo ""
echo "  pip install git+https://github.com/microsoft/azfuse.git"
echo ""
echo "Then set the environment variable:"
echo "  export AZFUSE_TSV_USE_FUSE=1"
echo ""
echo "The model will download to: output/GIT_LARGE_R_TEXTCAPS/snapshot/"

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Downloaded files:"
ls -lh pretrained_weights/
echo ""
echo "You can now start training with:"
echo "  python train.py"
