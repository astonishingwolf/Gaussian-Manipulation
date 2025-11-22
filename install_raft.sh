#!/bin/bash

# Script to install RAFT as a submodule and download weights

set -e  # Exit on error

echo "=========================================="
echo "Installing RAFT Submodule and Weights"
echo "=========================================="

# Check if RAFT directory exists
if [ ! -d "RAFT" ]; then
    echo "RAFT directory not found. Adding RAFT as a submodule..."
    git submodule add https://github.com/princeton-vl/RAFT.git RAFT
else
    echo "RAFT directory exists. Initializing and updating submodule..."
    git submodule update --init --recursive RAFT
fi

# Navigate to RAFT directory
cd RAFT

# Download RAFT weights
echo ""
echo "Downloading RAFT weights..."
if [ -f "download_models.sh" ]; then
    chmod +x download_models.sh
    ./download_models.sh
else
    echo "Warning: download_models.sh not found. Downloading weights manually..."
    wget -q --show-progress https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
    unzip -q models.zip
    rm models.zip
fi

# Check if weights directory exists in project root, if not create it
cd ..
if [ ! -d "weights" ]; then
    mkdir -p weights
fi

# Copy raft-things.pth to weights directory if it exists
if [ -f "RAFT/models/raft-things.pth" ]; then
    echo ""
    echo "Copying raft-things.pth to weights directory..."
    cp RAFT/models/raft-things.pth weights/raft-things.pth
    echo "✓ RAFT weights copied to weights/raft-things.pth"
fi

echo ""
echo "=========================================="
echo "RAFT installation completed successfully!"
echo "=========================================="
echo ""
echo "RAF weights are available at:"
echo "  - RAFT/models/raft-things.pth"
if [ -f "weights/raft-things.pth" ]; then
    echo "  - weights/raft-things.pth"
fi
echo ""

