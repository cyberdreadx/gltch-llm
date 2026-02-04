#!/bin/bash
# GLTCH Hive — Quick Join Script
# 
# Run this on any machine to join the training network:
#   curl -sSL https://gltch.app/join | bash
#
# Or with options:
#   curl -sSL https://gltch.app/join | bash -s -- --server ws://coordinator:8765 --size 10m
#
# Created by: cyberdreadx

set -e

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║   GLTCH HIVE — Quick Join                                                     ║"
echo "║   Generative Language Transformer with Contextual Hierarchy                   ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Defaults
SERVER="ws://localhost:8765"
SIZE="2.7m"
NAME="node-$(shuf -i 1000-9999 -n 1)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --server) SERVER="$2"; shift 2 ;;
        --size) SIZE="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Prompt for server if default
if [ "$SERVER" == "ws://localhost:8765" ]; then
    echo "🔗 Enter coordinator server URL (or press Enter for localhost):"
    read -p "   > " INPUT_SERVER
    if [ -n "$INPUT_SERVER" ]; then
        SERVER="$INPUT_SERVER"
    fi
fi

echo ""
echo "📋 Configuration:"
echo "   Server: $SERVER"
echo "   Size: $SIZE"
echo "   Name: $NAME"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Create temp directory
WORK_DIR=$(mktemp -d)
cd "$WORK_DIR"
echo "📁 Working directory: $WORK_DIR"

# Check if git is available
if command -v git &> /dev/null; then
    echo "📥 Cloning GLTCH..."
    git clone --depth 1 https://github.com/cyberdreadx/gltch-2.7m.git .
else
    echo "📥 Downloading GLTCH..."
    curl -sSL https://github.com/cyberdreadx/gltch-2.7m/archive/main.zip -o gltch.zip
    unzip -q gltch.zip
    mv gltch-2.7m-main/* .
fi

# Create virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet torch websockets requests

# Check for GPU
echo ""
python3 -c "import torch; print('🎮 GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
echo ""

# Run peer
echo "🚀 Starting GLTCH peer..."
echo "   Press Ctrl+C to stop"
echo ""
python3 hive/peer.py --server "$SERVER" --name "$NAME" --size "$SIZE"
