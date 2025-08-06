#!/bin/bash

# Audio Summarizer - Execution Script
echo "🎧 Starting Audio Summarizer..."

# Activate the virtual environment
echo "🔧 Activating virtual environment..."
source ~/venv/bin/activate

# Get the absolute path of the directory where this script is located
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Change to the script directory to ensure relative paths work correctly
cd "$SCRIPT_DIR"

# Run the main Python application
echo "🚀 Running Audio Summarizer..."
python3 main.py

# Deactivate the virtual environment
echo "🔧 Deactivating virtual environment..."
deactivate

echo "👋 Audio Summarizer finished."
