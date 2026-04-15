#!/bin/bash
# FAIM - One-time setup: create virtual environment and install dependencies.
# Double-click this file once after copying the FAIM folder to your Mac.

set -e
# Always install into the folder that contains this file (not Terminal's cwd)
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

echo "=============================================="
echo "  FAIM - Installing dependencies"
echo "=============================================="
echo ""

# Check for Python 3
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Please install Python 3 from https://www.python.org/downloads/ or run: brew install python3"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using Python $PYTHON_VERSION"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in $APP_DIR/.venv ..."
    python3 -m venv .venv
    echo "Done."
else
    echo "Virtual environment already exists."
fi
echo ""

# Activate and install
echo "Installing packages (this may take a few minutes)..."
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt
echo ""
echo "=============================================="
echo "  Installation complete."
echo "  Use 'Run FAIM.command' to start the app."
echo "=============================================="
echo ""
read -p "Press Enter to close."
