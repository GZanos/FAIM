#!/bin/bash
# FAIM - Start the app. Double-click to run (browser will open).
# Run 'Install FAIM.command' once first if you haven't already.

# Always use the folder that contains this .command file — not your home directory
# or wherever Terminal's current directory happens to be.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1

if [ ! -d ".venv" ]; then
    echo "First-time setup required."
    echo "Please double-click 'Install FAIM.command' once, then run this again."
    read -p "Press Enter to close."
    exit 1
fi

PY="$HERE/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "ERROR: Missing $PY"
    echo "If you moved the FAIM folder, delete the .venv folder here and run 'Install FAIM.command' again."
    read -p "Press Enter to close."
    exit 1
fi

# Use: python -m streamlit
# (Calling .venv/bin/streamlit directly breaks after moving the project — its shebang still points at the old path.)
if ! "$PY" -m streamlit --version >/dev/null 2>&1; then
    echo "ERROR: streamlit is not installed in this venv."
    echo "Run 'Install FAIM.command' again."
    read -p "Press Enter to close."
    exit 1
fi

echo "Starting FAIM from: $HERE"
echo "URL: http://127.0.0.1:8501"
echo "Close this window to stop the app."
echo ""

# Open browser only after the server answers (Safari often opens too early with a fixed sleep)
(
 for _ in $(seq 1 90); do
        if curl -sf -o /dev/null "http://127.0.0.1:8501/_stcore/health" 2>/dev/null || \
           curl -sf -o /dev/null "http://127.0.0.1:8501" 2>/dev/null; then
            open "http://127.0.0.1:8501" 2>/dev/null || true
            exit 0
        fi
        sleep 1
    done
    echo ""
    echo "WARNING: Server did not respond on port 8501. Read any Python errors above."
    echo "If the port is busy, quit other Streamlit apps or change --server.port in this script."
) &

"$PY" -m streamlit run "$HERE/wildfire_forecast_app_V1_5_5.py" \
    --server.headless true \
    --server.address 127.0.0.1 \
    --server.port 8501

