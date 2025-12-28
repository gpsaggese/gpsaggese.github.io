#!/bin/bash

# Start Flask Backend and Streamlit Frontend for PEFT Fake News Detector

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/Users/vikas/src/umd_classes1/.venv/bin/python"
STREAMLIT="/Users/vikas/src/umd_classes1/.venv/bin/streamlit"

echo "============================================================"
echo "🚀 PEFT Fake News Detector - Startup Script"
echo "============================================================"
echo ""
echo "⚠️  Note: Using port 5001 for backend (macOS AirPlay uses 5000)"
echo ""

# Check if Python exists
if [ ! -f "$PYTHON" ]; then
    echo "❌ Python not found at: $PYTHON"
    exit 1
fi

# Check if Streamlit exists
if [ ! -f "$STREAMLIT" ]; then
    echo "❌ Streamlit not found at: $STREAMLIT"
    exit 1
fi

echo "📦 Starting Flask Backend (port 5000)..."
echo "   Log: backend.log"
$PYTHON "$SCRIPT_DIR/backend.py" > "$SCRIPT_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
sleep 3

echo ""
echo "🎨 Starting Streamlit Frontend (port 8501)..."
echo "   Your browser will open automatically"
echo ""
echo "============================================================"
echo "✅ Both servers are running!"
echo "============================================================"
echo ""
echo "Backend API:     http://localhost:5001"
echo "Streamlit UI:    http://localhost:8501"
echo ""
echo "Backend PID:     $BACKEND_PID"
echo ""
echo "To stop the servers:"
echo "  - Press Ctrl+C in this terminal"
echo "  - Or run: kill $BACKEND_PID"
echo ""
echo "============================================================"
echo ""

# Start Streamlit (this will block)
$STREAMLIT run "$SCRIPT_DIR/app.py"

# If Streamlit exits, kill backend
echo ""
echo "🛑 Stopping backend server..."
kill $BACKEND_PID 2>/dev/null
echo "✅ All servers stopped"
