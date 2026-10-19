#!/usr/bin/env bash
# ==============================================================================
# SNN-AI Cognitive Health & Neuro-Monitoring Platform: 1-Click Runner
# ==============================================================================
# This script launches both the FastAPI backend (port 8000) and the
# React Vite frontend (port 5173) in one single command.
#
# Usage:
#   chmod +x run.sh
#   ./run.sh
# ==============================================================================

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "=================================================================="
echo " 🧠 SNN-AI Cognitive Health & Neuro-Monitoring Platform"
echo "    Spiking Neural Network (LIF) + Tabular Q-Learning Recommender"
echo "=================================================================="

# 1. Environment checks
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH."
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed or not in PATH."
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm is not installed or not in PATH."
    exit 1
fi

echo "  [✓] Python: $(python3 --version)"
echo "  [✓] Node:   $(node --version)"
echo "  [✓] npm:    $(npm --version)"

# 2. Check and install frontend dependencies if missing
if [ ! -d "$REPO_DIR/frontend/node_modules" ]; then
    echo ""
    echo "📦 Installing frontend dependencies (one-time setup)..."
    (cd "$REPO_DIR/frontend" && npm install)
fi

# 3. Prevent MNE lock errors
export MNE_DONTWRITE_HOME=true

# 4. Clean up any existing listeners on 8000 and 5173
echo ""
echo "🧹 Releasing ports 8000 and 5173 if busy..."
lsof -ti :8000 | xargs kill -9 2>/dev/null || true
lsof -ti :5173 | xargs kill -9 2>/dev/null || true

# 5. Trap SIGINT (Ctrl+C) to shut down both processes cleanly
cleanup() {
    echo ""
    echo "🛑 Shutting down servers gracefully..."
    if [ -n "$BACKEND_PID" ]; then kill "$BACKEND_PID" 2>/dev/null || true; fi
    if [ -n "$FRONTEND_PID" ]; then kill "$FRONTEND_PID" 2>/dev/null || true; fi
    echo "✓ All servers stopped. Thank you!"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 6. Start Backend (FastAPI / Uvicorn)
echo ""
echo "🚀 Starting FastAPI Backend on http://127.0.0.1:8000..."
(
    cd "$REPO_DIR/backend"
    python3 -m uvicorn snn_ai_optimizer.app:app --host 127.0.0.1 --port 8000 --reload
) &
BACKEND_PID=$!

# Wait for backend to initialize
sleep 2

# 7. Start Frontend (Vite)
echo "🚀 Starting React Frontend on http://localhost:5173..."
(
    cd "$REPO_DIR/frontend"
    ./node_modules/.bin/vite --host localhost --port 5173
) &
FRONTEND_PID=$!

sleep 2

echo ""
echo "=================================================================="
echo " 🎉 SYSTEM IS LIVE AND READY!"
echo "=================================================================="
echo " 🌐 Dual-Portal Landing Page : http://localhost:5173/"
echo " 🔬 Research & Benchmark Lab : http://localhost:5173/research"
echo " 🏥 Clinical Neuro-Hub       : http://localhost:5173/dashboard"
echo " 📡 Backend API Docs         : http://127.0.0.1:8000/docs"
echo "=================================================================="
echo " Press Ctrl+C at any time to stop both servers."
echo ""

# 8. Try to open the default browser automatically
if command -v open &> /dev/null; then
    open "http://localhost:5173/" 2>/dev/null || true
elif command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:5173/" 2>/dev/null || true
fi

# Keep script running and wait for background jobs
wait $BACKEND_PID $FRONTEND_PID
