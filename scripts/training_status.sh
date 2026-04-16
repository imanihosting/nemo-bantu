#!/usr/bin/env bash
# training_status.sh — quick health check for the FastPitch training session
# Usage: bash scripts/training_status.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
STATUS_FILE="$LOG_DIR/training_status.txt"

echo "══════════════════════════════════════════════════════"
echo "  FastPitch Shona — Training Status  [$(date)]"
echo "══════════════════════════════════════════════════════"

# tmux session
if tmux has-session -t fastpitch_train 2>/dev/null; then
    echo "  tmux session  : ✅ RUNNING (fastpitch_train)"
    echo "  Attach with   : tmux attach -t fastpitch_train"
else
    echo "  tmux session  : ❌ NOT RUNNING"
fi

# status file
if [[ -f "$STATUS_FILE" ]]; then
    echo ""
    echo "  --- Status File ---"
    cat "$STATUS_FILE"
fi

# latest log
LATEST_LOG=$(ls -t "$LOG_DIR"/fastpitch_*.log 2>/dev/null | head -1)
if [[ -n "$LATEST_LOG" ]]; then
    echo ""
    echo "  --- Last 15 lines of log ($LATEST_LOG) ---"
    tail -15 "$LATEST_LOG"
fi

# GPU
echo ""
echo "  --- GPU ---"
nvidia-smi --query-gpu=name,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader 2>/dev/null || echo "  nvidia-smi N/A"

# System resources
echo ""
echo "  --- System Resources ---"
echo "  Load avg : $(uptime | awk -F'load average: ' '{print $2}')"
echo "  Memory   : $(free -h | awk '/^Mem/ {printf "used %s / %s (avail: %s)", $3, $2, $7}')"

# checkpoint status
echo ""
echo "  --- Checkpoints ---"
find "$PROJECT_DIR/outputs" -name "*.nemo" -o -name "*.ckpt" 2>/dev/null \
    | xargs ls -lth 2>/dev/null | head -5 || echo "  No checkpoints yet"

echo "══════════════════════════════════════════════════════"
