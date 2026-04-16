#!/usr/bin/env bash
# =============================================================================
# train_safe.sh — Safe FastPitch training launcher
#
# Key design decisions:
#  - Training runs in the FOREGROUND of the tmux window so tqdm progress bars
#    render properly (tqdm needs a real TTY)
#  - tmux pipe-pane captures all terminal output to a log file simultaneously
#  - systemd-run enforces 80% CPU quota (1600% = 16 of 20 cores)
#  - Watchdog monitors RAM and CPU, sends SIGSTOP/SIGCONT if limits breached
#  - Session survives SSH disconnects — attach anytime with:
#      tmux attach -t fastpitch_train
#
# Usage:
#   bash scripts/train_safe.sh            # start training
#   bash scripts/train_safe.sh --attach   # reattach to running session
#   bash scripts/training_status.sh       # check status without attaching
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SESSION="fastpitch_train"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/fastpitch_${TIMESTAMP}.log"
STATUS_FILE="$LOG_DIR/training_status.txt"

mkdir -p "$LOG_DIR"

# ── Attach-only mode ──────────────────────────────────────────────────────────
if [[ "${1:-}" == "--attach" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        exec tmux attach-session -t "$SESSION"
    else
        echo "No session '$SESSION'. Run: bash scripts/train_safe.sh"
        exit 1
    fi
fi

# ── Guard against double-launch ───────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "⚠  Session '$SESSION' already running!"
    echo "   Attach : tmux attach -t $SESSION"
    echo "   Status : bash scripts/training_status.sh"
    echo "   Restart: tmux kill-session -t $SESSION && bash scripts/train_safe.sh"
    exit 1
fi

# ── Pre-flight checks ─────────────────────────────────────────────────────────
cd "$PROJECT_DIR"
echo "=== Pre-flight checks ==="
[[ -f ".venv/bin/activate" ]]          && echo "  ✓ virtualenv"        || { echo "ERROR: .venv missing"; exit 1; }
[[ -f "training/train_fastpitch.py" ]] && echo "  ✓ training script"   || { echo "ERROR: train script missing"; exit 1; }
MANIFEST="data/manifests/shona_train_manifest.jsonl"
[[ -f "$MANIFEST" ]] && echo "  ✓ manifest: $(wc -l < "$MANIFEST") entries" || { echo "ERROR: manifest missing"; exit 1; }
PITCH=$(find data/processed/sup_data_shona/pitch -name "*.pt" 2>/dev/null | wc -l)
echo "  ✓ pitch cache: $PITCH files"
systemd-run --user --scope --help >/dev/null 2>&1 && echo "  ✓ systemd-run available" || { echo "ERROR: systemd-run missing"; exit 1; }
echo ""
echo "  Log   : $LOG_FILE"
echo "  Status: $STATUS_FILE"
echo ""

# ── Build watchdog script ─────────────────────────────────────────────────────
WATCHDOG_SCRIPT="$LOG_DIR/watchdog_${TIMESTAMP}.sh"
cat > "$WATCHDOG_SCRIPT" << 'WATCHDOG_EOF'
#!/usr/bin/env bash
TRAIN_PID="$1"
LOG_FILE="$2"
CPU_PAUSE_THRESHOLD=95
CPU_RESUME_THRESHOLD=80
MEM_PAUSE_GB=4           # 119GB machine — only panic below 4GB
MEM_RESUME_GB=6
INTERVAL=10
PAUSED=0

_log() { echo "[WATCHDOG $(date +%H:%M:%S)] $1" | tee -a "$LOG_FILE"; }
_log "Started. Watching PID $TRAIN_PID | CPU pause >$CPU_PAUSE_THRESHOLD% | RAM pause <${MEM_PAUSE_GB}GB"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    read -r _ user nice system idle _ < /proc/stat
    sleep 1
    read -r _ user2 nice2 system2 idle2 _ < /proc/stat
    TOTAL=$(( (user2+nice2+system2+idle2) - (user+nice+system+idle) ))
    CPU_USED=$(( 100 - ((idle2-idle)*100/TOTAL) ))
    MEM_GB=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)

    if [[ $CPU_USED -gt $CPU_PAUSE_THRESHOLD || $MEM_GB -lt $MEM_PAUSE_GB ]] && [[ $PAUSED -eq 0 ]]; then
        _log "⚠  THROTTLE CPU=${CPU_USED}% RAM_FREE=${MEM_GB}GB — SIGSTOP"
        kill -STOP "$TRAIN_PID" 2>/dev/null; PAUSED=1
    elif [[ $CPU_USED -le $CPU_RESUME_THRESHOLD && $MEM_GB -ge $MEM_RESUME_GB ]] && [[ $PAUSED -eq 1 ]]; then
        _log "✓  RESUME   CPU=${CPU_USED}% RAM_FREE=${MEM_GB}GB — SIGCONT"
        kill -CONT "$TRAIN_PID" 2>/dev/null; PAUSED=0
    fi
    sleep $(( INTERVAL - 1 ))
done
_log "Process $TRAIN_PID exited. Watchdog done."
WATCHDOG_EOF
chmod +x "$WATCHDOG_SCRIPT"

# ── Build the inner script that runs INSIDE tmux ──────────────────────────────
INNER_SCRIPT="$LOG_DIR/inner_train_${TIMESTAMP}.sh"
cat > "$INNER_SCRIPT" << INNER_EOF
#!/usr/bin/env bash
# This script runs inside the tmux window.
# Training runs in the FOREGROUND so tqdm can detect the TTY and show the bar.

PROJECT_DIR="${PROJECT_DIR}"
LOG_FILE="${LOG_FILE}"
STATUS_FILE="${STATUS_FILE}"
WATCHDOG_SCRIPT="${WATCHDOG_SCRIPT}"
SESSION="${SESSION}"

cd "\$PROJECT_DIR"
source .venv/bin/activate

# Capture everything printed to this tmux window into the log file
tmux pipe-pane -t "$SESSION" -o "cat >> '\$LOG_FILE'"

echo "============================================================"
echo "  FastPitch Shona Training"
echo "  Started : \$(date)"
echo "  Host    : \$(hostname)"
echo "  GPU     : \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  RAM     : \$(free -h | awk '/^Mem/{print \$2}') total"
echo "  Log     : \$LOG_FILE"
echo ""
echo "  Detach (keep running): Ctrl+B then D"
echo "  Reattach             : tmux attach -t ${SESSION}"
echo "============================================================"
echo ""

# Write initial status
cat > "\$STATUS_FILE" << EOF
STATUS=running
STARTED=\$(date)
SESSION=${SESSION}
LOG=\$LOG_FILE
EOF

# ── Start watchdog in background ─────────────────────────────────────────────
bash "\$WATCHDOG_SCRIPT" "\$\$" "\$LOG_FILE" &
WATCHDOG_PID=\$!

# ── Run training in FOREGROUND under systemd CPU quota ───────────────────────
# Running in foreground = tqdm detects real TTY = live progress bar in this window
echo "[INFO] Launching under systemd CPUQuota=1600% (80% of 20 cores)..."
echo ""

systemd-run --user --scope \
    -p CPUQuota=1600% \
    -p MemoryHigh=100G \
    -p MemoryMax=110G \
    -- \
    bash -c "
        cd '\${PROJECT_DIR}'
        source .venv/bin/activate
        export PYTHONUNBUFFERED=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        exec python -u training/train_fastpitch.py \
            train_dataset=data/manifests/shona_train_manifest.jsonl \
            validation_datasets=data/manifests/shona_train_manifest.jsonl
    "

EXIT_CODE=\$?
kill "\$WATCHDOG_PID" 2>/dev/null || true
tmux pipe-pane -t "$SESSION" 2>/dev/null || true  # stop log capture

echo ""
echo "============================================================"
if [[ \$EXIT_CODE -eq 0 ]]; then
    echo "  ✅  TRAINING COMPLETED SUCCESSFULLY — \$(date)"
    echo "STATUS=completed" > "\$STATUS_FILE"
else
    echo "  ❌  TRAINING FAILED (exit \$EXIT_CODE) — \$(date)"
    echo "STATUS=failed EXIT_CODE=\$EXIT_CODE" > "\$STATUS_FILE"
fi
echo "  Log: \$LOG_FILE"
echo "============================================================"
echo "ENDED=\$(date)" >> "\$STATUS_FILE"

notify-send "FastPitch Training" "Exit: \$EXIT_CODE" 2>/dev/null || true
echo ""
echo "Press Enter to close this window."
read -r || true
INNER_EOF
chmod +x "$INNER_SCRIPT"

# ── Launch tmux session ───────────────────────────────────────────────────────
tmux new-session -d -s "$SESSION" -x 220 -y 50
# Run the inner script in the tmux window
tmux send-keys -t "$SESSION" "bash $INNER_SCRIPT" Enter

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Training launched!                                               ║"
echo "║                                                                   ║"
echo "║  👉 See live progress bar:  tmux attach -t ${SESSION}    ║"
echo "║     Detach safely:          Ctrl+B → D                           ║"
echo "║                                                                   ║"
echo "║  Check status:  bash scripts/training_status.sh                  ║"
echo "║  Tail log:      tail -f $LOG_FILE"
echo "║                                                                   ║"
echo "║  ✓ SSH-safe  ✓ CPU 80% capped  ✓ Watchdog active               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "👉  Now run this from your SSH terminal:"
echo "    tmux attach -t ${SESSION}"
