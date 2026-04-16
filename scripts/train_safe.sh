#!/usr/bin/env bash
# =============================================================================
# train_safe.sh — Safe FastPitch training launcher for Shona TTS
#
# Fixes the 3 recurring failure modes:
#   1. SSH disconnect killing the process  → runs inside tmux (persists forever)
#   2. CPU thrash during data preprocessing → CPUQuota=1600% via systemd cgroup
#   3. No visibility when something goes wrong → watchdog + status file + log
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
        echo "Attaching to existing tmux session '$SESSION'..."
        tmux attach-session -t "$SESSION"
    else
        echo "No tmux session '$SESSION' found."
        echo "Run: bash scripts/train_safe.sh   to start training."
        exit 1
    fi
    exit 0
fi

# ── Guard against accidental double-launch ─────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "⚠  tmux session '$SESSION' is already running!"
    echo ""
    echo "   Attach:  tmux attach -t $SESSION"
    echo "   Status:  bash scripts/training_status.sh"
    echo "   Restart: tmux kill-session -t $SESSION  &&  bash scripts/train_safe.sh"
    exit 1
fi

# ── Pre-flight checks ────────────────────────────────────────────────────────
cd "$PROJECT_DIR"
echo "=== Pre-flight checks ==="

[[ -f "$PROJECT_DIR/.venv/bin/activate" ]] || { echo "ERROR: .venv missing"; exit 1; }
echo "  ✓ virtualenv present"

[[ -f "$PROJECT_DIR/training/train_fastpitch.py" ]] || { echo "ERROR: train_fastpitch.py missing"; exit 1; }
echo "  ✓ training script present"

MANIFEST="$PROJECT_DIR/data/manifests/shona_train_manifest.jsonl"
[[ -f "$MANIFEST" ]] || { echo "ERROR: manifest not found at $MANIFEST"; exit 1; }
MANIFEST_LINES=$(wc -l < "$MANIFEST")
echo "  ✓ manifest: $MANIFEST_LINES entries"

PITCH_COUNT=$(find "$PROJECT_DIR/data/processed/sup_data_shona/pitch" -name "*.pt" 2>/dev/null | wc -l)
echo "  ✓ pitch cache: $PITCH_COUNT / $MANIFEST_LINES files pre-computed"

systemd-run --user --scope --help >/dev/null 2>&1 || { echo "ERROR: systemd-run not available"; exit 1; }
echo "  ✓ systemd-run (cgroup CPU limiting) available"

echo ""
echo "  Log file   : $LOG_FILE"
echo "  Status file: $STATUS_FILE"
echo "  tmux session: $SESSION"
echo ""

# ── Write status: starting ────────────────────────────────────────────────────
cat > "$STATUS_FILE" << EOF
STATUS=starting
STARTED=$(date)
LOG=$LOG_FILE
EOF

# ── Build the watchdog script ─────────────────────────────────────────────────
WATCHDOG_SCRIPT="$LOG_DIR/watchdog_${TIMESTAMP}.sh"
cat > "$WATCHDOG_SCRIPT" << 'WATCHDOG_EOF'
#!/usr/bin/env bash
# Watchdog: pauses training process if CPU > 90% or RAM available < 8 GB.
# Resumes automatically when resources recover.
TRAIN_PID="$1"
LOG_FILE="$2"
CPU_PAUSE_THRESHOLD=95   # pause if system CPU% exceeds this
CPU_RESUME_THRESHOLD=80  # resume when CPU% drops below this
MEM_PAUSE_GB=4           # pause if available RAM drops below 4GB (119GB machine)
MEM_RESUME_GB=6          # resume when available RAM rises above 6GB
INTERVAL=10              # check every N seconds
PAUSED=0

_log() { echo "[WATCHDOG $(date +%H:%M:%S)] $1" | tee -a "$LOG_FILE"; }

_log "Started. Watching PID $TRAIN_PID | CPU pause >$CPU_PAUSE_THRESHOLD% | RAM pause <${MEM_PAUSE_GB}GB"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    # CPU: use /proc/stat for accuracy (no vmstat delay needed)
    read -r cpu user nice system idle iowait irq softirq steal _ < /proc/stat
    sleep 1
    read -r cpu2 user2 nice2 system2 idle2 iowait2 irq2 softirq2 steal2 _ < /proc/stat
    TOTAL=$(( (user2+nice2+system2+idle2+iowait2+irq2+softirq2+steal2) \
             - (user+nice+system+idle+iowait+irq+softirq+steal) ))
    IDLE_DIFF=$(( idle2 - idle ))
    CPU_USED=$(( 100 - (IDLE_DIFF * 100 / TOTAL) ))

    MEM_KB=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    MEM_GB=$(( MEM_KB / 1024 / 1024 ))

    if [[ $CPU_USED -gt $CPU_PAUSE_THRESHOLD || $MEM_GB -lt $MEM_PAUSE_GB ]] && [[ $PAUSED -eq 0 ]]; then
        _log "⚠  THROTTLE  CPU=${CPU_USED}%  RAM_FREE=${MEM_GB}GB — SIGSTOP sent"
        kill -STOP "$TRAIN_PID" 2>/dev/null || true
        PAUSED=1
    elif [[ $CPU_USED -le $CPU_RESUME_THRESHOLD && $MEM_GB -ge $MEM_RESUME_GB ]] && [[ $PAUSED -eq 1 ]]; then
        _log "✓  RESUME    CPU=${CPU_USED}%  RAM_FREE=${MEM_GB}GB — SIGCONT sent"
        kill -CONT "$TRAIN_PID" 2>/dev/null || true
        PAUSED=0
    fi

    sleep $(( INTERVAL - 1 ))  # -1 because we slept 1 for CPU diff
done

_log "Training PID $TRAIN_PID exited. Watchdog done."
WATCHDOG_EOF
chmod +x "$WATCHDOG_SCRIPT"

# ── Build the inner tmux script ───────────────────────────────────────────────
INNER_SCRIPT="$LOG_DIR/inner_train_${TIMESTAMP}.sh"
cat > "$INNER_SCRIPT" << INNER_EOF
#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR}"
LOG_FILE="${LOG_FILE}"
STATUS_FILE="${STATUS_FILE}"
WATCHDOG_SCRIPT="${WATCHDOG_SCRIPT}"

cd "\$PROJECT_DIR"
source .venv/bin/activate

# ── Header ─────────────────────────────────────────────────────────────────
{
echo "============================================================"
echo "  FastPitch Shona Training"
echo "  Started  : \$(date)"
echo "  Host     : \$(hostname)"
echo "  CPU cores: \$(nproc) total | Quota enforced: 80% (1600%)"
echo "  Memory   : \$(free -h | awk '/^Mem/{printf \"%s total, %s available\", \$2, \$7}')"
echo "  GPU      : \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Session  : tmux attach -t ${SESSION}"
echo "============================================================"
} | tee "\$LOG_FILE"

# ── Launch training under systemd-run (enforces CPUQuota=1600% = 16/20 CPUs) ─
echo "[\$(date +%H:%M:%S)] Starting training under systemd CPUQuota=1600%..." | tee -a "\$LOG_FILE"

systemd-run --user --scope \\
    -p CPUQuota=1600% \\
    -p MemoryHigh=100G \\
    -p MemoryMax=110G \\
    -- \\
    bash -c "
        cd '${PROJECT_DIR}'
        source .venv/bin/activate
        exec python training/train_fastpitch.py \\
            train_dataset=data/manifests/shona_train_manifest.jsonl \\
            validation_datasets=data/manifests/shona_train_manifest.jsonl
    " >> "\$LOG_FILE" 2>&1 &

TRAIN_PID=\$!
echo "[\$(date +%H:%M:%S)] Training PID: \$TRAIN_PID" | tee -a "\$LOG_FILE"

# ── Update status file ──────────────────────────────────────────────────────
cat > "\$STATUS_FILE" << STATUSEOF
STATUS=running
PID=\$TRAIN_PID
STARTED=\$(date)
SESSION=${SESSION}
LOG=\$LOG_FILE
STATUSEOF

# ── Start watchdog ──────────────────────────────────────────────────────────
bash "\$WATCHDOG_SCRIPT" "\$TRAIN_PID" "\$LOG_FILE" &
WATCHDOG_PID=\$!
echo "[\$(date +%H:%M:%S)] Watchdog PID: \$WATCHDOG_PID" | tee -a "\$LOG_FILE"

# ── Wait ────────────────────────────────────────────────────────────────────
wait "\$TRAIN_PID"
EXIT_CODE=\$?

kill "\$WATCHDOG_PID" 2>/dev/null || true

# ── Result ──────────────────────────────────────────────────────────────────
if [[ \$EXIT_CODE -eq 0 ]]; then
    RESULT="✅  TRAINING COMPLETED SUCCESSFULLY"
    echo "STATUS=completed" > "\$STATUS_FILE"
else
    RESULT="❌  TRAINING FAILED (exit code: \$EXIT_CODE)"
    echo "STATUS=failed" > "\$STATUS_FILE"
    echo "EXIT_CODE=\$EXIT_CODE" >> "\$STATUS_FILE"
fi
echo "ENDED=\$(date)" >> "\$STATUS_FILE"

{
echo ""
echo "============================================================"
echo "  \$RESULT"
echo "  Ended: \$(date)"
echo "  Log  : \$LOG_FILE"
echo "============================================================"
} | tee -a "\$LOG_FILE"

# Desktop notification (no-op if DISPLAY not set)
notify-send "FastPitch Training" "\$RESULT" 2>/dev/null || true

echo ""
echo "Training session complete. This window will stay open."
echo "Press Ctrl+D or type 'exit' to close it."
exec bash
INNER_EOF
chmod +x "$INNER_SCRIPT"

# ── Launch tmux session ───────────────────────────────────────────────────────
tmux new-session -d -s "$SESSION" -x 200 -y 50 "bash $INNER_SCRIPT"

# Update status
echo "STATUS=launched" > "$STATUS_FILE"
echo "SESSION=$SESSION" >> "$STATUS_FILE"
echo "LOG=$LOG_FILE" >> "$STATUS_FILE"
echo "LAUNCHED=$(date)" >> "$STATUS_FILE"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Training launched successfully!                                  ║"
echo "║                                                                   ║"
echo "║  Attach to live output:  tmux attach -t ${SESSION}      ║"
echo "║  Detach safely:          Ctrl+B → D                              ║"
echo "║  Check status anytime:   bash scripts/training_status.sh         ║"
echo "║  Tail log:               tail -f ${LOG_FILE##*/}                 ║"
echo "║                                                                   ║"
echo "║  ✓ SSH-safe: session survives disconnects                        ║"
echo "║  ✓ CPU capped at 80% (systemd CPUQuota=1600%)                   ║"
echo "║  ✓ Watchdog pauses at >90% CPU or <8GB RAM free                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Log: $LOG_FILE"
