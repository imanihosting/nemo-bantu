#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="/home/blaquesoul/Desktop/nemo-bantu"
LOG_FILE="/home/blaquesoul/Desktop/nemo-bantu/logs/fastpitch_20260416_143639.log"
STATUS_FILE="/home/blaquesoul/Desktop/nemo-bantu/logs/training_status.txt"
WATCHDOG_SCRIPT="/home/blaquesoul/Desktop/nemo-bantu/logs/watchdog_20260416_143639.sh"

cd "$PROJECT_DIR"
source .venv/bin/activate

# ── Header ─────────────────────────────────────────────────────────────────
{
echo "============================================================"
echo "  FastPitch Shona Training"
echo "  Started  : $(date)"
echo "  Host     : $(hostname)"
echo "  CPU cores: $(nproc) total | Quota enforced: 80% (1600%)"
echo "  Memory   : $(free -h | awk '/^Mem/{printf \"%s total, %s available\", $2, $7}')"
echo "  GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Session  : tmux attach -t fastpitch_train"
echo "============================================================"
} | tee "$LOG_FILE"

# ── Launch training under systemd-run (enforces CPUQuota=1600% = 16/20 CPUs) ─
echo "[$(date +%H:%M:%S)] Starting training under systemd CPUQuota=1600%..." | tee -a "$LOG_FILE"

systemd-run --user --scope \
    -p CPUQuota=1600% \
    -p MemoryHigh=100G \
    -p MemoryMax=110G \
    -- \
    bash -c "
        cd '/home/blaquesoul/Desktop/nemo-bantu'
        source .venv/bin/activate
        exec python training/train_fastpitch.py \
            train_dataset=data/manifests/shona_train_manifest.jsonl \
            validation_datasets=data/manifests/shona_train_manifest.jsonl
    " >> "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "[$(date +%H:%M:%S)] Training PID: $TRAIN_PID" | tee -a "$LOG_FILE"

# ── Update status file ──────────────────────────────────────────────────────
cat > "$STATUS_FILE" << STATUSEOF
STATUS=running
PID=$TRAIN_PID
STARTED=$(date)
SESSION=fastpitch_train
LOG=$LOG_FILE
STATUSEOF

# ── Start watchdog ──────────────────────────────────────────────────────────
bash "$WATCHDOG_SCRIPT" "$TRAIN_PID" "$LOG_FILE" &
WATCHDOG_PID=$!
echo "[$(date +%H:%M:%S)] Watchdog PID: $WATCHDOG_PID" | tee -a "$LOG_FILE"

# ── Wait ────────────────────────────────────────────────────────────────────
wait "$TRAIN_PID"
EXIT_CODE=$?

kill "$WATCHDOG_PID" 2>/dev/null || true

# ── Result ──────────────────────────────────────────────────────────────────
if [[ $EXIT_CODE -eq 0 ]]; then
    RESULT="✅  TRAINING COMPLETED SUCCESSFULLY"
    echo "STATUS=completed" > "$STATUS_FILE"
else
    RESULT="❌  TRAINING FAILED (exit code: $EXIT_CODE)"
    echo "STATUS=failed" > "$STATUS_FILE"
    echo "EXIT_CODE=$EXIT_CODE" >> "$STATUS_FILE"
fi
echo "ENDED=$(date)" >> "$STATUS_FILE"

{
echo ""
echo "============================================================"
echo "  $RESULT"
echo "  Ended: $(date)"
echo "  Log  : $LOG_FILE"
echo "============================================================"
} | tee -a "$LOG_FILE"

# Desktop notification (no-op if DISPLAY not set)
notify-send "FastPitch Training" "$RESULT" 2>/dev/null || true

echo ""
echo "Training session complete. This window will stay open."
echo "Press Ctrl+D or type 'exit' to close it."
exec bash
