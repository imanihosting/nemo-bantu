#!/usr/bin/env bash
# =============================================================================
# train_hifigan_phase2_safe.sh — Phase 2 HiFi-GAN training launcher
#
# Phase 2 fine-tunes HiFi-GAN on FastPitch-generated mels.
# Runs in tmux for SSH safety.
#
# Usage:
#   bash scripts/train_hifigan_phase2_safe.sh
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_DIR/.venv/bin"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$PROJECT_DIR/logs/hifigan_phase2_${TIMESTAMP}.log"
SESSION="hifigan_phase2"

# ── Check if Phase 1 training is still running ──────────────────────────────
if tmux has-session -t hifigan_train 2>/dev/null; then
    echo "⚠️  Phase 1 HiFi-GAN is still running (tmux: hifigan_train)"
    echo "   Stop it first: tmux kill-session -t hifigan_train"
    exit 1
fi

# ── Generate FastPitch mels if not already done ─────────────────────────────
MEL_MANIFEST="$PROJECT_DIR/data/manifests/shona_fp_mels_manifest.jsonl"
if [ ! -f "$MEL_MANIFEST" ]; then
    echo "⏳ Generating FastPitch mels (one-time step)..."
    "$VENV/python" "$PROJECT_DIR/scripts/generate_fp_mels.py" 2>&1 | tee "$PROJECT_DIR/logs/generate_fp_mels_${TIMESTAMP}.log"
    echo "✅ Mel generation complete"
fi

# ── Create inner training script ────────────────────────────────────────────
INNER_SCRIPT="$PROJECT_DIR/logs/inner_hifigan_phase2_${TIMESTAMP}.sh"
cat > "$INNER_SCRIPT" << 'INNER'
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="__PROJECT_DIR__"
VENV="$PROJECT_DIR/.venv/bin"
LOGFILE="__LOGFILE__"

cd "$PROJECT_DIR"

echo "$(date): Starting HiFi-GAN Phase 2 training" | tee -a "$LOGFILE"

# script -f preserves the live TTY output (including \r-based progress bars)
# and flushes after every write, so the log file stays in sync.
script -q -f -e -a -c "\"$VENV/python\" \"$PROJECT_DIR/training/train_hifigan_phase2.py\" --epochs 200 --batch-size 8" "$LOGFILE"

echo "$(date): Phase 2 training finished" | tee -a "$LOGFILE"
INNER

sed -i "s|__PROJECT_DIR__|$PROJECT_DIR|g" "$INNER_SCRIPT"
sed -i "s|__LOGFILE__|$LOGFILE|g" "$INNER_SCRIPT"
chmod +x "$INNER_SCRIPT"

# ── Launch in tmux ──────────────────────────────────────────────────────────
# Kill the tmux session AND any orphan training python (script's pty can
# detach the grandchild from the tmux session lifecycle).
tmux kill-session -t "$SESSION" 2>/dev/null || true
pkill -f "train_hifigan_phase2.py" 2>/dev/null || true
sleep 2
tmux new-session -d -s "$SESSION" "bash $INNER_SCRIPT"

echo ""
echo "============================================================"
echo "  ✅ HiFi-GAN Phase 2 training launched!"
echo "  Session : $SESSION"
echo "  Log     : $LOGFILE"
echo ""
echo "  Monitor : tmux attach -t $SESSION"
echo "  Detach  : Ctrl+B then D"
echo "============================================================"

# Update status file
cat > "$PROJECT_DIR/logs/hifigan_phase2_status.txt" << EOF
STATUS=running
STARTED=$(date)
SESSION=$SESSION
LOG=$LOGFILE
EOF
