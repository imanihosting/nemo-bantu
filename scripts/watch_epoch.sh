#!/usr/bin/env bash
# =============================================================================
# watch_epoch.sh — Auto-test when training reaches a target epoch
#
# Polls the checkpoint directory every 60s. When a checkpoint at or beyond
# TARGET_EPOCH appears, it runs the HiFi-GAN test synthesis and plays the
# audio automatically, then sends a desktop notification.
#
# Usage:
#   bash scripts/watch_epoch.sh              # default: watch for epoch 200
#   bash scripts/watch_epoch.sh 300          # watch for epoch 300
#   bash scripts/watch_epoch.sh 200 --no-play  # synthesize but don't autoplay
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CKPT_DIR="$PROJECT_DIR/outputs/fastpitch_shona/FastPitch_Shona/checkpoints"

TARGET_EPOCH="${1:-200}"
AUTOPLAY=true
[[ "${2:-}" == "--no-play" ]] && AUTOPLAY=false

POLL_INTERVAL=60   # seconds between checks

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Epoch Watcher — waiting for epoch $TARGET_EPOCH                    ║"
echo "║  Checking every ${POLL_INTERVAL}s. Ctrl+C to cancel.                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

while true; do
    # Look for a non-last checkpoint at or beyond the target epoch
    FOUND=""
    while IFS= read -r ckpt; do
        # Extract epoch number from filename e.g. epoch=199
        epoch_num=$(echo "$ckpt" | grep -oP 'epoch=\K\d+' || true)
        if [[ -n "$epoch_num" && "$epoch_num" -ge "$TARGET_EPOCH" ]]; then
            FOUND="$ckpt"
            break
        fi
    done < <(find "$CKPT_DIR" -name "*.ckpt" ! -name "*-last.ckpt" 2>/dev/null | sort)

    if [[ -n "$FOUND" ]]; then
        FNAME="$(basename "$FOUND")"
        echo ""
        echo "✅ Found checkpoint at/beyond epoch $TARGET_EPOCH: $FNAME"
        echo "⏳ Running HiFi-GAN synthesis test..."
        echo ""

        cd "$PROJECT_DIR"
        source .venv/bin/activate

        OUTLOG="$PROJECT_DIR/logs/auto_test_epoch${TARGET_EPOCH}.log"
        python scripts/test_audio.py --checkpoint "$FOUND" 2>&1 | tee "$OUTLOG"

        # Find the output dir from the log
        OUTDIR=$(grep "📁 Output:" "$OUTLOG" | tail -1 | awk '{print $NF}')

        if [[ -n "$OUTDIR" && -d "$OUTDIR" ]]; then
            notify-send "🎤 Shona TTS" "Epoch $TARGET_EPOCH audio ready! Check: $OUTDIR" 2>/dev/null || true
            echo ""
            echo "🔊 Playing synthesized audio..."
            if $AUTOPLAY; then
                aplay "$OUTDIR"/*.wav 2>/dev/null || true
            fi
            echo ""
            echo "📁 WAVs saved to: $OUTDIR"
        fi

        echo ""
        echo "Done. Exiting watcher."
        exit 0
    fi

    # Show current best checkpoint while waiting
    LATEST=$(find "$CKPT_DIR" -name "*.ckpt" ! -name "*-last.ckpt" 2>/dev/null \
             | xargs ls -t 2>/dev/null | head -1 || true)
    LATEST_EPOCH="?"
    if [[ -n "$LATEST" ]]; then
        LATEST_EPOCH=$(basename "$LATEST" | grep -oP 'epoch=\K\d+' || echo "?")
    fi

    printf "\r  [%s] Current epoch: %-4s | Waiting for epoch %s ..." \
        "$(date +%H:%M:%S)" "$LATEST_EPOCH" "$TARGET_EPOCH"

    sleep "$POLL_INTERVAL"
done
