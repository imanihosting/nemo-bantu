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
