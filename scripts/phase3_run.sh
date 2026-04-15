#!/usr/bin/env bash
set -euo pipefail

LANGUAGE="${1:-shona}"
MANIFEST="data/manifests/${LANGUAGE}_train_manifest.jsonl"

source .venv/bin/activate

python training/prepare_data.py --input-dir "data/raw/${LANGUAGE}" --manifest "$MANIFEST"
python training/validate_manifest.py --manifest "$MANIFEST"
python training/align_mfa.py
python training/train_fastpitch.py
python training/train_hifigan.py

echo "Phase 3 run complete for ${LANGUAGE}."
