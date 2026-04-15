#!/usr/bin/env bash
set -euo pipefail

LANGUAGE="${1:-shona}"
MANIFEST="data/manifests/${LANGUAGE}_train_manifest.jsonl"
SKIP_MFA="${SKIP_MFA:-0}"

source .venv/bin/activate

python training/prepare_data.py --input-dir "data/raw/${LANGUAGE}" --manifest "$MANIFEST"
python training/validate_manifest.py --manifest "$MANIFEST"

if [[ "$SKIP_MFA" == "1" ]]; then
  echo "Skipping MFA alignment (SKIP_MFA=1)."
else
  python training/align_mfa.py \
    --corpus-dir "data/raw/${LANGUAGE}" \
    --dictionary "frontend/lexicons/${LANGUAGE}.txt" \
    --output-dir "data/processed/aligned/${LANGUAGE}"
fi

python training/train_fastpitch.py
python training/train_hifigan.py

echo "Phase 3 run complete for ${LANGUAGE}."
