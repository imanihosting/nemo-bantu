#!/usr/bin/env bash
set -euo pipefail

LANGUAGE="${1:-shona}"
MANIFEST="data/manifests/${LANGUAGE}_train_manifest.jsonl"

source .venv/bin/activate

echo "=== Phase 3: ${LANGUAGE} ==="

# Step 1: Prepare manifest from wav/txt pairs
echo "[1/4] Preparing manifest..."
python training/prepare_data.py --input-dir "data/raw/${LANGUAGE}" --manifest "$MANIFEST"

# Step 2: Validate manifest integrity
echo "[2/4] Validating manifest..."
python training/validate_manifest.py --manifest "$MANIFEST"

# Step 3: MFA alignment (optional — FastPitch learn_alignment=true can skip this)
if [ "${SKIP_MFA:-1}" = "0" ]; then
    echo "[3/4] Running MFA alignment..."
    python training/align_mfa.py \
        --corpus-dir "data/raw/${LANGUAGE}" \
        --dictionary "frontend/lexicons/${LANGUAGE}.txt" \
        --acoustic-model english_mfa \
        --output-dir "data/processed/aligned/${LANGUAGE}"
else
    echo "[3/4] MFA alignment skipped (SKIP_MFA=1, learn_alignment=true handles alignment internally)"
fi

# Step 4: Train FastPitch acoustic model
echo "[4/4] Training FastPitch..."
python training/train_fastpitch.py \
    train_dataset="$MANIFEST" \
    validation_datasets="$MANIFEST"

echo "=== Phase 3 FastPitch training launched for ${LANGUAGE} ==="
echo "Note: HiFi-GAN training should run after FastPitch converges."
echo "  python training/train_hifigan.py"
