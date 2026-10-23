#!/usr/bin/env bash
set -euo pipefail

AD="$HOME/adaptive-physio"
PA="$HOME/physiology_as_language_repro"

DATA="$AD/outputs/downstream/staging/adaptive_pasl"
MODELS="$AD/outputs/downstream/staging/retrained_cnn"
LOGS="$AD/outputs/logs/retrained_pasl"

GPU="${GPU:-0}"

mkdir -p "$MODELS" "$LOGS"

if [ "$#" -gt 0 ]; then
    CONDITIONS=("$@")
else
    CONDITIONS=(
        uniform64
        uniform128
        random80
        fixed80
        V7_80
    )
fi

echo "============================================"
echo "RETRAINED CNN-TCN — SHHS1"
echo "Conditions: ${CONDITIONS[*]}"
echo "GPU: $GPU"
echo "Epochs: 20"
echo "============================================"

for condition in "${CONDITIONS[@]}"; do

    echo
    echo "############################################"
    echo "CONDITION: $condition"
    echo "############################################"

    # -----------------------------------------
    # 1. Generate reconstructed EEG spectrograms
    # -----------------------------------------

    cd "$AD"

    for split in train val test; do

        echo
        echo "[$condition] Generating $split"

        PYTHONPATH="$AD/src:$AD/scripts" \
        CUDA_VISIBLE_DEVICES="$GPU" \
        python -u scripts/build_adaptive_pasl.py \
            --mode build \
            --condition "$condition" \
            --split "$split" \
            --microbatch 32 \
            2>&1 | tee "$LOGS/build_${condition}_${split}.log"

        FILE="$DATA/$condition/shhs1/$split/synthesized_eeg.mmap"
        COMPLETE="$DATA/$condition/shhs1/$split/.complete"

        if [ ! -s "$FILE" ] || [ ! -f "$COMPLETE" ]; then
            echo "ERROR: Incomplete data for $condition / $split"
            exit 1
        fi
    done

    echo
    echo "[$condition] All datasets prepared"

    # -----------------------------------------
    # 2. CNN-TCN training — 20 epochs
    # -----------------------------------------

    cd "$PA"

    OUT="$MODELS/$condition"
    mkdir -p "$OUT"

    if [ ! -f "$OUT/train_complete.flag" ]; then

        echo
        echo "[$condition] TRAINING CNN-TCN — 20 EPOCHS"

        CUDA_VISIBLE_DEVICES="$GPU" \
        python -u \
            paper_architecture/downstream/CNN_TCN/train.py \
            --data-root "$DATA/$condition" \
            --dataset shhs1 \
            --input-type synthesized \
            --output-dir "$OUT" \
            --epochs 20 \
            --warmup-epochs 2 \
            --batch-size 8 \
            --lr 3e-4 \
            --seed 42 \
            2>&1 | tee "$LOGS/train_${condition}.log"

        test -s "$OUT/checkpoint_best_macro_f1.pt"
        touch "$OUT/train_complete.flag"

    else
        echo "[$condition] Training already complete"
    fi

    # -----------------------------------------
    # 3. Independent test evaluation
    # -----------------------------------------

    echo
    echo "[$condition] TEST EVALUATION"

    CUDA_VISIBLE_DEVICES="$GPU" \
    python -u \
        paper_architecture/downstream/CNN_TCN/train.py \
        --data-root "$DATA/$condition" \
        --dataset shhs1 \
        --input-type synthesized \
        --output-dir "$OUT" \
        --eval-only \
        --checkpoint "$OUT/checkpoint_best_macro_f1.pt" \
        --split test \
        --batch-size 8 \
        2>&1 | tee "$LOGS/test_${condition}.log"

    touch "$OUT/test_complete.flag"

    echo
    echo "[$condition] COMPLETED"

done

echo
echo "============================================"
echo "ALL CONDITIONS COMPLETED"
echo "============================================"

cd "$AD"

for file in "$LOGS"/test_*.log; do
    [ -f "$file" ] || continue
    echo "===== $(basename "$file") ====="
    grep -E '^loss=.*acc=.*MF1=.*kappa=' "$file" || true
done
