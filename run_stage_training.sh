#!/bin/bash

# 2-Stage Training Script (Optimized for V100 32GB)
# Usage: ./run_stage_training.sh [stage]
# stage: 1 (KorQuad only), 2 (Final Fine-tuning), all (default)

STAGE=${1:-all}
BASE_MODEL="klue/roberta-large"
OUTPUT_DIR="../../../models" # Relative to code/src
DATASET_PATH="../../../raw/data/train_dataset" # Relative to code/src

# --- Hyperparameters (V100 32GB Optimization) ---
# Max Sequence Length increased to 512 to capture full context.
# FP16 enabled for speed and memory efficiency.
# Batch size tuned for stability with gradient accumulation.

MAX_SEQ_LEN=512
DOC_STRIDE=128

# Effective Batch Size = BATCH_SIZE * GRAD_ACCUM = 12 * 3 = 36
BATCH_SIZE=12
GRAD_ACCUM=3

echo "Starting training pipeline with stage: $STAGE"
echo "Config: SeqLen=$MAX_SEQ_LEN, Batch=$BATCH_SIZE, Accum=$GRAD_ACCUM, FP16=True"

# Move to source directory to ensure imports work correctly
cd code/src

# Stage 1: KorQuad Pre-training
# Goal: Learn general QA capability from large dataset.
if [ "$STAGE" == "1" ] || [ "$STAGE" == "all" ]; then
    echo "=================================================================="
    echo "STAGE 1: Pre-training on KorQuad v1 Only"
    echo "=================================================================="
    
    python train.py \
        --output_dir ${OUTPUT_DIR}/korquad_pretrain \
        --model_name_or_path $BASE_MODEL \
        --dataset_name $DATASET_PATH \
        --do_train \
        --do_eval \
        --add_korquad \
        --korquad_only \
        --max_seq_length $MAX_SEQ_LEN \
        --doc_stride $DOC_STRIDE \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --per_device_eval_batch_size 32 \
        --learning_rate 3e-5 \
        --num_train_epochs 2 \
        --warmup_ratio 0.1 \
        --fp16 \
        --logging_steps 100 \
        --save_steps 1000 \
        --eval_steps 1000 \
        --save_total_limit 2 \
        --evaluation_strategy steps \
        --load_best_model_at_end \
        --metric_for_best_model exact_match \
        --overwrite_output_dir
fi

# Stage 2: Fine-tuning on Competition Data (Train+Valid)
# Goal: Adapt to competition-specific domain/style without forgetting QA skills.
if [ "$STAGE" == "2" ] || [ "$STAGE" == "all" ]; then
    echo "=================================================================="
    echo "STAGE 2: Fine-tuning on Competition Data (Train + Valid)"
    echo "=================================================================="

    # Check for Stage 1 model
    PRETRAINED_PATH="${OUTPUT_DIR}/korquad_pretrain"
    if [ ! -d "$PRETRAINED_PATH" ]; then
        if [ "$STAGE" == "2" ]; then
            echo "WARNING: Stage 1 model not found at $PRETRAINED_PATH."
            echo "Starting from base model: $BASE_MODEL"
            PRETRAINED_PATH=$BASE_MODEL
        else
             echo "Using newly trained model from Stage 1."
        fi
    else
        echo "Loading pre-trained model from: $PRETRAINED_PATH"
    fi

    # Lower LR (1.5e-5) and include validation data for final tuning
    python train.py \
        --output_dir ${OUTPUT_DIR}/final_submission \
        --model_name_or_path $PRETRAINED_PATH \
        --dataset_name $DATASET_PATH \
        --do_train \
        --include_validation \
        --max_seq_length $MAX_SEQ_LEN \
        --doc_stride $DOC_STRIDE \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --learning_rate 1.5e-5 \
        --num_train_epochs 3 \
        --warmup_ratio 0.1 \
        --fp16 \
        --logging_steps 50 \
        --save_strategy epoch \
        --save_total_limit 2 \
        --overwrite_output_dir
fi

# Return to original directory
cd ../..