#!/bin/bash

# train_data: Path to the training data.
# eval_data: Path to the evaluation data.
# output_dir: Output directory to save models and logs.
# model_path: Path to the pre-trained model.
# pooling: Type of pooling to use.
# random_init: If set, initialize the model randomly.
# negative_ctxs: Number of negative contexts to use.
# negative_hard_ratio: Ratio of hard negative contexts (0.5 means half of the negative contexts are hard).
# negative_hard_min_idx: Minimum index for hard negative contexts (can be useful to skip the first contexts).
# eval_normalize_text: If set, normalize text during evaluation.
# maxload: Maximum number of examples to load.
# per_gpu_batch_size: Batch size per GPU.
# num_workers: Number of workers for the DataLoader.
# total_steps: Total number of training steps.
# log_freq: Logging frequency.
# eval_freq: Evaluation frequency.
# save_freq: Model saving frequency.
# dropout: Dropout rate.
# optim: Type of optimizer (e.g., adam, sam, asam).
# seed: Seed for reproducibility.
# chunk_length: Maximum length of passages.
# lora_r: r parameter for LoRA.
# lora_alpha: alpha parameter for LoRA.
# lora_dropout: Dropout rate for LoRA.
# lora_target_modules: Target modules for LoRA.
# use_rslora: Whether to use RSLORA.
# init_lora_weights: Initial weights for LoRA.

DATASET=INSERT_DATASET_NAME_HERE
TRAIN_DATA=./beir_datasets/$DATASET/training_data.jsonl
EVAL_DATA=./beir_datasets/$DATASET/test_data.jsonl
MODEL_PATH=facebook/contriever-msmarco
TOTAL_STEPS=5000 # they used 500000
SAVE_FREQ=10 # they used 20000
LOG_FREQ=10 # they used 20000
EVAL_FREQ=500
PER_GPU_BATCH_SIZE=32 # they used 64
NEGATIVE_CTXS=5
NEGATIVE_HARD_RATIO=0.8
NEGATIVE_HARD_MIN_IDX=0



# LoRA parameters
LORA_R=64
LORA_ALPHA=32
LORA_DROPOUT=0.1
LORA_TARGET_MODULES="query,key,value,output.dense,intermediate.dense"
USE_RSLORA=True
INIT_LORA_WEIGHTS=olora

if [[ -n "${LORA_R}" ]]; then # if 'LORA_R' is defined, finetune the model with LoRA

    OUTPUT_DIR=beir_results/$DATASET/lora_experiment_$(date +%m%d-%H%M)
    echo "Finetuning with LoRA. Results will be saved inside $OUTPUT_DIR"

    python ./finetuning.py \
    --model_path $MODEL_PATH \
    --train_data $TRAIN_DATA \
    --eval_data $EVAL_DATA \
    --total_steps $TOTAL_STEPS \
    --save_freq $SAVE_FREQ \
    --log_freq $LOG_FREQ \
    --eval_freq $EVAL_FREQ \
    --per_gpu_batch_size $PER_GPU_BATCH_SIZE \
    --output_dir $OUTPUT_DIR \
    --negative_ctxs $NEGATIVE_CTXS \
    --negative_hard_ratio $NEGATIVE_HARD_RATIO \
    --negative_hard_min_idx $NEGATIVE_HARD_MIN_IDX \
    --use_lora \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $LORA_TARGET_MODULES \
    --use_rslora $USE_RSLORA \
    --init_lora_weights $INIT_LORA_WEIGHTS

else # if 'LORA_R' is not defined, finetune the model without LoRA

    OUTPUT_DIR=beir_results/$DATASET/finetuned_basemodel_experiment_$(date +%m%d-%H%M)
    echo "Finetuning basemodel. Results will be saved inside $OUTPUT_DIR"
    
    DROPOUT=0.1

    python ./finetuning.py \
    --model_path $MODEL_PATH \
    --train_data $TRAIN_DATA \
    --eval_data $EVAL_DATA \
    --total_steps $TOTAL_STEPS \
    --dropout $DROPOUT \
    --save_freq $SAVE_FREQ \
    --log_freq $LOG_FREQ \
    --eval_freq $EVAL_FREQ \
    --per_gpu_batch_size $PER_GPU_BATCH_SIZE \
    --output_dir $OUTPUT_DIR \
    --negative_ctxs $NEGATIVE_CTXS \
    --negative_hard_ratio $NEGATIVE_HARD_RATIO \
    --negative_hard_min_idx $NEGATIVE_HARD_MIN_IDX
fi