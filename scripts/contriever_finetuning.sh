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
# lora_r: r parameter for LoRA.
# lora_alpha: alpha parameter for LoRA.
# lora_dropout: Dropout rate for LoRA.
# optim: Type of optimizer (e.g., adam, sam, asam).
# seed: Seed for reproducibility.
# chunk_length: Maximum length of passages.

DATASET=INSERT_DATASET_NAME_HERE
TRAIN_DATA=./beir_datasets/$DATASET/training_data.jsonl
EVAL_DATA=./beir_datasets/$DATASET/test_data.jsonl
MODEL_PATH=facebook/contriever-msmarco
TOTAL_STEPS=5000 # they used 500000
SAVE_FREQ=10 # they used 20000
LOG_FREQ=10 # they used 20000
EVAL_FREQ=500
PER_GPU_BATCH_SIZE=32 # they used 64
USE_RSLORA=True
INIT_LORA_WEIGHTS=pissa

# LoRA parameters
lora_r=64
lora_alpha=32
lora_dropout=0.1
lora_target_modules="query,key,value,output.dense,intermediate.dense"

name="finetuning-experiment"

echo "Running experiment $name"

python ./finetuning.py \
    --model_path $MODEL_PATH \
    --train_data $TRAIN_DATA \
    --eval_data $EVAL_DATA \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --lora_target_modules $lora_target_modules \
    --use_rslora $USE_RSLORA \
    --init_lora_weights $INIT_LORA_WEIGHTS \
    --total_steps $TOTAL_STEPS \
    --save_freq $SAVE_FREQ \
    --log_freq $LOG_FREQ \
    --eval_freq $EVAL_FREQ \
    --per_gpu_batch_size $PER_GPU_BATCH_SIZE \
    --name $name \
    --negative_ctxs 5 \
    --negative_hard_ratio 0.8 \
    --negative_hard_min_idx 0
