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
# label_smoothing: Label smoothing value for the cross-entropy loss [0.0, 1.0].
# optim: Type of optimizer (e.g., adam, sam, asam).
# seed: Seed for reproducibility.
# chunk_length: Maximum length of passages.
# lora_r: r parameter for LoRA.
# lora_alpha: alpha parameter for LoRA.
# lora_dropout: Dropout rate for LoRA.
# lora_target_modules: Target modules for LoRA.
# use_rslora: Whether to use RSLORA.
# init_lora_weights: Initial weights for LoRA.
# eval_datasets: list of datasets for which to evaluate the model.
# eval_split: Split to use for evaluation (e.g., dev, test).
# use_minicorpus: If set, use a mini-corpus for evaluation to speed up the process.
# score_function: Scoring function to use (e.g., dot, cos_sim).


DATASET=DATASET_NAME_HERE # dataset in BEIR format, e.g. nfcorpus, hotpotqa, scifact, etc.
MODEL_PATH=MODEL_NAME_HERE # e.g., facebook/contriever-msmarco, intfloat/e5-large-v2, sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco, ...
MODEL_ID=${MODEL_PATH##*/}

if [[ "$MODEL_ID" == "contriever-msmarco" ]]; then
    LORA_TARGET_MODULES="query key value output.dense intermediate.dense"
    SCORE_FUNCTION="dot"
    NORM_QUERY=""
    NORM_DOC=""
    POOLING="average"
    
elif [[ "$MODEL_ID" == "e5-large-v2" ]]; then
    LORA_TARGET_MODULES="query key value output.dense intermediate.dense"
    SCORE_FUNCTION="cos_sim"
    NORM_QUERY="--norm_query"
    NORM_DOC="--norm_doc"
    POOLING="average"
    
elif [[ "$MODEL_ID" == "distilbert-dot-tas_b-b256-msmarco" ]]; then
    LORA_TARGET_MODULES="q_lin k_lin v_lin out_lin lin1 lin2"
    SCORE_FUNCTION="dot"
    NORM_QUERY=""
    NORM_DOC=""
    POOLING="cls"
    MODEL_ID="msmarco-distilbert-base-tas-b"
    
else
    SCORE_FUNCTION="dot"
    NORM_QUERY=""
    NORM_DOC=""
    POOLING="average"
fi

if [[ "$DATASET" == "hotpotqa" || "$DATASET" == "nq-train" || "$DATASET" == "fever" ]]; then
    USE_MINICORPUS="--use_minicorpus"
else
    USE_MINICORPUS=""
fi

TRAIN_DATA=./beir_datasets/$DATASET/$MODEL_ID/training_data.jsonl
EVAL_DATA=./beir_datasets/$DATASET/$MODEL_ID/dev_data.jsonl
TOTAL_STEPS=5000
SCHEDULER=cosine
WARMUP_STEPS=$(($TOTAL_STEPS / 10))
SAVE_FREQ=2500
LOG_FREQ=10
EVAL_FREQ=200
LR=0.00005
PER_GPU_BATCH_SIZE=32
ACCUMULATION_STEPS=2 # they used 64 as batchsize
NEGATIVE_CTXS=4
NEGATIVE_HARD_RATIO=1
NEGATIVE_HARD_MIN_IDX=0
TEMPERATURE=0.05
CHUNK_LENGTH=256
LABEL_SMOOTHING=0.1


# LoRA parameters
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1
INIT_LORA_WEIGHTS=pissa


if [[ -n "${LORA_R}" ]]; then # if 'LORA_R' is defined, finetune the model with LoRA

OUTPUT_DIR=beir_results/$MODEL_ID/$DATASET/lora_experiment_$(date +%m%d-%H%M)
echo "Finetuning with LoRA. Results will be saved inside $OUTPUT_DIR"

python ./finetuning.py \
--model_path $MODEL_PATH \
--train_data $TRAIN_DATA \
--eval_data $EVAL_DATA \
--total_steps $TOTAL_STEPS \
--save_freq $SAVE_FREQ \
--log_freq $LOG_FREQ \
--eval_freq $EVAL_FREQ \
--lr $LR \
--temperature $TEMPERATURE \
--label_smoothing $LABEL_SMOOTHING \
--chunk_length $CHUNK_LENGTH \
--per_gpu_batch_size $PER_GPU_BATCH_SIZE \
--accumulation_steps $ACCUMULATION_STEPS \
--scheduler $SCHEDULER \
--warmup_steps $WARMUP_STEPS \
--output_dir $OUTPUT_DIR \
--negative_ctxs $NEGATIVE_CTXS \
--negative_hard_ratio $NEGATIVE_HARD_RATIO \
--negative_hard_min_idx $NEGATIVE_HARD_MIN_IDX \
--pooling $POOLING \
--score_function $SCORE_FUNCTION \
$NORM_QUERY \
$NORM_DOC \
--use_lora \
--lora_r $LORA_R \
--lora_alpha $LORA_ALPHA \
--lora_dropout $LORA_DROPOUT \
--lora_target_modules $LORA_TARGET_MODULES \
--init_lora_weights $INIT_LORA_WEIGHTS \
--use_rslora \
--eval_datasets $DATASET \
--eval_split dev \
$USE_MINICORPUS


else # if 'LORA_R' is not defined, finetune the model without LoRA

OUTPUT_DIR=beir_results/$MODEL_ID/$DATASET/finetuned_basemodel_experiment_$(date +%m%d-%H%M)
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
--lr $LR \
--temperature $TEMPERATURE \
--label_smoothing $LABEL_SMOOTHING \
--chunk_length $CHUNK_LENGTH \
--per_gpu_batch_size $PER_GPU_BATCH_SIZE \
--accumulation_steps $ACCUMULATION_STEPS \
--scheduler $SCHEDULER \
--warmup_steps $WARMUP_STEPS \
--output_dir $OUTPUT_DIR \
--negative_ctxs $NEGATIVE_CTXS \
--negative_hard_ratio $NEGATIVE_HARD_RATIO \
--negative_hard_min_idx $NEGATIVE_HARD_MIN_IDX \
--pooling $POOLING \
--score_function $SCORE_FUNCTION \
$NORM_QUERY \
$NORM_DOC \
--eval_datasets $DATASET \
--eval_split dev \
$USE_MINICORPUS

fi