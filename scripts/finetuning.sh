#!/bin/bash

# model paths, e.g., facebook/contriever-msmarco, intfloat/e5-base-v2, sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco, ...
models=(
  "facebook/contriever-msmarco"
  "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
  "intfloat/e5-base-v2"
)

# datasets in BEIR format, e.g. nfcorpus, hotpotqa, scifact, etc.
datasets=(
  "scifact"
  "nfcorpus"
  "fiqa"
  "legalbenchrag"
  "hotpotqa"
  "fever"
  "nq-train"
)

for MODEL_PATH in "${models[@]}"; do
  for DATASET in "${datasets[@]}"; do

    MODEL_ID=${MODEL_PATH##*/}

    if [[ "$MODEL_ID" == "contriever-msmarco" ]]; then
        LORA_TARGET_MODULES="query key value output.dense intermediate.dense"
        SCORE_FUNCTION="dot"
        NORM_QUERY=""
        NORM_DOC=""
        POOLING="average"
        MODEL_ID="contriever-base-msmarco"
        PREFIX_TYPE="none"
        
    elif [[ "$MODEL_ID" == "e5-base-v2" || "$MODEL_ID" == "e5-large-v2" ]]; then
        LORA_TARGET_MODULES="query key value output.dense intermediate.dense"
        SCORE_FUNCTION="cos_sim"
        NORM_QUERY="--norm_query"
        NORM_DOC="--norm_doc"
        POOLING="average"
        PREFIX_TYPE="query_or_passage"
        
    elif [[ "$MODEL_ID" == "distilbert-dot-tas_b-b256-msmarco" ]]; then
        LORA_TARGET_MODULES="q_lin k_lin v_lin out_lin lin1 lin2"
        SCORE_FUNCTION="dot"
        NORM_QUERY=""
        NORM_DOC=""
        POOLING="cls"
        MODEL_ID="msmarco-distilbert-base-tas-b"
        PREFIX_TYPE="none"
        
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
    ACCUMULATION_STEPS=2 # for contriever they used 64 as batchsize
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
    --prefix_type $PREFIX_TYPE \
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
    --prefix_type $PREFIX_TYPE \
    --eval_datasets $DATASET \
    --eval_split dev \
    $USE_MINICORPUS

    done
done