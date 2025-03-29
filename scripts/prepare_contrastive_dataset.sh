#!/bin/bash

MODEL_NAME=nthakur/contriever-base-msmarco
DATASET_NAME=DATASET_NAME_HERE
NUM_NEGATIVES=5
NUM_HARD_NEGATIVES=5
RANGE_MAX=50
BATCH_SIZE=512

python prepare_contrastive_dataset.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --num_negatives $NUM_NEGATIVES \
    --num_hard_negatives $NUM_HARD_NEGATIVES \
    --range_max $RANGE_MAX \
    --batch_size $BATCH_SIZE \
    --use_faiss \
