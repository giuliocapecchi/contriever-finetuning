#!/bin/bash

MODEL_NAME=nthakur/contriever-base-msmarco
DATASET_NAME=DATASET_NAME_HERE
NUM_NEGATIVES=100
NUM_HARD_NEGATIVES=100
RANGE_MAX=500
BATCH_SIZE=128

python prepare_contrastive_dataset.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --num_negatives $NUM_NEGATIVES \
    --num_hard_negatives $NUM_HARD_NEGATIVES \
    --range_max $RANGE_MAX \
    --batch_size $BATCH_SIZE \
    --use_faiss \
