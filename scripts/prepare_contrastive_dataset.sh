#!/bin/bash

MODEL_NAME=nthakur/contriever-base-msmarco
DATASET_NAME=DATASET_NAME_HERE
NUM_NEGATIVES=5
NUM_HARD_NEGATIVES=5
RELATIVE_MARGIN=0.05
RANGE_MAX=2000
BATCH_SIZE=512

python prepare_contrastive_dataset.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --num_negatives $NUM_NEGATIVES \
    --num_hard_negatives $NUM_HARD_NEGATIVES \
    --relative_margin $RELATIVE_MARGIN \
    --range_max $RANGE_MAX \
    --batch_size $BATCH_SIZE \
    --use_faiss
