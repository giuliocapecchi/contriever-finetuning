#!/bin/bash

NUM_NEGATIVES=5
NUM_HARD_NEGATIVES=5
RELATIVE_MARGIN=0.05
RANGE_MAX=2000
BATCH_SIZE=512


# model paths, e.g. nthakur/contriever-base-msmarco, intfloat/e5-base-v2, sentence-transformers/msmarco-distilbert-base-tas-b 
models=(
  "nthakur/contriever-base-msmarco"
  "sentence-transformers/msmarco-distilbert-base-tas-b"
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

for MODEL_NAME in "${models[@]}"; do
  for DATASET_NAME in "${datasets[@]}"; do

    MODEL_ID=${MODEL_NAME##*/}
    if [[ "$MODEL_ID" == "e5-base-v2" || "$MODEL_ID" == "e5-large-v2" ]]; then
        PREFIX_TYPE="query_or_passage"
        NORMALIZE_EMBEDDINGS="--normalize_embeddings"
    else
        PREFIX_TYPE="none"
        NORMALIZE_EMBEDDINGS=""
    fi

    python prepare_contrastive_dataset.py \
        --model_name $MODEL_NAME \
        --dataset_name $DATASET_NAME \
        --num_negatives $NUM_NEGATIVES \
        --num_hard_negatives $NUM_HARD_NEGATIVES \
        --relative_margin $RELATIVE_MARGIN \
        --prefix_type $PREFIX_TYPE \
        $NORMALIZE_EMBEDDINGS \
        --range_max $RANGE_MAX \
        --batch_size $BATCH_SIZE \
        --use_faiss

    done
done