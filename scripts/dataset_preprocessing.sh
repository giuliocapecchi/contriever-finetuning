#!/bin/bash
  
# DATASET_NAME: Evaluation dataset from the BEIR benchmark.
# NUM_NEGATIVES: Number of random negative documents per query.
# NUM_HARD_NEGATIVES: Number of hard negative documents per query.
# RANGE_MIN: Excludes the top 'range_min' most similar candidates.
# RANGE_MAX: Maximum rank of the closest matches to consider as negatives.
# MAX_SCORE: Allow negatives with a similarity score up to this value.
# MIN_SCORE: Exclude further negatives.
# MARGIN: Useful to skip candidates negatives whose similarity to the anchor is within a certain margin of the positive pair.
# SAMPLING_STRATEGY: 'top' will sample the hardest negatives, 'random' will sample randomly.
# BATCH_SIZE: Batch size for mining hard negatives.
# MAX_EXAMPLES: Maximum number of examples to generate.

DATASET_NAME=nfcorpus
NUM_NEGATIVES=5
NUM_HARD_NEGATIVES=5
RANGE_MIN=1
RANGE_MAX=50
MAX_SCORE=0.8
# MIN_SCORE=""
MARGIN=0.1
SAMPLING_STRATEGY="random"
BATCH_SIZE=32
# MAX_EXAMPLES=""

python mine_hard_negatives.py \
    --dataset_name $DATASET_NAME \
    --num_negatives $NUM_NEGATIVES \
    --num_hard_negatives $NUM_HARD_NEGATIVES \
    --range_min $RANGE_MIN \
    --range_max $RANGE_MAX \
    --max_score $MAX_SCORE \
    --margin $MARGIN \
    --sampling_strategy $SAMPLING_STRATEGY \
    --batch_size $BATCH_SIZE 
