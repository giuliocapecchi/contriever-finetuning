#!/bin/bash

# args:
# MODEL_NAME_OR_PATH: Model name or path to the model.
# DATASET: Dataset to evaluate.
# BEIR_DIR: Directory where the BEIR datasets are stored.
# LORA_ADAPTER_PATH: Path to a folder containing a LoRA adapter. 
# FINETUNED_BASEMODEL_CHECKPOINT: Path to a folder containing a checkpoint of a fine-tuned model.

MODEL_NAME_OR_PATH=MODEL_PATH_HERE  # e.g. facebook/contriever-msmarco, intfloat/e5-base-v2, sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco 
MODEL_ID=${MODEL_NAME_OR_PATH##*/}
DATASET=DATASET_NAME_HERE # dataset in BEIR format, e.g. nfcorpus, hotpotqa, scifact, etc.
BEIR_DIR=beir_datasets

USE_RERANKER=--use_reranker
RERANKER_MODEL_NAME="BAAI/bge-reranker-base" # e.g. BAAI/bge-reranker-base, cross-encoder/ms-marco-electra-base

# Path to a folder containing a LoRA adapter (optional)
LORA_ADAPTER_PATH=beir_results/contriever-base-msmarco/scifact/lora_experiment_0520-1006/lora_step-800 #
# Path to a folder containing a checkpoint of a fine-tuned model (optional)
FINETUNED_BASEMODEL_CHECKPOINT=beir_results/contriever-base-msmarco/scifact/finetuned_basemodel_experiment_0520-1209/checkpoint/step-600


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

save_results_path=./beir_results/${MODEL_ID}/${DATASET}/zero-shot-evaluation
PER_GPU_BATCH_SIZE=128


#########################################################################################################################################

if [[ ! -d "${save_results_path}" ]]; then # if the base-model evaluation folder does not exist, perform the evaluation
    echo "Evaluating basemodel"
    mkdir -p $save_results_path
    python eval_beir.py --model_name_or_path $MODEL_NAME_OR_PATH --dataset $DATASET --score_function $SCORE_FUNCTION --pooling $POOLING $NORM_QUERY $NORM_DOC --prefix_type $PREFIX_TYPE --beir_dir $BEIR_DIR --save_results_path $save_results_path --output_dir $save_results_path --per_gpu_batch_size $PER_GPU_BATCH_SIZE $USE_RERANKER --reranker_model_name $RERANKER_MODEL_NAME
else
    echo "Base model evaluation already performed and saved in ${save_results_path}/metrics.txt"
fi


if [[ -n "${LORA_ADAPTER_PATH}" ]]; then # if 'LORA_ADAPTER_PATH' is defined, evaluate the model with LoRA
    echo "Evaluating with LoRA"
    python eval_beir.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --dataset $DATASET \
        --score_function $SCORE_FUNCTION \
        --pooling $POOLING \
        $NORM_QUERY \
        $NORM_DOC \
        --prefix_type $PREFIX_TYPE \
        --beir_dir $BEIR_DIR \
        --lora_adapter_path $LORA_ADAPTER_PATH \
        --save_results_path $LORA_ADAPTER_PATH \
        --output_dir $LORA_ADAPTER_PATH \
        --per_gpu_batch_size $PER_GPU_BATCH_SIZE \
        $USE_RERANKER \
        --reranker_model_name $RERANKER_MODEL_NAME

    # Update LORA_ADAPTER_PATH to include the wildcard
    LORA_ADAPTER_PATH=$(echo $LORA_ADAPTER_PATH | awk -F'/' '{OFS="/"; $NF="*" $NF; print}')
    # create table with results for the LoRA model
    echo "Creating table with results for '$DATASET' in '$LORA_ADAPTER_PATH', confronting the base model with LoRA"
    python ./beir_results/visualize_results.py --zeroshot_folder $save_results_path --results_folder $LORA_ADAPTER_PATH
    # perform statistical test between LoRA and base model
    echo "Performing statistical test between LoRA and base model in zero-shot"
    python src/statistical_tests.py --modelA "${save_results_path}/perquery_scores.csv" --modelB "${LORA_ADAPTER_PATH}/perquery_scores.csv" --save_results_path "${LORA_ADAPTER_PATH}/zeroshot_vs_lora.txt"
fi


if [[ -n "${FINETUNED_BASEMODEL_CHECKPOINT}" ]]; then # if 'FINETUNED_BASEMODEL_CHECKPOINT' is defined, evaluate the fine-tuned model
    echo "Evaluating fine-tuned base model"
    python eval_beir.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --dataset $DATASET \
        --score_function $SCORE_FUNCTION \
        --pooling $POOLING \
        $NORM_QUERY \
        $NORM_DOC \
        --prefix_type $PREFIX_TYPE \
        --score_function $SCORE_FUNCTION \
        --beir_dir $BEIR_DIR \
        --finetuned_basemodel_checkpoint $FINETUNED_BASEMODEL_CHECKPOINT \
        --save_results_path $FINETUNED_BASEMODEL_CHECKPOINT \
        --output_dir $FINETUNED_BASEMODEL_CHECKPOINT \
        --per_gpu_batch_size $PER_GPU_BATCH_SIZE \
        $USE_RERANKER \
        --reranker_model_name $RERANKER_MODEL_NAME
    
    # Update FINETUNED_BASEMODEL_CHECKPOINT to include the wildcard
    FINETUNED_BASEMODEL_CHECKPOINT=$(echo $FINETUNED_BASEMODEL_CHECKPOINT | awk -F'/' '{OFS="/"; $NF="*" $NF; print}')
    # create table with results for the fine-tuned model
    echo "Creating table with results for '$DATASET' in '$FINETUNED_BASEMODEL_CHECKPOINT', confronting the base model with the fine-tuned model"
    python ./beir_results/visualize_results.py --zeroshot_folder $save_results_path --results_folder $FINETUNED_BASEMODEL_CHECKPOINT
    # perform statistical test between LoRA and fine-tuned base model
    echo "Performing statistical test between full-finetuned model and base model in zero-shot"
    python src/statistical_tests.py --modelA "${save_results_path}/perquery_scores.csv" --modelB "${FINETUNED_BASEMODEL_CHECKPOINT}/perquery_scores.csv" --save_results_path "${FINETUNED_BASEMODEL_CHECKPOINT}/zeroshot_vs_finetuned.txt"
fi


if [[ -n "${LORA_ADAPTER_PATH}" && -n "${FINETUNED_BASEMODEL_CHECKPOINT}" ]]; then # if both are defined, perform paired t-test between the two
    echo "Performing paired t-test between LoRA and fine-tuned base model"
    
    python src/statistical_tests.py --modelA "${FINETUNED_BASEMODEL_CHECKPOINT}/perquery_scores.csv" --modelB "${LORA_ADAPTER_PATH}/perquery_scores.csv" --save_results_path "${LORA_ADAPTER_PATH}/finetuned_vs_lora.txt"

else
    echo "No paired t-test performed, as one of the paths is not defined."
fi
