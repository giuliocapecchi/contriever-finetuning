#!/bin/bash

# args:
# MODEL_NAME_OR_PATH: Model name or path to the model.
# DATASET: Dataset to evaluate.
# BEIR_DIR: Directory where the BEIR datasets are stored.
# LORA_ADAPTER_PATH: Path to a folder containing a LoRA adapter. 
# FINETUNED_BASEMODEL_CHECKPOINT: Path to a folder containing a checkpoint of a fine-tuned model.

MODEL_NAME_OR_PATH=facebook/contriever-msmarco # e.g. nthakur/contriever-base-msmarco, intfloat/e5-large-v2, sentence-transformers/msmarco-distilbert-base-tas-b 
MODEL_ID=${MODEL_NAME_OR_PATH##*/}
DATASET=nfcorpus # dataset in BEIR format, e.g. nfcorpus, hotpotqa, scifact, etc.
BEIR_DIR=beir_datasets

save_results_path=./beir_results/${MODEL_ID}/${DATASET}/zero-shot-evaluation
PER_GPU_BATCH_SIZE=128

# LoRA parameters (optional)
LORA_ADAPTER_PATH= # path to a folder containing a LoRA adapter

# Finetuned model path (optional)
FINETUNED_BASEMODEL_CHECKPOINT= # path to a folder containing a checkpoint of a fine-tuned model



#########################################################################################################################################

if [[ ! -d "${save_results_path}" ]]; then # if the base-model evaluation folder does not exist, perform the evaluation
    echo "Evaluating basemodel"
    mkdir -p ./beir_results/${DATASET}/zero-shot-evaluation
    python eval_beir.py --model_name_or_path $MODEL_NAME_OR_PATH --dataset $DATASET --beir_dir $BEIR_DIR --save_results_path $save_results_path --output_dir $save_results_path --per_gpu_batch_size $PER_GPU_BATCH_SIZE
else
    echo "Base model evaluation already performed and saved in ${save_results_path}.txt"
fi


if [[ -n "${LORA_ADAPTER_PATH}" ]]; then # if 'LORA_ADAPTER_PATH' is defined, evaluate the model with LoRA
    echo "Evaluating with LoRA"
    python eval_beir.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --dataset $DATASET \
        --beir_dir $BEIR_DIR \
        --lora_adapter_path $LORA_ADAPTER_PATH \
        --save_results_path $LORA_ADAPTER_PATH \
        --output_dir $LORA_ADAPTER_PATH \
        --per_gpu_batch_size $PER_GPU_BATCH_SIZE

    # Update LORA_ADAPTER_PATH to include the wildcard
    LORA_ADAPTER_PATH=$(echo $LORA_ADAPTER_PATH | awk -F'/' '{OFS="/"; $NF="*" $NF; print}')
    echo "Creating table with results for "$DATASET" in "$LORA_ADAPTER_PATH", confronting the base model with LoRA"
    python ./beir_results/visualize_results.py --dataset $DATASET --results_folder $LORA_ADAPTER_PATH
fi


if [[ -n "${FINETUNED_BASEMODEL_CHECKPOINT}" ]]; then # if 'FINETUNED_BASEMODEL_CHECKPOINT' is defined, evaluate the fine-tuned model
    echo "Evaluating fine-tuned base model"
    python eval_beir.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --dataset $DATASET \
        --beir_dir $BEIR_DIR \
        --finetuned_basemodel_checkpoint $FINETUNED_BASEMODEL_CHECKPOINT \
        --save_results_path $FINETUNED_BASEMODEL_CHECKPOINT \
        --output_dir $FINETUNED_BASEMODEL_CHECKPOINT \
        --per_gpu_batch_size $PER_GPU_BATCH_SIZE
    
    # Update FINETUNED_BASEMODEL_CHECKPOINT to include the wildcard
    FINETUNED_BASEMODEL_CHECKPOINT=$(echo $FINETUNED_BASEMODEL_CHECKPOINT | awk -F'/' '{OFS="/"; $NF="*" $NF; print}')
    echo "Creating table with results for "$DATASET" in "$FINETUNED_BASEMODEL_CHECKPOINT", confronting the base model with the fine-tuned model"
    python ./beir_results/visualize_results.py --dataset $DATASET --results_folder $FINETUNED_BASEMODEL_CHECKPOINT
fi


if [[ -n "${LORA_ADAPTER_PATH}" && -n "${FINETUNED_BASEMODEL_CHECKPOINT}" ]]; then # if both are defined, perform paired t-test
    echo "Performing paired t-test between LoRA and fine-tuned base model"
    
    python src/statistical_tests.py --full-finetuned_model "${FINETUNED_BASEMODEL_CHECKPOINT}/perquery_scores.csv" --lora-finetuned_model "${LORA_ADAPTER_PATH}/perquery_scores.csv" --save_results_path "${LORA_ADAPTER_PATH}"

else
    echo "No paired t-test performed, as one of the paths is not defined."
fi
