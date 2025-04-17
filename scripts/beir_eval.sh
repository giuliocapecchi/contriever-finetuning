#!/bin/bash

# args:
# MODEL_NAME_OR_PATH: Model name or path to the model.
# DATASET: Dataset to evaluate.
# BEIR_DIR: Directory where the BEIR datasets are stored.
# LORA_ADAPTER_PATH: Path to a folder containing a LoRA adapter. 
# FINETUNED_BASEMODEL_CHECKPOINT: Path to a folder containing a checkpoint of a fine-tuned model.

MODEL_NAME_OR_PATH=facebook/contriever-msmarco
DATASET=DATASET_NAME_HERE # e.g., nfcorpus, hotpotqa, scifact, etc.
BEIR_DIR=beir_datasets
save_results_path=./beir_results/${DATASET}/contriever-beir-results/
PER_GPU_BATCH_SIZE=128

# LoRA parameters (optional)
LORA_ADAPTER_PATH=beir_results/hotpotqa/lora_experiment_0405-0202/lora_step-6000

# Finetuned model path (optional)
FINETUNED_BASEMODEL_CHECKPOINT=beir_results/hotpotqa/finetuned_basemodel_experiment_0405-2025/step-10000



#########################################################################################################################################

if [[ ! -d "${save_results_path}" ]]; then # if the base-model evaluation folder does not exist, perform the evaluation
    echo "Evaluating basemodel"
    mkdir -p ./beir_results/${DATASET}/contriever-beir-results/
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

    echo "Creating table with results for "$DATASET" in "$FINETUNED_BASEMODEL_CHECKPOINT", confronting the base model with the fine-tuned model"
    python ./beir_results/visualize_results.py --dataset $DATASET --results_folder $FINETUNED_BASEMODEL_CHECKPOINT
fi
