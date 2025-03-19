#!/bin/bash

# MODEL_NAME_OR_PATH: Model name or path to the model.
# DATASET: Dataset to evaluate.
# EXPERIMENT_NAME: Name of the experiment.
# LORA_MODEL_AT_STEP: Step of the LoRA model to evaluate.

MODEL_NAME_OR_PATH=facebook/contriever-msmarco
DATASET=DATASET_NAME_HERE
BEIR_DIR=beir_datasets
EXPERIMENT_NAME=EXPERIMENT_NAME_HERE
LORA_MODEL_AT_STEP=INSERT_LORA_MODEL_STEP_HERE

LORA_ADAPTER_PATH=beir_results/${DATASET}/${EXPERIMENT_NAME}/${LORA_MODEL_AT_STEP}
save_results_path=./beir_results/${DATASET}/contriever-beir-results/
lora_save_results_path=./beir_results/${DATASET}/${EXPERIMENT_NAME}/${LORA_MODEL_AT_STEP}

echo "Evaluating without LoRA"
if [[ ! -d "${save_results_path}" ]]; then # if the base-model evaluation folder does not exist, perform the evaluation
    mkdir -p ./beir_results/${DATASET}/contriever-beir-results/
    python eval_beir.py --model_name_or_path $MODEL_NAME_OR_PATH --dataset $DATASET --beir_dir $BEIR_DIR --save_results_path $save_results_path --output_dir $save_results_path
else
    echo "Base model evaluation already performed and saved in ${save_results_path}.txt"
fi

echo "Evaluating with LoRA"
python eval_beir.py --model_name_or_path $MODEL_NAME_OR_PATH --dataset $DATASET --beir_dir $BEIR_DIR --lora_adapter_path $LORA_ADAPTER_PATH --save_results_path $lora_save_results_path --output_dir $lora_save_results_path

echo "Creating table with results for $DATASET"
python ./beir_results/visualize_results.py --dataset $DATASET --experiment_name $EXPERIMENT_NAME --lora_model_at_step $LORA_MODEL_AT_STEP