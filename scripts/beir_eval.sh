#!/bin/bash

MODEL_NAME_OR_PATH=facebook/contriever-msmarco
DATASET=nfcorpus
EXPERIMENT_NAME=experiment_0314-1053
LORA_MODEL_AT_STEP=lora_step-0
LORA_ADAPTER_PATH=checkpoint/${EXPERIMENT_NAME}/${LORA_MODEL_AT_STEP}

save_results_path=./beir_results/${DATASET}/${EXPERIMENT_NAME}/contriever_beir_results
lora_save_results_path=./beir_results/${DATASET}/${EXPERIMENT_NAME}/lora_beir_results

mkdir -p ./beir_results/$DATASET/$EXPERIMENT_NAME


echo "Evaluating without LoRA"
if [[ ! -f "${save_results_path}" && ! -f "${save_results_path}.txt" ]]; then # if the base-model evaluation has already been performed, skip this step (results won't change for the base-model)
    python eval_beir.py --model_name_or_path $MODEL_NAME_OR_PATH --dataset $DATASET --save_results_path $save_results_path
else
    echo "Base model evaluation already performed and saved in ${save_results_path}.txt"
fi

echo "Evaluating with LoRA"
python eval_beir.py --model_name_or_path $MODEL_NAME_OR_PATH --dataset $DATASET --lora_adapter_path $LORA_ADAPTER_PATH --save_results_path $lora_save_results_path

echo "Creating table with results for $DATASET"
python ./beir_results/visualize_results.py --dataset $DATASET --experiment_name $EXPERIMENT_NAME