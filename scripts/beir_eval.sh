#!/bin/bash

MODEL_NAME_OR_PATH=facebook/contriever-msmarco
DATASET=nfcorpus
LORA_ADAPTER_PATH=checkpoint/experiment_0312-1849/lora_step-0
mkdir -p ./beir_results/$DATASET
save_results_path=./beir_results/${DATASET}/contriever_beir_results
lora_save_results_path=./beir_results/${DATASET}/lora_beir_results

echo "Evaluating without LoRA"
python eval_beir.py --model_name_or_path $MODEL_NAME_OR_PATH --dataset $DATASET --save_results_path $save_results_path

echo "Evaluating with LoRA"
python eval_beir.py --model_name_or_path $MODEL_NAME_OR_PATH --dataset $DATASET --lora_adapter_path $LORA_ADAPTER_PATH --save_results_path $lora_save_results_path

echo "Creating table with results for $DATASET"
python ./beir_results/visualize_results.py --dataset $DATASET