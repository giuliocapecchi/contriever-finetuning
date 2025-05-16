import argparse
import logging

import optuna
import os
import torch

from finetuning import apply_lora, finetuning
from src import contriever, inbatch, utils
from src.options import Options


def objective(trial, args):

    # setup options
    options = Options()
    opt = options.parse()
    
    opt.per_gpu_batch_size = 32
    opt.accumulation_steps = 2
    opt.model_path = args.model_path
    opt.train_data = [f"./beir_datasets/{args.dataset}/{args.model_path}/training_data.jsonl"]
    opt.eval_data = [f"./beir_datasets/{args.dataset}/{args.model_path}/dev_data.jsonl"]
    opt.eval_datasets = [args.dataset]
    opt.log_freq = 10
    opt.total_steps = 20
    opt.warmup_steps = opt.total_steps // 10
    opt.eval_freq = 20
    opt.save_freq = 2500
    opt.use_minicorpus = False if args.dataset in ["nfcorpus", "fiqa", "scifact", "legalbenchrag"] else True

    # set optuna hyperparameters
    if opt.use_lora:
        opt.lora_r = trial.suggest_int("lora_r", 4, 64)
        opt.lora_alpha = trial.suggest_int("lora_alpha", 8, 128)
        opt.use_rslora = trial.suggest_categorical("use_rslora", [True, False])
        opt.lora_x = trial.suggest_float("lora_dropout", 0.0, 0.3)
        opt.lora_target_modules = trial.suggest_categorical("target_modules", ["query","key","value","output.dense","intermediate.dense"])
        opt.init_lora_weights = trial.suggest_categorical("init_lora_weights", [True, "pissa"])
    opt.learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    opt.scheduler = trial.suggest_categorical("scheduler", ["linear", "cosine"])
    opt.dropout = trial.suggest_float("dropout", 0.0, 0.3)
    opt.negative_ctxs = trial.suggest_int("negative_ctxs", 1, 4)
    opt.negative_hard_ratio = trial.suggest_float("negative_hard_ratio", 0.0, 1.0)
    opt.temperature = trial.suggest_float("temperature", 0.05, 1.0)
    opt.label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.3)

    if opt.use_lora:
        opt.output_dir = f"hyperparameter_experiments/{args.dataset}/lorafinetune_trial_{trial.number}"
    else:
        opt.output_dir = f"hyperparameter_experiments/{args.dataset}/finetunedbasemodel_trial_{trial.number}"

    options.print_options(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    logger = utils.init_logger(opt)

    if opt.accumulation_steps > 1:
        logger.info(f"Gradient accumulation steps: {opt.accumulation_steps}. Because of this the actual batch size is {opt.per_gpu_batch_size} * {opt.accumulation_steps} = {opt.per_gpu_batch_size * opt.accumulation_steps}.")
    

    # load retriever and tokenizer
    retriever, tokenizer, retriever_model_id = contriever.load_retriever(opt.model_path, opt.pooling, opt.random_init)
    opt.retriever_model_id = retriever_model_id
    model = inbatch.InBatch(opt, retriever, tokenizer)
    
    logger.info(utils.get_parameters(model))
    if opt.use_lora:
        model.encoder = apply_lora(model.encoder, opt)
        logger.info(model)
        logger.info(utils.get_parameters(model, using_lora=True))
    model = model.cuda()

    optimizer, scheduler = utils.set_optim(opt, model)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = opt.dropout

    # finally, perform finetuning (the function returns the best evaluation result)
    _, best_metric = finetuning(opt, model, optimizer, scheduler, tokenizer, 0, trial = trial)

    return best_metric  

def main(args):    
    os.makedirs(f"hyperparameter_experiments/{args.dataset}", exist_ok=True)
    study = optuna.create_study(
    direction="maximize",
    study_name=f"hyperparameter_tuning_{args.dataset}",
    storage=f"sqlite:///hyperparameter_experiments/{args.dataset}/hyperparameter_tuning.db",
    load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, args), n_trials=10) # from optuna faq, a lambda is okay
    print("Best trial:")
    print(study.best_trial)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default="nfcorpus", help="Dataset to use")
    parser.add_argument("--model_path", type=str, default="facebook/contriever-msmarco", help="Path to the model")
    args, _ = parser.parse_known_args()
    
    main(args)

