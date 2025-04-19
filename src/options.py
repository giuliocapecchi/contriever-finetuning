# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import os


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize()

    def initialize(self):
        # basic parameters
        self.parser.add_argument(
            "--output_dir", type=str, default="./checkpoint/my_experiment", help="LoRA modules are saved here"
        )
        self.parser.add_argument(
            "--train_data",
            nargs="+",
            default=[],
            help="Data used for training, passed as a list of directories splitted into tensor files.",
        )
        self.parser.add_argument(
            "--eval_data",
            nargs="+",
            default=[],
            help="Data used for evaluation during finetuning, this option is not used during contrastive pre-training.",
        )
        self.parser.add_argument(
            "--eval_datasets", nargs="+", default=[], help="List of datasets used for evaluation, in BEIR format"
        )
        self.parser.add_argument(
            "--eval_datasets_dir", type=str, default="./beir_datasets", help="Directory where eval datasets are stored"
        )
        self.parser.add_argument("--model_path", type=str, default="none", help="path for retraining")
        self.parser.add_argument("--continue_training", action="store_true")
        self.parser.add_argument("--num_workers", type=int, default=5)

        self.parser.add_argument("--chunk_length", type=int, default=256)
        self.parser.add_argument("--loading_mode", type=str, default="split")
        self.parser.add_argument("--lower_case", action="store_true", help="perform evaluation after lowercasing")
        
        self.parser.add_argument("--dropout", type=float, default=0.1)

        self.parser.add_argument("--temperature", type=float, default=1.0)
        self.parser.add_argument("--eval_normalize_text", action="store_false")
        self.parser.add_argument("--norm_query", action="store_true")
        self.parser.add_argument("--norm_doc", action="store_true")
        self.parser.add_argument("--projection_size", type=int, default=768)

        self.parser.add_argument("--score_function", type=str, default="dot")
        self.parser.add_argument("--retriever_model_id", type=str, default="bert-base-uncased")
        self.parser.add_argument("--pooling", type=str, default="average")
        self.parser.add_argument("--random_init", action="store_true", help="init model with random weights")

        # dataset parameters
        self.parser.add_argument("--per_gpu_batch_size", default=64, type=int, help="Batch size per GPU for training.")
        self.parser.add_argument(
            "--per_gpu_eval_batch_size", default=256, type=int, help="Batch size per GPU for evaluation."
        )
        self.parser.add_argument("--use_minicorpus", action="store_true", help="wether to use a reduced corpus for evaluation (speeds up computations). The minicorpus is a subset of the original corpus and should be placed inside the dataset folder with name 'minicorpus.jsonl'.")
        self.parser.add_argument("--eval_split", type=str, default="dev", help="split used for evaluation")

        self.parser.add_argument("--total_steps", type=int, default=1000)
        self.parser.add_argument("--warmup_steps", type=int, default=-1)

        self.parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
        self.parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
        # training parameters
        self.parser.add_argument("--optim", type=str, default="adamw")
        self.parser.add_argument("--scheduler", type=str, default="linear")
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument(
            "--lr_min_ratio",
            type=float,
            default=0.0,
            help="minimum learning rate at the end of the optimization schedule as a ratio of the learning rate",
        )
        self.parser.add_argument("--weight_decay", type=float, default=0.01, help="learning rate")
        self.parser.add_argument("--beta1", type=float, default=0.9, help="beta1")
        self.parser.add_argument("--beta2", type=float, default=0.98, help="beta2")
        self.parser.add_argument("--eps", type=float, default=1e-6, help="eps")
        self.parser.add_argument(
            "--log_freq", type=int, default=100, help="log train stats every <log_freq> steps during training"
        )
        self.parser.add_argument(
            "--eval_freq", type=int, default=500, help="evaluate model every <eval_freq> steps during training"
        )
        self.parser.add_argument("--save_freq", type=int, default=50000)
        self.parser.add_argument("--maxload", type=int, default=None)
        self.parser.add_argument("--label_smoothing", type=float, default=0.0)

        # finetuning options
        self.parser.add_argument("--accumulation_steps", type=int, default=1)
        self.parser.add_argument("--negative_ctxs", type=int, default=1)
        self.parser.add_argument("--negative_hard_min_idx", type=int, default=0)
        self.parser.add_argument("--negative_hard_ratio", type=float, default=0.0)

        # LoRA options
        self.parser.add_argument("--use_lora", action="store_true", help="Whether to use LoRA or the base model.")
        self.parser.add_argument("--lora_r", type=int, default=8, help ="LoRA matrix rank")
        self.parser.add_argument("--use_rslora", action="store_true", help="Whether to use RSLORA for alpha scaling.") 
        self.parser.add_argument("--lora_alpha", type=float, default=32, help="LoRA alpha scaling factor")
        self.parser.add_argument("--lora_dropout", type=float, default=0.1, help="dropout for LoRA")
        self.parser.add_argument("--init_lora_weights", type=str, default="none", help="Choose between 'gaussian', 'eva', 'olora', 'pissa', 'pissa_niter_[number of iters]', 'loftq'")
        self.parser.add_argument("--lora_target_modules", nargs="+", default=None, help="LoRA target modules")
        

    def print_options(self, opt):
        message = ""
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = f"\t[default: %s]" % str(default)
            message += f"{str(k):>40}: {str(v):<40}{comment}\n"
        print(message, flush=True)
        if not os.path.exists(opt.output_dir):
            os.makedirs(opt.output_dir)
        file_name = os.path.join(opt.output_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self):
        opt, _ = self.parser.parse_known_args()
        # opt = self.parser.parse_args()
        return opt
