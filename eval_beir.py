# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
import logging
import os

# import src.slurm
import src.contriever
import src.beir_utils
import src.utils
import src.dist_utils
import src.contriever
from peft import PeftModel

from src import inbatch
from src.options import Options


logger = logging.getLogger(__name__)


def main(args):

    logger = src.utils.init_logger(args)
    model, tokenizer, _ = src.contriever.load_retriever(args.model_name_or_path)
    
    # load LoRA module
    if args.lora_adapter_path is not None and args.finetuned_basemodel_checkpoint is None:
        if not os.path.exists(args.lora_adapter_path):
            raise FileNotFoundError(f"LoRA adapter path '{args.lora_adapter_path}' not found.")
        logger.info(f"Loading LoRA module from {args.lora_adapter_path}...")
        model = PeftModel.from_pretrained(model, args.lora_adapter_path)
    # load finetuned base-model
    elif args.finetuned_basemodel_checkpoint is not None and args.lora_adapter_path is None:
        checkpoint_path = os.path.join(args.finetuned_basemodel_checkpoint, "checkpoint.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Base model checkpoint path '{checkpoint_path}' not found")
        logger.info(f"Loading finetuned base-model from {checkpoint_path}...")
        # load weights
        checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        logger.info("Model loaded successfully")
    else:
        logger.info("No LoRA adapter or finetuned base-model loaded. Evaluating base model...")


    model = model.cuda()
    model.eval()
    query_encoder = model
    doc_encoder = model

    logger.info("Model loaded")

    logger.info("Start indexing")

    metrics = src.beir_utils.evaluate_model(
        query_encoder=query_encoder,
        doc_encoder=doc_encoder,
        tokenizer=tokenizer,
        dataset=args.dataset,
        batch_size=args.per_gpu_batch_size,
        norm_query=args.norm_query,
        norm_doc=args.norm_doc,
        is_main=src.dist_utils.is_main(),
        split="dev" if args.dataset == "msmarco" else "test",
        score_function=args.score_function,
        beir_dir=args.beir_dir,
        save_results_path=args.save_results_path,
        lower_case=args.lower_case,
        normalize_text=args.normalize_text,
    )

    if src.dist_utils.is_main():
        for key, value in metrics.items():
            logger.info(f"{args.dataset} : {key}: {value:.1f}")

    if args.lora_adapter_path is not None:
        logger.info(f"LoRA adapter path: {args.lora_adapter_path}")
        parts = args.lora_adapter_path.split("/")
        parts[-1] = "*" + parts[-1]
        updated_path = "/".join(parts)
        os.rename(args.lora_adapter_path, updated_path)
        logger.info(f"Updated LoRA adapter path: {updated_path}")
    elif args.finetuned_basemodel_checkpoint is not None: 
        logger.info(f"Finetuned base-model checkpoint path: {args.finetuned_basemodel_checkpoint}")
        parts = args.finetuned_basemodel_checkpoint.split("/")
        parts[-1] = "*" + parts[-1]
        updated_path = "/".join(parts)
        os.rename(args.finetuned_basemodel_checkpoint, updated_path)
        logger.info(f"Updated finetuned base-model path: {updated_path}")
    
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, help="Evaluation dataset from the BEIR benchmark")
    parser.add_argument("--beir_dir", type=str, default="./", help="Directory to save and load beir datasets")
    parser.add_argument("--text_maxlength", type=int, default=512, help="Maximum text length")

    parser.add_argument("--per_gpu_batch_size", default=128, type=int, help="Batch size per GPU/CPU for indexing.")
    parser.add_argument("--output_dir", type=str, default="./my_experiment", help="Output directory")
    parser.add_argument("--model_name_or_path", type=str, help="Model name or path")
    parser.add_argument(
        "--score_function", type=str, default="dot", help="Metric used to compute similarity between two embeddings"
    )
    parser.add_argument("--norm_query", action="store_true", help="Normalize query representation")
    parser.add_argument("--norm_doc", action="store_true", help="Normalize document representation")
    parser.add_argument("--lower_case", action="store_true", help="lowercase query and document text")
    parser.add_argument(
        "--normalize_text", action="store_true", help="Apply function to normalize some common characters"
    )
    parser.add_argument("--save_results_path", type=str, default=None, help="Path to save the results of the evaluation (in a txt file)")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # parser.add_argument("--main_port", type=int, default=-1, help="Main port (for multi-node SLURM jobs)")

    parser.add_argument("--lora_adapter_path", type=str, default=None, help="Path to LoRA module")
    parser.add_argument("--finetuned_basemodel_checkpoint", type=str, default=None, help="Path to base model checkpoint")

    args, _ = parser.parse_known_args()
    main(args)
