# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import logging
import numpy as np
import torch.distributed as dist # allows for distributed training
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from src.options import Options
from src import dist_utils, utils, contriever, finetuning_data, inbatch
import train

# LoRA
from peft import get_peft_model, LoraConfig, TaskType

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__) # logger for this file


def apply_lora(model, opt):
    
    target_modules = opt.lora_target_modules
    if target_modules is not None and isinstance(target_modules, list):
        target_modules = target_modules[0].split(",")
        target_modules = [module.strip() for module in target_modules]

    logger.info(f"Applying LoRA to the following modules: {target_modules}")

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=opt.lora_r,
        lora_alpha=opt.lora_alpha,
        use_rslora= True if str(opt.use_rslora).lower() == "true" else False,
        lora_dropout=opt.lora_dropout,
        init_lora_weights=opt.init_lora_weights if opt.init_lora_weights != 'none' else True,
        target_modules=target_modules,
        
    )
    try:
        model = get_peft_model(model, lora_config)
    except ValueError as e:
        logger.error(f"Error: {e}")
        logger.error("LoRA could not be applied to the model. The training will proceed without LoRA. The base model is composed by the following modules:")
        for name, _ in model.named_modules():
            logger.error(f"{name}")
    return model


def save_lora_model(model, optimizer, scheduler, output_dir, opt,  step=None):
    if hasattr(model, "module"):
        model = model.module # remove the DataParallel wrapper
    if step:
        fp = os.path.join(output_dir, f"lora_step-{step}")
        model.save_pretrained(fp) # since model was wrapped with PEFT's LoRA, this should save only the LoRA weights and config
    else: # final model
        fp = os.path.join(output_dir, "final_lora_model")
        model.save_pretrained(fp)

    # save the optimizer and scheduler state separately
    checkpoint_fp = os.path.join(fp, "checkpoint.pth")
    checkpoint = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "opt": opt
    }
    torch.save(checkpoint, checkpoint_fp)
    logger.info(f"Saved LoRA model and training state to {fp}")


def finetuning(opt, model, optimizer, scheduler, tokenizer, step):

    run_stats = utils.WeightedAvgStats()

    tb_logger = utils.init_tb_logger(opt.output_dir) # tensorboard logger

    if hasattr(model, "module"): # if model is a DataParallel object (DDP)
        logger.info("Model is a DataParallel object")
        eval_model = model.module 
    else:
        logger.info("Model is not a DataParallel object")
        eval_model = model
    eval_model = eval_model.get_encoder()

    # load the training data
    train_dataset = finetuning_data.Dataset(
        datapaths=opt.train_data,
        negative_ctxs=opt.negative_ctxs,
        training=True,
        negative_hard_ratio=opt.negative_hard_ratio,
        negative_hard_min_idx=opt.negative_hard_min_idx, 
        global_rank=dist_utils.get_rank(),
        world_size=dist_utils.get_world_size(),
        maxload=opt.maxload,
    )

    # collator manages the batch creation
    collator = finetuning_data.Collator(tokenizer, passage_maxlength=opt.chunk_length)
    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=opt.num_workers,
        collate_fn=collator,
    )

    train.eval_model(opt, eval_model, None, tokenizer, tb_logger, step) # evaluate the model before training
    evaluate(opt, eval_model, tokenizer, tb_logger, step) # evaluate the model before training

    epoch = 1

    model.train() # sets model in training mode

    while step < opt.total_steps:
        logger.info(f"Start epoch {epoch}, number of batches: {len(train_dataloader)}")
        
        # logging to ensure the correct number of batches
        if len(train_dataloader) == 0:
            logger.warning("No batches found in train_dataloader. Check your training data.")
            break

        for i, batch in enumerate(train_dataloader):
            batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            step += 1

            train_loss, iter_stats = model(**batch, stats_prefix="train") # 
            train_loss.backward() 

            if opt.optim == "sam" or opt.optim == "asam": # if SAM or ASAM optimizer is used, applies a double update
                optimizer.first_step(zero_grad=True)

                sam_loss, _ = model(**batch, stats_prefix="train/sam_opt")
                sam_loss.backward()
                optimizer.second_step(zero_grad=True)

            else:
                optimizer.step()
            
            scheduler.step() # adjusts the learning rate
            optimizer.zero_grad()

            run_stats.update(iter_stats)

            if step % opt.log_freq == 0: # log the training statistics
                log = f"{step} / {opt.total_steps}"
                for k, v in sorted(run_stats.average_stats.items()): # average statistics
                    log += f" | {k}: {v:.3f}"
                    if tb_logger: # log to tensorboard
                        tb_logger.add_scalar(k, v, step)
                log += f" | lr: {scheduler.get_last_lr()[0]:0.3g}" # learning rate
                log += f" | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB" # memory usage

                logger.info(log)
                run_stats.reset()

            if step % opt.eval_freq == 0: # evaluate the model every eval_freq steps

                train.eval_model(opt, eval_model, None, tokenizer, tb_logger, step)
                evaluate(opt, eval_model, tokenizer, tb_logger, step)

                if step % opt.save_freq == 0 and dist_utils.get_rank() == 0: # save the model every save_freq steps
                    if opt.use_lora: # if LoRA is applied, save the LoRA module
                        save_lora_model(model, optimizer, scheduler, opt.output_dir, opt, step)
                    else: # otherwise, save the whole finetuned model
                        utils.save(
                        model,
                        optimizer,
                        scheduler,
                        step,
                        opt,
                        opt.output_dir,
                        f"step-{step}",
                        )
                    logger.info(f"Saved model at step {step}")
                model.train()

            if step >= opt.total_steps:
                break

        epoch += 1


def evaluate(opt, model, tokenizer, tb_logger, step):
    """ Evaluate the model on the evaluation data"""
    # first load the evaluation data
    dataset = finetuning_data.Dataset(
        datapaths=opt.eval_data,
        normalize=opt.eval_normalize_text,
        global_rank=dist_utils.get_rank(),
        world_size=dist_utils.get_world_size(),
        maxload=opt.maxload,
        training=False,
    )

    collator = finetuning_data.Collator(tokenizer, passage_maxlength=opt.chunk_length)
    sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=opt.num_workers,
        collate_fn=collator,
    )

    model.eval() # sets model in evaluation (inference) mode (normalisation layers use running statistics, de-activates Dropout layers)

    if hasattr(model, "module"): # if model is a DataParallel object (DDP) remove the DataParallel wrapper
        model = model.module

    all_q, all_g, all_n = [], [], [] # lists to store the query, positive and negative embeddings

    logger.info("Start embedding all queries and documents")

    with torch.no_grad(): # disables gradient calculation to save memory (speeds up computation)
        for i, batch in enumerate(dataloader):

            batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

            all_tokens = torch.cat([batch["g_tokens"], batch["n_tokens"]], dim=0) # concatenate the positive and negative tokens
            all_mask = torch.cat([batch["g_mask"], batch["n_mask"]], dim=0) # concatenate the positive and negative masks

            q_emb = model(input_ids=batch["q_tokens"], attention_mask=batch["q_mask"], normalize=opt.norm_query) # query embeddings
            all_emb = model(input_ids=all_tokens, attention_mask=all_mask, normalize=opt.norm_doc) # positive and negative embeddings

            g_emb, n_emb = torch.split(all_emb, [len(batch["g_tokens"]), len(batch["n_tokens"])])

            all_q.append(q_emb)
            all_g.append(g_emb)
            all_n.append(n_emb)

        # concatenate query, positive and negative embeddings
        all_q = torch.cat(all_q, dim=0) 
        all_g = torch.cat(all_g, dim=0)
        all_n = torch.cat(all_n, dim=0)

        if dist_utils.is_main():
            logger.info("Finished embedding all queries and documents")

        labels = torch.arange(0, len(all_q), device=all_q.device, dtype=torch.long) # create a tensor of labels for the queries (0, 1, 2, ..., len(all_q))

        all_sizes = dist_utils.get_varsize(all_g)
        all_g = dist_utils.varsize_gather_nograd(all_g)
        all_n = dist_utils.varsize_gather_nograd(all_n)
        labels = labels + sum(all_sizes[: dist_utils.get_rank()])

        # calcualte the similarity between the query and the positive and negative samples
        scores_pos = torch.einsum("id, jd->ij", all_q, all_g) # scalar product between all rows of all_q and all rows of all_g. Returns a matrix of size (all_q.size(0), all_g.size(0)), where each element represents the similarity between a query and a positive sample
        scores_neg = torch.einsum("id, jd->ij", all_q, all_n) # scalar product between all rows of all_q and all rows of all_n
        scores = torch.cat([scores_pos, scores_neg], dim=-1) 

        argmax_idx = torch.argmax(scores, dim=1) # Returns the index of the maximum value of the scores matrix along the second dimension (axis=1)
        sorted_scores, indices = torch.sort(scores, descending=True)
        isrelevant = indices == labels[:, None] # Returns a boolean matrix of the same size as the scores matrix, where each element is True if the corresponding element in the scores matrix is the correct label for the query
        rs = [r.cpu().numpy().nonzero()[0] for r in isrelevant] # returns the indices of the relevant documents for each query
        mrr = np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])

        acc = (argmax_idx == labels).sum() / all_q.size(0) 
        acc, total = dist_utils.weighted_average(acc, all_q.size(0))
        mrr, _ = dist_utils.weighted_average(mrr, all_q.size(0))
        acc = 100 * acc

        message = []
        if dist_utils.is_main():
            message = [f"eval acc: {acc:.2f}%", f"eval mrr: {mrr:.3f}"]
            logger.info(" | ".join(message))
            if tb_logger is not None:
                tb_logger.add_scalar(f"eval_acc", acc, step)
                tb_logger.add_scalar(f"mrr", mrr, step)


def main():
    logger.info("Start")

    options = Options()
    opt = options.parse()
    options.print_options(opt)

    torch.manual_seed(opt.seed)

    # Removed SLURM initialization
    # slurm.init_distributed_mode(opt)
    # slurm.init_signal_handler()

    directory_exists = os.path.isdir(opt.output_dir)
    if dist.is_initialized():
        dist.barrier()
    os.makedirs(opt.output_dir, exist_ok=True)
    if not directory_exists and dist_utils.is_main():
        options.print_options(opt)
    if dist.is_initialized():
        dist.barrier()
    utils.init_logger(opt)

    step = 0

    # ensure model_path is correctly set
    if not hasattr(opt, 'model_path') or not opt.model_path:
        raise ValueError("The model_path argument is required.")

    retriever, tokenizer, retriever_model_id = contriever.load_retriever(opt.model_path, opt.pooling, opt.random_init)
    opt.retriever_model_id = retriever_model_id
    model = inbatch.InBatch(opt, retriever, tokenizer)

    # print the number of trainable parameters in the model before and after applying LoRA
    logger.info(utils.get_parameters(model))
    if opt.use_lora:
        model = apply_lora(model, opt)
        logger.info(utils.get_parameters(model, using_lora=True))
    model = model.cuda()

    optimizer, scheduler = utils.set_optim(opt, model)
    # if dist_utils.is_main():
    #    utils.save(model, optimizer, scheduler, global_step, 0., opt, opt.output_dir, f"step-{0}")
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = opt.dropout

    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    finetuning(opt, model, optimizer, scheduler, tokenizer, step)

    # save the final model
    if dist_utils.get_rank() == 0:
        if opt.use_lora:
            save_lora_model(model, optimizer, scheduler, opt.output_dir, opt, step)
        else:
            utils.save(
                model,
                optimizer,
                scheduler,
                step,
                opt,
                opt.output_dir,
                "final_model",
            )

if __name__ == "__main__":
    main()
