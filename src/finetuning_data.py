# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import random
import json
from src import normalize_text


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datapaths,
        negative_ctxs=1,
        negative_hard_ratio=0.0,
        negative_hard_min_idx=0,
        training=False,
        prefix_type=None,
        global_rank=-1,
        world_size=-1,
        maxload=None,
        normalize=True,
    ):
        self.negative_ctxs = negative_ctxs
        self.negative_hard_ratio = negative_hard_ratio
        self.negative_hard_min_idx = negative_hard_min_idx
        self.training = training
        self.prefix_type = prefix_type
        self.normalize_fn = normalize_text.normalize if normalize and normalize_text else lambda x: x
        self._load_data(datapaths, global_rank, world_size, maxload)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = example["question"]
        if self.training:
            gold = random.choice(example["positive_ctxs"])

            n_hard_negatives, n_random_negatives = self.sample_n_hard_negatives(example)
            negatives = []
            if n_random_negatives > 0: # if there are negative examples, sample them
                random_negatives = random.sample(example["negative_ctxs"], n_random_negatives)
                negatives += random_negatives
            if n_hard_negatives > 0: # if there are hard negative examples, sample them
                hard_negatives = random.sample(
                    example["hard_negative_ctxs"][self.negative_hard_min_idx :], n_hard_negatives
                )
                negatives += hard_negatives
        else:
            gold = example["positive_ctxs"][0]
            nidx = 0
            if "negative_ctxs" in example:
                negatives = [example["negative_ctxs"][nidx]]
            else:
                negatives = []

        # gold is composed by title and text
        gold = gold["title"] + " " + gold["text"] if "title" in gold and len(gold["title"]) > 0 else gold["text"]
        
        # the same is true for negatives
        negatives = [
            n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] for n in negatives
        ]

        if self.prefix_type == 'query_or_passage':  # for e5 models, add prefixes "query: " and "passage: " to input texts to avoid performance degradation
            question = "query: " + question
            gold = "passage: " + gold
            negatives = ["passage: " + n for n in negatives]
        elif self.prefix_type is not None:
            raise ValueError(f"Unsupported prefix type: {self.prefix_type}. Supported types are None and 'query_or_passage'.")


        example = {
            "query": self.normalize_fn(question),
            "gold": self.normalize_fn(gold),
            "negatives": [self.normalize_fn(n) for n in negatives],
        }
        return example

    def _load_data(self, datapaths, global_rank, world_size, maxload):
        counter = 0
        self.data = []
        for path in datapaths:
            path = str(path)
            if path.endswith(".jsonl"):
                file_data, counter = self._load_data_jsonl(path, global_rank, world_size, counter, maxload)
            elif path.endswith(".json"):
                file_data, counter = self._load_data_json(path, global_rank, world_size, counter, maxload)
            self.data.extend(file_data)
            if maxload is not None and maxload > 0 and counter >= maxload:
                break

    def _load_data_json(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            data = json.load(fin)
        for example in data:
            counter += 1
            if global_rank > -1 and not counter % world_size == global_rank:
                continue
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break

        return examples, counter

    def _load_data_jsonl(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            for line in fin:
                counter += 1
                if global_rank > -1 and not counter % world_size == global_rank:
                    continue
                example = json.loads(line)
                examples.append(example)
                if maxload is not None and maxload > 0 and counter == maxload:
                    break

        return examples, counter

    def sample_n_hard_negatives(self, ex):

        if "hard_negative_ctxs" in ex:
            n_hard_negatives = sum([random.random() < self.negative_hard_ratio for _ in range(self.negative_ctxs)])
            n_hard_negatives = min(n_hard_negatives, len(ex["hard_negative_ctxs"][self.negative_hard_min_idx :]))
        else:
            n_hard_negatives = 0
        n_random_negatives = self.negative_ctxs - n_hard_negatives # this will be self.negative_ctxs if there are no hard negatives
        if "negative_ctxs" in ex:
            n_random_negatives = min(n_random_negatives, len(ex["negative_ctxs"]))
        else:
            n_random_negatives = 0
        return n_hard_negatives, n_random_negatives


class Collator(object):
    """
    A collator class that processes and tokenizes batches of query and passage data for fine-tuning.
    Attributes:
        tokenizer: A tokenizer object used to encode the text data.
        passage_maxlength: An integer representing the maximum length of the passages to be tokenized.
    Methods:
        __call__(batch):
            Processes a batch of data, tokenizes the queries and passages, and returns a dictionary
            containing tokenized queries, passages, gold passages, and negative passages along with their
            corresponding attention masks.
    """
    def __init__(self, tokenizer, passage_maxlength=200):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength

    def __call__(self, batch):
        queries = [ex["query"] for ex in batch]
        golds = [ex["gold"] for ex in batch]
        negs = [item for ex in batch for item in ex["negatives"]]
        allpassages = golds + negs

        qout = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        kout = self.tokenizer.batch_encode_plus(
            allpassages,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].bool()
        k_tokens, k_mask = kout["input_ids"], kout["attention_mask"].bool()

        num_golds = len(golds)
        num_negs = len(negs)

        batch = {
            "q_tokens": q_tokens,
            "q_mask": q_mask,
            "k_tokens": k_tokens,
            "k_mask": k_mask,
            "num_golds": num_golds,
            "num_negs": num_negs,
        }

        return batch
