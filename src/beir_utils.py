# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
import pytrec_eval
import torch
import torch.distributed as dist
import csv
import logging

import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch

from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank

import src.dist_utils as dist_utils
from src import normalize_text
import glob


logger = logging.getLogger(__name__)


class DenseEncoderModel:
    def __init__(
        self,
        query_encoder,
        doc_encoder=None,
        tokenizer=None,
        max_length=512,
        add_special_tokens=True,
        norm_query=False,
        norm_doc=False,
        lower_case=False,
        normalize_text=False,
        **kwargs,
    ):
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.lower_case = lower_case
        self.normalize_text = normalize_text

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:

        if dist.is_initialized():
            idx = np.array_split(range(len(queries)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(queries))

        queries = [queries[i] for i in idx]
        if self.normalize_text:
            queries = [normalize_text.normalize(q) for q in queries]
        if self.lower_case:
            queries = [q.lower() for q in queries]

        allemb = []
        nbatch = (len(queries) - 1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(queries))

                qencode = self.tokenizer.batch_encode_plus(
                    queries[start_idx:end_idx],
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                )
                qencode = {key: value.cuda() for key, value in qencode.items()}
                emb = self.query_encoder(**qencode, normalize=self.norm_query)
                allemb.append(emb.cpu())

        allemb = torch.cat(allemb, dim=0)
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):

        if dist.is_initialized():
            idx = np.array_split(range(len(corpus)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(corpus))
        corpus = [corpus[i] for i in idx]
        corpus = [c["title"] + " " + c["text"] if len(c["title"]) > 0 else c["text"] for c in corpus]
        if self.normalize_text:
            corpus = [normalize_text.normalize(c) for c in corpus]
        if self.lower_case:
            corpus = [c.lower() for c in corpus]

        allemb = []
        nbatch = (len(corpus) - 1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(corpus))

                cencode = self.tokenizer.batch_encode_plus(
                    corpus[start_idx:end_idx],
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                )
                cencode = {key: value.cuda() for key, value in cencode.items()}
                emb = self.doc_encoder(**cencode, normalize=self.norm_doc)
                allemb.append(emb.cpu())

        allemb = torch.cat(allemb, dim=0)
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb
    
def aggregate_results(
        scores: Dict[str, Dict[str, float]],
        k_values: List[int],
        ) ->  Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Aggregate the queries results into a single dictionary.
    """
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0
    
    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id].get("ndcg_cut_" + str(k), 0.0)
            _map[f"MAP@{k}"] += scores[query_id].get("map_cut_" + str(k), 0.0)
            recall[f"Recall@{k}"] += scores[query_id].get("recall_" + str(k), 0.0)
            precision[f"P@{k}"] += scores[query_id].get("P_" + str(k), 0.0)
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)

    for eval in [ndcg, _map, recall, precision]:
        logging.info("\n")
        for k in eval.keys():
            logging.info("{}: {:.4f}".format(k, eval[k]))

    return ndcg, _map, recall, precision


def evaluate_model(
    query_encoder,
    doc_encoder,
    tokenizer,
    dataset,
    use_minicorpus=False,
    batch_size=128,
    add_special_tokens=True,
    norm_query=False,
    norm_doc=False,
    is_main=True,
    split="test",
    score_function="dot",
    beir_dir="BEIR/datasets",
    save_results_path=None,
    lower_case=False,
    normalize_text=False,
    save_perquery_scores=False,
    prefix_type=None,
    use_reranker=False,
    reranker_model_name="BAAI/bge-reranker-base",
):

    metrics = defaultdict(list)  # store final results

    if hasattr(query_encoder, "module"):
        query_encoder = query_encoder.module
    query_encoder.eval()

    if doc_encoder is not None:
        if hasattr(doc_encoder, "module"):
            doc_encoder = doc_encoder.module
        doc_encoder.eval()
    else:
        doc_encoder = query_encoder

    dmodel = DenseRetrievalExactSearch(
        DenseEncoderModel(
            query_encoder=query_encoder,
            doc_encoder=doc_encoder,
            tokenizer=tokenizer,
            add_special_tokens=add_special_tokens,
            norm_query=norm_query,
            norm_doc=norm_doc,
            lower_case=lower_case,
            normalize_text=normalize_text,
        ),
        batch_size=batch_size,
    )
    retriever = EvaluateRetrieval(dmodel, score_function=score_function)
    data_path = os.path.join(beir_dir, dataset)

    if not os.path.isdir(data_path) and is_main:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        data_path = beir.util.download_and_unzip(url, beir_dir)
    dist_utils.barrier()

    if not dataset == "cqadupstack":
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path, corpus_file="minicorpus.jsonl" if use_minicorpus else "corpus.jsonl").load(split=split)
        
        if prefix_type == 'query_or_passage': # necessary for the E5 model family 
            # prepend 'query:' to queries and 'passage:' to corpus
            queries = {k: f"query: {v}" for k, v in queries.items()}
            for _, doc in corpus.items():
                doc['title'] = f"passage: {doc['title']}" if doc.get('title', '') else 'passage:'
        elif prefix_type is not None:
            raise ValueError(f"Unsupported prefix type: {prefix_type}. Supported types are None and 'query_or_passage'.")


        map_string = "map_cut." + ",".join([str(k) for k in retriever.k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in retriever.k_values])
        recall_string = "recall." + ",".join([str(k) for k in retriever.k_values])
        precision_string = "P." + ",".join([str(k) for k in retriever.k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        
        results = retriever.retrieve(corpus, queries)

        if save_perquery_scores and save_results_path is not None and is_main:
            scores = evaluator.evaluate(results) # these are the scores query by query
            ndcg, _map, recall, precision = aggregate_results(scores, retriever.k_values)
            
            with open(os.path.join(save_results_path, "perquery_scores.csv"), "w") as f:
                writer = csv.writer(f)
                header = ["query_id"] + list(next(iter(scores.values())).keys())
                writer.writerow(header)
                for qid, values in scores.items():
                    row = [qid] + [values[k] for k in header[1:]]
                    writer.writerow(row)
            
            if save_results_path is not None:
                with open(f"{save_results_path}/metrics.txt", "w") as f:
                    for metric_dict in [ndcg, _map, recall, precision]:
                        for key, value in metric_dict.items():
                            f.write(f"{key}: {value}\n")

        elif is_main:
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
            for metric in (ndcg, _map, recall, precision): # "mrr", "recall_cap", "hole"
                if isinstance(metric, str):
                    metric = retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
                for key, value in metric.items():
                    metrics[key].append(value)

        if use_reranker and reranker_model_name is not None and is_main :
            logger.info(f"Reranking results with {reranker_model_name}...")
            cross_encoder = CrossEncoder(reranker_model_name)
            reranker = Rerank(cross_encoder)
            logger.info(retriever.k_values)
            rerank_results = reranker.rerank(corpus, queries, results, top_k=max(retriever.k_values))
            
            rerank_scores = evaluator.evaluate(rerank_results) # these are the scores query by query
            ndcg, _map, recall, precision = aggregate_results(rerank_scores, retriever.k_values)

            if save_results_path is not None: # save rerank results
                os.makedirs(f"{save_results_path}/rerank_results", exist_ok=True)
                with open(os.path.join(save_results_path, "rerank_results", "rerank_metrics.txt"), "w") as f:
                    for metric_dict in [ndcg, _map, recall, precision]:
                        for key, value in metric_dict.items():
                            f.write(f"{key}: {value}\n")
                if save_perquery_scores:
                    with open(os.path.join(os.path.join(save_results_path, "rerank_results", "perquery_scores.csv")), "w") as f:
                        writer = csv.writer(f)
                        header = ["query_id"] + list(next(iter(rerank_scores.values())).keys())
                        writer.writerow(header)
                        for qid, values in rerank_scores.items():
                            row = [qid] + [values[k] for k in header[1:]]
                            writer.writerow(row)

    elif dataset == "cqadupstack":  # compute macroaverage over datasets
        paths = glob.glob(data_path)
        for path in paths:
            corpus, queries, qrels = GenericDataLoader(data_folder=path).load(split=split)
            results = retriever.retrieve(corpus, queries)
            if is_main:
                ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
                for metric in (ndcg, _map, recall, precision): # "mrr", "recall_cap", "hole"
                    if isinstance(metric, str):
                        metric = retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
                    for key, value in metric.items():
                        metrics[key].append(value)
        for key, value in metrics.items():
            assert (
                len(value) == 12
            ), f"cqadupstack includes 12 datasets, only {len(value)} values were compute for the {key} metric"

    metrics = {key: 100 * np.mean(value) for key, value in metrics.items()}

    return metrics
