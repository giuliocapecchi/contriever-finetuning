import numpy as np
import torch
from sentence_transformers import SentenceTransformer, SimilarityFunction
from tqdm import tqdm
from typing import Any, Dict, List, Literal


def mine_hard_negatives(
    queries: Dict[str, Any], 
    corpus: Dict[str, Any], 
    qrels: Dict[str, Dict[str, float]], 
    model: SentenceTransformer = None,
    range_max: int = 50, 
    relative_margin: float = None,
    positive_score_to_use: Literal["min", "max", "mean"] = "mean",
    num_hard_negatives: int = 10,
    use_faiss: bool = True,
    batch_size: int = 64,
    use_multi_process: bool = False,
    target_devices: list[str] = None,
    verbose = True,
    include_docids_and_scores: bool = False
) -> List[Dict[str, Any]]:
    """
    Mines hard negative samples for a given set of queries and a corpus using a SentenceTransformer model.
    
    Args:
        queries (Dict[str, Any]): A dictionary where keys are query IDs and values are dictionaries containing query data (e.g., text).
        corpus (Dict[str, Any]): A dictionary where keys are document IDs and values are dictionaries containing document data (e.g., title, text).
        qrels (Dict[str, Dict[str, float]]): A dictionary where keys are query IDs and values are dictionaries of relevant document IDs with their relevance scores.
        model (SentenceTransformer): A SentenceTransformer model used for embedding queries and corpus documents.
        range_max (int, optional): The maximum number of documents to consider for similarity search. Defaults to 50.
        relative_margin (float, optional): Defines a margin to exclude candidates that are too similar to the positive document. For example, a relative_margin of 0.1 ensures that hard negatives are at most 90% as similar to the query as the positive document. Defaults to None.
        positive_score_to_use (Literal["min", "max", "mean"], optional): Specifies which positive score to use for each query. Defaults to "mean".
        num_hard_negatives (int, optional): The number of hard negatives to mine per query. Defaults to 10.
        use_faiss (bool, optional): Whether to use FAISS for efficient similarity search. Defaults to True. Recommended for large datasets.
        batch_size (int, optional): Batch size for encoding queries and corpus documents. Defaults to 64.
        use_multi_process (bool, optional): Whether to use multi-process encoding for faster computation. Defaults to False.
        target_devices (list[str], optional): List of target devices for multi-process encoding. Defaults to None.
        verbose (bool, optional): Whether to print detailed logs and statistics. Defaults to True.
        include_docids_and_scores (bool, optional): Whether to include document IDs and similarity scores in the output. Defaults to False.
    
    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing:
            - 'qid': Query ID (if include_docids_and_scores is True).
            - 'question': Query text.
            - 'positive_ctxs': List of positive contexts (relevant documents).
            - 'hard_negative_ctxs': List of hard negative contexts (irrelevant but similar documents).
    
    Raises:
        ValueError: If `range_max` is less than `num_hard_negatives`.
        ValueError: If there is a mismatch between the number of query embeddings and the number of query IDs.
    
    Notes:
        - Hard negatives are documents that are not relevant to the query but are highly similar based on the model's embeddings.
        - The function excludes positives and optionally excludes negatives that are too similar to the first positive based on the `relative_margin`.
        - FAISS is used for efficient similarity search when enabled, with GPU support if available.
        - Multi-process encoding can be used for faster embedding computation, but requires proper configuration of `target_devices`.
    """

    if range_max < num_hard_negatives:
        raise ValueError("range_max must be greater than num_hard_negatives.")

    # remove the queries that don't have a qrel associated
    relevant_queries = {qid: q for qid, q in queries.items() if qid in qrels.keys()}
    positives_per_query = [len(qrels[qid]) for qid in qrels.keys()]

    # prepare ordered list of queries and corpus texts
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[doc_id]["title"] + " " + corpus[doc_id]["text"] for doc_id in corpus_ids]
    query_ids = list(relevant_queries.keys())
    query_texts = [queries[query_id]["text"] for query_id in query_ids]

    if verbose: # print infos
        print(f"Using device: {model.device}")
        print(f"First corpus document -> docid = {corpus_ids[0]} : {corpus_texts[0][:100]}...")
        print(f"First query -> {query_ids[0]} : {query_texts[0]}")
        print(f"Removed {len(queries) - len(relevant_queries)} queries without qrels. Remaining queries: {len(relevant_queries)}")
        avg_positives_per_query = np.mean(positives_per_query)
        print(f"Found an average of {avg_positives_per_query:.3f} positives per query.")        
        print(f"Found {len(corpus)} documents in the corpus.")
    
    # conversion maps between doc_id and index. will be used since faiss returns indices
    corpus_id_to_index = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    index_to_corpus_id = {idx: doc_id for doc_id, idx in corpus_id_to_index.items()} # simply inverse the above mapping
    
    max_positives = max(positives_per_query)
    model.similarity_fn_name = SimilarityFunction.DOT_PRODUCT

    # embed the corpus and the queries
    if use_multi_process: # from SentenceTransformer documentation: first create a pool, then embed documents, finally model.stop_multi_process_pool(pool)
        print("Using multi-process for encoding.")
        pool = model.start_multi_process_pool(
            target_devices=target_devices,
        )
        query_embeddings = model.encode_multi_process(
            query_texts, pool, batch_size=batch_size, normalize_embeddings=False, show_progress_bar=True
        )
        corpus_embeddings = model.encode_multi_process(
            corpus_texts, pool, batch_size=batch_size, normalize_embeddings=False, show_progress_bar=True
        )
        model.stop_multi_process_pool(pool)


    else:
        print("Using single process for encoding.")
        query_embeddings = model.encode(
            query_texts,
            batch_size=batch_size,
            normalize_embeddings=False,
            convert_to_numpy=True,
            show_progress_bar=True,
            device=model.device
        )
        corpus_embeddings = model.encode(
            corpus_texts,
            batch_size=batch_size,
            normalize_embeddings=False,
            convert_to_numpy=True,
            show_progress_bar=True,
            device=model.device
        )

        query_embeddings = torch.from_numpy(query_embeddings)
        corpus_embeddings = torch.from_numpy(corpus_embeddings)

    if relative_margin:  # check for the ["min", "max", "mean"] positive score for each query
        positive_scores = {}
        for query_id in tqdm(query_ids, desc=f"Calculating ({positive_score_to_use}) positive scores for each query...", unit="query"):
            positive_docids = set(qrels.get(query_id, {}).keys())
            scores = []
            for doc_id in positive_docids:
                if doc_id not in corpus_id_to_index:
                    if verbose:
                        print(f"Warning: doc_id {doc_id} not found in corpus_id_to_index. Skipping.")
                    continue
                score = model.similarity(
                    query_embeddings[query_ids.index(query_id)],
                    corpus_embeddings[corpus_id_to_index[doc_id]]
                )

                scores.append(score.item())
            
            if not scores:
                if verbose:
                    print(f"Warning: No positive scores found for query_id {query_id}. Skipping.")
                continue

            if positive_score_to_use == "min":
                selected_score = min(scores)
            elif positive_score_to_use == "max":
                selected_score = max(scores)
            elif positive_score_to_use == "mean":
                selected_score = np.mean(scores)
            else:
                raise ValueError(f"Invalid value for positive_score_to_use: {positive_score_to_use}. Must be 'min', 'max', or 'mean'.")

            positive_scores[query_id] = selected_score
       

    effective_range_max = range_max + max_positives

    if effective_range_max > 2048 and use_faiss:
        effective_range_max = 2048 # faiss gpu can only retrieve up to 2048 documents per query
        if verbose:
            print("Using FAISS, we can only retrieve up to 2048 documents per query. Setting range_max to 2048.")
    
    k = min(effective_range_max, len(corpus_embeddings))
    if verbose:
        print(f"Searching for {k} most similar documents per query. This is given by the range_max ({range_max}) parameter + the maximum number of positives per query ({max_positives}).")

    if use_faiss:
        import faiss
        
        index = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())
        
        # use gpu if available
        try:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            index = faiss.index_cpu_to_all_gpus(index, co=co)
            print("Using FAISS GPU.")
        except Exception:
            print("FAISS GPU not available, using CPU.")
        
        index.add(corpus_embeddings)
        
        query_batch_size = 512
        all_scores = []
        all_indices = []

        for start in tqdm(range(0, len(query_embeddings), query_batch_size), desc="FAISS search"):
            end = start + query_batch_size
            batch_queries = query_embeddings[start:end]
            batch_scores, batch_indices = index.search(batch_queries, k=k)
            all_scores.append(batch_scores)
            all_indices.append(batch_indices)

        # chain results
        scores = np.vstack(all_scores)
        indices = np.vstack(all_indices)
        scores = torch.from_numpy(scores).to('cpu')
        indices = torch.from_numpy(indices).to('cpu')
    
    else:
        # similarity scores without faiss
        scores = model.similarity(
            torch.from_numpy(query_embeddings), 
            torch.from_numpy(corpus_embeddings)
        ).to('cpu')
        
        k = min(k, scores.size(1))
        scores, indices = torch.topk(scores, k, dim=1)
    
    del query_embeddings
    del corpus_embeddings
    
    indices = indices.cpu().numpy()
    scores = scores.cpu().numpy()

    final_queries = {}
    num_queries_with_hard_negatives = 0
    
     # ensure indices and query_embeddings have matching dimensions
    if indices.shape[0] != len(query_ids):
        raise ValueError(f"Mismatch between indices rows ({indices.shape[0]}) and query_ids ({len(query_ids)}).")
    
    skipped_because_of_relative_margin = 0
    actual_hard_negatives = 0

    for query_idx, query_id in enumerate(tqdm(query_ids, desc="Processing queries", unit="query")):
        positive_docids = set(qrels.get(query_id, {}).keys())
        # print("Positive docids for this query: ", positive_docids)
        
        margin_positive_score = positive_scores[query_id] if relative_margin else None
        
        query_hard_negatives = []
        for score_idx, indice in enumerate(indices[query_idx]):
            
            # convert from faiss index to corpus id
            doc_id = index_to_corpus_id[indice]

            # exclude positives 
            if doc_id not in positive_docids and len(query_hard_negatives) < num_hard_negatives:

                # exclude based on margin
                if relative_margin is not None and relative_margin > 0.0 and scores[query_idx][score_idx].item() >= ((1-relative_margin) * margin_positive_score):
                    skipped_because_of_relative_margin += 1
                    continue
                

                # check if the document is already in query_hard_negatives (might happen if the corpus has duplicates)
                if corpus[doc_id]['title'] in [d['title'] for d in query_hard_negatives]:
                    if verbose:
                        print(f"Warning: Document {doc_id} already in query_hard_negatives. Skipping.")
                    continue

                # include the document in the hard_negatives
                hard_negative = {}
                if include_docids_and_scores:
                    hard_negative['docid'] = doc_id
                    hard_negative['simscore'] = round(scores[query_idx][score_idx].item(), 3)
                hard_negative.update({
                    'title': corpus[doc_id]['title'],
                    'text': corpus[doc_id]['text']
                })
                actual_hard_negatives += 1
                query_hard_negatives.append(hard_negative)
        
        positive_contexts = []
        for doc_id in positive_docids:
            if doc_id not in corpus:
                if verbose:
                    print(f"Warning: positive doc_id {doc_id} not found in corpus. Skipping.")
                continue
            positive_context = {}
            if include_docids_and_scores:
                positive_context['docid'] = doc_id
                positive_context['maxpos_score'] = round(positive_scores[query_id], 3)
            positive_context.update({
                'title': corpus[doc_id]['title'], 
                'text': corpus[doc_id]['text']
                })
            positive_contexts.append(positive_context)
        
        entry = {}
        if include_docids_and_scores:
            entry['qid'] = query_id
        entry.update({
            'question': queries[query_id]["text"],
            'positive_ctxs': positive_contexts
        })
        if query_hard_negatives:
            entry['hard_negative_ctxs'] = query_hard_negatives
            num_queries_with_hard_negatives += 1

        final_queries[query_id] = entry

    if verbose:
        print(f"Found {num_queries_with_hard_negatives} queries with hard negatives. This is the {num_queries_with_hard_negatives / len(relevant_queries) * 100:.2f}% of the queries.")
        print(f"Found {actual_hard_negatives} hard negatives, so there's a mean of {actual_hard_negatives / len(final_queries):.2f} hard negatives per query.")
        if relative_margin is not None:
            print(f"Excluding {skipped_because_of_relative_margin} potential negatives based on relative margin ({positive_score_to_use}).")

    return final_queries
