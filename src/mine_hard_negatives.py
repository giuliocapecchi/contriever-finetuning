from sentence_transformers import SentenceTransformer
import torch
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm


def mine_hard_negatives(
    queries: Dict[str, Any], 
    corpus: Dict[str, Any], 
    qrels: Dict[str, Dict[str, float]], 
    model: SentenceTransformer = None,
    range_max: int = 50, 
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
        queries (Dict[str, Any]): A dictionary where keys are query IDs and values are query texts.
        corpus (Dict[str, Any]): A dictionary where keys are document IDs and values are document data (e.g., text, title).
        qrels (Dict[str, Dict[str, float]]): A dictionary where keys are query IDs and values are dictionaries of relevant document IDs with their relevance scores.
        model (SentenceTransformer): A SentenceTransformer model used for embedding queries and corpus documents.
        range_max (int, optional): The maximum number of documents to consider for similarity search. Defaults to 50.
        num_hard_negatives (int, optional): The number of hard negatives to mine per query. Defaults to 10.
        use_faiss (bool, optional): Whether to use FAISS for efficient similarity search. Defaults to True. It's recommended to use for large datasets.
        batch_size (int, optional): Batch size for encoding queries and corpus documents. Defaults to 64.
        use_multi_process (bool, optional): Whether to use multi-process encoding for faster computation. Defaults to False.
        target_devices (list[str], optional): List of target devices for multi-process encoding. Defaults to None.
        verbose (bool, optional): Whether to print detailed logs and statistics. Defaults to True.
        include_docids_and_scores (bool, optional): Whether to include document IDs and similarity scores in the output. Adding them will increase the size of the output. Defaults to False. 
    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing:
            - 'qid': Query ID.
            - 'question': Query text.
            - 'positive_ctxs': List of positive contexts (relevant documents).
            - 'hard_negative_ctxs': List of hard negative contexts (irrelevant but similar documents).
    Raises:
        ValueError: If `range_max` is less than `num_hard_negatives`.
        ValueError: If there is a mismatch between the number of query embeddings and the number of query IDs.
    Notes:
        - Hard negatives are documents that are not relevant to the query but are highly similar based on the model's embeddings.
        - They can be ranked above relevant documents or below, the function will just return the top-k exluding positives.
    """

    if range_max < num_hard_negatives:
        raise ValueError("range_max must be greater than num_hard_negatives.")

    # remove the queries that don't have a qrel associated
    relevant_queries = {qid: q for qid, q in queries.items() if qid in qrels.keys()}
    positives_per_query = [len(qrels[qid]) for qid in qrels.keys()]

    if verbose: # print collection statistics
        print("First query -> \"", list(queries.keys())[0], ":",list(queries.values())[0]["text"],"\"")
        print(f"Removed {len(queries) - len(relevant_queries)} queries without qrels. Remaining queries: {len(relevant_queries)}")
        avg_positives_per_query = np.mean(positives_per_query)
        print(f"Found an average of {avg_positives_per_query:.3f} positives per query.")        
        print(f"Found {len(corpus)} documents in the corpus.")

    # prepare ordered list of queries and corpus texts
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]
    query_ids = list(relevant_queries.keys())
    query_texts = [queries[query_id] for query_id in query_ids]
    
    # conversion maps between doc_id and index. will be used since faiss returns indices
    corpus_id_to_index = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    index_to_corpus_id = {idx: doc_id for doc_id, idx in corpus_id_to_index.items()} # simply inverse the above mapping
    
    max_positives = max(positives_per_query)


    # embed the corpus and the queries
    if use_multi_process: # from SentenceTransformer documentation: first create a pool, then embed documents, finally model.stop_multi_process_pool(pool)
        print("Using multi-process for encoding.")
        pool = model.start_multi_process_pool(
            target_devices=target_devices,
        )
        query_embeddings = model.encode_multi_process(
            query_texts, pool, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
        )
        corpus_embeddings = model.encode_multi_process(
            corpus_texts, pool, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
        )
        model.stop_multi_process_pool(pool)


    else:
        print("Using single process for encoding.")
        query_embeddings = model.encode(
            query_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        corpus_embeddings = model.encode(
            corpus_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        

    k = min(range_max + max_positives, len(corpus_embeddings))
    if verbose:
        print(f"Searching for {k} most similar documents per query. This is given by the range_max ({range_max}) parameter + the maximum number of positives per query ({max_positives}).")

    if use_faiss:
        import faiss
        
        index = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())
        
        # use gpu if available
        try:
            print("Using FAISS GPU.")
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            index = faiss.index_cpu_to_all_gpus(index, co=co)
        except Exception:
            print("FAISS GPU not available, using CPU.")
        
        index.add(corpus_embeddings)
        
        scores, indices = index.search(query_embeddings, k=k)
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

    queries_with_hard_negatives = {}
    
     # ensure indices and query_embeddings have matching dimensions
    if indices.shape[0] != len(query_ids):
        raise ValueError(f"Mismatch between indices rows ({indices.shape[0]}) and query_ids ({len(query_ids)}).")
    
    for query_idx, query_id in enumerate(tqdm(query_ids, desc="Processing queries", unit="query")):
        positive_docids = set(qrels.get(query_id, {}).keys())
        # print("Positive docids for this query: ", positive_docids)
        query_hard_negatives = []
        for score_idx, indice in enumerate(indices[query_idx]):
            
            # convert from faiss index to corpus id
            doc_id = index_to_corpus_id[indice]

            # exclude positives 
            if doc_id not in positive_docids and len(query_hard_negatives) < num_hard_negatives:
                
                hard_negative = {}
                if include_docids_and_scores:
                    hard_negative['docid'] = doc_id
                    hard_negative['simscore'] = round(scores[query_idx][score_idx].item(), 3)
                hard_negative.update({
                    'title': corpus[doc_id]['title'],
                    'text': corpus[doc_id]['text']
                })
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
            'positive_ctxs': positive_contexts,
            'hard_negative_ctxs': query_hard_negatives
        })

        queries_with_hard_negatives[query_id] = entry
    
    return queries_with_hard_negatives
