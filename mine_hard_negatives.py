import json
import random
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from sentence_transformers.util import mine_hard_negatives


def load_jsonl(file_path):
    """Reads a JSONL file and returns a dictionary."""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            data[entry['_id']] = entry  # each entry has an '_id' field
    return data

def load_tsv(file_path):
    """Reads a qrel TSV file and returns a dictionary with qid -> {doc_id: score}."""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            qid, doc_id, score = line.strip().split('\t')
            if qid not in data:
                data[qid] = {}  # initialize the dictionary
            data[qid][doc_id] = float(score)
    return data

def save_jsonl(data, file_path):
    """Saves a list of dictionaries to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def prepare_dataset_with_negatives(
    queries_file, 
    corpus_file, 
    qrels_file, 
    output_file, 
    retriever=None,
    num_negatives=5, 
    num_hard_negatives=5, 
    max_examples=None,
    batch_size=32
):
    """Prepares training data with both random and hard negatives.
    
    Args:
        queries_file: Path to a JSONL file with queries.
        corpus_file: Path to a JSONL file with the corpus.
        qrels_file: Path to a TSV file with query-document relevance labels.
        output_file: Path to save the training data.
        retriever: The SentenceTransformer model for mining hard negatives (optional).
        num_negatives: Number of random negative documents per query.
        num_hard_negatives: Number of hard negative documents per query.
        max_examples: Maximum number of training examples to generate.
        batch_size: Batch size for mining hard negatives.
    """
    queries = load_jsonl(queries_file)
    corpus = load_jsonl(corpus_file)
    qrels = load_tsv(qrels_file)

    all_doc_ids = set(corpus.keys())
    dataset_entries = []
    
    # group queries and maintain a counter for examples
    examples_count = 0
    
    # process by query to gather all positives for each query
    query_data = {}
    for qid, relevant_docs in qrels.items():
        if qid not in queries:  # skip qrels without queries
            print(f"Query {qid} not found in the queries file.")
            continue
        
        query_text = queries[qid]["text"]
        positive_doc_ids = [doc_id for doc_id in relevant_docs.keys()]
        
        if not positive_doc_ids:
            print(f"No relevant documents found for query {qid}.")
            continue # skip queries without relevant documents
            
        # get all positives for this query
        valid_positives = []
        for pos_doc_id in positive_doc_ids:
            if pos_doc_id in corpus:
                valid_positives.append(pos_doc_id)
            else:
                print(f"Document {pos_doc_id} not found in the corpus.")
                
        if not valid_positives:
            print(f"No valid positive documents found for query {qid}.")
            continue # skip if no valid positives
            
        # sample random negatives
        negative_doc_ids = random.sample(
            list(all_doc_ids - set(relevant_docs.keys())), 
            min(num_negatives, len(all_doc_ids) - len(relevant_docs)) # ensure we don't sample more negatives than available
        )
        
        query_data[qid] = {
            "question": query_text,
            "positive_doc_ids": valid_positives,
            "negative_doc_ids": negative_doc_ids
        }
        
        examples_count += 1
        if max_examples is not None and examples_count >= max_examples:
            break
    
    # prepare dataset entries with all positive examples per query
    for qid, data in query_data.items():
        
        query_text = data["question"]
        
        # create a list of all positive contexts
        positive_ctxs = []
        for pos_doc_id in data["positive_doc_ids"]:
            positive_ctxs.append({
                "title": corpus[pos_doc_id].get("title", ""),
                "text": corpus[pos_doc_id]["text"]
            })
        
        # create a list of all negative contexts
        negative_ctxs = []
        for neg_id in data["negative_doc_ids"]:
            negative_ctxs.append({
                "title": corpus[neg_id].get("title", ""),
                "text": corpus[neg_id]["text"]
            })
        
        # finally create an entry with multiple positives and negatives
        entry = {
            "question": query_text,
            "positive_ctxs": positive_ctxs,
            "negative_ctxs": negative_ctxs,
        }
        
        dataset_entries.append(entry)
    
    # if a retriever is provided, we mine for the hard negatives
    # otherwise, 'dataset_entries' will be used as is

    final_data = []
    
    if retriever is not None and num_hard_negatives > 0:
        # Prepare data for hard negative mining - we'll use the first positive for each query
        # since mine_hard_negatives typically expects a single positive per query
        combined_dataset = []
        for entry in dataset_entries:
            
            for positive_ctx in entry["positive_ctxs"]:
                combined_entry = {
                    "question": entry["question"],
                    "positive": positive_ctx.get("title", "") + " " + positive_ctx["text"]
                }
                combined_dataset.append(combined_entry)
        


        # convert to HuggingFace Dataset format
        hf_dataset = Dataset.from_list(combined_dataset)
        
        # finally mine_hard_negatives
        hard_negatives_dataset = mine_hard_negatives(
            dataset=hf_dataset,
            anchor_column_name="question",
            positive_column_name="positive",
            model=retriever,
            corpus=corpus,
            num_negatives=num_hard_negatives,
            sampling_strategy="top",
            as_triplets=False,
            batch_size=batch_size
        )

        # save_jsonl(hard_negatives_dataset, "hard_negatives.jsonl")
        
        # Add hard negatives to original entries
        hard_negatives_map = defaultdict(set)  # query_text -> hard negatives

        for entry in hard_negatives_dataset:
            query_text = entry["question"]
            # extract all the negative_X for the query
            hard_negatives = [entry[f"negative_{i+1}"] for i in range(num_hard_negatives)]
            hard_negatives_map[query_text].update(hard_negatives)  # add the hard negatives to the set

        # add hard negatives to the original entries
        for original_entry in dataset_entries:
            query_text = original_entry["question"]
            
            # if there are hard negatives for this query, add them to the entry (if they are not positives)
            if query_text in hard_negatives_map:
                hard_negative_ctxs = [
                    {"title": "", "text": neg} 
                    for neg in hard_negatives_map[query_text] 
                    if neg not in [pos.get("title", "") + " " + pos["text"] for pos in original_entry["positive_ctxs"]]
                ]
                original_entry["hard_negative_ctxs"] = hard_negative_ctxs
            
            final_data.append(original_entry)
    else:
        # If no retriever, just use the entries with random negatives
        final_data = dataset_entries
    
    save_jsonl(final_data, output_file)
    print(f"Saved {len(final_data)} examples to {output_file}.")
    

if __name__ == "__main__":
    # Load the SentenceTransformer model -> https://huggingface.co/nishimoto/contriever-sentencetransformer
    model = SentenceTransformer("nishimoto/contriever-sentencetransformer")

    dataset_name = "nfcorpus"
    
    # training data with negatives and hard negatives
    prepare_dataset_with_negatives(
        queries_file=f"{dataset_name}/queries.jsonl",
        corpus_file=f"{dataset_name}/corpus.jsonl",
        qrels_file=f"{dataset_name}/qrels/train.tsv",
        output_file=f"{dataset_name}/training_data.jsonl",
        retriever=model,
        num_negatives=20,
        num_hard_negatives=20,
        max_examples=None,
        batch_size=32
    )
    
    # test data
    prepare_dataset_with_negatives(
        queries_file=f"{dataset_name}/queries.jsonl",
        corpus_file=f"{dataset_name}/corpus.jsonl",
        qrels_file=f"{dataset_name}/qrels/test.tsv",
        output_file=f"{dataset_name}/test_data.jsonl",
        retriever=None,  # no hard negatives for test data
        num_negatives=20,
        num_hard_negatives=0,
        max_examples=None
    )