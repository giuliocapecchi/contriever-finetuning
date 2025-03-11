import json
import random

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
        next(f) # skip header
        for line in f:
            qid, doc_id, score = line.strip().split('\t')
            if qid not in data:
                data[qid] = {} # initialize the dictionary
            data[qid][doc_id] = float(score)
    return data

def save_jsonl(data, file_path):
    """Saves a list of dictionaries to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def prepare_data(queries_file, corpus_file, qrels_file, output_file, num_negatives=5, max_examples=None):
    """Prepares training data for contrastive learning.
    
    Args:
        queries_file: Path to a JSONL file with queries.
        corpus_file: Path to a JSONL file with the corpus.
        qrels_file: Path to a TSV file with query-document relevance labels.
        output_file: Path to save the training data.
        num_negatives: Number of negative documents to sample per positive document.
        max_examples: Maximum number of training examples to generate.
    """
    queries = load_jsonl(queries_file)
    corpus = load_jsonl(corpus_file)
    qrels = load_tsv(qrels_file)
    
    training_data = []
    all_doc_ids = set(corpus.keys())

    for qid, relevant_docs in qrels.items():
        if qid not in queries: # skip qrels without queries
            continue
        
        query_text = queries[qid]["text"]
        positive_doc_ids = [doc_id for doc_id, _ in relevant_docs.items()]

        if not positive_doc_ids:
            continue

        for pos_doc_id in positive_doc_ids:
            if pos_doc_id not in corpus:
                continue
            
            negative_doc_ids = random.sample(list(all_doc_ids - set(relevant_docs.keys())), min(num_negatives, len(all_doc_ids) - len(relevant_docs))) # with the min we ensure that we don't sample more negatives than available

            example = {
                "question": query_text,
                "positive_ctxs": [{"title": corpus[pos_doc_id].get("title", ""), "text": corpus[pos_doc_id]["text"]}],
                "negative_ctxs": [{"title": corpus[neg_id].get("title", ""), "text": corpus[neg_id]["text"]} for neg_id in negative_doc_ids],
                # "hard_negative_ctxs": [{"text": ""}], 
                "title": corpus[pos_doc_id].get("title", ""),
                "text": corpus[pos_doc_id]["text"] # can be probably removed
            }
            training_data.append(example)

            if max_examples is not None and len(training_data) >= max_examples:
                break
        if max_examples is not None and len(training_data) >= max_examples:
            break

    save_jsonl(training_data, output_file)
    print(f"Saved {len(training_data)} examples to {output_file}.")


# usage on NFcorpus
dataset_name = 'nfcorpus'

prepare_data(
    queries_file=f"{dataset_name}/queries.jsonl",
    corpus_file=f"{dataset_name}/corpus.jsonl",
    qrels_file=f"{dataset_name}/qrels/train.tsv",
    output_file=f"{dataset_name}/training_data.jsonl",
    num_negatives=5,
    max_examples=5000  # Limit the number of examples
)

prepare_data(
    queries_file=f"{dataset_name}/queries.jsonl",
    corpus_file=f"{dataset_name}/corpus.jsonl",
    qrels_file=f"{dataset_name}/qrels/test.tsv",
    output_file=f"{dataset_name}/test_data.jsonl",
    num_negatives=5,
    max_examples=2500  # Limit the number of examples
)
