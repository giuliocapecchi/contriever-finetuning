import json
import random
from sentence_transformers import SentenceTransformer
import os
import beir.util
import argparse
from tqdm import tqdm
from src.mine_hard_negatives import mine_hard_negatives


def load_jsonl(file_path):
    """Reads a JSONL file and returns a dictionary."""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            entry_id = entry.pop('_id') # each entry has an '_id' field
            data[entry_id] = entry        
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
        for entry in data.values():
            f.write(json.dumps(entry) + '\n')



def create_negative_corpus(qrels, corpus):
    "Precompute the negative corpus (all documents - relevant documents)"
    
    negative_corpus = set(corpus.keys())
    
    for _, relevant_docs in tqdm(qrels.items(), desc="Creating negative corpus", unit="queries", total=len(qrels)):
        # remove the positive document IDs for this query from the corpus
        positive_doc_ids = set(relevant_docs.keys())
        negative_corpus -= positive_doc_ids
    
    print(f"Initial corpus size: {len(corpus)}\nNegative corpus size: {len(negative_corpus)}")
    return list(negative_corpus)


def prepare_dataset_with_negatives(
    dataset_name: str,
    output_file: str, 
    target: str = "train",
    data_path: str = "./beir_datasets",
    num_negatives: int = 5, 
    include_docids_and_scores: bool = False,
    # hard negatives parameters
    model: SentenceTransformer = None,
    num_hard_negatives: int = None,
    use_faiss: bool = True,
    use_multi_process: bool = False,
    range_max: int = 30,               
    batch_size: int = 512,
) -> None:
    
    target_path = os.path.join(data_path, dataset_name)
    if not os.path.isdir(target_path):
        print(f"{dataset_name} not present locally. Downloading and extracting it.")
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
        data_path = beir.util.download_and_unzip(url, data_path)
        # delete the zip file
        os.remove(data_path + ".zip")
        print(f"Downloaded and extracted {dataset_name} to {data_path}.")

    queries_path = os.path.join(target_path, "queries.jsonl")
    corpus_path = os.path.join(target_path, "corpus.jsonl")

    if target == "train":
        qrels_path = os.path.join(target_path, "qrels/train.tsv")
    elif target == "test":
        qrels_path = os.path.join(target_path, "qrels/test.tsv")
    elif target == "dev":
        qrels_path = os.path.join(target_path, "qrels/dev.tsv")
    else:
        print("Invalid target")

    if not os.path.exists(qrels_path):
            print(f"{target} qrels not found for {dataset_name}. Skipping")
            return
    
    queries = load_jsonl(queries_path)
    corpus = load_jsonl(corpus_path)
    qrels = load_tsv(qrels_path)

    print(f"Loaded {len(queries)} queries, {len(corpus)} documents, and {len(qrels)} qrels for {dataset_name} ({target})")

    negative_dataset = {}

    # mine hard negatives if a model is provided and num_hard_negatives > 0
    if model is not None and num_hard_negatives > 0:
        negative_dataset = mine_hard_negatives(
            queries, corpus, qrels, 
            model, 
            range_max=range_max, 
            num_hard_negatives=num_hard_negatives,
            use_faiss=use_faiss,
            batch_size=batch_size,
            use_multi_process=use_multi_process,
            include_docids_and_scores=include_docids_and_scores
        )

    negative_corpus = create_negative_corpus(qrels, corpus)

    # filter away queries without qrels
    valid_qids = [qid for qid in qrels if qid in queries]
    
    print(f"Processing {len(valid_qids)} valid queries")

    for qid in tqdm(valid_qids, desc="Extracting negatives", unit="queries", total=len(valid_qids)):
        
        # sample random negatives
        negative_doc_ids =  random.sample(negative_corpus, min(num_negatives, len(negative_corpus)))
        negative_ctxs = []
        for neg_docid in negative_doc_ids:
            entry = {
                "title": corpus[neg_docid].get("title", ""),
                "text": corpus[neg_docid]["text"]
            }
            if include_docids_and_scores:
                entry["docid"] = neg_docid
            negative_ctxs.append(entry)
        
        if model is not None and num_hard_negatives > 0:
            # if condition is true, the dict has already been updated by src.mine_hard_negatives with question|positive_ctxs|hard_negatives_ctxs
            # just add the random negatives to the entry
            negative_dataset[qid].update({"negative_ctxs" : negative_ctxs})
            continue
        
        else: # otherwise create the entry from scratch
            relevant_docs = qrels[qid]
            query_text = queries[qid]["text"]
            
            valid_positives = [doc_id for doc_id in relevant_docs if doc_id in corpus]

            if not valid_positives:
                continue  # skip if no valid positives
            
            positive_ctxs = []
            for pos_docid in valid_positives:
                positive_doc = {}
                if include_docids_and_scores:
                    positive_doc["docid"] = pos_docid
                positive_doc.update({
                    "title": corpus[pos_docid].get("title", ""),
                    "text": corpus[pos_docid]["text"]
                })
            positive_ctxs.append(positive_doc)
            
            # put together the entry
            entry = {
                "question": query_text,
                "positive_ctxs": positive_ctxs,
                "negative_ctxs": negative_ctxs,
            }
            
            negative_dataset[qid] = entry
  
    
    save_jsonl(negative_dataset, output_file)
    print(f"Saved {len(negative_dataset)} examples to {output_file}.")


def main(args):

    model = SentenceTransformer(args.model_name)

    # training data with negatives and hard negatives
    prepare_dataset_with_negatives(
        dataset_name= args.dataset_name,
        target = "train",
        output_file=f"beir_datasets/{args.dataset_name}/training_data.jsonl",
        model=model,
        num_negatives=args.num_negatives,
        num_hard_negatives=args.num_hard_negatives,
        range_max=args.range_max,
        batch_size=args.batch_size,
        use_faiss=args.use_faiss,
        use_multi_process=args.use_multi_process,
        include_docids_and_scores=args.include_docids_and_scores,
    )

    # save the settings used for mining hard negatives
    with open(f"beir_datasets/{args.dataset_name}/mining_settings.json", "w") as f:
        json.dump(vars(args), f)

    # dev data
    prepare_dataset_with_negatives(
        dataset_name= args.dataset_name,
        target = "dev",
        output_file=f"beir_datasets/{args.dataset_name}/dev_data.jsonl",
         model=model,
        num_negatives=args.num_negatives,
        num_hard_negatives=args.num_hard_negatives,
        range_max=args.range_max,
        batch_size=args.batch_size,
        use_faiss=args.use_faiss,
        use_multi_process=args.use_multi_process,
        include_docids_and_scores=args.include_docids_and_scores,
    )
    
    # test data
    prepare_dataset_with_negatives(
        dataset_name= args.dataset_name,
        target = "test",
        output_file=f"beir_datasets/{args.dataset_name}/test_data.jsonl",
        model=None,  # no hard negatives for test data
        num_negatives=args.num_negatives,
        num_hard_negatives=0
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Load the SentenceTransformer model -> https://huggingface.co/nthakur/contriever-base-msmarco
    parser.add_argument("--model_name", type=str, help="SentenceTransformer model name", default="nthakur/contriever-base-msmarco")
    parser.add_argument("--dataset_name", type=str, help="Evaluation dataset from the BEIR benchmark")
    parser.add_argument("--num_negatives", type=int, help="Number of random negative documents per query", default=5)
    # hard negatives parameters
    parser.add_argument("--num_hard_negatives", type=int, help="Number of hard negative documents per query", default=5)
    parser.add_argument("--range_max", type=int, help="Maximum rank of the closest matches to consider as negatives", default=None)
    parser.add_argument("--batch_size", type=int, help="Batch size for mining hard negatives", default=32)
    parser.add_argument("--use_faiss", action="store_true", help="Use FAISS for efficient similarity search")
    parser.add_argument("--use_multi_process", action="store_true", help="Use multi-process for mining hard negatives")
    parser.add_argument("--include_docids_and_scores", action="store_true", help="Whether to include document IDs and scores in the output. This will increase the size of the output.")
    args, _ = parser.parse_known_args()

    # print the arguments
    print("Arguments provided: ",args)


    main(args)
