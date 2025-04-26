import os
import random
from collections import defaultdict


def split_qrels(input_train_file, output_dev_file, split_ratio=0.8):
    """
    Splits a qrel file into training and development sets based on queries.

    This function reads a QRELs file, groups the lines by query ID, and splits the queries
    into training and development sets according to the specified split ratio. The training
    set is written back to the original file, and the development set is written to a new file.
    Additionally, the original file is renamed.

    Args:
        input_train_file (str): Path to the input QRELs file containing the full dataset.
        output_dev_file (str): Path to the output file where the development set will be saved.
        split_ratio (float, optional): Proportion of queries to include in the training set.
                                       Defaults to 0.8 (80% training, 20% development).

    Returns:
        None

    Side Effects:
        - Writes the development set to `output_dev_file`.
        - Renames the original input file to indicate it contains the full dataset.
        - Overwrites the original input file with the training set.

    Notes:
        - Assumes the input file is tab-separated and that the query ID is the first column.
    """
    qrels = defaultdict(list)
    with open(input_train_file, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            query_id = line.split('\t')[0]
            qrels[query_id].append(line)

    all_queries = list(qrels.keys())
    random.seed(42)  # reproducibility
    random.shuffle(all_queries)

    # split queries
    split_index = int(len(all_queries) * split_ratio)
    train_queries = all_queries[:split_index]
    dev_queries = all_queries[split_index:]

    train_qrels = [line for qid in train_queries for line in qrels[qid]]
    dev_qrels = [line for qid in dev_queries for line in qrels[qid]]

    with open(output_dev_file, 'w', encoding='utf-8') as f:
        f.write("query-id\tcorpus-id\tscore\n")
        f.writelines(dev_qrels)

    # rename the input file to indicate that it's the full train qrels set
    os.rename(input_train_file, input_train_file.replace('.tsv', '_full.tsv'))
    with open(input_train_file, 'w', encoding='utf-8') as f:
        f.write("query-id1\tcorpus-id\tscore\n")
        f.writelines(train_qrels)

    print(f"Dataset split by query into {len(train_queries)} train queries and {len(dev_queries)} dev queries.")

if __name__ == "__main__":
    dataset_name = "INSERT_DATASET_NAME_HERE"
    input_train_file = f"beir_datasets/{dataset_name}/qrels/train.tsv"
    output_dev_file = f"beir_datasets/{dataset_name}/qrels/dev.tsv"

    if os.path.exists(output_dev_file):
        print(f"File {output_dev_file} already exists. Are you sure you want to continue? (y/n)")
        answer = input().strip().lower()
        if answer != 'y':
            print("Exiting without overwriting files.")
            exit(0)

    split_qrels(input_train_file, output_dev_file)
