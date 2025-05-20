import os
import sys
import json
import torch
import random
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.contriever import load_retriever

def parse_args():
    parser = argparse.ArgumentParser(description="Check hard negatives distribution")
    parser.add_argument("--dataset", type=str, required=True, help="Beir dataset name (e.g., 'nfcorpus', 'hotpotqa', 'scifact', ...)")
    parser.add_argument("--model_name", type=str, default="facebook/contriever-msmarco", help="Model name or path")
    parser.add_argument("--num_examples", type=int, default=1000, help="Number of examples to check")
    parser.add_argument("--normalize_embeddings", action="store_true", help="Whether to normalize embeddings")
    return parser.parse_args()

def embed(texts, tokenizer, retriever, normalize_embeddings, use_cuda):
    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    if use_cuda:
        batch = {k: v.cuda() for k, v in batch.items()}
    with torch.no_grad():
        embeddings = retriever(**batch)
    if normalize_embeddings:
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    return embeddings

def main():
    args = parse_args()
    print(args)

    dataset = args.dataset
    num_examples = args.num_examples
    data_path = f"./beir_datasets/{dataset}/{args.model_name.split('/')[-1]}/training_data.jsonl"
    plot_save_folder = f"./utils/hard_negatives_distribution/{dataset}/{args.model_name.split('/')[-1]}/"

    use_cuda = torch.cuda.is_available()

    # load model
    retriever, tokenizer, _ = load_retriever(args.model_name, random_init=False)
    retriever.eval()

    if use_cuda:
        retriever = retriever.cuda()

    # load dataset
    with open(data_path, "r") as f:
        lines = f.readlines()

    random.seed(42)
    random.shuffle(lines)

    if num_examples > len(lines):
        num_examples = len(lines)
        print(f"[Warning] num_examples > dataset size. Setting num_examples to {num_examples}.")

    examples = []
    queries_without_hn = 0

    for line in lines:
        item = json.loads(line)
        if "positive_ctxs" in item and "negative_ctxs" in item and "hard_negative_ctxs" in item:
            if len(examples) < num_examples:
                examples.append(item)
        if "hard_negative_ctxs" not in item:
            queries_without_hn += 1

    # compute similarities
    positives, negatives, hards = [], [], []
    hard_over_pos, neg_over_pos = 0, 0

    for ex in tqdm(examples, desc="Embedding documents..."):
        query = ex["question"]

        pos_texts = [p["title"] + " " + p["text"] for p in ex["positive_ctxs"]]
        neg_texts = [n["title"] + " " + n["text"] for n in ex["negative_ctxs"]]
        hard_texts = [h["title"] + " " + h["text"] for h in ex["hard_negative_ctxs"]]

        if 'e5' in args.model_name: # add 'query:' and 'passage:' prefixes
            query = "query: " + query
            pos_texts = ["passage: " + text for text in pos_texts]
            neg_texts = ["passage: " + text for text in neg_texts]
            hard_texts = ["passage: " + text for text in hard_texts]


        query_emb = embed([query], tokenizer, retriever, args.normalize_embeddings, use_cuda)

        pos_embs = embed(pos_texts, tokenizer, retriever, args.normalize_embeddings, use_cuda)
        neg_embs = embed(neg_texts, tokenizer, retriever, args.normalize_embeddings, use_cuda)
        hard_embs = embed(hard_texts, tokenizer, retriever, args.normalize_embeddings, use_cuda)
    

        pos_scores = torch.matmul(pos_embs, query_emb.T).squeeze()
        neg_scores = torch.matmul(neg_embs, query_emb.T).squeeze()
        hard_scores = torch.matmul(hard_embs, query_emb.T).squeeze()

        max_pos_score = pos_scores.max().item()
        max_neg_score = neg_scores.max().item()
        max_hard_score = hard_scores.max().item()

        positives.append(max_pos_score)
        negatives.append(max_neg_score)
        hards.append(max_hard_score)

        if max_neg_score > max_pos_score:
            neg_over_pos += 1
        if max_hard_score > max_pos_score:
            hard_over_pos += 1
            print(f"\nFound a hard negative with higher score than the positive!\nQuery: {query}")
            print(f"Positive: {pos_texts[pos_scores.argmax()][:50]}")
            print(f"Hard Negative: {hard_texts[hard_scores.argmax()][:50]}")
            print()

    total_examples = len(lines)
    print(f"\nQueries without hard negatives: {queries_without_hn}/{total_examples} ({queries_without_hn / total_examples:.2%})")

    # create save folder if needed
    os.makedirs(plot_save_folder, exist_ok=True)

    # plot average similarities
    labels = ['Positives', 'Hard Negatives', 'Random Negatives']
    all_scores = [positives, hards, negatives]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, [sum(scores) / len(scores) for scores in all_scores], color=["green", "red", "orange"])
    plt.title(f"({dataset}) Average Similarity (Dot Product)")
    plt.ylabel("Score")
    plt.text(0.98, 0.95,
             f"Examples used: {len(positives)}/{total_examples}\nQueries without HNs: {queries_without_hn/total_examples:.2%}",
             ha='right', va='top', fontsize=10, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig(os.path.join(plot_save_folder, "barplot.png"))

    # plot score distributions
    plt.figure(figsize=(10, 6))
    plt.boxplot(all_scores, tick_labels=labels)
    plt.title(f"({dataset}) Score Distribution by Type")
    plt.ylabel("Dot Product Similarity")
    plt.grid(True)
    plt.text(0.98, 0.95,
             f"Examples used: {len(positives)}/{total_examples}\nQueries without HNs: {queries_without_hn/total_examples:.2%}",
             ha='right', va='top', fontsize=10, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig(os.path.join(plot_save_folder, "boxplot.png"))

    # Final stats
    print(f"Hard negatives > positives: {hard_over_pos}/{len(positives)} ({hard_over_pos / len(positives):.2%})")
    print(f"Random negatives > positives: {neg_over_pos}/{len(positives)} ({neg_over_pos / len(positives):.2%})")

if __name__ == "__main__":
    main()
