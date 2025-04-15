import json
import statistics

def calculate_document_statistics(file_path):
    document_lengths = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            document = json.loads(line)
            text = document.get("text", "")
            document_lengths.append(len(text.split())) # splitting by whitespace

    if document_lengths:
        mean_length = sum(document_lengths) / len(document_lengths)
        median_length = statistics.median(document_lengths)
        min_length = min(document_lengths)
        max_length = max(document_lengths)
    else:
        mean_length = median_length = min_length = max_length = 0

    return {
        "mean": mean_length,
        "median": median_length,
        "min": min_length,
        "max": max_length
    }

if __name__ == "__main__":

    dataset_name = 'fever'
    print(f"Calculating statistics for {dataset_name} dataset...")

    corpus_file = f"beir_datasets/{dataset_name}/corpus.jsonl"
    stats = calculate_document_statistics(corpus_file)
    print("Document statistics:")
    print(f"  Mean length: {stats['mean']:.2f} words")
    print(f"  Median length: {stats['median']:.2f} words")
    print(f"  Min length: {stats['min']} words")
    print(f"  Max length: {stats['max']} words")