import csv
import numpy as np
from scipy.stats import ttest_rel
import argparse

def load_scores_from_csv(file_path):
    scores = {}
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            query_id = row[0]
            metrics = {header[i]: float(row[i]) for i in range(1, len(header))}
            scores[query_id] = metrics
    return scores

def perform_ttest(full_finetuned_scores, lora_finetuned_scores, metrics, save_results_path):
    results = []
    for metric in metrics:
        common_qids = sorted(set(full_finetuned_scores) & set(lora_finetuned_scores))
        metric_full = [full_finetuned_scores[qid][metric] for qid in common_qids]
        metric_lora = [lora_finetuned_scores[qid][metric] for qid in common_qids]

        # perform paired t-test
        if len(metric_full) != len(metric_lora):
            raise ValueError(f"The number of samples in both models must be the same for the t-test (metric: {metric}).")
        t_stat, p_val = ttest_rel(metric_full, metric_lora)
        mean1 = sum(metric_full) / len(metric_full)
        mean2 = sum(metric_lora) / len(metric_lora)
        std1 = np.std(metric_full)
        std2 = np.std(metric_lora)
        result_str = (f"{metric} | Full: {mean1:.4f} ± {std1:.4f}, LoRA: {mean2:.4f} ± {std2:.4f} | t-stat: {t_stat:.4f}, p-value: {p_val:.4f}")
        print(result_str)
        results.append(result_str)

    if save_results_path:
        if not save_results_path.endswith(".txt"):
            save_results_path = save_results_path + "/ttest_results.txt"
        with open(save_results_path, "w") as f:
            for line in results:
                f.write(line + "\n")

def main():
    parser = argparse.ArgumentParser(description="Paired t-test on perquery_scores.csv of two models.")
    parser.add_argument('--full-finetuned_model', type=str, required=True, help='Path to perquery_scores.csv for the full finetuned model')
    parser.add_argument('--lora-finetuned_model', type=str, required=True, help='Path to perquery_scores.csv for the model finetuned with LoRA')
    parser.add_argument('--save_results_path', type=str, default=None, help='Path to save the results of the evaluation (in a txt file)')
    parser.add_argument('--metrics', type=str, nargs='+', required=False, default=["ndcg_cut_10", "recall_100", "recall_1000"], help='List of metrics to test')
    args = parser.parse_args()

    full_finetuned_scores = load_scores_from_csv(args.full_finetuned_model)
    lora_finetuned_scores = load_scores_from_csv(args.lora_finetuned_model)

    perform_ttest(full_finetuned_scores, lora_finetuned_scores, args.metrics, args.save_results_path)

if __name__ == "__main__":
    main()
