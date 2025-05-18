import csv
import numpy as np
from scipy.stats import ttest_rel
import argparse
from statsmodels.stats.multitest import multipletests


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

def perform_ttest(modelA_scores, modelB_scores, metrics_to_consider, save_results_path):
    results = []
    raw_pvals = []
    metric_stats = []

    common_qids = sorted(set(modelA_scores) & set(modelB_scores))

    for metric in metrics_to_consider:
        metric_A = [modelA_scores[qid][metric] for qid in common_qids]
        metric_B = [modelB_scores[qid][metric] for qid in common_qids]

        if len(metric_A) != len(metric_B):
            raise ValueError(f"The number of samples in both models must be the same for the t-test (metric: {metric}).")

        t_stat, p_val = ttest_rel(metric_A, metric_B)
        mean1 = sum(metric_A) / len(metric_A)
        mean2 = sum(metric_B) / len(metric_B)
        std1 = np.std(metric_A)
        std2 = np.std(metric_B)

        raw_pvals.append(p_val)
        metric_stats.append((metric, mean1, std1, mean2, std2, t_stat))

    # Bonferroni & Holm corrections
    reject_bonf, pvals_bonf, _, _ = multipletests(raw_pvals, alpha=0.05, method='bonferroni')
    reject_holm, pvals_holm, _, _ = multipletests(raw_pvals, alpha=0.05, method='holm')

    for i, (metric, mean1, std1, mean2, std2, t_stat) in enumerate(metric_stats):
        result_str = (
            f"{metric} | Model A: {mean1:.4f} ± {std1:.4f}, Model B: {mean2:.4f} ± {std2:.4f} | "
            f"t-stat: {t_stat:.4f}, raw p: {raw_pvals[i]:.4f}, "
            f"Bonf-corr p: {pvals_bonf[i]:.4f} ({'sig' if reject_bonf[i] else 'ns'}), "
            f"Holm-corr p: {pvals_holm[i]:.4f} ({'sig' if reject_holm[i] else 'ns'})"
        )
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
    parser.add_argument('--modelA', type=str, required=True, help='Path to perquery_scores.csv for the full finetuned model')
    parser.add_argument('--modelB', type=str, required=True, help='Path to perquery_scores.csv for the model finetuned with LoRA')
    parser.add_argument('--save_results_path', type=str, default=None, help='Path to save the results of the evaluation (in a txt file)')
    parser.add_argument('--metrics_to_consider', type=str, nargs='+', required=False, default=["ndcg_cut_10", "recall_100", "recall_1000"], help='List of metrics_to_consider to test')
    args = parser.parse_args()

    modelA_scores = load_scores_from_csv(args.modelA)
    modelB_scores = load_scores_from_csv(args.modelB)

    perform_ttest(modelA_scores, modelB_scores, args.metrics_to_consider, args.save_results_path)

if __name__ == "__main__":
    main()
