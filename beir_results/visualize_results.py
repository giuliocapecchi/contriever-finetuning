import argparse
import logging

# import src.slurm

logger = logging.getLogger(__name__)

"""
This script is used to visualize the results of the BEIR evaluation for the Contriever and LoRA models.
It reads the results from the files contriever_beir_results.txt and lora_beir_results.txt and creates a markdown table
"""
def load_results(filepath):
    results = {}
    with open(filepath, 'r') as file:
        for line in file:
            metric, value = line.split(':')
            results[metric.strip()] = float(value.strip().strip('[]'))
    return results


def main(args):
    
    contriever_results = load_results(f'./beir_results/{args.dataset}/contriever-beir-results/metrics.txt')
    lora_results = load_results(f'{args.results_folder}/metrics.txt')

    # extract metrics (they're the same for both models)
    metrics = contriever_results.keys()

    if "lora" in args.results_folder.lower():
        table = "| Metric   |   Contriever  | Lora          |\n"
    else:
        table = "| Metric   |   Contriever  | full-finetune |\n"


    table += "|----------|---------------|---------------|\n"

    for metric in metrics:
        contriever_metric_value = contriever_results[metric]
        lora_metric_value = lora_results[metric]

        if contriever_metric_value >= lora_metric_value:
            table += f"| {metric} | **{contriever_metric_value:.5f}** | {lora_metric_value:.5f} |\n"
        else:
            table += f"| {metric} | {contriever_metric_value:.5f} | **{lora_metric_value:.5f}** |\n"

    # save the table to a file
    with open(f'{args.results_folder}/comparison_table.md', 'w') as file:
        file.write(table)

    logger.info("Comparison table saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, help="Evaluation dataset from the BEIR benchmark")
    parser.add_argument("--results_folder", type=str, help="Step number at which the LoRA model was saved")

    args, _ = parser.parse_known_args()
    main(args)
