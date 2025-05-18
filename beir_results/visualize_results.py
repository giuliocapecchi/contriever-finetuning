import argparse
import logging
import os


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
    
    zeroshot_results = load_results(os.path.join(args.zeroshot_folder, 'metrics.txt'))
    finetuned_results = load_results(os.path.join(args.results_folder, 'metrics.txt'))

    # extract metrics (they're the same for both models)
    metrics = zeroshot_results.keys()

    if "lora" in args.results_folder.lower():
        table = "| Metric   |   Zero-shot  | Lora          |\n"
    else:
        table = "| Metric   |   Zero-shot  | full-finetune |\n"


    table += "|----------|---------------|---------------|\n"

    for metric in metrics:
        zeroshot_metric_value = zeroshot_results[metric]
        finetuned_metric_value = finetuned_results[metric]

        if zeroshot_metric_value >= finetuned_metric_value:
            table += f"| {metric} | **{zeroshot_metric_value:.5f}** | {finetuned_metric_value:.5f} |\n"
        else:
            table += f"| {metric} | {zeroshot_metric_value:.5f} | **{finetuned_metric_value:.5f}** |\n"

    # save the table to a file
    with open(f'{args.results_folder}/comparison_table.md', 'w') as file:
        file.write(table)

    logger.info("Comparison table saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--zeroshot_folder", type=str, help="Folder containing the model evaluation in zero-shot")
    parser.add_argument("--results_folder", type=str, help="Step number at which the LoRA model was saved")

    args, _ = parser.parse_known_args()
    main(args)
