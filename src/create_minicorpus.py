import random
import logging
 
logger = logging.getLogger(__name__)


def create_minicorpus(dataset_name: str, beir_directory: str = "beir_datasets", sample_size: float = 0.1) -> None:
    """
    Creates a mini-corpus by sampling a subset of documents from a BEIR dataset.

    This function reads the original corpus from a specified BEIR dataset, samples a subset 
    of the documents, and writes the sampled subset to a new file.

    Args:
        dataset_name (str): The name of the BEIR dataset (e.g., "nq").
        beir_directory (str, optional): The directory where the BEIR datasets are stored. 
            Defaults to "beir_datasets".
        sample_size (float, optional): The size of the sample to create. If an integer is passed,
            it will be treated as the number of documents to sample. If a float smaller than 1 is passed, it will be
            treated as a percentage of the original corpus size. For example, 0.1 means 10% of the original
            corpus. If the sample size is larger than the original corpus, it will sample the entire corpus.
            Defaults to 0.1 (10% of the original corpus).
    Returns:
        None: Does not return anything. It writes the sampled subset to a new file.
    Raises:
        ValueError: If the sample size is not a positive integer or a float between 0 and 1.
        FileNotFoundError: If the input file does not exist.
    """

    input_file = f"{beir_directory}/{dataset_name}/corpus.jsonl"
    output_file = f"{beir_directory}/{dataset_name}/minicorpus.jsonl"

    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    total_lines = len(lines)

    if sample_size < 1 and sample_size > 0: # sample a percentage of the corpus
        sample_size = int(total_lines * sample_size)
    elif sample_size > total_lines: # sample the entire corpus
        sample_size = total_lines
    elif sample_size <= 0: # error
        raise ValueError("Sample size must be a positive integer or a float between 0 and 1.")

    random.seed(42)
    sampled_lines = random.sample(lines, sample_size)

    with open(output_file, 'w') as outfile:
        outfile.writelines(sampled_lines)
    
    if not logger.hasHandlers(): # called from main
        print(f"Sampled {sample_size} lines from {total_lines} total lines. Output written to {output_file}.")
    else: # called from another module
        logger.info(f"Sampled {sample_size} lines from {total_lines} total lines. Output written to {output_file}.")


if __name__ == "__main__": # example usage
    dataset_name = "fever"
    beir_directory = "beir_datasets" 
    create_minicorpus(dataset_name, beir_directory, 0.1) # 10% of the original corpus
