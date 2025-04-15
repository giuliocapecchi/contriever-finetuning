import json
import random

def split_dataset(input_file, output_dev_file, split_ratio=0.8):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]

    random.shuffle(data)

    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    dev_data = data[split_index:]

    # save the training set
    with open(input_file, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry) + '\n')

    # save the dev set
    with open(output_dev_file, 'w', encoding='utf-8') as f:
        for entry in dev_data:
            f.write(json.dumps(entry) + '\n')
    print(f"Dataset split into {len(train_data)} training samples and {len(dev_data)} development samples.")


if __name__ == "__main__":
    dataset = "scifact"
    input_file = f"beir_datasets/{dataset}/training_data.jsonl"
    output_dev_file = f"beir_datasets/{dataset}/dev_data.jsonl"

    import os
    if os.path.exists(output_dev_file):
        print(f"File {output_dev_file} already exists. Are you sure you want to continue? (y/n)")
        answer = input().strip().lower()
        if answer != 'y':
            print("Exiting without overwriting files.")
            exit(0)

    split_dataset(input_file, output_dev_file)