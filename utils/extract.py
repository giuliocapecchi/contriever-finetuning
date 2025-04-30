import json

dataset_name = "INSERT_DATASET_NAME_HERE"
input_file = f"./beir_datasets/{dataset_name}/corpus.jsonl"

with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        if "love" in data.get("question", []) and any("taylor" in e.get("title","").lower() + e.get("text","").lower() for e in data.get("hard_negative_ctxs", [])): 
            print("Question: ", data["question"])
            print("-" * 100)
            print("Positive Contexts: ")
            for context in data.get("positive_ctxs", []):
                print("Title: ", context.get("title", ""))
                print("Text: ", context.get("text", "")[:300],"...")
            print("-" * 100)
            print("Hard Negative Contexts: ")
            for context in data.get("hard_negative_ctxs", [])[:2]:
                print("Title: ", context.get("title", ""))
                print("Text: ", context.get("text", "")[:300],"...")
            print("-" * 100)
        