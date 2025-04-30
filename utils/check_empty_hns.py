import json

input_file = './beir_datasets/scifact/training_data.jsonl'
output_file = './beir_datasets/scifact/training_data.jsonl'

with open(input_file, 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

cleaned_lines = []
queries_without_hn = 0
for line in lines:
    ex = json.loads(line)
    if "hard_negative_ctxs" in ex and isinstance(ex["hard_negative_ctxs"], list) and not ex["hard_negative_ctxs"]:
        del ex["hard_negative_ctxs"]
        queries_without_hn += 1
    cleaned_lines.append(json.dumps(ex))

with open(output_file, 'w', encoding='utf-8') as outfile:
    outfile.write('\n'.join(cleaned_lines))

print(f"File saved in: {output_file}")
print(f"Queries without hard negative: {queries_without_hn}, which is the {queries_without_hn / len(lines) * 100:.2f}%  of the total ({len(lines)}).")