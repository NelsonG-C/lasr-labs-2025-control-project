import json

# Define the problematic keywords to filter
problematic_keywords = ["bomb", "kill", "blood", "harm", "attack", "weapon", "assault"]

# Input and output files
input_file = "/mnt/user-data/outputs/train_messages_shuffled.jsonl"
output_file = "/mnt/user-data/outputs/train_messages_filtered.jsonl"

# Read and filter samples
filtered_samples = []
removed_count = 0

with open(input_file, "r") as f:
    for line_num, line in enumerate(f, 1):
        data = json.loads(line)

        # Convert entire sample to lowercase string for keyword checking
        sample_text = json.dumps(data).lower()

        # Check if any problematic keyword is present
        has_problematic_keyword = any(
            keyword in sample_text for keyword in problematic_keywords
        )

        if not has_problematic_keyword:
            filtered_samples.append(data)
        else:
            removed_count += 1

# Write filtered samples to output file
with open(output_file, "w") as f:
    for sample in filtered_samples:
        f.write(json.dumps(sample) + "\n")

# Count the distribution in the filtered dataset
yes_count = 0
no_count = 0

for sample in filtered_samples:
    for message in sample["messages"]:
        if message["role"] == "assistant":
            if "<answer>YES</answer>" in message["content"]:
                yes_count += 1
            elif "<answer>NO</answer>" in message["content"]:
                no_count += 1
            break

print("Filtering complete!")
print("\nOriginal shuffled dataset: 628 samples")
print(f"Samples removed: {removed_count}")
print(f"Samples remaining: {len(filtered_samples)}")
print("\nFiltered distribution:")
print(f"YES samples: {yes_count}")
print(f"NO samples: {no_count}")
print(f"\nOutput saved to: {output_file}")
