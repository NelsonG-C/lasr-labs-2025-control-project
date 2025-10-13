# import json


# def extract_answer_from_messages(messages):
#     """Extract the answer (YES/NO) from the assistant message."""
#     for msg in messages:
#         if msg["role"] == "assistant":
#             content = msg["content"]
#             answer_start = content.find("<answer>")
#             answer_end = content.find("</answer>")

#             if answer_start != -1 and answer_end != -1:
#                 answer = content[answer_start + len("<answer>") : answer_end].strip()
#                 return answer.upper()
#     return None


# def extract_balanced_samples(input_file, output_file, yes_count=60, no_count=60):
#     """Extract the first N YES and N NO samples from the dataset."""

#     # Load the data - handle both JSON and JSONL formats
#     data = []
#     with open(input_file, "r") as f:
#         first_char = f.read(1)
#         f.seek(0)

#         if first_char == "[":
#             # Standard JSON array format
#             data = json.load(f)
#         else:
#             # JSONL format (one JSON object per line)
#             for line in f:
#                 line = line.strip()
#                 if line:
#                     data.append(json.loads(line))

#     # Handle both formats: {"data": [...]} or just [...]
#     if isinstance(data, dict) and "data" in data:
#         data = data["data"]

#     yes_samples = []
#     no_samples = []

#     # Iterate through data and collect samples
#     for item in data:
#         if len(yes_samples) >= yes_count and len(no_samples) >= no_count:
#             break

#         try:
#             answer = extract_answer_from_messages(item["messages"])

#             if answer == "YES" and len(yes_samples) < yes_count:
#                 yes_samples.append(item)
#             elif answer == "NO" and len(no_samples) < no_count:
#                 no_samples.append(item)

#         except (KeyError, TypeError) as e:
#             print(f"Warning: Skipping item due to error: {e}")
#             continue

#     # Combine the samples
#     balanced_samples = yes_samples + no_samples

#     # Save to output file
#     with open(output_file, "w") as f:
#         json.dump(balanced_samples, f, indent=2)

#     print(f"Extracted {len(yes_samples)} YES samples and {len(no_samples)} NO samples")
#     print(f"Total samples: {len(balanced_samples)}")
#     print(f"Saved to: {output_file}")

#     return balanced_samples


# if __name__ == "__main__":
#     input_file = (
#         "src/lasr_labs_2025_control_project/finetuning/data/apps/train_messages.jsonl"
#     )
#     output_file = (
#         "src/lasr_labs_2025_control_project/dspy/data/balanced_apps_train_data_120.json"
#     )

#     extract_balanced_samples(input_file, output_file, yes_count=60, no_count=60)
