import json
from typing import List


def process_json_file(input_file: str, output_file: str, properties: List[str]):
    """
    Process JSON file and clean specified property in each dict.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        property_to_clean: Name of the property to clean (e.g., 'solution_A', 'solution_B')
    """
    # Read the JSON file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Process each dictionary
    for item in data:
        for property_to_clean in properties:
            if property_to_clean in item:
                if item[property_to_clean] == "o4-mini":
                    item[property_to_clean] = "openai/o4-mini"
                elif item[property_to_clean] == "claude-3-5-haiku-20241022":
                    item[property_to_clean] = "anthropic/claude-3-5-haiku-20241022"

    # Write the cleaned data to new file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Processed {len(data)} items. Cleaned '{property_to_clean}' property.")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    # params
    input_path = (
        "src/lasr_labs_2025_control_project/self_rec/data/merged_solutions.json"
    )
    output_path = "src/lasr_labs_2025_control_project/self_rec/data/merged_solutions_model_name.json"
    properties_to_clean = ["model_A", "model_B"]

    # run the code
    process_json_file(input_path, output_path, properties_to_clean)
