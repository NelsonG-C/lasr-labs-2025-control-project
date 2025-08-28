import json
from typing import List

from lasr_labs_2025_control_project.utils.utils import clean_markdown_from_code


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
                item[property_to_clean] = clean_markdown_from_code(
                    item[property_to_clean]
                )

    # Write the cleaned data to new file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Processed {len(data)} items. Cleaned '{property_to_clean}' property.")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    # params
    input_path = "src/lasr_labs_2025_control_project/experiments/self_rec/data/claude-3-5-haiku-20241022_o4-mini.json"
    output_path = "src/lasr_labs_2025_control_project/experiments/self_rec/data/claude-3-5-haiku-20241022_o4-mini_cleaned.json"
    properties_to_clean = ["solution_A", "solution_B"]

    # run the code
    process_json_file(input_path, output_path, properties_to_clean)
