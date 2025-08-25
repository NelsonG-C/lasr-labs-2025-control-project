import json
from typing import Any

from inspect_ai.dataset import Sample


def record_to_sample(record: dict[str, Any]) -> Sample:
    test_list = []
    if record["input_output"]:
        input_output = json.loads(record["input_output"])
        inputs = input_output.get("inputs", [])
        outputs = input_output.get("outputs", [])

        # Pair each input with its corresponding output
        for test_input, expected_output in zip(inputs, outputs):
            # Handle cases where input/output might be lists or other types
            test_input_repr = repr(test_input)
            expected_output_repr = repr(expected_output)

            test_case = f"assert solution({test_input_repr}) == {expected_output_repr}"
            test_list.append(test_case)

    # Create a limited version for display in the prompt
    prompt_test_list = test_list[:3]

    return Sample(
        input=record["question"],
        target=test_list,  # Full test list for verification
        id=record["problem_id"],
        metadata={
            "question": record["question"],
            "test_list": test_list,  # Full test list
            "test_list_str": "\n".join(
                prompt_test_list
            ),  # Limited test list for display
            "problem_id": record["problem_id"],
            "difficulty": record["difficulty"],
        },
    )
