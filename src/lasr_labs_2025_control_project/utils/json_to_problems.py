import json
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Solution:
    model: str
    solution: str
    metadata: Dict[str, Any]


@dataclass
class ProblemSolutions:
    apps_problem_id: str
    problem_statement: str
    solutions: List[Solution]


def extract_code_from_content(content: str) -> str:
    """Extract code from JSON content or return empty string."""
    if not content:
        return ""

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "code" in parsed:
            return parsed["code"]
    except Exception:
        pass

    return ""


def convert_conversation_to_problem_solutions(
    conversation_data: Any,
) -> ProblemSolutions:
    """Convert conversation data to ProblemSolutions format using output method only."""
    problem_id = str(conversation_data.get("id", "unknown"))
    problem_statement = conversation_data.get("input", "")
    solutions = []

    if "output" in conversation_data and "choices" in conversation_data["output"]:
        output = conversation_data["output"]
        for i, choice in enumerate(output["choices"]):
            if "message" in choice:
                message = choice["message"]

                # Check for tool calls with submit function
                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        # Safe access to function and name with type checking
                        function = tool_call.get("function")
                        if function and isinstance(function, dict):
                            if function.get("name") == "submit":
                                args = function.get("arguments", {})
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except Exception:
                                        continue

                                if isinstance(args, dict) and "code" in args:
                                    solution = Solution(
                                        model=output.get("model", "unknown"),
                                        solution=args["code"],
                                        metadata={
                                            "choice_index": i,
                                            "stop_reason": choice.get(
                                                "stop_reason", ""
                                            ),
                                            "usage": output.get("usage", {}),
                                            "secret_input": args.get(
                                                "secret_input", ""
                                            ),
                                            "tool_call_id": tool_call.get("id", ""),
                                        },
                                    )
                                    solutions.append(solution)

                # Fallback: check message content for JSON-encoded code
                content = message.get("content", "")
                code = extract_code_from_content(content)
                if code:
                    solution = Solution(
                        model=output.get("model", "unknown"),
                        solution=code,
                        metadata={
                            "choice_index": i,
                            "stop_reason": choice.get("stop_reason", ""),
                            "usage": output.get("usage", {}),
                            "original_content": content,
                        },
                    )
                    solutions.append(solution)

    return ProblemSolutions(
        apps_problem_id=problem_id,
        problem_statement=problem_statement,
        solutions=solutions,
    )


def convert_file(input_file: str, output_file: str):
    """Convert JSON file from conversation format to ProblemSolutions format."""
    with open(input_file, "r") as f:
        data = json.load(f)

    # Handle the evaluation data structure with "samples" array
    if "samples" in data:
        conversations = data["samples"]
    elif isinstance(data, list):
        conversations = data
    else:
        conversations = [data]

    all_problem_solutions = [
        convert_conversation_to_problem_solutions(conv) for conv in conversations
    ]

    output_data = []
    for ps in all_problem_solutions:
        ps_dict = {
            "apps_problem_id": ps.apps_problem_id,
            "problem_statement": ps.problem_statement,
            "solutions": [
                {"model": sol.model, "solution": sol.solution, "metadata": sol.metadata}
                for sol in ps.solutions
            ],
        }
        output_data.append(ps_dict)

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(
        f"Converted {len(output_data)} problems with {sum(len(ps['solutions']) for ps in output_data)} solutions"
    )


if __name__ == "__main__":
    # Usage: convert_file("input.json", "output.json")
    convert_file(
        "src/lasr_labs_2025_control_project/apps/results/logs/base/2025-09-08T15-48-38+01-00_apps-untrusted-monitoring-protocol-attack-18d39942_E2cimgd9vrvKfCgTvhBkb2.json",
        "src/lasr_labs_2025_control_project/apps/results/logs/base/output.json",
    )
