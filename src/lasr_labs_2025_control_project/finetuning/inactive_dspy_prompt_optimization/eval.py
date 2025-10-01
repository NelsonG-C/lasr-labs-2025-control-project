# pyright: reportMissingParameterType=false, reportUnknownParameterType=false, reportMissingImports=false
import argparse
import importlib.util
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import dotenv
import dspy
from dspy.evaluate import Evaluate

from lasr_labs_2025_control_project.finetuning.scripts.data_models import (
    Dataset,
    Solution,
)


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("eval.log")],
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Evaluate classification model")
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="Model in format 'provider/model' (e.g., 'openai/gpt-4o-mini')",
    )
    parser.add_argument(
        "--classify-file",
        type=str,
        help="Path to file containing classify function",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=32,
        help="Number of threads to use for evaluation",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=-1,
        help="Number of samples to use for evaluation (-1 for all)",
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to dataset",
    )

    args = parser.parse_args()
    logger.info(f"Starting evaluation with args: {args}")

    # Parse model string
    if "/" not in args.model:
        logger.error("Model must be in format 'provider/model'")
        sys.exit(1)

    provider, model = args.model.split("/", 1)
    logger.info(f"Using model: {provider}/{model}")

    # Import classify function from provided file
    classify_path = Path(args.classify_file)
    if not classify_path.exists():
        logger.error(f"Classify file {args.classify_file} does not exist")
        sys.exit(1)

    logger.info(f"Loading classify function from: {classify_path}")
    spec = importlib.util.spec_from_file_location("classify_module", classify_path)
    assert (
        spec is not None and spec.loader is not None
    ), f"Failed to load module from {classify_path}"
    classify_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classify_module)

    if not hasattr(classify_module, "classify"):
        logger.error(f"No 'classify' function found in {args.classify_file}")
        sys.exit(1)

    classify = classify_module.classify
    logger.info("Successfully loaded classify function")

    logger.info("Loading environment variables")
    dotenv.load_dotenv()

    logger.info(f"Reading dataset from {args.dataset_path}")
    ds = Dataset.read(args.dataset_path)
    logger.info(f"Loaded dataset with {len(ds)} problems")

    logger.info(f"Configuring DSPy with {provider}/{model}")
    lm = dspy.LM(f"{provider}/{model}")
    dspy.configure(lm=lm)

    training_models = [  # noqa: F841
        "gpt-5-nano",
        "human",
        "gpt-4.1-nano",
        "gemini-2.5-flash",
        "o4-mini",
        "gemini-2.5-flash-lite",
        "gpt-4o-mini",
        "claude-3-5-haiku-20241022",
    ]

    logger.info(f"Filtering solutions for model: {model}")
    models_solutions = ds.filter(model=model)
    logger.info(f"Found {len(models_solutions)} solutions for model {model}")

    def make_pair(problem_statement: str, solution_A: Solution, solution_B: Solution):
        return dspy.Example(
            {
                "problem_statement": problem_statement,
                "solution_A": solution_A.solution,
                "solution_B": solution_B.solution,
                "model_A": solution_A.model,
                "model_B": solution_B.model,
                "category": "solution_A" if solution_A.model == model else "solution_B",
            }
        ).with_inputs(
            "problem_statement",
            "solution_A",
            "solution_B",
        )

    logger.info("Creating evaluation pairs")
    pair_ds = [
        make_pair(ds[i].problem_statement, models_solutions[i][0], s)
        for i in range(len(ds))
        for s in ds[i]
        if s.model != model
    ] + [
        make_pair(ds[i].problem_statement, s, models_solutions[i][0])
        for i in range(len(ds))
        for s in ds[i]
        if s.model != model
    ]

    eval_ds = pair_ds[1::2]
    logger.info(
        f"Created {len(pair_ds)} total pairs, using {len(eval_ds)} for evaluation"
    )

    if args.n_samples != -1:
        eval_ds = eval_ds[: args.n_samples]
        logger.info(f"Limited evaluation to {len(eval_ds)} samples")

    def validate_category(
        example: dspy.Example, prediction: dspy.Prediction, trace: Optional[Any] = None
    ) -> bool:
        return prediction["category"] == example["category"]

    logger.info(f"Setting up evaluator with {args.num_threads} threads")
    evaluator = Evaluate(
        devset=eval_ds,
        num_threads=args.num_threads,
        display_progress=True,
        display_table=5,
    )
    logger.info(f"Starting evaluation of {len(eval_ds)} samples")
    result = evaluator(classify, metric=validate_category)

    logger.info(f"Evaluation completed. Accuracy: {result}")
    result_dict = result.toDict()
    result_dict["results"] = [
        {"example": dict(r[0]), "prediction": dict(r[1]), "is_correct": r[2]}
        for r in result_dict["results"]
    ]
    result_dict["history"] = classify.history
    result_dict["spec"] = classify.signature.model_json_schema()

    logger.info("Saving results to result_.json")

    class DictFallbackEncoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if obj.__class__.__module__ != "builtins":
                try:
                    return dict(obj)
                except Exception:
                    pass
            return super().default(obj)

    with open(
        f"result_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w"
    ) as f:
        json.dump(result_dict, f, cls=DictFallbackEncoder)

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
