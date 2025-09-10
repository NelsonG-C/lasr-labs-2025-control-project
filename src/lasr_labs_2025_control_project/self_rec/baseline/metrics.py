import numpy as np  # type: ignore
import scipy.stats as stats  # type: ignore
from inspect_ai.scorer import Metric, SampleScore, metric  # type: ignore
from typing import Any


@metric  # type: ignore
def cohens_d() -> Any:  # Metric
    """Cohen's d effect size vs chance (0.5)."""

    def metric(scores: list[Any]  # list[SampleScore]) -> float:
        results = []
        for item in scores:
            results.append(item.score.value)
        if len(results) < 2:
            return 0.0
        accuracy = np.mean(results)
        std_dev = np.std(results, ddof=1)
        return float((accuracy - 0.5) / std_dev if std_dev > 0 else 0.0)

    return metric


@metric  # type: ignore
def sem() -> Any:  # Metric
    """Standard error of the mean."""

    def metric(scores: list[Any]  # list[SampleScore]) -> float:
        results = []
        for item in scores:
            results.append(item.score.value)
        std_dev = np.std(results, ddof=1)
        return std_dev / np.sqrt(len(results))

    return metric


@metric  # type: ignore
def ci_lower() -> Any:  # Metric
    """Lower bound of 95% confidence interval."""

    def metric(scores: list[Any]  # list[SampleScore]) -> float:
        results = []
        for item in scores:
            results.append(item.score.value)
        n = len(results)
        accuracy = np.mean(results)
        std_dev = np.std(results, ddof=1)
        sem_val = std_dev / np.sqrt(n)
        t_critical = stats.t.ppf(0.975, df=n - 1)
        return accuracy - t_critical * sem_val

    return metric


@metric  # type: ignore
def ci_upper() -> Any:  # Metric
    """Upper bound of 95% confidence interval."""

    def metric(scores: list[Any]  # list[SampleScore]) -> float:
        results = []
        for item in scores:
            results.append(item.score.value)
        n = len(results)
        accuracy = np.mean(results)
        std_dev = np.std(results, ddof=1)
        sem_val = std_dev / np.sqrt(n)
        t_critical = stats.t.ppf(0.975, df=n - 1)
        return accuracy + t_critical * sem_val

    return metric
