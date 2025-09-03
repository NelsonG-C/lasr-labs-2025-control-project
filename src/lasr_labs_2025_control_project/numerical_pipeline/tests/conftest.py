"""Pytest configuration and shared fixtures for numerical_pipeline tests."""

# Ensure the repository's `src` directory is importable for tests
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("montecarlo")
    group.addoption(
        "--mc-trials",
        action="store",
        type=int,
        default=20000,
        help="Number of Monte Carlo trials to run in stochastic tests (default: 20000)",
    )
    group.addoption(
        "--mc-seed",
        action="store",
        type=int,
        default=42,
        help="Random seed for Monte Carlo tests (default: 42)",
    )


@pytest.fixture
def mc_trials(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--mc-trials"))


@pytest.fixture
def mc_seed(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--mc-seed"))
