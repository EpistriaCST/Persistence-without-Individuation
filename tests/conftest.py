"""Shared fixtures for the test suite."""

import numpy as np
import pytest


@pytest.fixture
def witness_params() -> dict:
    """Default witness parameters used across most tests.

    omega_0 = 1.0 : natural frequency
    k = 0.3       : coupling strength
    gamma = 0.1   : damping coefficient
    beta = 1.0    : inverse temperature
    """
    return {"omega_0": 1.0, "k": 0.3, "gamma": 0.1, "beta": 1.0}


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random number generator for reproducible tests."""
    return np.random.default_rng(seed=20260425)
