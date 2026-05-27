"""Shared pytest fixtures for the tambora test suite."""

import numpy as np
import pytest

# Fixed seed used to make every test's random draws reproducible.
RANDOM_SEED = 42


@pytest.fixture(autouse=True)
def _seed_numpy_rng():
    """Reseed NumPy's global RNG before every test.

    Tests draw from ``np.random`` (a single process-wide stream). Seeding once
    at module import is order-dependent: the values a test receives depend on
    how many draws earlier tests consumed and on pytest's collection order,
    which varies across OS and Python versions. Reseeding before each test
    guarantees identical inputs regardless of execution order or parallelism.
    """
    np.random.seed(RANDOM_SEED)
