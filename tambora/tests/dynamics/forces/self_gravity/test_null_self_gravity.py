"""
Tests for NullSelfGravity, the solver of a run without self-gravity.
"""

import numpy as np
import pytest

from tambora.dynamics import NullSelfGravity

POS = np.random.default_rng(3).normal(size=(10, 3))
MASS = np.ones(10)


def test_no_self_gravity_gives_zeros_shaped_like_each_call():
    null = NullSelfGravity()
    for n in (10, 4, 10):
        acc, phi = null.acc_and_potential(POS[:n], MASS[:n])
        assert acc.shape == (n, 3) and phi.shape == (n,)
        assert not acc.any() and not phi.any()


def test_no_self_gravitys_zeros_cannot_be_changed_in_place():
    acc, phi = NullSelfGravity().acc_and_potential(POS, MASS)
    for zeros in (acc, phi):
        with pytest.raises(ValueError, match="read-only"):
            zeros += 1
