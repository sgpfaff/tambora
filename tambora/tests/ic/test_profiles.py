"""
Tests for tambora's profiles (``ic.Plummer``, ``ic.Hernquist``, ``ic.King``), which need no package.
"""

import dataclasses

import numpy as np
import pytest

from tambora import ic

PROFILES = [
    pytest.param(ic.Plummer, dict(M=1e5, rscale=0.01), 'Plummer(M=100000, rscale=0.01)', id='Plummer'),
    pytest.param(ic.Hernquist, dict(M=1e10, rscale=2.), 'Hernquist(M=1e+10, rscale=2)', id='Hernquist'),
    pytest.param(ic.King, dict(M=1e5, W0=5., rt=0.03), 'King(M=100000, W0=5, rt=0.03)', id='King'),
]


@pytest.mark.parametrize("cls, params, label", PROFILES)
def test_a_profile_keeps_its_parameters_as_floats(cls, params, label):
    profile = cls(**{k: np.float32(v) if k == 'M' else v for k, v in params.items()})
    assert dataclasses.asdict(profile) == pytest.approx(params, rel=1e-6)
    assert all(type(v) is float for v in dataclasses.asdict(profile).values())


@pytest.mark.parametrize("cls, params, label", PROFILES)
def test_describe_gives_the_profile_and_its_parameters(cls, params, label):
    assert cls(**params).describe() == label


@pytest.mark.parametrize("cls, params, label", PROFILES)
def test_a_profile_is_frozen_and_compares_by_value(cls, params, label):
    profile = cls(**params)
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.M = 2.
    assert profile == cls(**params) and hash(profile) == hash(cls(**params))


@pytest.mark.parametrize("value", [0., -1., np.inf, np.nan])
@pytest.mark.parametrize("cls, params, label", PROFILES)
def test_every_parameter_must_be_positive_and_finite(cls, params, label, value):
    for name in params:
        with pytest.raises(ValueError, match=f"{cls.__name__}'s {name} must be positive and finite"):
            cls(**{**params, name: value})


@pytest.mark.parametrize("value", [True, '1', None, [1.]])
def test_a_parameter_that_isnt_a_number_is_a_type_error(value):
    with pytest.raises(TypeError, match=r"Plummer's rscale must be a number \[kpc\]"):
        ic.Plummer(M=1., rscale=value)
