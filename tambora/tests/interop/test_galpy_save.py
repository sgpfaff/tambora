"""A galpy potential saved and loaded by its backend: the record a saved run will hold."""

import os
import pickle
import subprocess
import sys
import warnings

import numpy as np
import pytest

gp = pytest.importorskip("galpy.potential")
import galpy                                                      # noqa: E402
from galpy.util.conversion import get_physical                    # noqa: E402

import tambora                                                    # noqa: E402
from tambora.interop._backend import PotentialBackend             # noqa: E402
from tambora.interop._galpy.potential import GalpyPotential       # noqa: E402
from tambora.tests.interop.galpy_classes import EVERY_GALPY_POTENTIAL, build, satellite_orbit  # noqa: E402

POS = np.array([[8., 0., 0.5], [3., -2., 1.], [0.5, 4., -2.]])   # kpc
TIMES = (0., 0.01, -0.02)                                        # Gyr, inside the moving object's orbit


def _assert_same_forces(a, b):
    for t in TIMES:
        np.testing.assert_array_equal(a.acc(POS, t), b.acc(POS, t))
        np.testing.assert_array_equal(a.potential(POS, t), b.potential(POS, t))


def _nfw(**units):
    return gp.NFWPotential(a=2., normalize=0.35, **(units or dict(ro=8., vo=220.)))


def _reloaded(backend):
    return GalpyPotential.load(backend.save())


@pytest.mark.parametrize("name", EVERY_GALPY_POTENTIAL)
def test_every_galpy_potential_that_pickles_comes_back_with_the_same_forces(name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')     # no units set; galpy's own warnings
        backend = GalpyPotential(build(name))
        try:
            pickle.dumps(backend.obj)
        except Exception as e:              # holds a function pickle can't save
            with pytest.raises(type(e)):
                backend.save()
            return
        _assert_same_forces(_reloaded(backend), backend)


@pytest.mark.parametrize("obj", [
    lambda: gp.MWPotential2014,
    lambda: _nfw() + gp.PlummerPotential(b=0.5, ro=8., vo=220.),
    lambda: [_nfw(), gp.PlummerPotential(b=0.5, ro=9., vo=230.)],
    lambda: gp.DehnenSmoothWrapperPotential(pot=gp.DehnenBarPotential(ro=8., vo=220.), tform=-1., tsteady=0.5),
    lambda: gp.MovingObjectPotential(satellite_orbit(), pot=gp.PlummerPotential(amp=0.1, b=0.1)),
], ids=['MWPotential2014', 'composite', 'units differ', 'wrapper', 'moving object'])
def test_lists_composites_wrappers_and_moving_objects_come_back(obj):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        backend = GalpyPotential(obj())
    loaded = _reloaded(backend)
    _assert_same_forces(loaded, backend)
    assert loaded.describe() == backend.describe()
    assert loaded.obj is not backend.obj


def test_a_record_holds_the_galpy_version_and_the_pickle():
    record = GalpyPotential(_nfw()).save()
    assert set(record) == {'galpy_version', 'data'}
    assert record['galpy_version'] == galpy.__version__
    assert isinstance(record['data'], bytes)


def test_the_units_are_those_it_was_added_with():
    nfw = _nfw()
    backend = GalpyPotential(nfw)
    nfw.turn_physical_on(ro=9., vo=230.)
    assert not np.array_equal(GalpyPotential(nfw).acc(POS, 0.), backend.acc(POS, 0.))
    _assert_same_forces(_reloaded(backend), backend)
    assert get_physical(nfw) == {'ro': 9., 'vo': 230.}      # yours is left alone


def test_the_parameters_are_those_it_has_when_saved():
    nfw = _nfw()
    backend = GalpyPotential(nfw)
    before = backend.acc(POS, 0.)
    nfw._amp *= 2.
    np.testing.assert_allclose(backend.acc(POS, 0.), 2. * before)
    _assert_same_forces(_reloaded(backend), backend)


def test_the_components_are_those_it_was_added_with():
    galaxy = _nfw() + gp.PlummerPotential(b=0.5, ro=8., vo=220.)
    backend = GalpyPotential(galaxy)
    galaxy[1] = gp.HernquistPotential(a=1., ro=8., vo=220.)
    loaded = _reloaded(backend)
    _assert_same_forces(loaded, backend)
    assert loaded.describe() == 'NFWPotential+PlummerPotential'


def test_loading_doesnt_warn_again_about_units_not_set():
    with pytest.warns(UserWarning, match="does not have physical units explicitly set"):
        backend = GalpyPotential(gp.NFWPotential())
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _assert_same_forces(_reloaded(backend), backend)


def test_a_potential_holding_a_lambda_cant_be_saved():
    backend = GalpyPotential(gp.TimeDependentAmplitudeWrapperPotential(A=lambda t: 1., pot=_nfw()))
    with pytest.raises((pickle.PicklingError, AttributeError)):
        backend.save()


_LOAD_IN_A_NEW_PROCESS = """
import pickle, sys
import numpy as np
from tambora.interop._galpy.potential import GalpyPotential
with open(sys.argv[1], 'rb') as f:
    backend = GalpyPotential.load(pickle.load(f))
pos = np.array([[8., 0., 0.5], [3., -2., 1.], [0.5, 4., -2.]])
np.save(sys.argv[2], np.stack([backend.acc(pos, t) for t in (0., 0.01, -0.02)]))
"""


def test_a_record_loads_in_a_new_process(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')     # no units set
        backend = GalpyPotential([gp.MWPotential2014,
                                  gp.MovingObjectPotential(satellite_orbit(), pot=gp.PlummerPotential(amp=0.1, b=0.1))])
    with open(tmp_path / 'record.pkl', 'wb') as f:
        pickle.dump(backend.save(), f)
    root = os.path.dirname(os.path.dirname(tambora.__file__))
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([root, os.environ.get('PYTHONPATH', '')])}
    subprocess.run([sys.executable, '-c', _LOAD_IN_A_NEW_PROCESS, str(tmp_path / 'record.pkl'),
                    str(tmp_path / 'acc.npy')], check=True, env=env, cwd=tmp_path)
    np.testing.assert_array_equal(np.load(tmp_path / 'acc.npy'),
                                  np.stack([backend.acc(POS, t) for t in TIMES]))


class _Unsaved(PotentialBackend):
    name = 'unsaved'

    @classmethod
    def accepts(cls, obj):
        return False

    def __init__(self, obj):
        self.obj = obj

    def acc(self, pos, t):
        return np.zeros_like(pos)

    def potential(self, pos, t):
        return np.zeros(len(pos))


def test_a_backend_that_doesnt_save_gives_no_record_and_cant_load():
    assert _Unsaved(None).save() is None
    with pytest.raises(NotImplementedError, match="unsaved potentials can't be loaded"):
        _Unsaved.load({})
