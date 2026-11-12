"""Sim.load: a streamed run read back from its directory."""

import os
import shutil
import warnings

import h5py
import numpy as np
import pytest
from galpy import potential as gp

from tambora.dynamics.hooks import EveryStep
from tambora.io import run_files
from tambora.io.run_files import RUN_FILE, snapshot_name
from tambora.simulation import Sim
from tambora.tests.io.test_streaming_sim import RUN, _assert_same_run, _Drag, _files, _sim, _Stop


def _load(directory):
    with warnings.catch_warnings():
        warnings.simplefilter('error')      # a finished run loads without warnings
        return Sim.load(directory)


@pytest.mark.parametrize("run", [
    {}, dict(cache_self_gravity_acc=False, cache_self_gravity_pot=False), dict(cache_self_gravity_pot=False),
    dict(method=None), dict(method='falcON', theta=0.4), dict(t_end=-0.2, dt=-0.01, dt_out=-0.05),
    dict(t0=1.5, t_end=1.73), dict(monitors=False),
], ids=['default', 'no caches', 'no pot', 'no self-gravity', 'falcON', 'backward', 'later t0, uneven end',
        'no monitor'])
def test_a_loaded_run_is_the_run_kept_in_memory_bit_for_bit(tmp_path, run):
    disk, memory = _sim(stream_to=tmp_path / 'run'), _sim()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')     # the uneven end
        disk.run(**{**RUN, **run})
        memory.run(**{**RUN, **run})
    sim = _load(tmp_path / 'run')
    _assert_same_run(sim, memory)
    assert sim._slices == memory._slices
    for name in ('_mass', '_eps'):
        a, b = getattr(sim, name), getattr(memory, name)
        assert a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True), name
    np.testing.assert_array_equal(sim.b.vel(t=2), memory.b.vel(t=2))
    np.testing.assert_array_equal(sim.a.KE(), memory.a.KE())
    if memory._cached_self_pot is not None:
        np.testing.assert_array_equal(sim.system_energy(), memory.system_energy())
    assert repr(sim).splitlines()[:2] == repr(memory).splitlines()[:2]


def test_a_run_without_softening_loads(tmp_path):
    def test_particles(**kwargs):
        rng = np.random.default_rng(1)
        sim = Sim(**kwargs)
        sim.add_particles(rng.normal(size=(10, 3)) + 8, rng.normal(size=(10, 3)) * 200, np.zeros(10), name='stars')
        sim.add_external_pot(gp.NFWPotential(ro=8, vo=220))
        return sim
    disk, memory = test_particles(stream_to=tmp_path / 'run'), test_particles()
    disk.run(**{**RUN, 'method': None})
    memory.run(**{**RUN, 'method': None})
    with pytest.warns(UserWarning, match=r"ExternalPotential\(NFWPotential\)\ couldn't be loaded."):
        sim = Sim.load(tmp_path / 'run')
    assert np.isnan(sim._eps).all()
    _assert_same_run(sim, memory)
    sim.add_external_pot(gp.NFWPotential(ro=8, vo=220))
    np.testing.assert_array_equal(sim.external_acc(t=2), memory.external_acc(t=2))


def test_load_takes_a_path_as_sim_does(tmp_path, monkeypatch):
    _sim(stream_to=tmp_path / 'run').run(**RUN)
    monkeypatch.setenv('HOME', str(tmp_path))
    for path in ('~/run', str(tmp_path / 'run') + os.sep, tmp_path / 'run'):
        assert _load(path)._stream_to == str(tmp_path / 'run')


def test_load_reads_only_run_h5_and_the_last_snapshots_clock(tmp_path, monitor_reads):
    _sim(stream_to=tmp_path / 'run').run(**RUN)
    sim = Sim.load(str(tmp_path / 'run'))
    assert monitor_reads == [RUN_FILE, snapshot_name(4)]
    monitor_reads.clear()
    sim.a.pos(t=1)
    assert monitor_reads == [snapshot_name(1)]


@pytest.fixture
def monitor_reads(monkeypatch):
    opened, open_for_reading = [], run_files._open_for_reading

    def spy(path):
        opened.append(os.path.basename(path))
        return open_for_reading(path)
    monkeypatch.setattr(run_files, '_open_for_reading', spy)
    return opened


def _with_forces(sim):
    sim.add_external_force(_Drag())
    sim.add_external_pot([gp.NFWPotential(ro=8, vo=220), gp.MiyamotoNagaiPotential(ro=8, vo=220)])
    return sim


FORCES_NOT_LOADED = (r"ExternalPotential\(NFWPotential\+MiyamotoNagaiPotential\), _Drag\ couldn't be loaded. Add them again "
                     r"with add_external_pot or add_external_force before computing anything that needs them")


def test_loading_a_run_with_external_forces_warns_naming_them(tmp_path):
    _with_forces(_sim(stream_to=tmp_path / 'run')).run(**RUN)
    with pytest.warns(UserWarning, match=FORCES_NOT_LOADED) as caught:
        Sim.load(tmp_path / 'run')
    assert len(caught) == 1


@pytest.mark.parametrize("compute", [
    lambda sim: sim.compute_external_pot(t=1), lambda sim: sim.external_acc(t=1),
    lambda sim: sim.a.external_acc(), lambda sim: sim.energy(t=1), lambda sim: sim.PE(t=1),
    lambda sim: sim.system_energy(), lambda sim: sim.b.energy(t=0), lambda sim: sim.virial_ratio(t=1),
], ids=['compute_external_pot', 'external_acc', 'component external_acc', 'energy', 'PE', 'system_energy',
        'component energy', 'virial_ratio'])
def test_with_its_forces_added_again_a_loaded_run_gives_what_the_run_gave(tmp_path, compute):
    _with_forces(_sim(stream_to=tmp_path / 'run')).run(**RUN)
    memory = _with_forces(_sim())
    memory.run(**RUN)
    with pytest.warns(UserWarning, match=FORCES_NOT_LOADED):
        sim = _with_forces(Sim.load(tmp_path / 'run'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')     # on-the-fly external potential
        np.testing.assert_array_equal(compute(sim), compute(memory))


def test_loading_a_run_that_stopped_early_warns_and_keeps_its_outputs(tmp_path):
    disk = _sim(stream_to=tmp_path / 'run')
    disk.add_hook(_Stop(at=12), EveryStep())
    with pytest.warns(UserWarning, match="stopped early"), pytest.raises(RuntimeError, match="stopped by the test"):
        disk.run(**RUN)
    (tmp_path / 'run' / (snapshot_name(3) + '.1234-0a1b2c3d.tmp')).write_bytes(b'')  # as a killed writer leaves
    with pytest.warns(UserWarning, match=r"holds 3 of the 5 outputs its run was to make. The run stopped early, or is still running."):
        sim = Sim.load(tmp_path / 'run')
    _assert_same_run(sim, disk)


def test_a_loaded_sim_cant_run_again_or_take_particles_or_hooks(tmp_path):
    _sim(stream_to=tmp_path / 'run').run(**RUN)
    files = _files(tmp_path / 'run')
    sim = _load(tmp_path / 'run')
    assert sim.monitor is None and sim.hooks == ()
    with pytest.raises(RuntimeError, match=r"This Sim was loaded from .* and can't run\. Make a new Sim"):
        sim.run(**RUN)
    with pytest.raises(RuntimeError, match="after run"):
        sim.add_particles(np.zeros((1, 3)), np.zeros((1, 3)), np.ones(1), name='c')
    with pytest.raises(RuntimeError, match="after run"):
        sim.add_hook(lambda state: None)
    assert _files(tmp_path / 'run') == files
    shutil.rmtree(tmp_path / 'run')
    with pytest.raises(RuntimeError, match="can't run"):
        sim.run(**RUN)
    assert not (tmp_path / 'run').exists()


def test_setting_eps_on_a_loaded_sim_goes_back_to_the_first_output(tmp_path):
    _sim(stream_to=tmp_path / 'run').run(**RUN)
    memory = _sim()
    sim = _load(tmp_path / 'run')
    sim.a.eps = 0.2
    memory.a.eps = 0.2
    _assert_same_run(sim, memory)


# --- what load refuses -------------------------------------------------------------- #

def test_a_directory_without_a_run_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="holds no run"):
        Sim.load(tmp_path)
    with pytest.raises(FileNotFoundError, match="holds no run"):
        Sim.load(tmp_path / 'missing')


def test_the_run_file_instead_of_its_directory_is_refused(tmp_path):
    _sim(stream_to=tmp_path / 'run').run(**RUN)
    with pytest.raises(NotADirectoryError, match=r"run\.h5' is a file. Pass the run's directory"):
        Sim.load(tmp_path / 'run' / RUN_FILE)


def test_a_run_without_outputs_is_refused(tmp_path):
    _sim(stream_to=tmp_path / 'run').run(**RUN)
    for f in os.listdir(tmp_path / 'run'):
        if f.startswith('snap_'):
            os.remove(tmp_path / 'run' / f)
    with pytest.raises(ValueError, match="holds no outputs"):
        Sim.load(tmp_path / 'run')


def test_another_file_called_run_h5_is_refused(tmp_path):
    with h5py.File(tmp_path / RUN_FILE, 'w') as f:
        f.attrs['format'] = 'something else'
    with pytest.raises(ValueError, match="isn't a tambora run file"):
        Sim.load(tmp_path)


def test_another_format_version_is_refused(tmp_path):
    _sim(stream_to=tmp_path / 'run').run(**RUN)
    with h5py.File(tmp_path / 'run' / RUN_FILE, 'r+') as f:
        f.attrs['format_version'] = (2, 0)
    with pytest.raises(ValueError, match=r"format version 2\.0; this tambora reads 1\.x"):
        Sim.load(tmp_path / 'run')


@pytest.mark.parametrize("attr, value", [('t', 0.25), ('step', 21)])
def test_snapshots_that_dont_match_run_h5_are_refused(tmp_path, attr, value):
    _sim(stream_to=tmp_path / 'run').run(**RUN)
    with h5py.File(tmp_path / 'run' / snapshot_name(4), 'r+') as f:
        f.attrs[attr] = value
    with pytest.raises(ValueError, match=r"snap_000004.h5 in .* but run.h5 puts output 4 at step 20, t = 0.2"):
        Sim.load(tmp_path / 'run')
