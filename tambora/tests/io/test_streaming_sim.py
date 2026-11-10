"""Sim(stream_to=...): a run written to a directory as it runs, and read back from it."""

import os
import pathlib
import warnings

import h5py
import numpy as np
import pytest

import tambora.simulation.simulation as simulation_module
from tambora.dynamics.hooks import EveryOutput, EveryStep, Hook
from tambora.io import run_files
from tambora.io.run_files import RUN_FILE, _read_settings, _SnapshotArray, snapshot_name
from tambora.simulation import Sim

RUN = dict(t_end=0.2, dt=0.01, dt_out=0.05, method='direct', progress=False)
OUTPUTS = ('_positions', '_velocities', '_times', '_cached_self_acc', '_cached_self_pot', '_steps')


def _sim(**kwargs):
    rng = np.random.default_rng(0)
    sim = Sim(**kwargs)
    sim.add_particles(rng.normal(size=(30, 3)), rng.normal(size=(30, 3)) * 10, np.full(30, 1e5), name='a', eps=0.05)
    sim.add_particles(rng.normal(size=(20, 3)) + 3, rng.normal(size=(20, 3)) * 10, np.full(20, 2e5), name='b', eps=0.1)
    return sim


def _assert_same_run(disk, memory):
    for name in OUTPUTS:
        a, b = getattr(disk, name), getattr(memory, name)
        if b is None:
            assert a is None, name
            continue
        a = np.asarray(a)
        assert (a.dtype, a.shape) == (b.dtype, b.shape) and np.array_equal(a, b), name
    assert disk._run_settings == memory._run_settings


def _files(directory):
    return {f: (directory / f).read_bytes() for f in sorted(os.listdir(directory))}


def _settings(directory):
    with h5py.File(directory / RUN_FILE, 'r') as f:
        return _read_settings(f['settings'])


class _Stop(Hook):
    """Raises at step *at*."""
    def __init__(self, at):
        self.at = at

    def __call__(self, state):
        if state.step == self.at:
            raise RuntimeError("stopped by the test")


class _Steps(Hook):
    def __init__(self):
        self.steps = []

    def reset(self):
        self.steps = []

    def __call__(self, state):
        self.steps.append(state.step)


# --- a streamed run is the run that would be kept in memory --------------------------- #

@pytest.mark.parametrize("run", [
    {}, dict(cache_self_gravity_acc=False, cache_self_gravity_pot=False), dict(cache_self_gravity_pot=False),
    dict(method=None), dict(method='falcON', theta=0.4), dict(t_end=-0.2, dt=-0.01, dt_out=-0.05),
], ids=['default', 'no caches', 'no pot', 'no self-gravity', 'falcON', 'backward'])
def test_a_streamed_run_is_the_run_kept_in_memory_bit_for_bit(tmp_path, run):
    disk, memory = _sim(stream_to=tmp_path / 'run'), _sim()
    disk.run(**{**RUN, **run})
    memory.run(**{**RUN, **run})
    for name in ('_positions', '_velocities', '_cached_self_acc', '_cached_self_pot'):
        assert isinstance(getattr(disk, name), _SnapshotArray) or getattr(memory, name) is None, name
    _assert_same_run(disk, memory)
    assert sorted(os.listdir(tmp_path / 'run')) == [RUN_FILE] + [snapshot_name(i) for i in range(len(memory.times))]


def test_a_run_that_stops_early_keeps_the_same_outputs_streamed_as_in_memory(tmp_path):
    disk, memory = _sim(stream_to=tmp_path / 'run'), _sim()
    for sim in (disk, memory):
        sim.add_hook(_Stop(at=12), EveryStep())
        with pytest.warns(UserWarning, match="stopped early: the 3 outputs"), \
                pytest.raises(RuntimeError, match="stopped by the test"):
            sim.run(**RUN)
    np.testing.assert_array_equal(disk._steps, [0, 5, 10])
    _assert_same_run(disk, memory)
    assert disk._run_settings['n_steps'] == 20
    assert sorted(os.listdir(tmp_path / 'run')) == [RUN_FILE] + [snapshot_name(i) for i in range(3)]


@pytest.mark.parametrize("stop", [RuntimeError, KeyboardInterrupt])
def test_a_streamed_run_that_stops_before_its_first_output_leaves_no_run_behind(tmp_path, stop):
    class _StopAtStart(Hook):
        def __call__(self, state):
            raise stop("stopped by the test")
    (tmp_path / 'notes.txt').write_text('mine')
    sim = _sim(stream_to=tmp_path)
    hook = _StopAtStart()
    sim.add_hook(hook, EveryStep())
    with pytest.raises(stop, match="stopped by the test"):
        sim.run(**RUN)
    assert os.listdir(tmp_path) == ['notes.txt']
    assert not sim._has_run
    sim._hooks.remove((hook, sim._hooks[0][1]))
    sim.run(**RUN)              # the directory is free again
    assert (tmp_path / snapshot_name(len(sim.times) - 1)).exists()


def test_a_run_refused_by_the_writer_leaves_the_other_runs_files(tmp_path, monkeypatch):
    first, second = _sim(stream_to=tmp_path / 'run'), _sim(stream_to=tmp_path / 'run')
    first.run(**RUN)
    files = _files(tmp_path / 'run')
    monkeypatch.setattr(simulation_module, 'existing_run', lambda directory: None)
    with pytest.raises(FileExistsError, match=r"already holds a run \(run.h5\)\.$"):
        second.run(**RUN)
    assert _files(tmp_path / 'run') == files


# --- run.h5 holds what Sim.load needs ------------------------------------------------- #

def test_run_h5_holds_the_components_particles_and_run_settings(tmp_path):
    sim = _sim(stream_to=tmp_path / 'run')
    sim.run(t_end=1, dt=0.01, dt_out=0.05, method='falcON', progress=False)
    settings = _settings(tmp_path / 'run')
    assert list(settings) == ['components', 'particles', 'run']
    assert settings['components']['names'].tolist() == ['a', 'b']
    np.testing.assert_array_equal(settings['components']['start'], [0, 30])
    np.testing.assert_array_equal(settings['components']['stop'], [30, 50])
    np.testing.assert_array_equal(settings['particles']['mass'], sim._mass)
    np.testing.assert_array_equal(settings['particles']['eps'], sim._eps)
    assert settings['run'] == sim._run_settings == {
        't0': 0.0, 't_end': 1.0, 'dt': 0.01, 'dt_out': 0.05, 'n_steps': 100, 'steps_per_output': 5,
        'integration_method': 'leapfrog', 'self_gravity': {'method': 'falcON', 'kernel': 1, 'theta': 0.6},
        'cache_self_gravity_acc': True, 'cache_self_gravity_pot': True, 'monitors': 'auto'}
    assert all(type(settings['run'][k]) is float for k in ('t0', 't_end', 'dt', 'dt_out'))
    with h5py.File(tmp_path / 'run' / RUN_FILE, 'r') as f:
        assert (f.attrs['expected_outputs'], f.attrs['n_particles']) == (21, 50)
    for i, step in enumerate(sim._steps):
        with h5py.File(tmp_path / 'run' / snapshot_name(i), 'r') as f:
            assert (f.attrs['step'], f.attrs['t']) == (step, sim.times[i])


@pytest.mark.parametrize("times, n_steps, steps_per_output", [
    (dict(t0=2.0, t_end=2.2, dt=0.01, dt_out=0.05), 20, 5),
    (dict(t0=1.0, t_end=0.8, dt=-0.01, dt_out=-0.05), 20, 5),
    (dict(t0=0.0, t_end=0.23, dt=0.01, dt_out=0.05), 23, 5),
    (dict(t0=-1, t_end=1, dt=0.25, dt_out=0.5), 8, 2),
], ids=['later t0', 'backward', 'uneven end', 'ints'])
def test_the_run_settings_are_the_arguments_given(tmp_path, times, n_steps, steps_per_output):
    sim = _sim(stream_to=tmp_path / 'run')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sim.run(**{**RUN, **times})
    expected = {**{k: float(v) for k, v in times.items()}, 'n_steps': n_steps, 'steps_per_output': steps_per_output}
    for settings in (sim._run_settings, _settings(tmp_path / 'run')['run']):
        assert {k: settings[k] for k in expected} == expected
        assert all(type(settings[k]) is float for k in times)


def test_the_times_are_those_the_settings_give(tmp_path):
    """float32 or int times are run in float64, as recorded; every snapshot's t is a float."""
    sim = _sim(stream_to=tmp_path / 'run')
    sim.run(**{**RUN, 't0': np.float32(0.1), 't_end': 2, 'dt': np.float32(0.25), 'dt_out': 0.5})
    settings = _settings(tmp_path / 'run')['run']
    assert sim.times.dtype == np.float64
    np.testing.assert_array_equal(sim.times, settings['t0'] + sim._steps * settings['dt'])
    with h5py.File(tmp_path / 'run' / snapshot_name(0), 'r') as f:
        assert type(f.attrs['t']) is np.float64


@pytest.mark.parametrize("method, kwargs, expected", [
    ('falcON', {}, {'method': 'falcON', 'kernel': 1, 'theta': 0.6}),
    ('falcON', dict(theta=0.3, kernel=0), {'method': 'falcON', 'kernel': 0, 'theta': 0.3}),
    ('direct', {}, {'method': 'direct', 'use_C': True}),
    ('direct', dict(use_C=False), {'method': 'direct', 'use_C': False}),
    (None, {}, {'method': None}),
])
def test_the_solver_is_recorded_with_every_setting(tmp_path, method, kwargs, expected):
    sim = _sim(stream_to=tmp_path / 'run')
    sim.run(**{**RUN, 'method': method, **kwargs})
    assert sim._run_settings['self_gravity'] == expected
    assert _settings(tmp_path / 'run')['run']['self_gravity'] == expected


@pytest.mark.parametrize("monitors, kept, on_disk", [
    ('auto', 'auto', 'auto'), (False, False, False), (None, None, None), ('energy', 'energy', 'energy'),
    (('energy',), ('energy',), ['energy']), (['energy'], ('energy',), ['energy']), ((), (), []),
])
def test_monitors_are_recorded_as_given(tmp_path, monitors, kept, on_disk):
    sim = _sim(stream_to=tmp_path / 'run')
    sim.run(**RUN, monitors=monitors)
    assert sim._run_settings['monitors'] == kept
    recorded = _settings(tmp_path / 'run')['run']['monitors']
    assert recorded == on_disk and type(recorded) is type(on_disk)


@pytest.mark.parametrize("acc, pot", [(1, np.False_), (np.True_, None), ('yes', 0)])
def test_the_cache_flags_are_recorded_as_true_or_false(tmp_path, acc, pot):
    sim, memory = _sim(stream_to=tmp_path / 'run'), _sim()
    for each in (sim, memory):
        each.run(**RUN, cache_self_gravity_acc=acc, cache_self_gravity_pot=pot)
    _assert_same_run(sim, memory)
    with h5py.File(tmp_path / 'run' / RUN_FILE, 'r') as f:
        flags = [f.attrs['keep_self_acc'], f.attrs['keep_self_pot']]
    for settings in (sim._run_settings, _settings(tmp_path / 'run')['run']):
        assert [settings['cache_self_gravity_acc'], settings['cache_self_gravity_pot']] == flags == [bool(acc), bool(pot)]
        assert type(settings['cache_self_gravity_acc']) is bool and type(settings['cache_self_gravity_pot']) is bool


def test_an_unset_softening_is_recorded_as_nan(tmp_path):
    sim = Sim(stream_to=tmp_path / 'run')
    sim.add_particles(np.zeros((2, 3)), np.ones((2, 3)), np.ones(2), name='a')
    sim.run(**{**RUN, 'method': None})
    assert np.isnan(_settings(tmp_path / 'run')['particles']['eps']).all()


# --- the Sim keeps each output's step and its run's settings, in memory too ---------- #

@pytest.mark.parametrize("streamed", [False, True])
def test_each_output_keeps_the_step_that_made_it(tmp_path, streamed):
    sim = _sim(stream_to=tmp_path / 'run') if streamed else _sim()
    hook = _Steps()
    sim.add_hook(hook, EveryOutput())
    sim.run(**RUN)
    assert sim._steps.dtype == np.int64
    np.testing.assert_array_equal(sim._steps, hook.steps)
    np.testing.assert_array_equal(sim._steps, [0, 5, 10, 15, 20])


def test_the_steps_and_settings_go_with_the_outputs():
    sim = Sim()
    assert sim._steps is None and sim._run_settings is None
    sim.add_particles(np.zeros((2, 3)), np.ones((2, 3)), np.ones(2), name='a', eps=0.1)
    np.testing.assert_array_equal(sim._steps, [0])
    sim.run(**RUN)
    assert sim._run_settings['n_steps'] == 20 and len(sim._steps) == 5
    sim.eps = 0.2       # clears the outputs
    np.testing.assert_array_equal(sim._steps, [0])
    assert sim._run_settings is None


def test_a_rerun_replaces_the_steps_and_settings():
    sim = _sim()
    sim.run(**RUN)
    sim.run(**{**RUN, 'dt_out': 0.1})
    np.testing.assert_array_equal(sim._steps, [0, 10, 20])
    assert sim._run_settings['dt_out'] == 0.1


def test_a_failed_rerun_that_recorded_nothing_keeps_the_last_runs_steps_and_settings():
    sim = _sim()
    stop = _Stop(at=-1)
    sim.add_hook(stop, EveryStep())
    sim.run(**RUN)
    settings, steps, times = sim._run_settings, sim._steps, sim.times
    stop.at = 0
    with pytest.raises(RuntimeError, match="stopped by the test"):
        sim.run(**{**RUN, 'dt_out': 0.1})
    assert sim._run_settings is settings and sim._steps is steps and sim.times is times


# --- a directory that holds a run is refused (Q8), others are used ------------------- #

@pytest.mark.parametrize("taken", [RUN_FILE, snapshot_name(0), snapshot_name(3) + '.123-0badc0de.tmp'])
def test_a_directory_holding_a_run_is_refused_when_the_sim_is_made_and_left_alone(tmp_path, taken):
    (tmp_path / taken).write_bytes(b'an earlier run')
    (tmp_path / 'notes.txt').write_text('mine')
    files = _files(tmp_path)
    with pytest.raises(FileExistsError, match=rf"already holds a run \({taken}\)\. Delete it, or stream to another"):
        Sim(stream_to=tmp_path)
    assert _files(tmp_path) == files


@pytest.mark.parametrize("other", ['notes.txt', 'run.h5.tmp', 'run.h5.12345-0badc0de.tmp', 'snapshots'])
def test_other_files_in_the_directory_are_left_alone(tmp_path, other):
    (tmp_path / other).write_text('mine')
    sim = _sim(stream_to=tmp_path)
    sim.run(**RUN)
    assert (tmp_path / other).read_text() == 'mine'
    assert len(os.listdir(tmp_path)) == 1 + 1 + len(sim.times)


def test_a_missing_directory_is_made_by_the_run_not_by_the_sim(tmp_path):
    sim = _sim(stream_to=tmp_path / 'a' / 'b')
    assert not (tmp_path / 'a').exists()
    sim.run(**RUN)
    assert (tmp_path / 'a' / 'b' / RUN_FILE).exists()


def test_a_relative_directory_is_fixed_when_the_sim_is_made(tmp_path, monkeypatch):
    (tmp_path / 'here').mkdir()
    (tmp_path / 'there').mkdir()
    monkeypatch.chdir(tmp_path / 'here')
    sim = _sim(stream_to='run')
    monkeypatch.chdir(tmp_path / 'there')
    sim.run(**RUN)
    assert os.listdir(tmp_path / 'there') == []
    assert (tmp_path / 'here' / 'run' / RUN_FILE).exists()


def test_a_directory_in_the_home_directory_can_be_written_with_a_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))
    monkeypatch.chdir(tmp_path)
    sim = _sim(stream_to='~/run')
    sim.run(**RUN)
    assert (tmp_path / 'run' / RUN_FILE).exists() and not (tmp_path / '~').exists()


def test_a_path_object_works(tmp_path):
    sim = _sim(stream_to=pathlib.Path(tmp_path) / 'run')
    sim.run(**RUN)
    assert (tmp_path / 'run' / RUN_FILE).exists()


@pytest.mark.parametrize("bad, error, match", [
    (5, TypeError, "stream_to must be a path"),
    (b'run', TypeError, "stream_to must be a path"),
    ('', ValueError, "stream_to can't be empty"),
])
def test_stream_to_must_name_a_directory(bad, error, match):
    with pytest.raises(error, match=match):
        Sim(stream_to=bad)


def test_stream_to_naming_a_file_is_refused(tmp_path):
    (tmp_path / 'run').write_text('a file')
    with pytest.raises(NotADirectoryError, match="is a file, not a directory"):
        Sim(stream_to=tmp_path / 'run')


def test_the_options_are_keyword_only(tmp_path):
    with pytest.raises(TypeError):
        Sim(tmp_path)


# --- a run that would overwrite a run is refused (Q4) before anything changes -------- #

def test_a_run_into_a_directory_another_sim_filled_is_refused_and_changes_nothing(tmp_path):
    first, second = _sim(stream_to=tmp_path / 'run'), _sim(stream_to=tmp_path / 'run')
    hook = _Steps()
    second.add_hook(hook)
    hook.steps = ['before']
    solver = second._self_gravity_force
    first.run(**RUN)
    files = _files(tmp_path / 'run')
    with pytest.raises(FileExistsError, match=r"already holds a run \(run.h5\), which this run would overwrite"):
        second.run(**RUN)
    assert _files(tmp_path / 'run') == files
    assert second._self_gravity_force is solver and second.monitor is None
    assert hook.steps == ['before'] and second.hooks == ((hook, second.hooks[0][1]),)
    assert not second._has_run and second._run_settings is None
    np.testing.assert_array_equal(second.times, [0.0])


def test_a_rerun_after_the_outputs_were_cleared_is_still_refused(tmp_path):
    sim = _sim(stream_to=tmp_path / 'run')
    sim.run(**RUN)
    files = _files(tmp_path / 'run')
    sim.eps = 0.3
    with pytest.raises(FileExistsError, match="already holds a run"):
        sim.run(**RUN)
    assert _files(tmp_path / 'run') == files


def test_a_directory_filled_after_the_check_is_still_refused_by_the_writer(tmp_path, monkeypatch):
    sim = _sim(stream_to=tmp_path / 'run')
    sim.run(**RUN)
    files, times, steps, settings = _files(tmp_path / 'run'), sim.times, sim._steps, sim._run_settings
    monkeypatch.setattr(simulation_module, 'existing_run', lambda directory: None)
    with pytest.raises(FileExistsError, match=r"already holds a run \(run.h5\)\.$"):
        sim.run(**RUN)
    assert _files(tmp_path / 'run') == files
    assert sim.times is times and sim._steps is steps and sim._run_settings is settings


# --- every file reaches the disk ----------------------------------------------------- #

def test_every_file_is_pushed_to_the_disk(tmp_path, monkeypatch):
    calls = []
    real = os.fsync
    monkeypatch.setattr(run_files.os, 'fsync', lambda fd: (calls.append(fd), real(fd))[1])
    sim = _sim(stream_to=tmp_path / 'run')
    sim.run(**RUN)
    files = 1 + len(sim.times)
    per_file = 1 if os.name == 'nt' else 2                  # each file, then the directory (not on Windows)
    assert len(calls) == per_file * files
