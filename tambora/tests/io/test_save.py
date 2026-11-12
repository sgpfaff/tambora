"""sim.save: a run written to a directory as streaming would have written it."""

import os
import warnings

import h5py
import numpy as np
import pytest
from galpy import potential as gp

from tambora.dynamics.hooks import EveryStep
from tambora.io import run_files
from tambora.io.run_files import RUN_FILE, read_run, snapshot_name
from tambora.simulation import Sim
from tambora.tests.io.test_streaming_sim import RUN, _assert_same_run, _Drag, _files, _sim, _Stop


def _same(a, b):
    """Equal nested dicts, arrays compared by value."""
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind == 'f')
    return type(a) is type(b) and a == b


def _assert_same_files(saved, streamed):
    """*saved* holds what *streamed* holds: run.h5's record, and each snapshot's attributes and arrays."""
    assert _same(read_run(saved), read_run(streamed))
    for i in range(read_run(streamed)['n_outputs']):
        with h5py.File(saved / snapshot_name(i), 'r') as a, h5py.File(streamed / snapshot_name(i), 'r') as b:
            assert _same(dict(a.attrs), dict(b.attrs))
            assert list(a) == list(b)
            assert all(a[k].dtype == b[k].dtype and np.array_equal(a[k][...], b[k][...]) for k in a)


@pytest.mark.parametrize("run", [
    {}, dict(cache_self_gravity_acc=False, cache_self_gravity_pot=False), dict(cache_self_gravity_pot=False),
    dict(method=None), dict(method='falcON', theta=0.4), dict(t_end=-0.2, dt=-0.01, dt_out=-0.05),
    dict(t0=1.5, t_end=1.73), dict(monitors=False),
], ids=['default', 'no caches', 'no pot', 'no self-gravity', 'falcON', 'backward', 'later t0, uneven end',
        'no monitor'])
def test_a_saved_run_is_what_streaming_writes(tmp_path, run):
    memory, disk = _sim(), _sim(stream_to=tmp_path / 'streamed')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')     # the uneven end
        memory.run(**{**RUN, **run})
        disk.run(**{**RUN, **run})
    memory.save(tmp_path / 'saved')
    _assert_same_files(tmp_path / 'saved', tmp_path / 'streamed')
    with warnings.catch_warnings():
        warnings.simplefilter('error')      # a finished run loads without warnings
        _assert_same_run(Sim.load(tmp_path / 'saved'), memory)


def test_saving_a_streamed_or_loaded_sim_copies_its_run(tmp_path):
    streamed = _sim(stream_to=tmp_path / 'run')
    streamed.run(**RUN)
    streamed.save(tmp_path / 'copy')
    Sim.load(tmp_path / 'run').save(str(tmp_path / 'copy of loaded'))
    for copy in ('copy', 'copy of loaded'):
        _assert_same_files(tmp_path / copy, tmp_path / 'run')


def test_save_takes_a_path_as_sim_does(tmp_path, monkeypatch):
    sim = _sim()
    sim.run(**RUN)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))
    monkeypatch.chdir(tmp_path)
    sim.save('~/saved')
    _assert_same_run(Sim.load(tmp_path / 'saved'), sim)
    assert not (tmp_path / '~').exists()


def test_a_saved_run_that_stopped_early_loads_with_the_warning(tmp_path):
    sim = _sim()
    sim.add_hook(_Stop(at=12), EveryStep())
    with pytest.warns(UserWarning, match="stopped early"), pytest.raises(RuntimeError, match="stopped by the test"):
        sim.run(**RUN)
    sim.save(tmp_path / 'saved')
    with pytest.warns(UserWarning, match="holds 3 of the 5 outputs its run was to make"):
        loaded = Sim.load(tmp_path / 'saved')
    _assert_same_run(loaded, sim)


def test_a_saved_run_lists_its_forces(tmp_path):
    sim = _sim()
    sim.add_external_force(_Drag())
    sim.add_external_pot(gp.NFWPotential(ro=8, vo=220))
    sim.run(**RUN)
    sim.save(tmp_path / 'saved')
    with pytest.warns(UserWarning, match=r"ExternalPotential\(NFWPotential\), _Drag couldn't be loaded"):
        Sim.load(tmp_path / 'saved')


# --- what save refuses, and a save that stops partway -------------------------------- #

def test_a_sim_that_hasnt_run_isnt_saved(tmp_path):
    sim = _sim()
    with pytest.raises(RuntimeError, match="hasn't run. There's nothing to save"):
        sim.save(tmp_path / 'saved')
    sim.run(**RUN)
    sim.eps = 0.3                                   # clears the outputs
    with pytest.raises(RuntimeError, match="hasn't run"):
        sim.save(tmp_path / 'saved')
    assert not (tmp_path / 'saved').exists()


def test_a_directory_holding_a_run_is_refused_and_left_alone(tmp_path):
    _sim(stream_to=tmp_path / 'run').run(**RUN)
    files = _files(tmp_path / 'run')
    sim = _sim()
    sim.run(**RUN)
    with pytest.raises(FileExistsError, match=r"already holds a run \(run.h5\)"):
        sim.save(tmp_path / 'run')
    assert _files(tmp_path / 'run') == files


def test_a_file_is_refused(tmp_path):
    sim = _sim()
    sim.run(**RUN)
    (tmp_path / 'file').write_text('')
    with pytest.raises(NotADirectoryError, match="path=.* is a file, not a directory"):
        sim.save(tmp_path / 'file')


@pytest.mark.parametrize("stop", ['at the first output', 'at the third output', 'just after a rename'])
def test_a_save_that_stops_partway_removes_what_it_wrote(tmp_path, monkeypatch, stop):
    sim = _sim()
    sim.run(**RUN)
    (tmp_path / 'saved').mkdir()
    (tmp_path / 'saved' / 'notes.txt').write_text('mine')
    if stop == 'just after a rename':           # the third snapshot has its name, but isn't counted yet
        calls, fsync_directory = [], run_files._fsync_directory

        def interrupted(path):
            calls.append(path)
            if len(calls) == 4:                 # run.h5, then snapshots 0, 1, 2
                raise KeyboardInterrupt
            fsync_directory(path)
        monkeypatch.setattr(run_files, '_fsync_directory', interrupted)
    else:
        stop_at, record = (0 if stop == 'at the first output' else 2), run_files._FileRecorder.record

        def interrupted(self, step, *args):
            if step == stop_at * 5:
                raise KeyboardInterrupt
            record(self, step, *args)
        monkeypatch.setattr(run_files._FileRecorder, 'record', interrupted)
    with pytest.raises(KeyboardInterrupt):
        sim.save(tmp_path / 'saved')
    assert os.listdir(tmp_path / 'saved') == ['notes.txt']
    monkeypatch.undo()
    sim.save(tmp_path / 'saved')                    # the directory can be used again
    _assert_same_run(Sim.load(tmp_path / 'saved'), sim)
    assert (tmp_path / 'saved' / 'notes.txt').read_text() == 'mine'
