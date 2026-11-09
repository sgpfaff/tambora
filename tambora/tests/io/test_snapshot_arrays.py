"""Arrays over a run's snapshot files: read on demand, indexed exactly like the in-memory arrays."""

import copy
import gc
import inspect
import multiprocessing
import pickle

import h5py
import numpy as np
import pytest

import tambora.simulation.simulation as simulation_module
from tambora.dynamics import DirectSummationGravity, NullForce, NullSelfGravity
from tambora.dynamics.forces.CompositeForce import _CompositeConservative
from tambora.dynamics.forces.external_force import ExternalConservativeForce
from tambora.dynamics.integration import _runner
from tambora.dynamics.integration.LeapfrogIntegrator import LeapfrogIntegrator
from tambora.dynamics.integration.recorder import _MemoryRecorder
from tambora.io import run_files
from tambora.io.run_files import _FileRecorder, _SnapshotArray, snapshot_name
from tambora.simulation import Sim

N, NSNAPS = 50, 11


def _run(recorder):
    rng = np.random.default_rng(0)
    pos, vel, mass = rng.normal(size=(N, 3)), rng.normal(size=(N, 3)) * 10, np.full(N, 1e5)
    return _runner(pos, vel, mass, LeapfrogIntegrator(), DirectSummationGravity(), _CompositeConservative([]),
                   NullForce(), t0=0.0, t_end=1.0, dt=0.01, dt_out=0.1, progress=False, eps=np.full(N, 0.05),
                   recorder=recorder)


@pytest.fixture(scope='module')
def outputs(tmp_path_factory):
    """(on disk, in memory): pos, vel, times, self acc and self pot of the same run."""
    return _run(_FileRecorder(tmp_path_factory.mktemp('run'))), _run(_MemoryRecorder())


KEYS = [0, 3, -1, -11, np.int64(2), ..., (...,), slice(None), slice(2, 5), slice(None, None, 3),
        slice(None, None, -2), slice(5, 2), slice(3, 3), [0, 3, 3, 10], np.array([1, 0]), np.arange(NSNAPS) % 2 == 0,
        (0, slice(10, 20)), (slice(None), slice(10, 20)), (-1, 5), (slice(None), -5), (2, [1, 4, 2]),
        (2, np.arange(N) % 3 == 0), (0, slice(10, 10)), (0, []), ([0, 2], 5), ([0, 2], slice(3, 7)),
        ([1, 1], [2, 3]), (slice(8, 1, -3), slice(40, 5, -7)), (..., slice(1, 3)), (slice(1, 4), ...),
        (1, 2, ...), (..., 1, 2), ([0, 1], ..., [1, 2])]
KEYS_3D = KEYS + [(slice(1, 4), slice(None), 0), (..., 0), (2, ..., 1), (0, slice(None), [0, 2]),
                  (0, [1, 2], [0, 1]), (slice(None), [3, 1], 2), (slice(8, 1, -3), slice(40, 5, -7), 1),
                  ([0, 2], slice(None), [1, 1]), (slice(None), 0, ..., [2]), (slice(None), [1, 3], ..., [0, 2]),
                  (0, 0, 0, ...)]
BAD_KEYS = [11, -12, (0, 50), (0, -51), 1.5, (0, 0, 0, 0), [11], (0, [50]), (..., ...), (0, ..., ...)]


@pytest.mark.parametrize("key", KEYS_3D, ids=repr)
def test_a_3d_array_indexes_like_numpy(outputs, key):
    on_disk, in_memory = outputs
    for lazy, array in ((on_disk[0], in_memory[0]), (on_disk[3], in_memory[3])):
        got, want = lazy[key], array[key]
        assert isinstance(got, np.ndarray) and got.shape == want.shape and got.dtype == want.dtype
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("key", KEYS, ids=repr)
def test_a_2d_array_indexes_like_numpy(outputs, key):
    on_disk, in_memory = outputs
    got, want = on_disk[4][key], in_memory[4][key]
    assert got.shape == want.shape and got.dtype == want.dtype
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("key", BAD_KEYS, ids=repr)
def test_an_index_numpy_refuses_is_refused(outputs, key):
    on_disk, in_memory = outputs
    with pytest.raises(IndexError):
        in_memory[0][key]
    with pytest.raises(IndexError):
        on_disk[0][key]


def test_a_new_axis_is_refused(outputs):
    with pytest.raises(IndexError, match="newaxis"):
        outputs[0][0][None]


@pytest.mark.parametrize("key", [True, False, np.True_, (0, False), (..., True),
                                 np.ones((NSNAPS, N), bool), [[True] * N] * NSNAPS], ids=repr)
def test_boolean_scalars_and_masks_over_several_axes_are_refused(outputs, key):
    with pytest.raises(IndexError, match="Boolean"):
        outputs[0][0][key]


def test_it_looks_like_the_array_it_stands_for(outputs):
    on_disk, in_memory = outputs
    for lazy, array in zip(on_disk, in_memory):
        if isinstance(lazy, _SnapshotArray):
            assert (lazy.shape, lazy.ndim, lazy.dtype, len(lazy)) == (array.shape, array.ndim, array.dtype, len(array))
            np.testing.assert_array_equal(np.asarray(lazy), array)
            assert np.asarray(lazy, dtype=np.float32).dtype == np.float32
    assert type(on_disk[2]) is np.ndarray       # times stay in memory
    np.testing.assert_array_equal(on_disk[2], in_memory[2])


def test_the_times_in_memory_are_the_times_in_the_files(tmp_path):
    times = _run(_FileRecorder(tmp_path))[2]
    for i, t in enumerate(times):
        with h5py.File(tmp_path / snapshot_name(i), 'r') as f:
            assert f.attrs['t'] == t


# --- reading on demand ---------------------------------------------------------------- #

def _count_opens(monkeypatch):
    opened = []
    open_for_reading = run_files._open_for_reading

    def counting(path):
        opened.append(path.rsplit('/', 1)[-1])
        return open_for_reading(path)
    monkeypatch.setattr(run_files, '_open_for_reading', counting)
    return opened


def _log_reads(monkeypatch):
    read = []
    dataset_getitem = h5py.Dataset.__getitem__

    def logged(dataset, selection, *args, **kwargs):
        read.append((dataset.name, selection))
        return dataset_getitem(dataset, selection, *args, **kwargs)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', logged)
    return read


def test_nothing_is_read_until_an_array_is_indexed_and_then_only_what_it_selects(tmp_path, monkeypatch):
    recorder = _FileRecorder(tmp_path)
    _run(recorder)
    opened, read = _count_opens(monkeypatch), _log_reads(monkeypatch)
    pos = recorder.outputs()[0]
    assert opened == [] and read == []
    pos[3, 7]
    assert opened == [snapshot_name(3)] and read == [('/pos', slice(7, 8))]
    opened.clear()
    pos[:, 10:20]
    assert opened == [snapshot_name(i) for i in range(NSNAPS)] and read[-1] == ('/pos', slice(10, 20))


@pytest.mark.parametrize("key, snapshots, selection", [
    ((2, slice(10, 20, 3)), [2], slice(10, 20, 3)),
    ((2, slice(19, 9, -3)), [2], slice(10, 20, 3)),
    (([4, 1, 4], 5), [1, 4], slice(5, 6)),
    ((np.arange(NSNAPS) == 6, slice(None, 5)), [6], slice(0, 5)),
], ids=repr)
def test_a_strided_or_repeated_index_reads_each_snapshot_once_and_only_its_particles(tmp_path, monkeypatch, key,
                                                                                    snapshots, selection):
    recorder = _FileRecorder(tmp_path)
    _run(recorder)
    opened, read = _count_opens(monkeypatch), _log_reads(monkeypatch)
    recorder.outputs()[0][key]
    assert opened == [snapshot_name(i) for i in snapshots]
    assert read == [('/pos', selection)] * len(snapshots)


def test_an_output_written_again_after_a_failed_write_keeps_its_own_time(tmp_path, monkeypatch):
    recorder = _FileRecorder(tmp_path)
    recorder.begin(2, 2, False, False)
    replace = run_files.os.replace

    def fail_once(src, dst):
        monkeypatch.setattr(run_files.os, 'replace', replace)
        raise OSError("disk full")
    monkeypatch.setattr(run_files.os, 'replace', fail_once)
    with pytest.raises(OSError):
        recorder.record(0, 0.0, np.zeros((2, 3)), np.zeros((2, 3)), None, None)
    recorder.record(0, 0.5, np.zeros((2, 3)), np.zeros((2, 3)), None, None)
    recorder.record(1, 1.0, np.zeros((2, 3)), np.zeros((2, 3)), None, None)
    np.testing.assert_array_equal(recorder.outputs()[2], [0.5, 1.0])


def test_a_snapshot_held_open_with_h5py_can_still_be_read(tmp_path):
    recorder = _FileRecorder(tmp_path)
    _run(recorder)
    with h5py.File(tmp_path / snapshot_name(3), 'r') as mine:
        np.testing.assert_array_equal(recorder.outputs()[0][3], mine['pos'][...])


def test_every_file_a_read_opens_is_closed_again_also_when_the_read_fails(tmp_path, monkeypatch):
    recorder = _FileRecorder(tmp_path)
    _run(recorder)
    with h5py.File(tmp_path / snapshot_name(5), 'a') as f:
        del f['pos']
    files = []

    def tracked(*args, **kwargs):
        files.append(open(*args, **kwargs))
        return files[-1]
    monkeypatch.setattr(run_files, 'open', tracked, raising=False)
    open_hdf5_files = h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_FILE)
    pos = recorder.outputs()[0]
    pos[:5, 7]
    with pytest.raises(KeyError):
        pos[:, 7]
    assert len(files) == 11 and all(f.closed for f in files)
    assert h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_FILE) == open_hdf5_files


def _read_shared_snapshot(args):
    """In a worker: many reads of the snapshot every worker reads; how many came back wrong."""
    pos, expected, seed = args
    rng = np.random.default_rng(seed)
    return sum(not np.array_equal(pos[0, j], expected[j]) for j in rng.integers(0, len(expected), 2000))


@pytest.mark.skipif('fork' not in multiprocessing.get_all_start_methods(), reason="needs fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded, use of fork:DeprecationWarning")
def test_forked_workers_read_the_same_snapshot_correctly(tmp_path):
    # A file the parent held open while forking would share its read position with the workers.
    # 20000 particles: more than HDF5 keeps of a dataset in memory, so every read goes to the file.
    n = 20_000
    recorder = _FileRecorder(tmp_path, fsync=False)
    _runner(np.random.default_rng(0).normal(size=(n, 3)), np.ones((n, 3)), np.ones(n), LeapfrogIntegrator(),
            NullSelfGravity(), _CompositeConservative([]), NullForce(), t0=0.0, t_end=0.01, dt=0.01, dt_out=0.01,
            return_self_gravity_acc=False, return_self_gravity_pot=False, progress=False, recorder=recorder)
    pos = recorder.outputs()[0]
    expected = pos[0]
    with multiprocessing.get_context('fork').Pool(4) as pool:
        wrong = pool.map(_read_shared_snapshot, [(pos, expected, seed) for seed in range(8)], chunksize=1)
    assert wrong == [0] * 8


def test_a_sim_on_disk_can_be_pickled_and_copied_after_it_was_read(sims):
    on_disk, in_memory = sims
    on_disk.a.pos(t=3)
    for clone in (pickle.loads(pickle.dumps(on_disk)), copy.deepcopy(on_disk)):
        np.testing.assert_array_equal(clone.a.pos(), in_memory.a.pos())


def test_a_snapshot_missing_its_positions_is_an_error_not_zeros(tmp_path):
    recorder = _FileRecorder(tmp_path)
    _run(recorder)
    with h5py.File(tmp_path / snapshot_name(2), 'a') as f:
        del f['pos']
    with pytest.raises(KeyError, match="snap_000002.h5 .* has no 'pos'"):
        recorder.outputs()[0][2]


def test_a_snapshot_without_a_cache_reads_as_zeros(tmp_path):
    recorder = _FileRecorder(tmp_path)
    recorder.begin(2, 2, True, True)
    recorder.record(0, 0.0, np.ones((2, 3)), np.ones((2, 3)), np.ones((2, 3)), np.ones(2))
    recorder.record(1, 0.1, np.ones((2, 3)), np.ones((2, 3)), None, None)
    acc, pot = recorder.outputs()[3:]
    np.testing.assert_array_equal(acc[:, 1], [[1, 1, 1], [0, 0, 0]])
    np.testing.assert_array_equal(pot[1], [0, 0])


# --- a Sim whose run is on disk reads exactly what it would from memory --------------- #

class _Halo(ExternalConservativeForce):
    """A softened point mass, so the accessors that need an external force have one."""
    def acc(self, pos, t):
        r2 = np.sum(pos**2, axis=1, keepdims=True) + 1.0
        return -1e4 * pos / r2**1.5

    def potential(self, pos, t):
        return -1e4 / np.sqrt(np.sum(pos**2, axis=1) + 1.0)


def _sim():
    rng = np.random.default_rng(1)
    sim = Sim()
    sim.add_particles(rng.normal(size=(30, 3)) + 5, rng.normal(size=(30, 3)) * 20, np.full(30, 1e5), name='a', eps=0.05)
    sim.add_particles(rng.normal(size=(20, 3)) - 5, rng.normal(size=(20, 3)) * 20, np.full(20, 2e5), name='b', eps=0.1)
    sim.add_external_force(_Halo())
    return sim


@pytest.fixture(scope='module')
def sims(tmp_path_factory):
    run = dict(t_end=0.5, dt=0.01, dt_out=0.05, method='direct', progress=False)
    on_disk, in_memory = _sim(), _sim()
    directory = tmp_path_factory.mktemp('sim_run')
    memory_recorder = simulation_module._MemoryRecorder
    simulation_module._MemoryRecorder = lambda: _FileRecorder(directory)
    try:
        on_disk.run(**run)
    finally:
        simulation_module._MemoryRecorder = memory_recorder
    in_memory.run(**run)
    assert isinstance(on_disk._positions, _SnapshotArray)
    return on_disk, in_memory


def _accessors(view):
    """The public methods of *view* that take a time t."""
    return [name for name, member in inspect.getmembers(type(view), callable)
            if not name.startswith('_') and 't' in inspect.signature(member).parameters]


def _same(a, b):
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc')


def _call(method, t):
    try:
        return 'returned', method(t=t)
    except Exception as e:
        return 'raised', type(e)


@pytest.mark.parametrize("view", ['sim', 'a', 'b'])
@pytest.mark.parametrize("t", [..., 0, 4, -1, 0.35], ids=repr)
def test_every_accessor_gives_the_same_from_disk_as_from_memory(sims, view, t):
    on_disk, in_memory = sims
    disk_view, memory_view = (on_disk, in_memory) if view == 'sim' else (getattr(on_disk, view), getattr(in_memory, view))
    names = _accessors(memory_view)
    assert len(names) > 20
    differ = []
    for name in names:
        disk, memory = _call(getattr(disk_view, name), t), _call(getattr(memory_view, name), t)
        if disk[0] != memory[0] or not (_same(disk[1], memory[1]) if disk[0] == 'returned' else disk[1] is memory[1]):
            differ.append(name)
    assert differ == []


def test_a_sim_run_to_disk_reports_its_run_like_one_in_memory(sims):
    on_disk, in_memory = sims
    np.testing.assert_array_equal(on_disk.times, in_memory.times)
    assert repr(on_disk) == repr(in_memory)
