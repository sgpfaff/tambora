"""The file recorder: a run directory with run.h5 and one HDF5 file per output."""

import datetime
import errno
import os
import signal
import stat
import subprocess
import sys
import time

import h5py
import numpy as np
import pytest

import tambora
from tambora.dynamics import DirectSummationGravity, NullForce, NullSelfGravity
from tambora.dynamics.forces.CompositeForce import _CompositeConservative
from tambora.dynamics.hooks import EveryStep
from tambora.dynamics.integration import _runner
from tambora.dynamics.integration.LeapfrogIntegrator import LeapfrogIntegrator
from tambora.dynamics.integration.recorder import _MemoryRecorder, _Recorder
from tambora.io import run_files
from tambora.io.run_files import UNITS, _FileRecorder, _read_settings, snapshot_name

N = 50
POSIX = sys.platform != 'win32'


def _run(recorder, method='direct', keep_acc=True, keep_pot=True, t_end=1.0, dt=0.01, dt_out=0.1, hooks=()):
    """A 50-particle cluster run to *t_end*, its outputs handed to *recorder*."""
    rng = np.random.default_rng(0)
    pos, vel, mass = rng.normal(size=(N, 3)), rng.normal(size=(N, 3)) * 10, np.full(N, 1e5)
    solver = DirectSummationGravity() if method == 'direct' else NullSelfGravity()
    return _runner(pos, vel, mass, LeapfrogIntegrator(), solver, _CompositeConservative([]), NullForce(),
                   t0=0.0, t_end=t_end, dt=dt, dt_out=dt_out, return_self_gravity_acc=keep_acc,
                   return_self_gravity_pot=keep_pot, hooks=hooks, progress=False, eps=np.full(N, 0.05),
                   recorder=recorder)


def _assert_same_outputs(a, b):
    for x, y in zip(a, b, strict=True):
        if x is None:
            assert y is None
        else:
            assert x.dtype == y.dtype
            np.testing.assert_array_equal(x, y)


def _subprocess_env():
    """The environment for a child Python that imports this tambora."""
    root = os.path.dirname(os.path.dirname(tambora.__file__))
    return {**os.environ, 'PYTHONPATH': os.pathsep.join([root, os.environ.get('PYTHONPATH', '')])}


# --- what is written reads back exactly ------------------------------------------ #

def test_a_file_recorder_is_a_recorder():
    assert isinstance(_FileRecorder('unused'), _Recorder)
    assert _FileRecorder('unused').n == 0


@pytest.mark.parametrize("method, keep_acc, keep_pot", [('direct', True, True), ('direct', False, False),
                                                        ('direct', True, False), ('direct', False, True),
                                                        (None, True, True)])
def test_a_run_written_to_files_reads_back_bit_for_bit(tmp_path, method, keep_acc, keep_pot):
    _assert_same_outputs(_run(_FileRecorder(tmp_path / 'run'), method, keep_acc, keep_pot),
                         _run(_MemoryRecorder(), method, keep_acc, keep_pot))


def test_a_backward_run_reads_back_bit_for_bit(tmp_path):
    backward = dict(t_end=-1.0, dt=-0.01, dt_out=-0.1)
    _assert_same_outputs(_run(_FileRecorder(tmp_path / 'run'), **backward), _run(_MemoryRecorder(), **backward))


def test_missing_self_gravity_arrays_read_back_as_zeros(tmp_path):
    # The recorder interface lets a run hand over None for them.
    file, memory = _FileRecorder(tmp_path), _MemoryRecorder()
    for recorder in (file, memory):
        recorder.begin(1, 2, True, True)
        recorder.record(0, 0.0, np.ones((2, 3)), np.ones((2, 3)), None, None)
    with h5py.File(tmp_path / snapshot_name(0), 'r') as f:
        assert sorted(f) == ['pos', 'vel']
    _assert_same_outputs(file.outputs(), memory.outputs())


# --- the run directory ---------------------------------------------------------------- #

def test_the_directory_holds_run_h5_and_one_file_per_output(tmp_path):
    _run(_FileRecorder(tmp_path / 'run'))
    assert sorted(os.listdir(tmp_path / 'run')) == ['run.h5'] + [snapshot_name(i) for i in range(11)]


@pytest.mark.parametrize("keep_acc, keep_pot", [(True, False), (False, True)])
def test_run_h5_describes_the_run(tmp_path, keep_acc, keep_pot):
    _run(_FileRecorder(tmp_path / 'run', settings={'dt': 0.01}), keep_acc=keep_acc, keep_pot=keep_pot)
    with h5py.File(tmp_path / 'run' / 'run.h5', 'r') as f:
        assert f.attrs['format'] == 'tambora-run'
        assert tuple(f.attrs['format_version']) == (1, 0)
        assert f.attrs['tambora_version'] == tambora.__version__
        assert datetime.datetime.fromisoformat(f.attrs['created']).tzinfo is not None
        assert f.attrs['expected_outputs'] == 11
        assert f.attrs['n_particles'] == N
        assert (f.attrs['keep_self_acc'], f.attrs['keep_self_pot']) == (keep_acc, keep_pot)
        assert dict(f['units'].attrs) == UNITS
        assert _read_settings(f['settings']) == {'dt': 0.01}


@pytest.mark.parametrize("keep", [True, False])
def test_each_snapshot_file_holds_its_index_step_and_time(tmp_path, keep):
    times = _run(_FileRecorder(tmp_path / 'run'), keep_acc=keep, keep_pot=keep)[2]
    for i, t in enumerate(times):
        with h5py.File(tmp_path / 'run' / snapshot_name(i), 'r') as f:
            assert (f.attrs['index'], f.attrs['step'], f.attrs['t']) == (i, 10 * i, t)
            assert tuple(f.attrs['format_version']) == (1, 0)
            expected = {'pos': (N, 3), 'vel': (N, 3)}
            if keep:
                expected.update(self_acc=(N, 3), self_pot=(N,))
            assert {name: f[name].shape for name in f} == expected
            for data in f.values():     # contiguous: one particle is read without decompressing a chunk
                assert (data.dtype, data.chunks, data.compression) == (np.float64, None, None)


def _assert_same_settings(a, b, path='settings'):
    """Equal, with the same types and the same key order, NaN equal to NaN."""
    assert type(a) is type(b), (path, a, b)
    if isinstance(a, dict):
        assert list(a) == list(b), path
        for key in a:
            _assert_same_settings(a[key], b[key], f'{path}/{key}')
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for x, y in zip(a, b):
            _assert_same_settings(x, y, path)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype, path
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, float) and np.isnan(a):
        assert np.isnan(b), path
    else:
        assert a == b, path


def test_settings_read_back_as_written(tmp_path):
    settings = {'method': 'falcON', 'theta': 0.6, 'kernel': 1, 'fsync': True, 'eps': None, 'nan': np.nan,
                'name': np.str_('lmc'), 'greek': 'Ωμ', 'monitors': ('energy',),
                'track': ['energy', 'angular_momentum'], 'pair': (1, 2.5), 'none': (),
                'mass': np.arange(3.0), 'names': np.array(['disk', 'halo']),
                'solver': {'type': 'FalcONGravity', 'inner': {'n': 2}}, 'empty': {}, 'last': 1}
    _FileRecorder(tmp_path, settings=settings).begin(1, 1, False, False)
    with h5py.File(tmp_path / 'run.h5', 'r') as f:
        back = _read_settings(f['settings'])
    _assert_same_settings(back, {
        'method': 'falcON', 'theta': 0.6, 'kernel': 1, 'fsync': True, 'eps': None, 'nan': np.nan,
        'name': 'lmc', 'greek': 'Ωμ', 'monitors': ['energy'],
        'track': ['energy', 'angular_momentum'], 'pair': [1.0, 2.5], 'none': [],
        'mass': np.arange(3.0), 'names': np.array(['disk', 'halo'], dtype=object),
        'solver': {'type': 'FalcONGravity', 'inner': {'n': 2}}, 'empty': {}, 'last': 1})


@pytest.mark.parametrize("key", ['lmc/halo', '', '.', '__order__', 1])
def test_a_settings_key_hdf5_would_misread_is_refused(tmp_path, key):
    with pytest.raises(ValueError, match="can't be a settings key"):
        _FileRecorder(tmp_path, settings={'forces': {key: 1}}).begin(1, 1, False, False)
    assert os.listdir(tmp_path) == []


def test_a_relative_directory_stays_where_it_was_when_the_working_directory_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recorder = _FileRecorder('run')
    (tmp_path / 'elsewhere').mkdir()
    monkeypatch.chdir(tmp_path / 'elsewhere')
    _run(recorder, t_end=0.1)
    assert sorted(os.listdir(tmp_path / 'run')) == ['run.h5', 'snap_000000.h5', 'snap_000001.h5']
    assert os.listdir(tmp_path / 'elsewhere') == []


# --- claiming the directory -------------------------------------------------------------- #

@pytest.mark.parametrize("name", ['run.h5', 'snap_000000.h5'])
def test_a_directory_that_already_holds_a_run_is_refused(tmp_path, name):
    (tmp_path / name).write_bytes(b'earlier run')
    recorder = _FileRecorder(tmp_path)
    with pytest.raises(FileExistsError, match="already holds a run"):
        recorder.begin(1, 1, False, False)
    assert recorder.n == 0                  # the earlier run's files aren't this recorder's outputs
    assert sorted(os.listdir(tmp_path)) == [name]
    assert (tmp_path / name).read_bytes() == b'earlier run'


def test_a_run_into_a_directory_that_holds_a_run_is_refused_before_it_starts(tmp_path):
    _run(_FileRecorder(tmp_path), t_end=0.1)
    before = {name: (tmp_path / name).read_bytes() for name in os.listdir(tmp_path)}
    recorder = _FileRecorder(tmp_path)
    with pytest.raises(FileExistsError):
        _run(recorder, t_end=0.1)
    assert recorder.n == 0
    assert {name: (tmp_path / name).read_bytes() for name in os.listdir(tmp_path)} == before


@pytest.mark.parametrize("hard_links", [True, False])
def test_of_two_runs_starting_together_the_first_to_write_run_h5_keeps_the_directory(tmp_path, monkeypatch,
                                                                                    hard_links):
    first = _FileRecorder(tmp_path, settings={'run': 'first'})
    first.begin(1, 1, False, False)
    before = (tmp_path / 'run.h5').read_bytes()
    if not hard_links:
        def no_hard_links(src, dst):
            raise PermissionError(errno.EPERM, "Operation not permitted")
        monkeypatch.setattr(run_files.os, 'link', no_hard_links)
    monkeypatch.setattr(run_files.os, 'listdir', lambda path: [])   # it checked before run.h5 appeared
    second = _FileRecorder(tmp_path, settings={'run': 'second'})
    with pytest.raises(FileExistsError, match="already holds a run"):
        second.begin(1, 1, False, False)
    monkeypatch.undo()
    assert os.listdir(tmp_path) == ['run.h5']
    assert (tmp_path / 'run.h5').read_bytes() == before


def test_without_hard_links_a_run_still_writes_its_directory(tmp_path, monkeypatch):
    def no_hard_links(src, dst):
        raise PermissionError(errno.EPERM, "Operation not permitted")
    monkeypatch.setattr(run_files.os, 'link', no_hard_links)
    _run(_FileRecorder(tmp_path / 'run'), t_end=0.1)
    assert sorted(os.listdir(tmp_path / 'run')) == ['run.h5', 'snap_000000.h5', 'snap_000001.h5']


@pytest.mark.parametrize("left_over", ['run.h5.tmp', 'run.h5.12345-0badc0de.tmp'])
def test_a_temporary_left_by_a_killed_run_does_not_block_the_next(tmp_path, left_over):
    (tmp_path / left_over).write_bytes(b'half written')
    _run(_FileRecorder(tmp_path), t_end=0.1)
    assert sorted(os.listdir(tmp_path)) == ['run.h5', left_over, 'snap_000000.h5', 'snap_000001.h5']


def test_a_missing_directory_is_created_and_other_files_are_left_alone(tmp_path):
    _run(_FileRecorder(tmp_path / 'a' / 'b'), t_end=0.1)
    assert sorted(os.listdir(tmp_path / 'a' / 'b')) == ['run.h5', 'snap_000000.h5', 'snap_000001.h5']
    (tmp_path / 'c').mkdir()
    (tmp_path / 'c' / 'notes.txt').write_text('mine')
    _run(_FileRecorder(tmp_path / 'c'), t_end=0.1)
    assert (tmp_path / 'c' / 'notes.txt').read_text() == 'mine'


# --- failures and interrupts -------------------------------------------------------------- #

@pytest.mark.parametrize("error", [OSError("disk full"), KeyboardInterrupt()])
def test_a_failed_write_leaves_neither_the_file_nor_a_temporary(tmp_path, monkeypatch, error):
    recorder = _FileRecorder(tmp_path)
    recorder.begin(1, 2, False, False)
    create_dataset = h5py.Group.create_dataset

    def fail_at_vel(group, name, *args, **kwargs):
        if name == 'vel':
            raise error
        return create_dataset(group, name, *args, **kwargs)
    monkeypatch.setattr(h5py.Group, 'create_dataset', fail_at_vel)
    with pytest.raises(type(error)):
        recorder.record(0, 0.0, np.zeros((2, 3)), np.zeros((2, 3)), None, None)
    assert sorted(os.listdir(tmp_path)) == ['run.h5']
    assert recorder.n == 0
    assert [len(a) for a in recorder.outputs()[:3]] == [0, 0, 0]


def test_a_failed_cleanup_does_not_replace_the_original_error(tmp_path, monkeypatch):
    recorder = _FileRecorder(tmp_path)
    recorder.begin(1, 2, False, False)

    def interrupted(group, name, *args, **kwargs):
        raise KeyboardInterrupt

    def read_only(path):
        raise PermissionError(errno.EROFS, "Read-only file system")
    monkeypatch.setattr(h5py.Group, 'create_dataset', interrupted)
    monkeypatch.setattr(run_files.os, 'remove', read_only)
    with pytest.raises(KeyboardInterrupt):
        recorder.record(0, 0.0, np.zeros((2, 3)), np.zeros((2, 3)), None, None)


_FULL_DISK = """
import resource, signal, sys
import numpy as np
from tambora.io.run_files import _FileRecorder
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
recorder = _FileRecorder(sys.argv[1])
recorder.begin(1, 200, False, False)
resource.setrlimit(resource.RLIMIT_FSIZE, (1000, 1000))     # no file may grow past 1000 bytes
try:
    recorder.record(0, 0.0, np.zeros((200, 3)), np.zeros((200, 3)), None, None)
except OSError as e:
    print('OSError', e.errno, recorder.n)
"""


@pytest.mark.skipif(not POSIX, reason="needs RLIMIT_FSIZE")
def test_a_full_disk_raises_an_error_and_leaves_no_temporary(tmp_path):
    # h5py's own file driver segfaults on close here (h5py 3.12, HDF5 1.14).
    result = subprocess.run([sys.executable, '-c', _FULL_DISK, str(tmp_path)], env=_subprocess_env(),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ['OSError', str(errno.EFBIG), '0']
    assert os.listdir(tmp_path) == ['run.h5']


def test_an_interrupt_just_after_a_rename_still_counts_the_file(tmp_path, monkeypatch):
    recorder = _FileRecorder(tmp_path)
    recorder.begin(1, 2, False, False)
    replace = os.replace

    def replace_then_interrupt(src, dst):
        replace(src, dst)
        raise KeyboardInterrupt
    monkeypatch.setattr(run_files.os, 'replace', replace_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        recorder.record(0, 0.0, np.ones((2, 3)), np.ones((2, 3)), None, None)
    assert sorted(os.listdir(tmp_path)) == ['run.h5', 'snap_000000.h5']
    assert recorder.n == 1
    pos, _, times = recorder.outputs()[:3]
    np.testing.assert_array_equal(pos, np.ones((1, 2, 3)))
    np.testing.assert_array_equal(times, [0.0])


def test_a_run_stopped_early_keeps_the_files_written_so_far(tmp_path):
    def interrupt_at_35(state):
        if state.step == 35:
            raise KeyboardInterrupt
    recorder = _FileRecorder(tmp_path / 'run')
    with pytest.raises(KeyboardInterrupt):
        _run(recorder, hooks=[(interrupt_at_35, EveryStep())])
    assert recorder.n == 4
    assert sorted(os.listdir(tmp_path / 'run')) == ['run.h5'] + [snapshot_name(i) for i in range(4)]
    full = _run(_MemoryRecorder())
    _assert_same_outputs(recorder.outputs(), tuple(a[:4] for a in full))


# --- reaching the disk -------------------------------------------------------------------- #

@pytest.mark.parametrize("fsync", [True, False])
def test_each_file_is_whole_and_on_disk_before_its_rename(tmp_path, monkeypatch, fsync):
    directory, events = tmp_path / 'run', []
    os_fsync, os_replace, os_link = os.fsync, os.replace, os.link

    def whole(path):
        with h5py.File(path, 'r', locking=False) as f:
            return 'format' in f.attrs or 'pos' in f

    def logged_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            events.append(('fsync', 'directory'))
        else:
            [tmp] = [p for p in directory.iterdir() if p.name.endswith('.tmp')]
            events.append(('fsync', tmp.name.split('.h5.')[0] + '.h5', whole(tmp)))
        os_fsync(fd)

    def logged(rename):
        def renamed(src, dst):
            events.append(('rename', os.path.basename(dst), whole(src)))
            rename(src, dst)
        return renamed
    monkeypatch.setattr(run_files.os, 'fsync', logged_fsync)
    monkeypatch.setattr(run_files.os, 'replace', logged(os_replace))
    monkeypatch.setattr(run_files.os, 'link', logged(os_link))
    _run(_FileRecorder(directory) if fsync else _FileRecorder(directory, fsync=False), t_end=0.1)
    names = ['run.h5', 'snap_000000.h5', 'snap_000001.h5']
    if fsync:       # the default
        assert events == [e for name in names for e in
                          (('fsync', name, True), ('rename', name, True), ('fsync', 'directory'))]
    else:
        assert events == [('rename', name, True) for name in names]


@pytest.mark.skipif(not POSIX, reason="needs a directory fsync")
@pytest.mark.parametrize("code, raised", [(errno.EINVAL, False), (errno.EIO, True)])
def test_a_directory_that_cant_be_synced_is_skipped_but_a_failing_disk_is_not(tmp_path, monkeypatch,
                                                                              code, raised):
    os_fsync = os.fsync

    def no_directory_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(code, os.strerror(code))
        os_fsync(fd)
    monkeypatch.setattr(run_files.os, 'fsync', no_directory_fsync)
    if raised:
        with pytest.raises(OSError):
            _run(_FileRecorder(tmp_path), t_end=0.1)
    else:
        _run(_FileRecorder(tmp_path), t_end=0.1)
        assert sorted(os.listdir(tmp_path)) == ['run.h5', 'snap_000000.h5', 'snap_000001.h5']


@pytest.mark.skipif(not POSIX, reason="needs flock")
def test_reading_back_takes_no_file_lock(tmp_path):
    # HDF5's locks fail on NFS mounts without a lock manager; finished files never change.
    import fcntl
    recorder = _FileRecorder(tmp_path)
    _run(recorder, t_end=0.1)
    with open(tmp_path / snapshot_name(0), 'rb') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert recorder.outputs()[0][0].shape == (N, 3)


_KILLED_WRITER = """
import sys
import numpy as np
from tambora.dynamics import NullForce, NullSelfGravity
from tambora.dynamics.forces.CompositeForce import _CompositeConservative
from tambora.dynamics.integration import _runner
from tambora.dynamics.integration.LeapfrogIntegrator import LeapfrogIntegrator
from tambora.io.run_files import _FileRecorder
n = 200_000
_runner(np.zeros((n, 3)), np.ones((n, 3)), np.ones(n), LeapfrogIntegrator(), NullSelfGravity(),
        _CompositeConservative([]), NullForce(), t0=0.0, t_end=1e6, dt=1.0, dt_out=1.0,
        return_self_gravity_acc=False, return_self_gravity_pot=False, progress=False,
        recorder=_FileRecorder(sys.argv[1], fsync=False))
"""


@pytest.mark.skipif(not hasattr(signal, 'SIGKILL'), reason="needs SIGKILL")
@pytest.mark.parametrize("files_before_kill", [3, 6])
def test_a_killed_writer_leaves_only_whole_files(tmp_path, files_before_kill):
    with subprocess.Popen([sys.executable, '-c', _KILLED_WRITER, str(tmp_path)], env=_subprocess_env(),
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as writer:
        try:
            deadline = time.monotonic() + 60
            while True:
                names = os.listdir(tmp_path)
                done = sum(f.startswith('snap_') and f.endswith('.h5') for f in names)
                if done >= files_before_kill and any(f.endswith('.tmp') for f in names):
                    break                                       # kill it in the middle of a write
                assert writer.poll() is None, writer.stderr.read().decode()
                assert done < files_before_kill + 20 and time.monotonic() < deadline, \
                    "never caught the writer in the middle of a write"
                time.sleep(0.0005)
        finally:
            writer.send_signal(signal.SIGKILL)
            writer.wait()
    names = sorted(os.listdir(tmp_path))
    snaps = [f for f in names if f.startswith('snap_') and f.endswith('.h5')]
    assert len([f for f in names if f.endswith('.tmp')]) <= 1
    assert snaps == [snapshot_name(i) for i in range(len(snaps))]
    for i, name in enumerate(snaps):
        with h5py.File(tmp_path / name, 'r') as f:
            assert f.attrs['index'] == i and f.attrs['t'] == float(i)
            np.testing.assert_array_equal(f['vel'][...], np.ones((200_000, 3)))
    for name in names:                  # ~10 MB each; pytest keeps its last few temporary directories
        os.remove(tmp_path / name)


def test_abandon_removes_the_run_file_only_while_nothing_is_recorded(tmp_path):
    empty, started = _FileRecorder(tmp_path / 'empty'), _FileRecorder(tmp_path / 'started')
    for recorder in (empty, started):
        recorder.begin(2, 2, False, False)
    started.record(0, 0.0, np.zeros((2, 3)), np.zeros((2, 3)), None, None)
    for recorder in (empty, started):
        recorder.abandon()
    assert os.listdir(tmp_path / 'empty') == []
    assert sorted(os.listdir(tmp_path / 'started')) == ['run.h5', 'snap_000000.h5']


def test_abandon_leaves_a_run_file_the_recorder_did_not_write(tmp_path):
    _run(_FileRecorder(tmp_path), t_end=0.1)
    refused = _FileRecorder(tmp_path)
    with pytest.raises(FileExistsError):
        refused.begin(2, 2, False, False)
    refused.abandon()
    assert sorted(os.listdir(tmp_path)) == ['run.h5', 'snap_000000.h5', 'snap_000001.h5']
