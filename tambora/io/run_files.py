"""The run directory: ``run.h5`` with the run's settings, then ``snap_NNNNNN.h5`` per output.

Every file is written under a temporary name and renamed when complete, so a
file with its final name is always whole.
"""

import contextlib
import datetime
import errno
import os
import secrets

import h5py
import numpy as np

from ..dynamics.integration.recorder import _Recorder
from ..version import version as _tambora_version

FORMAT = "tambora-run"
FORMAT_VERSION = (1, 0)
UNITS = {'length': 'kpc', 'velocity': 'kpc/Gyr', 'mass': 'Msun', 'time': 'Gyr',
         'acceleration': 'kpc/Gyr^2', 'potential': 'kpc^2/Gyr^2'}
RUN_FILE = "run.h5"
_ORDER = '__order__'    # the keys of a settings group, in order


def snapshot_name(index):
    return f"snap_{index:06d}.h5"


def existing_run(directory):
    """The first of a run's files (``run.h5``, ``snap_*``) in *directory*, or None."""
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return None
    taken = sorted(f for f in names if f == RUN_FILE or f.startswith('snap_'))
    return taken[0] if taken else None


@contextlib.contextmanager
def _open_for_reading(path):
    """*path* opened for reading, through a Python file."""
    with open(path, 'rb') as raw, h5py.File(raw, 'r') as f:
        yield f


class _FileRecorder(_Recorder):
    """A run's outputs written to *directory*: ``run.h5`` at the start, then one file per output.

    *settings* is a nested dict saved under ``run.h5``'s ``settings`` group (see
    ``_write_settings``). With *fsync*, each file reaches the disk before it gets
    its final name.
    """

    def __init__(self, directory, settings=None, fsync=True):
        self.directory = os.path.abspath(os.fspath(directory))
        self.settings = {} if settings is None else settings
        self.fsync = fsync
        self._done = 0
        self._writing = None    # index of the file being written
        self._claimed = False   # run.h5 is this recorder's
        self._times = []
        self.steps = []

    @property
    def n(self):
        # An interrupt can land between a file's rename and the count: the file on disk counts.
        if self._writing is not None and os.path.exists(self._path(snapshot_name(self._writing))):
            return self._writing + 1
        return self._done

    def begin(self, nsnaps, n_particles, keep_acc, keep_pot):
        os.makedirs(self.directory, exist_ok=True)
        taken = existing_run(self.directory)
        if taken:
            raise FileExistsError(f"{self.directory!r} already holds a run ({taken}).")
        self.n_particles, self.keep_acc, self.keep_pot = n_particles, keep_acc, keep_pot
        self._done, self._times, self.steps = 0, [], []

        def fill(f):
            f.attrs['format'] = FORMAT
            f.attrs['format_version'] = FORMAT_VERSION
            f.attrs['tambora_version'] = _tambora_version
            f.attrs['created'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
            f.attrs['expected_outputs'] = nsnaps
            f.attrs['n_particles'] = n_particles
            f.attrs['keep_self_acc'] = keep_acc
            f.attrs['keep_self_pot'] = keep_pot
            f.create_group('units').attrs.update(UNITS)
            _write_settings(f.create_group('settings'), self.settings)
        try:
            self._write_atomically(RUN_FILE, fill, exclusive=True)
        except FileExistsError:     # another run claimed the directory after the check above
            raise FileExistsError(f"{self.directory!r} already holds a run ({RUN_FILE}).") from None
        self._claimed = True

    def abandon(self):
        """Remove ``run.h5`` if this recorder wrote it and no output was recorded."""
        if self._claimed and self.n == 0:
            with contextlib.suppress(OSError):
                os.remove(self._path(RUN_FILE))
            self._claimed = False

    def record(self, step, t, pos, vel, self_acc, self_pot):
        i = self._writing = self._done
        self._times[i:], self.steps[i:] = [t], [step]

        def fill(f):
            f.attrs['format_version'] = FORMAT_VERSION
            f.attrs['index'] = i
            f.attrs['step'] = step
            f.attrs['t'] = t
            f.create_dataset('pos', data=pos)
            f.create_dataset('vel', data=vel)
            if self.keep_acc and self_acc is not None:
                f.create_dataset('self_acc', data=self_acc)
            if self.keep_pot and self_pot is not None:
                f.create_dataset('self_pot', data=self_pot)
        self._write_atomically(snapshot_name(i), fill)
        self._done, self._writing = i + 1, None

    def outputs(self):
        """The outputs recorded so far, as arrays that read their snapshot files when indexed."""
        n = self.n
        pos, vel, acc, pot = _snapshot_arrays(self.directory, n, self.n_particles, self.keep_acc, self.keep_pot)
        return pos, vel, np.array(self._times[:n], dtype=np.float64), acc, pot

    def _path(self, name):
        return os.path.join(self.directory, name)

    def _write_atomically(self, name, fill, exclusive=False):
        """Write *name* through *fill(h5file)* under a temporary name, then rename it.

        With *exclusive*, raise FileExistsError rather than replace an existing *name*.
        """
        final = self._path(name)
        tmp = self._path(f"{name}.{os.getpid()}-{secrets.token_hex(4)}.tmp")
        try:
            with open(tmp, 'x+b') as out:
                with h5py.File(out, 'w') as f:
                    fill(f)
                if self.fsync:
                    out.flush()
                    os.fsync(out.fileno())
            if exclusive:
                _rename_exclusive(tmp, final)
            else:
                os.replace(tmp, final)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(tmp)
            raise
        if self.fsync:
            _fsync_directory(self.directory)


def read_run(directory):
    """What ``Sim.load`` needs from a run directory, without reading snapshot data.

    A dict: ``run.h5``'s ``settings``, ``expected_outputs``, ``n_particles``,
    ``keep_self_acc`` and ``keep_self_pot``; ``n_outputs``, the snapshot files
    written, counted from the first; and ``last``, the step and time of the last
    of them (None if there's none).
    """
    if os.path.isfile(directory):
        raise NotADirectoryError(f"{directory!r} is a file. Pass the run's directory.")
    path = os.path.join(directory, RUN_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{directory!r} holds no run. It has no {RUN_FILE}.")
    with _open_for_reading(path) as f:
        if f.attrs.get('format') != FORMAT:
            raise ValueError(f"{path!r} isn't a tambora run file.")
        major, minor = (int(v) for v in f.attrs['format_version'])
        if major != FORMAT_VERSION[0]:
            raise ValueError(f"{path!r} has format version {major}.{minor}; "
                             f"this tambora reads {FORMAT_VERSION[0]}.x.")
        run = {key: f.attrs[key].item()
               for key in ('expected_outputs', 'n_particles', 'keep_self_acc', 'keep_self_pot')}
        run['settings'] = _read_settings(f['settings'])
    names = set(os.listdir(directory))
    n = 0
    while snapshot_name(n) in names:
        n += 1
    run['n_outputs'], run['last'] = n, None
    if n:
        with _open_for_reading(os.path.join(directory, snapshot_name(n - 1))) as f:
            run['last'] = (int(f.attrs['step']), float(f.attrs['t']))
    return run


def _snapshot_arrays(directory, n, n_particles, keep_acc, keep_pot):
    """pos, vel, self_acc and self_pot over the first *n* snapshot files; None for a cache not kept."""
    acc = _SnapshotArray(directory, 'self_acc', n, (n_particles, 3)) if keep_acc else None
    pot = _SnapshotArray(directory, 'self_pot', n, (n_particles,)) if keep_pot else None
    pos = _SnapshotArray(directory, 'pos', n, (n_particles, 3))
    vel = _SnapshotArray(directory, 'vel', n, (n_particles, 3))
    return pos, vel, acc, pot


class _SnapshotArray:
    """One dataset (pos, vel, self_acc or self_pot) across a run's snapshot files, read on demand.

    Indexes like the (n_snapshots, N, ...) array it stands for and returns numpy
    arrays, reading only the snapshots, and the particles in them, that the index
    selects, each file opened only for the read. np.newaxis, boolean scalars and
    boolean masks over several axes are refused. A snapshot without self_acc or
    self_pot reads as zeros.
    """

    dtype = np.dtype(np.float64)

    def __init__(self, directory, name, n, row_shape):
        self.directory, self.name = directory, name
        self.shape = (n,) + tuple(row_shape)

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return f"<{type(self).__name__} {self.name!r} {self.shape} in {self.directory!r}>"

    def __array__(self, dtype=None, copy=None):
        a = self[...]
        return a if dtype is None else a.astype(dtype)

    def __getitem__(self, key):
        key, ellipsis = _expand_index(key, self.ndim)
        snapshots, snapshot_key = _rows(key[0], self.shape[0])
        particles, particle_key = _block(key[1], self.shape[1])
        needed = np.empty((len(snapshots), len(range(self.shape[1])[particles])) + self.shape[2:], self.dtype)
        for j, i in enumerate(snapshots):
            with _open_for_reading(os.path.join(self.directory, snapshot_name(i))) as f:
                if self.name in f:
                    needed[j] = f[self.name][particles]
                elif self.name in ('self_acc', 'self_pot'):
                    needed[j] = 0.0
                else:
                    raise KeyError(f"{snapshot_name(i)} in {self.directory!r} has no {self.name!r}.")
        final = (snapshot_key, particle_key) + key[2:]
        if ellipsis is not None:        # put it back: a zero-width ... still matters to numpy
            final = final[:ellipsis[0]] + (Ellipsis,) + final[ellipsis[1]:]
        return needed[final]


def _expand_index(key, ndim):
    """*key* as one index per axis, ``...`` expanded as numpy does, and the axes the ``...`` covered."""
    key = key if isinstance(key, tuple) else (key,)
    for k in key:
        if k is None:
            raise IndexError("np.newaxis isn't supported here.")
        if isinstance(k, (bool, np.bool_)) or (isinstance(k, (list, np.ndarray))
                                               and np.asarray(k).dtype == bool and np.ndim(k) != 1):
            raise IndexError("Boolean scalars and boolean masks over several axes aren't supported here.")
    ellipses = [i for i, k in enumerate(key) if k is Ellipsis]
    if len(ellipses) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if len(key) - len(ellipses) > ndim:
        raise IndexError(f"too many indices: the array has {ndim} dimensions, "
                         f"{len(key) - len(ellipses)} were indexed")
    covered = None
    if ellipses:
        i, width = ellipses[0], ndim - len(key) + 1
        key = key[:i] + (slice(None),) * width + key[i + 1:]
        covered = (i, i + width)
    return key + (slice(None),) * (ndim - len(key)), covered


def _rows(index, length):
    """The snapshots *index* selects, each read once, and *index* rewritten to point into them."""
    if isinstance(index, (int, np.integer)):
        return [range(length)[index]], 0
    if isinstance(index, slice):
        return list(range(length)[index]), slice(None)
    chosen = np.arange(length)[index]
    rows = np.unique(chosen)
    return rows.tolist(), np.searchsorted(rows, chosen)


def _block(index, length):
    """The particles to read, as a slice with a positive step, and *index* rewritten to point into them."""
    if isinstance(index, (int, np.integer)):
        i = range(length)[index]
        return slice(i, i + 1), 0
    if isinstance(index, slice):
        r = range(length)[index]
        if not r:
            return slice(0, 0), slice(None)
        step = abs(r.step) if abs(r.step) != 1 else None
        if r.step > 0:
            return slice(r.start, r[-1] + 1, step), slice(None)
        return slice(r[-1], r.start + 1, step), slice(None, None, -1)
    chosen = np.arange(length)[index]
    if chosen.size == 0:
        return slice(0, 0), chosen
    lo = int(chosen.min())
    return slice(lo, int(chosen.max()) + 1), chosen - lo


def _rename_exclusive(tmp, final):
    try:
        os.link(tmp, final)     # fails if final exists, unlike a rename
    except FileExistsError:
        raise
    except OSError:             # no hard links on this filesystem
        if os.path.exists(final):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), final) from None
        os.replace(tmp, final)
        return
    os.remove(tmp)


def _fsync_directory(path):
    """Make the renames in *path* durable, where the filesystem can."""
    if os.name == 'nt':         # Windows can't open a directory
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as e:
        if e.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
            raise
    finally:
        os.close(fd)


def _write_settings(group, settings):
    """Write a nested dict, keys in order: dicts become groups, arrays datasets, the rest attributes.

    Attributes hold str, bool, int, float, None, or a tuple/list of str or numbers,
    which comes back as a list.
    """
    for key in settings:
        if not isinstance(key, str) or key in ('', '.', _ORDER) or '/' in key:
            raise ValueError(f"{key!r} can't be a settings key (in {group.name!r}).")
    group.attrs[_ORDER] = np.array(list(settings), dtype=h5py.string_dtype())
    for key, value in settings.items():
        if isinstance(value, dict):
            _write_settings(group.create_group(key), value)
        elif isinstance(value, np.ndarray):
            strings = value.dtype.kind in 'UO'
            group.create_dataset(key, data=value.astype(h5py.string_dtype()) if strings else value)
        elif value is None:
            group.attrs[key] = h5py.Empty('f8')
        elif isinstance(value, (tuple, list)):
            strings = value and all(isinstance(v, str) for v in value)
            group.attrs[key] = np.array(value, dtype=h5py.string_dtype() if strings else None)
        elif isinstance(value, str):
            group.attrs[key] = str(value)        # also np.str_
        else:
            group.attrs[key] = value


def _read_settings(group):
    """The nested dict ``_write_settings`` wrote."""
    settings = {}
    for key in group.attrs[_ORDER].tolist():
        if key in group:
            item = group[key]
            if isinstance(item, h5py.Group):
                settings[key] = _read_settings(item)
            else:
                settings[key] = item.asstr()[...] if h5py.check_string_dtype(item.dtype) else item[...]
        else:
            value = group.attrs[key]
            if isinstance(value, h5py.Empty):
                value = None
            elif isinstance(value, (np.ndarray, np.generic)):
                value = value.tolist()
            settings[key] = value
    return settings
