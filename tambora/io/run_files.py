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


def _open_for_reading(path):
    return h5py.File(path, 'r', locking=False)


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

    @property
    def n(self):
        # An interrupt can land between a file's rename and the count: the file on disk counts.
        if self._writing is not None and os.path.exists(self._path(snapshot_name(self._writing))):
            return self._writing + 1
        return self._done

    def begin(self, nsnaps, n_particles, keep_acc, keep_pot):
        os.makedirs(self.directory, exist_ok=True)
        taken = sorted(f for f in os.listdir(self.directory) if f == RUN_FILE or f.startswith('snap_'))
        if taken:
            raise FileExistsError(f"{self.directory!r} already holds a run ({taken[0]}).")
        self.n_particles, self.keep_acc, self.keep_pot = n_particles, keep_acc, keep_pot
        self._done = 0

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

    def record(self, step, t, pos, vel, self_acc, self_pot):
        i = self._writing = self._done

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
        """Every output, read back into memory."""
        n, N = self.n, self.n_particles
        pos, vel, times = np.empty((n, N, 3)), np.empty((n, N, 3)), np.empty(n)
        acc = np.zeros((n, N, 3)) if self.keep_acc else None
        pot = np.zeros((n, N)) if self.keep_pot else None
        for i in range(n):
            with _open_for_reading(self._path(snapshot_name(i))) as f:
                pos[i], vel[i], times[i] = f['pos'][...], f['vel'][...], f.attrs['t']
                if acc is not None and 'self_acc' in f:
                    acc[i] = f['self_acc'][...]
                if pot is not None and 'self_pot' in f:
                    pot[i] = f['self_pot'][...]
        return pos, vel, times, acc, pot

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
