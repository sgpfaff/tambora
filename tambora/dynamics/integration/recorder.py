from abc import ABC, abstractmethod
import numpy as np


class _Recorder(ABC):
    """Receives a run's outputs as they are made: ``begin`` once, then ``record`` per output."""

    n = 0  # outputs recorded so far
    steps = ()  # each output's step; the first n are recorded

    @abstractmethod
    def begin(self, nsnaps, n_particles, keep_acc, keep_pot):
        """Prepare for up to *nsnaps* outputs of *n_particles*, with or without the self-gravity caches."""

    @abstractmethod
    def record(self, step, t, pos, vel, self_acc, self_pot):
        """Store one output; *self_acc*/*self_pot* may be None (no self-gravity)."""

    @abstractmethod
    def outputs(self):
        """(pos, vel, times, self acc, self pot) of the outputs recorded so far."""

    def abandon(self):
        """Undo ``begin`` for a run that recorded nothing."""


class _MemoryRecorder(_Recorder):
    """A run's outputs in memory, one row per output, filled in as the run makes them."""

    def begin(self, nsnaps, n_particles, keep_acc, keep_pot):
        self.times = np.empty(nsnaps)
        self.steps = np.empty(nsnaps, dtype=np.int64)
        self.pos = np.zeros((nsnaps, n_particles, 3))
        self.vel = np.zeros((nsnaps, n_particles, 3))
        self.acc = np.zeros((nsnaps, n_particles, 3)) if keep_acc else None
        self.pot = np.zeros((nsnaps, n_particles)) if keep_pot else None
        self.n = 0

    def record(self, step, t, pos, vel, self_acc, self_pot):
        i = self.n
        self.times[i], self.steps[i] = t, step
        self.pos[i], self.vel[i] = pos, vel
        if self.acc is not None and self_acc is not None:
            self.acc[i] = self_acc
        if self.pot is not None and self_pot is not None:
            self.pot[i] = self_pot
        self.n += 1

    def outputs(self):
        n = self.n
        return (self.pos[:n], self.vel[:n], self.times[:n],
                None if self.acc is None else self.acc[:n],
                None if self.pot is None else self.pot[:n])
