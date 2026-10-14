from abc import ABC, abstractmethod
from ..integration import StepState
from .cadence import EveryOutput


class Hook(ABC):
    """
    Abstract base class for integration hooks that run between integration steps.

    A hook is a callable that receives a :class:`StepState` (a live, read-only
    view of the simulation at the current step) and observes it, typically
    accumulating results on ``self`` for inspection after the run.

    Attributes
    ----------
    default_cadence : Cadence
        Cadence used if none is supplied to ``Sim.add_hook``.
    """

    default_cadence = EveryOutput()

    @abstractmethod
    def __call__(self, state: StepState):
        """
        Apply the hook to the given StepState.

        Parameters
        ----------
        state : StepState
            The current state of the simulation at this step.
        """
        ...

    def reset(self):
        """
        Clear what the hook has recorded, so that a run starts afresh.

        ``Sim.run`` calls it at the start of every run, so running a simulation
        again doesn't mix the two runs' records. The default does nothing.
        """

    def _dedup_key(self):
        """
        Identity used by ``Sim.add_hook`` to reject duplicate registrations.

        Return a hashable/comparable key built from the hook's *meaningful
        configuration*. These are the flags that make two instances mean different
        things (e.g. ``component``, ``eps``, an output ``path``, an index set).
        Two hooks with equal, non-``None`` keys are treated as duplicates.
        Never include mutable/result state (accumulated logs, sampled times).

        The default is ``None``: opt out of dedup entirely (any number of
        instances allowed). 
        """
        return None
