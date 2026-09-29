import importlib.util as _importlib_util

from .ExternalConservativeForce import ExternalConservativeForce
from .ExternalPotential import ExternalPotential

# galpy-only until a package-independent LinearTidalForce replaces it. Imported
# only when galpy is installed, and without a blanket `except ImportError`, so a
# real error inside it isn't disguised as "galpy is missing".
if _importlib_util.find_spec("galpy") is not None:
    from .TidalTensorGalpyForce import TidalTensorGalpyForce
else:
    class TidalTensorGalpyForce:
        """Stand-in without galpy, so the name exists and using it explains why it can't work."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TidalTensorGalpyForce requires galpy, which isn't installed. "
                "See https://docs.galpy.org/en/stable/installation.html")
