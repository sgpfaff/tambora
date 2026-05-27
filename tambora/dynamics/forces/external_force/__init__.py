try:
    from .ExternalGalpyPotential import ExternalGalpyPotential
except ImportError:
    class ExternalGalpyPotential:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ExternalGalpyPotential requires galpy. "
                "Install with: pip install 'tambora[galpy]'"
            )