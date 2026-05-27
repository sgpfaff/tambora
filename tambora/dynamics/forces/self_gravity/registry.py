from .SelfGravity import FalcONGravity, DirectSummationGravity, NullSelfGravity

SELF_GRAVITY_METHODS = {
    'falcON': FalcONGravity,
    'direct': DirectSummationGravity,
    None: NullSelfGravity
}