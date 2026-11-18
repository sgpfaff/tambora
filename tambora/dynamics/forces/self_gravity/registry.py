from .SelfGravity import FalcON, DirectSum, NullSelfGravity

SELF_GRAVITY_METHODS = {
    'falcON': FalcON,
    'direct': DirectSum,
    None: NullSelfGravity
}