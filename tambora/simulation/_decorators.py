"""
Decorators for Sim and Component classes.
"""

import functools

_USE_CACHED_DEFAULT = object()  # sentinel for "caller didn't pass use_cached"


def _resolve_use_cached(func):
    '''
    Decorator that resolves *use_cached* dynamically.

    The cache only ever holds the *all-components* self-gravity computed
    during ``run()``, so it can only satisfy requests with
    ``include_all_components=True``. The default for *use_cached* is therefore
    tied to *include_all_components*:

      - If the caller didn't pass use_cached:
          * method given                       -> use_cached = False  (compute on-the-fly)
          * method absent, include_all=True    -> use_cached = _has_run (cache if available)
          * method absent, include_all=False   -> use_cached = False  (cache can't serve it;
                                                  a method must be provided)
      - If the caller explicitly passed use_cached=True:
          * before run()                       -> error (no cache exists)
          * with method given                  -> error (conflicting intent)
          * with include_all_components=False  -> error (cache is all-components only)
      - If the caller explicitly passed use_cached=False:
          * method absent                      -> error (need a method to compute)
    '''
    @functools.wraps(func)
    def wrapper(*args, use_cached=_USE_CACHED_DEFAULT, method=None, **kwargs):
        sim = args[0]
        explicit = use_cached is not _USE_CACHED_DEFAULT
        # The cache is an all-components quantity; it cannot represent a single
        # component in isolation, so include_all_components gates its use.
        include_all = kwargs.get('include_all_components', True)
        solver = kwargs.get('solver')       # a solver object, instead of method and its keywords
        if solver is not None:
            if method is not None:
                raise ValueError("Give solver or method, not both.")
            if explicit and use_cached:
                raise ValueError("A solver computes afresh, so use_cached can't be True with solver=.")
            return func(*args, use_cached=False, method=None, **kwargs)

        if not explicit:
            if method is not None:
                use_cached = False
            else:
                use_cached = sim._has_run and include_all
        else:
            if use_cached and not sim._has_run:
                raise ValueError("Cannot use cached results before run(). "
                    "Please set use_cached to False and provide a method "
                    "for computing self-gravity.")
            if use_cached and method is not None:
                raise ValueError("`method` should not be specified if "
                    "`use_cached` is True, since the cached self-gravity "
                    "was computed using a specific method. Please set "
                    "`use_cached` to False to specify a method for "
                    "computing self-gravity.")
            if use_cached and not include_all:
                raise ValueError("`use_cached` is True but "
                    "`include_all_components` is False. The cached self-gravity "
                    "is computed over all components and cannot represent a "
                    "single component in isolation. Set `use_cached` to False "
                    "and provide a `method`, or leave "
                    "`include_all_components=True` to use the cache.")

        if not use_cached and method is None:
            if not explicit and not sim._has_run:
                raise ValueError("No cached results available — the simulation "
                    "has not been run yet. Please call run() first, or provide "
                    "a method (e.g. method='direct') to compute on-the-fly.")
            if not explicit and not include_all:
                raise ValueError("`include_all_components` is False, so the "
                    "all-components cache cannot be used. Please provide a "
                    "`method` (e.g. method='direct') to compute the "
                    "single-component self-gravity on-the-fly, or set "
                    "`include_all_components=True` to use the cache.")
            raise ValueError("`use_cached` is False but no `method` was provided. "
                "Please specify a method (e.g. method='direct') to compute "
                "self-gravity, or set `use_cached` to True.")

        return func(*args, use_cached=use_cached, method=method, **kwargs)
    return wrapper


def _resolve_t(func):
    '''
    Decorator that resolves *t* based on *use_cached*:
      - use_cached=True  -> t can be ... (all snapshots) or int/float
      - use_cached=False -> t must be a single snapshot (... is rejected)

    Must be applied AFTER @_resolve_use_cached (i.e. listed BEFORE it
    in stacked decorator order).
    '''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sim = args[0]
        use_cached = kwargs.get('use_cached', True)

        # Extract t from positional args or kwargs
        if len(args) > 1:
            t = args[1]
            args = (args[0],) + args[2:]
        else:
            t = kwargs.pop('t', ...)

        if use_cached:
            t = sim._ti(t, vectorized=True)
        else:
            if t is ...:
                raise TypeError(
                    "Cannot compute on-the-fly for all times. "
                    "Please provide an integer index or a float time for t. "
                    "You will have to manually loop over snapshots.")
            t = sim._ti(t, vectorized=False)

        return func(*args, t=t, **kwargs)
    return wrapper