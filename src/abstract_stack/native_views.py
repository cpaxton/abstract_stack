"""Keep native view wrappers alive for their environment's lifetime."""

_GROUPS = (
    ("body", "nbody"),
    ("joint", "njnt"),
    ("geom", "ngeom"),
    ("site", "nsite"),
    ("camera", "ncam"),
    ("light", "nlight"),
    ("actuator", "nu"),
    ("tendon", "ntendon"),
    ("sensor", "nsensor"),
)


def _retain(owner, model):
    return {
        kind: tuple(getattr(owner, kind)(index) for index in range(getattr(model, count)))
        for kind, count in _GROUPS
    }


def retain_model_views(model):
    """Retain wrappers without changing any model or simulation array.

    On the recorded MuJoCo 3.17/Python 3.11 environment, a primitive-only
    transient named-view probe intermittently segfaults. Retaining model body
    wrappers survived the matched probes. Keep all commonly used view groups
    here to avoid repeatedly destroying/recreating their Python wrappers.
    This is a local lifetime mitigation, not a claim about the native cause.
    """
    return _retain(model, model)


def retain_data_views(model, data):
    """Retain data wrappers too; model-only retention failed a native suite."""
    return _retain(data, model)
