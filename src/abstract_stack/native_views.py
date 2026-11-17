"""Keep native model view wrappers alive for their environment's lifetime."""


def retain_model_views(model):
    """Retain wrappers without changing any model or simulation array.

    On the recorded MuJoCo 3.17/Python 3.11 environment, a primitive-only
    transient named-view probe intermittently segfaults. Retaining model body
    wrappers survived the matched probes. Keep all commonly used view groups
    here to avoid repeatedly destroying/recreating their Python wrappers.
    This is a local lifetime mitigation, not a claim about the native cause.
    """
    groups = (
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
    return {
        kind: tuple(getattr(model, kind)(index) for index in range(getattr(model, count)))
        for kind, count in groups
    }
