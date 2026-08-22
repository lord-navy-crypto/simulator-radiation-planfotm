from .planar import build_planar
from .helical import build_helical
from .elliptical import build_elliptical
from .apple2 import build_apple2
from .wiggler import build_wiggler

BUILDERS = {
    "Planar": build_planar,
    "Helical": build_helical,
    "Elliptical": build_elliptical,
    "APPLE-II": build_apple2,
    "Wiggler": build_wiggler,
}

def build_device(rad, kind, params):
    try:
        builder = BUILDERS[kind]
    except KeyError:
        raise ValueError(f"Unknown device type: {kind}")
    return builder(rad, params)
