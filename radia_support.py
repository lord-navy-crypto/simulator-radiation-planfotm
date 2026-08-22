from __future__ import annotations
import importlib
import os
import sys
from pathlib import Path

DEFAULT_RADIA_PATH = Path.home() / "Desktop" / "Radia-master" / "cpp" / "gcc"

def load_radia():
    """Load the compiled RADIA Python extension from env/default path."""
    candidates = []
    env_path = os.environ.get("RADIA_PYTHONPATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(DEFAULT_RADIA_PATH)

    # Normal import first (works if the user already exported PYTHONPATH).
    try:
        return importlib.import_module("radia")
    except ImportError:
        pass

    errors = []
    for path in candidates:
        if not path.exists():
            errors.append(f"missing: {path}")
            continue
        sp = str(path)
        if sp not in sys.path:
            sys.path.insert(0, sp)
        try:
            return importlib.import_module("radia")
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    raise ImportError(
        "RADIA Python extension could not be loaded. Checked: "
        + "; ".join(errors)
    )

def create_linear_ndfeb(rad, br_t=1.20, mu_parallel=1.05, mu_perpendicular=1.05):
    """
    RADIA MatLin expects magnetic susceptibilities [ksi_parallel, ksi_perpendicular]
    and remanent magnetization magnitude/direction.
    Relative permeability mu_r = 1 + ksi.
    """
    ksi_parallel = max(float(mu_parallel) - 1.0, 0.0)
    ksi_perpendicular = max(float(mu_perpendicular) - 1.0, 0.0)
    return rad.MatLin([ksi_parallel, ksi_perpendicular], float(br_t))

def normalize(v):
    import math
    n = math.sqrt(sum(float(x)*float(x) for x in v))
    if n <= 0:
        raise ValueError("Magnetization/easy-axis vector cannot be zero.")
    return [float(x)/n for x in v]

def make_block(
    rad,
    center_mm,
    size_mm,
    easy_axis,
    *,
    br_t=1.20,
    material=None,
    segmentation=(1,1,1),
):
    """
    Create one rectangular permanent-magnet block.

    RADIA geometry units are millimetres. If `material` is supplied, the
    ObjRecMag magnetization vector defines the easy-axis direction and MatApl
    supplies the remanent magnetization. Otherwise fixed remanent magnetization
    is used directly.
    """
    center = [float(v) for v in center_mm]
    size = [float(v) for v in size_mm]
    if any(v <= 0 for v in size):
        raise ValueError(f"All block dimensions must be positive; got {size}.")
    axis = normalize(easy_axis)

    if material is None:
        magnetization = [float(br_t)*a for a in axis]
    else:
        magnetization = axis

    obj = rad.ObjRecMag(center, size, magnetization)

    if material is not None:
        rad.MatApl(obj, material)

    seg = [max(1, int(v)) for v in segmentation]
    if seg != [1,1,1]:
        rad.ObjDivMag(obj, seg)

    return obj
