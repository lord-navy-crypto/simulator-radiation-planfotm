import math

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


def validate_device_params(params):
    required = (
        "period_mm", "periods", "gap_mm", "block_width_mm",
        "block_height_mm", "br_t", "material_mode",
    )
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError("Missing device parameter(s): " + ", ".join(missing))

    positive = ("period_mm", "gap_mm", "block_width_mm", "block_height_mm", "br_t")
    for key in positive:
        value = float(params[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a finite positive number; got {params[key]!r}.")
    periods = int(params["periods"])
    if periods < 1 or periods != float(params["periods"]):
        raise ValueError("periods must be a positive integer.")
    bpp = int(params.get("blocks_per_period", 4))
    if bpp < 4:
        raise ValueError("blocks_per_period must be at least 4.")
    fill = float(params.get("longitudinal_fill", 0.90))
    if not math.isfinite(fill) or not 0 < fill <= 1:
        raise ValueError("longitudinal_fill must be in the interval (0, 1].")
    if params["material_mode"] not in ("Fixed remanence", "Linear NdFeB + relaxation"):
        raise ValueError(f"Unsupported material_mode: {params['material_mode']!r}.")
    segmentation = tuple(params.get("segmentation", (1, 1, 1)))
    if len(segmentation) != 3 or any(int(v) < 1 or int(v) != float(v) for v in segmentation):
        raise ValueError("segmentation must contain three positive integers.")
    for key in (
        "field_error_pct", "longitudinal_error_mm", "transverse_error_mm",
        "angle_error_deg",
    ):
        value = float(params.get(key, 0.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite non-negative number.")


def build_device(rad, kind, params):
    validate_device_params(params)
    try:
        builder = BUILDERS[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown device type: {kind}") from exc
    return builder(rad, params)
