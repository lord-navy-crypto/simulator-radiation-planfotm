from __future__ import annotations
import numpy as np
from magnet_studio.devices.factory import build_device
from magnet_studio.solver.pipeline import solve_model, sample_on_axis
from magnet_studio.analysis.geometry_bounds import geometry_field_range

def peak_transverse(B):
    B = np.asarray(B, float)
    return float(np.max(np.sqrt(B[:, 0] ** 2 + B[:, 1] ** 2)))

def _measure_b0(rad, model, p, mode, samples):
    period = float(p["period_mm"])
    if mode == "Central-period peak B⊥":
        z = np.linspace(-0.5 * period, 0.5 * period, int(samples))
    elif mode == "Central 3-period peak B⊥":
        z = np.linspace(-1.5 * period, 1.5 * period, int(samples))
    elif mode == "Global peak B⊥":
        lo, hi = geometry_field_range(model["blocks"], period, margin_periods=1.0)
        z = np.linspace(lo, hi, int(samples))
    else:
        raise ValueError(f"Unknown B0 calibration mode: {mode}")
    B = sample_on_axis(rad, model["obj"], z)
    return peak_transverse(B)

def calibrate_br(
    rad, kind, params, target_b0_t, *,
    mode="Central-period peak B⊥",
    relax=False, precision=1e-4, max_iter=1000,
    iterations=4, samples=241
):
    """
    Calibrate Br against a selected ideal-device B0 definition.

    Default is central-period peak transverse field, avoiding accidental
    calibration to a fringe/end-field overshoot.
    """
    target = float(target_b0_t)
    if target <= 0:
        raise ValueError("Target B0 must be > 0.")

    p = dict(params)
    p["errors_enabled"] = False
    br = float(p["br_t"])
    history = []

    niter = 1 if p.get("material_mode") == "Fixed remanence" else max(2, int(iterations))
    for _ in range(niter):
        if hasattr(rad, "UtiDelAll"):
            rad.UtiDelAll()
        p["br_t"] = br
        model = build_device(rad, kind, p)
        solve_model(rad, model, relax=relax, precision=precision, max_iter=max_iter, method=4)
        peak = _measure_b0(rad, model, p, mode, samples)
        if not np.isfinite(peak) or peak <= 1e-12:
            raise RuntimeError("B0 calibration failed because computed transverse field is zero/non-finite.")
        history.append({"Br_T": br, "B0_T": peak, "mode": mode})
        scale = target / peak
        if abs(scale - 1.0) < 2e-4:
            return br, history
        scale = min(5.0, max(0.2, float(scale)))
        br *= scale

    # Final verification sample after last scaling operation.
    if hasattr(rad, "UtiDelAll"):
        rad.UtiDelAll()
    p["br_t"] = br
    model = build_device(rad, kind, p)
    solve_model(rad, model, relax=relax, precision=precision, max_iter=max_iter, method=4)
    peak = _measure_b0(rad, model, p, mode, samples)
    history.append({"Br_T": br, "B0_T": peak, "mode": mode})
    return br, history
