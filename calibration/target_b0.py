from __future__ import annotations
import numpy as np
from devices.factory import build_device
from solver.pipeline import solve_model, sample_on_axis
from analysis.geometry_bounds import geometry_field_range

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
    iterations=8, samples=241, relative_tolerance=5e-3
):
    """Calibrate Br and fail explicitly unless the requested B0 is achieved."""
    target = float(target_b0_t)
    if target <= 0:
        raise ValueError("Target B0 must be > 0.")
    tol = float(relative_tolerance)
    if not (0.0 < tol < 1.0):
        raise ValueError("relative_tolerance must be between 0 and 1.")

    p = dict(params)
    p["errors_enabled"] = False
    br = float(p["br_t"])
    history = []
    niter = max(2, int(iterations))

    for _ in range(niter):
        if hasattr(rad, "UtiDelAll"):
            rad.UtiDelAll()
        p["br_t"] = br
        model = build_device(rad, kind, p)
        solve_model(
            rad, model, relax=relax,
            precision=precision, max_iter=max_iter, method=4
        )
        peak = _measure_b0(rad, model, p, mode, samples)
        if not np.isfinite(peak) or peak <= 1e-12:
            raise RuntimeError(
                "B0 calibration failed because computed transverse field is zero/non-finite."
            )

        rel_err = abs(peak - target) / target
        history.append({
            "Br_T": br,
            "B0_T": peak,
            "target_B0_T": target,
            "relative_error": rel_err,
            "mode": mode,
        })
        if rel_err <= tol:
            return br, history

        scale = target / peak
        scale = min(5.0, max(0.2, float(scale)))
        br *= scale

    if hasattr(rad, "UtiDelAll"):
        rad.UtiDelAll()
    p["br_t"] = br
    model = build_device(rad, kind, p)
    solve_model(
        rad, model, relax=relax,
        precision=precision, max_iter=max_iter, method=4
    )
    peak = _measure_b0(rad, model, p, mode, samples)
    rel_err = abs(peak - target) / target if np.isfinite(peak) else float("inf")
    history.append({
        "Br_T": br,
        "B0_T": peak,
        "target_B0_T": target,
        "relative_error": rel_err,
        "mode": mode,
    })

    if (not np.isfinite(peak)) or peak <= 1e-12 or rel_err > tol:
        raise RuntimeError(
            "B0 calibration did not reach the requested target: "
            f"target={target:.9g} T, actual={peak:.9g} T, "
            f"relative_error={rel_err:.3%}, Br={br:.9g} T, "
            f"tolerance={tol:.3%}."
        )
    return br, history

