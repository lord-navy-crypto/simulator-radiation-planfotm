from __future__ import annotations
import math
import numpy as np

C_BRHO = 0.299792458  # GeV/c per (T*m)
ELECTRON_REST_GEV = 0.00051099895

def trapezoid_integral(y, x):
    """
    Trapezoidal integral compatible with current and older NumPy releases.
    Uses numpy.trapezoid when available and falls back to an explicit
    vectorized trapezoidal sum otherwise.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 1:
        raise ValueError("trapezoid_integral expects 1D x and y arrays.")
    if y.size != x.size:
        raise ValueError("x and y must have the same length.")
    if y.size < 2:
        return 0.0

    fn = getattr(np, "trapezoid", None)
    if fn is not None:
        return float(fn(y, x))

    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def cumulative_trapz(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    out = np.zeros_like(y)
    if len(y) > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out

def harmonic_ratios(z_mm, signal, period_mm, exclude_end_periods=1.0):
    """Estimate H1/H3/H5 with a leakage-resistant joint sinusoidal fit."""
    z = np.asarray(z_mm, float)
    s = np.asarray(signal, float)
    period = float(period_mm)
    if len(z) < 8 or period <= 0:
        return {"H1": 0.0, "H3/H1": 0.0, "H5/H1": 0.0}
    if z.ndim != 1 or s.ndim != 1 or len(z) != len(s):
        raise ValueError("harmonic_ratios expects equal-length 1D arrays.")
    if not np.all(np.diff(z) > 0) or not np.all(np.isfinite(z)) or not np.all(np.isfinite(s)):
        raise ValueError("Harmonic-analysis inputs must be finite with increasing z.")

    margin = max(0.0, float(exclude_end_periods)) * period
    lo, hi = z[0] + margin, z[-1] - margin
    usable_periods = int(np.floor((hi - lo) / period))
    if usable_periods < 1:
        lo, hi = z[0], z[-1]
        usable_periods = max(1, int(np.floor((hi - lo) / period)))
    center = 0.5 * (lo + hi)
    half_width = 0.5 * usable_periods * period
    mask = (z >= center - half_width) & (z <= center + half_width)
    zz, ss = z[mask], s[mask]
    if len(zz) < 8:
        zz, ss = z, s
        center = float(np.mean(z))

    u = (zz - center) / period
    columns = [np.ones_like(u), u]
    for n in (1, 3, 5):
        columns.extend((np.cos(2.0 * np.pi * n * u), np.sin(2.0 * np.pi * n * u)))
    coeff, _, _, _ = np.linalg.lstsq(np.column_stack(columns), ss, rcond=None)
    h1 = float(np.hypot(coeff[2], coeff[3]))
    h3 = float(np.hypot(coeff[4], coeff[5]))
    h5 = float(np.hypot(coeff[6], coeff[7]))
    return {
        "H1": h1,
        "H3/H1": h3 / h1 if h1 else 0.0,
        "H5/H1": h5 / h1 if h1 else 0.0,
    }

def zero_crossing_phase_rms_deg(z_mm, signal, period_mm):
    """
    Field-shape diagnostic only. This is NOT the standard electron phase error.
    """
    z = np.asarray(z_mm, float)
    s = np.asarray(signal, float) - float(np.mean(signal))
    crossings = []
    for i in range(len(s) - 1):
        if s[i] == 0:
            crossings.append(z[i])
        elif s[i] * s[i + 1] < 0:
            frac = -s[i] / (s[i + 1] - s[i])
            crossings.append(z[i] + frac * (z[i + 1] - z[i]))
    if len(crossings) < 4:
        return float("nan")
    c = np.asarray(crossings)
    expected = c[0] + np.arange(len(c)) * float(period_mm) / 2.0
    err_deg = (c - expected) * 360.0 / float(period_mm)
    return float(np.sqrt(np.mean((err_deg - np.mean(err_deg)) ** 2)))

def trajectory_from_field(z_mm, B, electron_energy_GeV):
    """
    Ultra-relativistic small-angle electron trajectory derived from on-axis
    transverse field. Coordinates are referenced to zero entrance angle/offset.
    """
    z_m = np.asarray(z_mm, float) * 1e-3
    B = np.asarray(B, float)
    energy = max(float(electron_energy_GeV), 1e-12)
    k = C_BRHO / energy
    xp =  k * cumulative_trapz(B[:, 1], z_m)   # By -> horizontal angle
    yp = -k * cumulative_trapz(B[:, 0], z_m)   # Bx -> vertical angle
    x_m = cumulative_trapz(xp, z_m)
    y_m = cumulative_trapz(yp, z_m)
    return {
        "x_mm": x_m * 1e3,
        "y_mm": y_m * 1e3,
        "xp_rad": xp,
        "yp_rad": yp,
    }

def electron_phase_error(z_mm, B, period_mm, electron_energy_GeV, exclude_end_periods=2.5):
    """
    Trajectory/slippage-based undulator electron phase error.

    Uses the ultra-relativistic slippage relation
      dS/dz = 1/(2 gamma^2) + (x'^2 + y'^2)/2
    and removes the best-fit linear slippage. The residual is converted to
    degrees using the fitted radiation slippage per undulator period.

    RMS is evaluated at nominal half-period pole locations in the regular
    central region, excluding configurable end periods.
    """
    z_mm = np.asarray(z_mm, float)
    B = np.asarray(B, float)
    if len(z_mm) < 10:
        return {
            "rms_deg": float("nan"),
            "radiation_wavelength_m": float("nan"),
            "positions_mm": np.array([]),
            "phase_error_deg": np.array([]),
        }

    z_m = z_mm * 1e-3
    period_m = float(period_mm) * 1e-3
    energy = max(float(electron_energy_GeV), ELECTRON_REST_GEV * 1.001)
    gamma = energy / ELECTRON_REST_GEV

    tr = trajectory_from_field(z_mm, B, energy)
    xp = np.asarray(tr["xp_rad"], float)
    yp = np.asarray(tr["yp_rad"], float)

    dsdz = 0.5 / (gamma * gamma) + 0.5 * (xp * xp + yp * yp)
    slippage_m = cumulative_trapz(dsdz, z_m)

    # Central regular region; avoid end-pole/fringe-field domination.
    margin = max(0.0, float(exclude_end_periods)) * period_m
    fit_mask = (z_m >= z_m[0] + margin) & (z_m <= z_m[-1] - margin)
    if np.count_nonzero(fit_mask) < 6:
        fit_mask = np.ones_like(z_m, dtype=bool)

    coeff = np.polyfit(z_m[fit_mask], slippage_m[fit_mask], 1)
    slope, intercept = float(coeff[0]), float(coeff[1])
    lambda_r = slope * period_m
    if not np.isfinite(lambda_r) or lambda_r <= 0:
        return {
            "rms_deg": float("nan"),
            "radiation_wavelength_m": float("nan"),
            "positions_mm": np.array([]),
            "phase_error_deg": np.array([]),
        }

    # Evaluate at half-period positions, analogous to pole-to-pole phase checks.
    z0 = z_m[fit_mask][0]
    z1 = z_m[fit_mask][-1]
    half = period_m / 2.0
    n0 = math.ceil((z0 - z_m[0]) / half)
    n1 = math.floor((z1 - z_m[0]) / half)
    pole_z = z_m[0] + np.arange(n0, n1 + 1) * half
    if len(pole_z) < 3:
        return {
            "rms_deg": float("nan"),
            "radiation_wavelength_m": lambda_r,
            "positions_mm": np.array([]),
            "phase_error_deg": np.array([]),
        }

    slip_at = np.interp(pole_z, z_m, slippage_m)
    fitted = slope * pole_z + intercept
    phase_deg = 360.0 * (slip_at - fitted) / lambda_r
    phase_deg -= np.mean(phase_deg)
    rms = float(np.sqrt(np.mean(phase_deg ** 2)))

    return {
        "rms_deg": rms,
        "radiation_wavelength_m": float(lambda_r),
        "positions_mm": pole_z * 1e3,
        "phase_error_deg": phase_deg,
    }

def classify_k(k_peak):
    k = float(k_peak)
    if k < 1.0:
        return "Undulator regime (Kpeak < 1)"
    if k < 3.0:
        return "Strong-undulator / transition regime"
    return "Wiggler-like regime (Kpeak is large)"

def compare_metrics(ideal, perturbed):
    keys = [
        "Bx_peak_T", "By_peak_T", "Bperp_peak_T",
        "Kx_peak", "Ky_peak", "K_peak", "K_vector_norm",
        "resonance_K2_over_2",
        "I1x_Tm", "I1y_Tm", "I2x_Tm2", "I2y_Tm2",
        "H3_over_H1", "H5_over_H1",
        "zero_crossing_field_phase_rms_deg",
        "electron_phase_error_rms_deg",
    ]
    out = {}
    for k in keys:
        a = ideal.get(k)
        b = perturbed.get(k)
        try:
            out[k] = {"ideal": float(a), "error": float(b), "delta": float(b) - float(a)}
        except (TypeError, ValueError):
            out[k] = {"ideal": a, "error": b, "delta": None}
    return out

def analyze(z_mm, B, period_mm, electron_energy_GeV=3.0):
    z = np.asarray(z_mm, float)
    B = np.asarray(B, float)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("z_mm must be a 1D array with at least two samples.")
    if B.shape != (z.size, 3):
        raise ValueError(f"B must have shape ({z.size}, 3); got {B.shape}.")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(B)):
        raise ValueError("Analysis inputs must be finite.")
    if not np.all(np.diff(z) > 0):
        raise ValueError("z_mm must be strictly increasing.")
    if float(period_mm) <= 0:
        raise ValueError("period_mm must be positive.")
    if float(electron_energy_GeV) <= ELECTRON_REST_GEV:
        raise ValueError("electron_energy_GeV must exceed the electron rest energy.")
    bperp = np.sqrt(B[:, 0] ** 2 + B[:, 1] ** 2)

    bx_peak = float(np.max(np.abs(B[:, 0]))) if len(B) else 0.0
    by_peak = float(np.max(np.abs(B[:, 1]))) if len(B) else 0.0
    bperp_peak = float(np.max(bperp)) if len(B) else 0.0
    period_cm = float(period_mm) / 10.0

    # Component K amplitudes. By bends x; Bx bends y.
    kx_peak = 0.934 * by_peak * period_cm
    ky_peak = 0.934 * bx_peak * period_cm

    # K_peak comes from the actual peak transverse field magnitude. For a
    # circular helical device this equals the conventional rotating-field K,
    # instead of incorrectly adding two equal components by sqrt(2).
    k_peak = 0.934 * bperp_peak * period_cm

    # Kept separately because this norm is useful but is NOT the conventional
    # helical K. The resonance denominator uses
    #   1 + (Kx^2 + Ky^2)/2
    k_vector_norm = math.sqrt(kx_peak * kx_peak + ky_peak * ky_peak)
    resonance_k2_over_2 = 0.5 * (kx_peak * kx_peak + ky_peak * ky_peak)

    z_m = z * 1e-3
    i1x = float(trapezoid_integral(B[:, 0], z_m))
    i1y = float(trapezoid_integral(B[:, 1], z_m))
    cix = cumulative_trapz(B[:, 0], z_m)
    ciy = cumulative_trapz(B[:, 1], z_m)
    i2x = float(trapezoid_integral(cix, z_m))
    i2y = float(trapezoid_integral(ciy, z_m))

    dominant = B[:, 1] if by_peak >= bx_peak else B[:, 0]
    harmonics = harmonic_ratios(z, dominant, period_mm)
    zero_phase = zero_crossing_phase_rms_deg(z, dominant, period_mm)
    tr = trajectory_from_field(z, B, electron_energy_GeV)
    ephase = electron_phase_error(z, B, period_mm, electron_energy_GeV)

    return {
        "Bx_peak_T": bx_peak,
        "By_peak_T": by_peak,
        "Bperp_peak_T": bperp_peak,
        "Kx_peak": kx_peak,
        "Ky_peak": ky_peak,
        "K_peak": k_peak,
        "K_vector_norm": k_vector_norm,
        "resonance_K2_over_2": resonance_k2_over_2,
        "resonance_factor_on_axis": 1.0 + resonance_k2_over_2,
        # Backward compatibility only; UI should not present this as physical "total K".
        "K_total_legacy_vector_norm": k_vector_norm,
        "I1x_Tm": i1x,
        "I1y_Tm": i1y,
        "I2x_Tm2": i2x,
        "I2y_Tm2": i2y,
        "H3_over_H1": harmonics["H3/H1"],
        "H5_over_H1": harmonics["H5/H1"],
        "zero_crossing_field_phase_rms_deg": zero_phase,
        "electron_phase_error_rms_deg": ephase["rms_deg"],
        "fitted_radiation_wavelength_m": ephase["radiation_wavelength_m"],
        "max_abs_x_mm": float(np.max(np.abs(tr["x_mm"]))),
        "max_abs_y_mm": float(np.max(np.abs(tr["y_mm"]))),
        "trajectory": tr,
        "electron_phase": ephase,
    }
