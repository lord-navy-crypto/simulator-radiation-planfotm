from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go


OVERVIEW_HEIGHT = 520
TABLE_HEIGHT = 680

SCAN_AXIS_LABELS = {
    "velocity": "Electron speed β = v/c",
    "gamma": "Lorentz factor γ",
    "K": "Undulator parameter K",
    "N_periods": "Number of periods N",
    "observer_distance": "Observer distance R (m)",
    "angle": "Observer θx (mrad)",
    "B0": "Target central field B0 (T)",
    "gap": "Magnetic gap (mm)",
    "ellipticity": "Ellipticity",
    "apple_phase": "APPLE-II row phase (deg)",
    "error_field_amplitude": "Field-amplitude error σ (%)",
    "error_longitudinal_position": "Longitudinal-position error σ (µm)",
    "error_transverse_position": "Transverse-position error σ (µm)",
    "error_magnetization_angle": "Magnetization-angle error σ (mrad)",
    "error_gap_asymmetry": "Gap asymmetry (µm)",
    "error_bank_strength_imbalance": "Bank strength imbalance (%)",
    "error_all_multiplier": "All selected error strengths (× nominal)",
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    unit: str
    group: str
    description: str
    y_scale: str = "linear"
    reference_key: str | None = None
    reference_label: str = "Theory"


# Every primary scan plot contains ONE dependent physical observable. A theory
# trace is allowed only when it is the reference prediction of that same
# observable. This prevents the visual ambiguity of plotting unrelated outputs
# against one another.
METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("gamma_avg", "Average Lorentz factor", "", "Beam", "Trajectory-averaged relativistic Lorentz factor."),
    MetricSpec("v_z_avg_over_c", "Average longitudinal speed", "c", "Beam", "Mean longitudinal electron speed normalized by c."),
    MetricSpec("Kx", "Horizontal deflection parameter Kx", "", "Device response", "Effective K component associated with horizontal bending."),
    MetricSpec("Ky", "Vertical deflection parameter Ky", "", "Device response", "Effective K component associated with vertical bending."),
    MetricSpec("f0", "Fundamental frequency", "Hz", "Resonance", "Simulated spectral fundamental peak.", "log", "f_expected", "Theory"),
    MetricSpec("lambda0", "Fundamental wavelength", "m", "Resonance", "Wavelength corresponding to the simulated fundamental.", "log", "lam_theory", "Theory"),
    MetricSpec("photon_energy_eV", "Fundamental photon energy", "eV", "Resonance", "Photon energy h f0 of the simulated fundamental.", "log"),
    MetricSpec("P_larmor", "Mean radiated power", "W", "Radiation", "Trajectory-averaged single-electron radiated power.", "log", "P_schwinger", "Theory"),
    MetricSpec("equivalent_photon_rate_s^-1", "Fundamental-equivalent photon rate", "s⁻¹", "Radiation", "Integrated radiated energy divided by h f0 and duration; an equivalent count, not a detector flux.", "log"),
    MetricSpec("spectral_photon_rate_estimate_s^-1", "Spectrum-weighted photon-rate estimate", "s⁻¹", "Radiation", "Photon-rate estimate using the simulated spectral shape.", "log"),
    MetricSpec("peak_amplitude", "Observer electric-field pulse peak", "V/m", "Observer signal", "Peak amplitude of the selected transverse observer-field component.", "log"),
    MetricSpec("avg_fwhm", "Observer pulse FWHM", "s", "Observer signal", "Average FWHM of detected field pulses.", "log"),
    MetricSpec("repetition_freq", "Pulse repetition frequency", "Hz", "Observer signal", "Pulse repetition rate inferred from the observer waveform.", "log"),
    MetricSpec("spectral_fwhm_hz", "Spectral FWHM", "Hz", "Spectrum", "Full width at half maximum around the fundamental spectral line.", "log"),
    MetricSpec("relative_linewidth", "Relative linewidth", "Δf/f", "Spectrum", "Spectral FWHM divided by the fundamental frequency.", "log"),
    MetricSpec("spectral_quality_factor", "Spectral quality factor Q", "", "Spectrum", "Fundamental frequency divided by spectral FWHM.", "log"),
    MetricSpec("radiation_H3_over_H1", "Radiation H3/H1", "", "Spectrum", "Integrated third-harmonic radiation power relative to the fundamental.", "log"),
    MetricSpec("radiation_H5_over_H1", "Radiation H5/H1", "", "Spectrum", "Integrated fifth-harmonic radiation power relative to the fundamental.", "log"),
    MetricSpec("P_circ", "Circular polarization degree", "", "Polarization", "Stokes V/I at the simulated fundamental."),
    MetricSpec("P_lin", "Linear polarization degree", "", "Polarization", "sqrt(Q²+U²)/I at the simulated fundamental."),
    MetricSpec("R_avg", "Average transverse orbit radius", "m", "Trajectory", "Mean transverse radial excursion of the electron trajectory.", "log"),
    MetricSpec("R_max", "Maximum transverse orbit radius", "m", "Trajectory", "Maximum transverse radial excursion of the electron trajectory.", "log"),
    MetricSpec("circularity", "Orbit circularity", "", "Trajectory", "Ratio of minor to major transverse orbit amplitude; 1 is circular."),
    MetricSpec("max_transverse_excursion_m", "Maximum transverse excursion", "m", "Trajectory", "Maximum distance from the beam axis during the trajectory.", "log"),
    MetricSpec("period_repeatability_rms_m", "Period-to-period orbit repeatability RMS", "m", "Trajectory quality", "RMS change in sampled transverse orbit from one magnetic period to the next.", "log"),
    MetricSpec("orbit_phase_error_rms_rad", "Orbit phase-step RMS error", "rad", "Trajectory quality", "RMS error of the transverse orbit phase progression.", "log"),
    MetricSpec("exit_xprime_rad", "Exit steering x′", "rad", "Trajectory quality", "Horizontal exit slope vx/vz."),
    MetricSpec("exit_yprime_rad", "Exit steering y′", "rad", "Trajectory quality", "Vertical exit slope vy/vz."),
    MetricSpec("frequency_relative_residual", "Frequency theory residual", "(f−fth)/fth", "Validation", "Relative difference between simulated and theoretical fundamental frequency."),
    MetricSpec("energy_mismatch", "Energy-accounting residual", "relative", "Validation", "Relative mismatch between particle energy loss and integrated radiated energy."),
    MetricSpec("chi_max", "Maximum quantum parameter χe", "", "Quantum monitor", "Maximum strong-field quantum parameter along the trajectory.", "log"),
    MetricSpec("g_min", "Minimum Gaunt factor", "", "Quantum monitor", "Minimum semiclassical radiation-power reduction factor."),
    MetricSpec("wiggler_critical_energy_eV", "Wiggler critical energy estimate", "eV", "Quantum monitor", "Critical photon-energy estimate useful in the wiggler-like regime.", "log"),
)

METRIC_BY_KEY = {m.key: m for m in METRICS}
GROUP_ORDER = [
    "Beam", "Device response", "Resonance", "Radiation", "Observer signal",
    "Spectrum", "Polarization", "Trajectory", "Trajectory quality", "Validation",
    "Quantum monitor",
]

# Focused defaults are scan-aware. A user can still request the comprehensive
# view, but the first screen shows the observables that answer the scan's main
# physical question instead of dumping every available quantity at once.
FOCUSED_GROUPS = {
    "velocity": ["Beam", "Resonance", "Radiation", "Spectrum", "Polarization", "Trajectory", "Quantum monitor", "Validation"],
    "gamma": ["Beam", "Resonance", "Radiation", "Spectrum", "Polarization", "Trajectory", "Quantum monitor", "Validation"],
    "K": ["Device response", "Resonance", "Radiation", "Spectrum", "Polarization", "Trajectory", "Validation"],
    "N_periods": ["Resonance", "Radiation", "Spectrum", "Trajectory quality", "Validation"],
    "observer_distance": ["Observer signal", "Resonance", "Spectrum", "Polarization", "Validation"],
    "angle": ["Observer signal", "Resonance", "Spectrum", "Polarization", "Validation"],
    "B0": ["Device response", "Resonance", "Radiation", "Spectrum", "Trajectory", "Validation"],
    "gap": ["Device response", "Resonance", "Radiation", "Spectrum", "Trajectory", "Validation"],
    "ellipticity": ["Device response", "Resonance", "Polarization", "Trajectory", "Validation"],
    "apple_phase": ["Device response", "Resonance", "Polarization", "Trajectory", "Validation"],
}


def axis_title(scan_name: str) -> str:
    return SCAN_AXIS_LABELS.get(str(scan_name), str(scan_name))


def scan_point_context(scan_name: str, scan_value: float, v11=None) -> dict[str, float | str]:
    """Describe the fixed operating condition of one selected scan row."""
    name = str(scan_name)
    value = float(scan_value)
    out: dict[str, float | str] = {
        "scan_variable": name,
        "scan_axis_label": axis_title(name),
        "scan_value": value,
    }
    if name == "velocity":
        beta = value
        out["beta_v_over_c"] = beta
        out["speed_m_per_s"] = beta * 299792458.0
        if 0.0 <= beta < 1.0:
            out["gamma"] = float(v11.gamma_from_beta(beta)) if v11 is not None else float(1.0 / np.sqrt(1.0-beta**2))
    elif name == "gamma":
        gamma = value
        out["gamma"] = gamma
        if gamma >= 1.0:
            beta = float(v11.beta_from_gamma(gamma)) if v11 is not None else float(np.sqrt(1.0-gamma**-2))
            out["beta_v_over_c"] = beta
            out["speed_m_per_s"] = beta * 299792458.0
    return out


def scan_point_context_text(scan_name: str, scan_value: float, v11=None) -> str:
    ctx = scan_point_context(scan_name, scan_value, v11=v11)
    base = f"{ctx['scan_axis_label']}={float(ctx['scan_value']):.10e}"
    if "beta_v_over_c" in ctx:
        base += (
            f" · β={float(ctx['beta_v_over_c']):.10e}"
            f" · v={float(ctx['speed_m_per_s']):.10e} m/s"
            f" · γ={float(ctx['gamma']):.10e}"
        )
    return base


def _numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)


def _x_transform(scan_name: str, x: np.ndarray) -> np.ndarray:
    if scan_name in {"gamma", "observer_distance"} and np.all(x[np.isfinite(x)] > 0):
        return np.log10(x)
    return x


def representative_indices(
    df: pd.DataFrame,
    x_col: str = "scan_x",
    n: int = 4,
    strategy: str = "Coverage",
) -> np.ndarray:
    """Choose existing scan rows for deeper analysis; never interpolate.

    Strategies:
      Coverage       - evenly cover the scanned range, including endpoints.
      Feature-aware  - endpoints plus rows where several normalized observables
                       change most rapidly; remaining slots are filled by coverage.
    """
    if not isinstance(df, pd.DataFrame) or df.empty or x_col not in df.columns:
        return np.array([], dtype=int)
    x = _numeric(df, x_col)
    valid = np.flatnonzero(np.isfinite(x))
    if len(valid) == 0:
        return np.array([], dtype=int)
    n = max(1, min(int(n), len(valid)))
    if n == 1:
        return np.array([valid[len(valid)//2]], dtype=int)

    order = valid[np.argsort(x[valid])]
    coverage_pos = np.unique(np.rint(np.linspace(0, len(order)-1, n)).astype(int))
    coverage = list(order[coverage_pos])
    if str(strategy).lower().startswith("coverage") or len(order) < 5:
        return np.asarray(sorted(set(coverage), key=lambda i: x[i]), dtype=int)

    # Feature-aware score: normalized local changes of several scan-level
    # observables. This does not change any result; it only proposes interesting
    # existing rows for expensive full-case analysis.
    candidate_keys = [
        "f0", "P_larmor", "photon_energy_eV", "relative_linewidth",
        "P_circ", "P_lin", "R_avg", "frequency_relative_residual",
    ]
    xo = _x_transform("gamma" if np.all(x[order] > 0) and (x[order].max()/max(x[order].min(), 1e-300) > 100) else "linear", x[order])
    scores = np.zeros(len(order), dtype=float)
    for key in candidate_keys:
        if key not in df.columns:
            continue
        y = _numeric(df, key)[order]
        good = np.isfinite(y)
        if np.count_nonzero(good) < 4:
            continue
        yf = y.copy()
        # interpolate only for the *selection score*, never for displayed data.
        idx = np.arange(len(yf))
        yf[~good] = np.interp(idx[~good], idx[good], yf[good])
        med = float(np.nanmedian(yf))
        scale = float(np.nanpercentile(np.abs(yf-med), 75))
        if not np.isfinite(scale) or scale <= 1e-30:
            scale = float(np.nanstd(yf))
        if not np.isfinite(scale) or scale <= 1e-30:
            continue
        yn = (yf-med)/scale
        # local change + curvature identify transitions without treating a large
        # absolute unit as intrinsically more important than another metric.
        g1 = np.gradient(yn, xo)
        g2 = np.gradient(g1, xo)
        s = np.nan_to_num(np.abs(g1), nan=0.0, posinf=0.0, neginf=0.0)
        s += 0.5*np.nan_to_num(np.abs(g2), nan=0.0, posinf=0.0, neginf=0.0)
        mx = float(np.max(s)) if len(s) else 0.0
        if mx > 0:
            scores += s/mx

    chosen = [int(order[0]), int(order[-1])]
    if n >= 3:
        chosen.append(int(order[len(order)//2]))
    ranked = list(order[np.argsort(scores)[::-1]])
    for idx in ranked + coverage:
        if int(idx) not in chosen:
            chosen.append(int(idx))
        if len(chosen) >= n:
            break
    return np.asarray(sorted(chosen[:n], key=lambda i: x[i]), dtype=int)


def _num_cfg(st, df: pd.DataFrame):
    cfg = {}
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            cfg[col] = st.column_config.NumberColumn(str(col), format="%.10e")
        elif pd.api.types.is_integer_dtype(df[col]):
            cfg[col] = st.column_config.NumberColumn(str(col), format="%d")
    return cfg


def available_metric_specs(df: pd.DataFrame, groups: Iterable[str] | None = None) -> list[MetricSpec]:
    group_set = set(groups) if groups is not None else None
    out = []
    for spec in METRICS:
        if spec.key not in df.columns:
            continue
        if group_set is not None and spec.group not in group_set:
            continue
        y = _numeric(df, spec.key)
        if np.any(np.isfinite(y)):
            out.append(spec)
    return out


def default_groups_for_scan(scan_name: str) -> list[str]:
    if str(scan_name).startswith("error_"):
        return ["Device response", "Resonance", "Radiation", "Spectrum", "Polarization", "Trajectory quality", "Validation"]
    return list(FOCUSED_GROUPS.get(str(scan_name), GROUP_ORDER))


def metric_dictionary_dataframe(df: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for spec in METRICS:
        if df is not None and spec.key not in df.columns:
            continue
        rows.append({
            "column": spec.key,
            "observable": spec.label,
            "unit": spec.unit or "dimensionless",
            "group": spec.group,
            "meaning": spec.description,
            "recommended_y_scale": spec.y_scale,
        })
    return pd.DataFrame(rows)


def _series(df: pd.DataFrame, key: str, log_y: bool = False):
    if key not in df.columns:
        return None
    x = _numeric(df, "scan_x")
    y = _numeric(df, key)
    ok = np.isfinite(x) & np.isfinite(y)
    if log_y:
        ok &= y > 0
    if not np.any(ok):
        return None
    return x, y, ok


def build_observable_figure(
    df: pd.DataFrame,
    scan_name: str,
    spec: MetricSpec,
    rep_idx: np.ndarray | None = None,
):
    """One dependent observable vs the selected scan variable.

    The only additional trace permitted is a theory/reference value for the same
    observable. This is the core visualization invariant of the v9.3 design.
    """
    raw_y = _numeric(df, spec.key)
    finite_y = raw_y[np.isfinite(raw_y)]
    # Preserve physically meaningful zeros. A requested log axis is used only
    # when every finite value is strictly positive; otherwise the chart falls
    # back to linear instead of silently dropping zero/negative scan rows.
    log_y = bool(spec.y_scale == "log" and len(finite_y) and np.all(finite_y > 0))
    primary = _series(df, spec.key, log_y=log_y)
    if primary is None:
        return None
    x, y, ok = primary
    x_label = axis_title(scan_name)
    y_label = f"{spec.label} ({spec.unit})" if spec.unit else spec.label
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x[ok], y=y[ok], mode="lines+markers", name="Simulation",
        marker=dict(size=6),
        customdata=np.flatnonzero(ok),
        hovertemplate=(
            "scan row #%{customdata}<br>" + x_label + "=%{x:.10e}<br>" +
            spec.label + "=%{y:.10e}" + (f" {spec.unit}" if spec.unit else "") +
            "<extra></extra>"
        ),
    ))

    if spec.reference_key and spec.reference_key in df.columns:
        ref = _series(df, spec.reference_key, log_y=log_y)
        if ref is not None:
            xr, yr, okr = ref
            fig.add_trace(go.Scatter(
                x=xr[okr], y=yr[okr], mode="lines", name=spec.reference_label,
                line=dict(dash="dash"),
                hovertemplate=(
                    spec.reference_label + "<br>" + x_label + "=%{x:.10e}<br>" +
                    spec.label + "=%{y:.10e}" + (f" {spec.unit}" if spec.unit else "") +
                    "<extra></extra>"
                ),
            ))

    ridx = np.asarray(rep_idx if rep_idx is not None else [], dtype=int)
    ridx = ridx[(ridx >= 0) & (ridx < len(df))]
    if len(ridx):
        rx, ry = x[ridx], y[ridx]
        rok = np.isfinite(rx) & np.isfinite(ry)
        if log_y:
            rok &= ry > 0
        if np.any(rok):
            fig.add_trace(go.Scatter(
                x=rx[rok], y=ry[rok], mode="markers", name="Representative points",
                marker=dict(size=14, symbol="circle-open", line=dict(width=3)),
                customdata=ridx[rok],
                hovertemplate=(
                    "representative row #%{customdata}<br>" + x_label + "=%{x:.10e}<br>" +
                    spec.label + "=%{y:.10e}" + (f" {spec.unit}" if spec.unit else "") +
                    "<extra></extra>"
                ),
            ))

    fig.update_layout(
        title={"text": f"FULL SCAN TREND · {spec.label} vs {x_label}", "x": 0.02, "xanchor": "left"},
        height=OVERVIEW_HEIGHT,
        margin=dict(l=90, r=35, t=95, b=80),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0.0),
        xaxis_title=x_label,
        yaxis_title=y_label,
        font=dict(size=14),
    )
    fig.update_xaxes(automargin=True, tickformat=".6g", hoverformat=".10e")
    fig.update_yaxes(automargin=True, tickformat=".6g", hoverformat=".10e")
    if scan_name in {"gamma", "observer_distance"}:
        xv = x[np.isfinite(x)]
        if len(xv) and np.all(xv > 0):
            fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")
    validate_scan_figure_axis(fig, df, scan_name)
    return fig


def validate_scan_figure_axis(fig, df: pd.DataFrame, scan_name: str) -> None:
    """Fail loudly if a scan-level chart stops using the selected scan quantity.

    A z array is valid inside a single operating-point trajectory/field plot,
    but it is never valid as the independent coordinate of a scan-level trend.
    This runtime invariant protects both the numeric x data and the visible
    axis title instead of merely relabelling an unrelated array.
    """
    if fig is None:
        raise AssertionError("A scan-level figure was not created.")
    expected_title = axis_title(scan_name)
    actual_title = str(getattr(getattr(fig.layout, "xaxis", None), "title", {}).text or "")
    if actual_title != expected_title:
        raise AssertionError(
            f"Scan-axis title mismatch: expected {expected_title!r}, got {actual_title!r}."
        )
    scan_x = _numeric(df, "scan_x")
    finite_scan_x = scan_x[np.isfinite(scan_x)]
    for trace in fig.data:
        plotted_x = np.asarray(trace.x, dtype=float)
        for value in plotted_x[np.isfinite(plotted_x)]:
            if not np.any(np.isclose(value, finite_scan_x, rtol=1e-12, atol=1e-15)):
                raise AssertionError(
                    f"Scan chart contains x={value!r}, which is not a selected {scan_name!r} scan point."
                )


# Backward-compatible helper retained for tests/callers from v9.2. It now
# rejects mixed observables by rendering only the first requested metric.
def build_scan_figure(df, scan_name, metric_specs, title, y_title, rep_idx, log_y=False):
    if not metric_specs:
        return None
    key, label = metric_specs[0]
    base = METRIC_BY_KEY.get(key)
    spec = base or MetricSpec(key, label, y_title, "Other", "User-selected scan observable.", "log" if log_y else "linear")
    if base is not None and log_y != (base.y_scale == "log"):
        spec = MetricSpec(base.key, base.label, base.unit, base.group, base.description, "log" if log_y else "linear", base.reference_key, base.reference_label)
    return build_observable_figure(df, scan_name, spec, rep_idx)


def _plot(st, fig, key=None):
    if fig is None:
        return
    st.plotly_chart(
        fig, width="stretch", key=key,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
            "toImageButtonOptions": {"format": "png", "scale": 2},
        },
    )


def _representative_table(df: pd.DataFrame, rep_idx: np.ndarray, scan_name: str, v11=None) -> pd.DataFrame:
    if len(rep_idx) == 0:
        return pd.DataFrame()
    preferred = [
        "scan_x", "gamma0", "gamma_avg", "v_z_avg_over_c", "Kx", "Ky",
        "f0", "f_expected", "lambda0", "lam_theory", "P_larmor", "P_schwinger",
        "photon_energy_eV", "spectral_fwhm_hz", "relative_linewidth", "spectral_quality_factor",
        "P_circ", "P_lin", "radiation_H3_over_H1", "radiation_H5_over_H1",
        "R_avg", "R_max", "circularity", "period_repeatability_rms_m",
        "orbit_phase_error_rms_rad", "frequency_relative_residual", "energy_mismatch", "chi_max",
    ]
    keep = [c for c in preferred if c in df.columns]
    if "scan_x" not in keep:
        keep.insert(0, "scan_x")
    out = df.iloc[rep_idx][keep].copy()
    out.insert(0, "scan_point_index", [int(i) for i in rep_idx])
    out.insert(1, "representative_case", [f"Case {i+1}" for i in range(len(out))])
    if scan_name == "gamma" and v11 is not None and "beta_v_over_c" not in out.columns:
        beta = [float(v11.beta_from_gamma(float(g))) for g in out["scan_x"]]
        out.insert(3, "beta_v_over_c", beta)
    elif scan_name == "velocity":
        if "beta_v_over_c" not in out.columns:
            out.insert(3, "beta_v_over_c", pd.to_numeric(out["scan_x"], errors="coerce"))
        if v11 is not None and "gamma_from_scan_speed" not in out.columns:
            gamma = [float(v11.gamma_from_beta(float(b))) for b in out["scan_x"]]
            out.insert(4, "gamma_from_scan_speed", gamma)
    return out


def render_scan_overview(
    st,
    df: pd.DataFrame,
    scan_name: str,
    v11=None,
    representative_count: int = 4,
    representative_strategy: str = "Coverage",
    comprehensive: bool = False,
):
    """Research scan view: scan variable → one dependent observable per chart."""
    if not isinstance(df, pd.DataFrame) or df.empty or "scan_x" not in df.columns:
        st.info("No completed scan data are available yet.")
        return np.array([], dtype=int)

    rep_idx = representative_indices(
        df, "scan_x", representative_count, strategy=representative_strategy
    )
    x_label = axis_title(scan_name)

    st.markdown("## Scan-level trends")
    st.success(
        f"SCAN X-AXIS: {x_label}. These plots compare different scan points; none of them uses z as x."
    )
    st.caption(
        f"Primary rule: **{x_label} is the horizontal axis of every scan-level trend**. "
        "Each chart shows one dependent physical observable. A second curve is drawn only when it is "
        "the theory/reference for that same observable. Open circles mark existing scan rows proposed "
        "for deeper full-case analysis."
    )
    st.caption(
        "Scope: the complete successful scan, not a selected-point diagnostic. Other natural axes "
        "such as position, time, frequency, angle, or phase-space coordinates appear only after a "
        "specific scan row has been selected for internal analysis."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Successful scan rows", f"{len(df):d}")
    x = _numeric(df, "scan_x")
    xv = x[np.isfinite(x)]
    if len(xv):
        c2.metric("Scan minimum", f"{np.min(xv):.8e}")
        c3.metric("Scan maximum", f"{np.max(xv):.8e}")
    c4.metric("Representative cases", f"{len(rep_idx):d}")

    st.markdown(f"### Exact horizontal-axis points — {x_label}")
    axis_df = pd.DataFrame({x_label: xv})
    if scan_name in {"velocity", "gamma"}:
        beta_values = xv if scan_name == "velocity" else np.sqrt(1.0 - xv**-2)
        gamma_values = 1.0 / np.sqrt(1.0 - beta_values**2)
        axis_df["Speed β = v/c"] = beta_values
        axis_df["Speed v (m/s)"] = beta_values * 299792458.0
        axis_df["Lorentz factor γ"] = gamma_values
    st.dataframe(axis_df, width="stretch", hide_index=True, height=min(360, 72 + 35 * len(axis_df)))

    groups = GROUP_ORDER if comprehensive else default_groups_for_scan(scan_name)
    specs = available_metric_specs(df, groups)

    for group in groups:
        group_specs = [s for s in specs if s.group == group]
        if not group_specs:
            continue
        st.markdown(f"### {group}")
        for spec in group_specs:
            _plot(st, build_observable_figure(df, scan_name, spec, rep_idx), key=f"scan_{scan_name}_{spec.key}")

    st.markdown("### Representative scan rows")
    st.caption(
        "Representative rows do not replace the full scan. They are existing calculated points selected "
        "only to make expensive trajectory / field / waveform / spectrum analysis manageable."
    )
    rep_df = _representative_table(df, rep_idx, scan_name, v11=v11)
    if not rep_df.empty:
        st.dataframe(
            rep_df, width="stretch", height=min(520, 110 + 68 * len(rep_df)),
            hide_index=True, column_config=_num_cfg(st, rep_df),
        )
        st.download_button(
            "Download representative rows CSV", rep_df.to_csv(index=False).encode("utf-8"),
            file_name=f"v11_{scan_name}_representative_rows.csv", mime="text/csv",
            key=f"download_rep_cases_{scan_name}", use_container_width=True,
        )

    st.markdown("### Complete per-point scan data")
    st.dataframe(df, width="stretch", height=TABLE_HEIGHT, hide_index=True, column_config=_num_cfg(st, df))
    st.download_button(
        "Download complete scan CSV", df.to_csv(index=False).encode("utf-8"),
        file_name=f"v11_{scan_name}_complete_scan.csv", mime="text/csv",
        key=f"download_complete_scan_{scan_name}", use_container_width=True,
    )

    with st.expander("Metric dictionary — what each value means", expanded=False):
        md = metric_dictionary_dataframe(df)
        st.dataframe(md, width="stretch", hide_index=True)

    return rep_idx


def selected_point_metric_table(r) -> pd.DataFrame:
    rows = []
    # full-run result keys differ slightly from scalar scan keys; map them here.
    values = {
        "f0": r.get("f0"),
        "lambda0": r.get("lambda0"),
        "P_larmor": r.get("P_larmor"),
        "photon_energy_eV": (r.get("photon_energy") or {}).get("eV"),
        "spectral_fwhm_hz": r.get("spectral_fwhm_hz"),
        "relative_linewidth": r.get("relative_linewidth"),
        "spectral_quality_factor": r.get("spectral_quality_factor"),
        "P_circ": (r.get("Stokes") or {}).get("P_circ"),
        "P_lin": (r.get("Stokes") or {}).get("P_lin"),
        "radiation_H3_over_H1": (r.get("harmonic_ratios") or {}).get("H3_over_H1"),
        "radiation_H5_over_H1": (r.get("harmonic_ratios") or {}).get("H5_over_H1"),
        "chi_max": (r.get("quantum") or {}).get("chi_max"),
        "energy_mismatch": (r.get("energy_accounting") or {}).get("relative_mismatch"),
    }
    for key, value in values.items():
        spec = METRIC_BY_KEY.get(key)
        if spec is None or value is None:
            continue
        try:
            val = float(value)
        except Exception:
            continue
        rows.append({
            "observable": spec.label,
            "value": val,
            "unit": spec.unit or "dimensionless",
            "meaning": spec.description,
        })
    return pd.DataFrame(rows)


def render_selected_point_summary(st, scan_name: str, scan_index: int, scan_value: float, r, v11):
    """Header + interpretation table for one existing scan row's full analysis."""
    x_label = axis_title(scan_name)
    st.markdown("## Selected scan-point deep analysis")
    st.warning(
        "FIXED SCAN POINT — the scan quantity no longer changes inside this section. "
        + scan_point_context_text(scan_name, scan_value, v11=v11)
    )
    st.caption(
        "The scan variable is now fixed at one existing scan row. The plots below therefore use the "
        "natural independent coordinate for each physics question: z for fields/trajectory, source or "
        "observer time for waveforms, frequency for spectra, photon energy for photon spectra, etc. "
        "Cross-variable loci are secondary diagnostics, not scan trends."
    )
    ctx = scan_point_context(scan_name, scan_value, v11=v11)
    c1, c2, c3 = st.columns(3)
    c1.metric("Scan row", f"#{int(scan_index)}")
    c2.metric(x_label, f"{float(scan_value):.10e}")
    c3.metric("Photon energy", f"{float(r['photon_energy']['eV']):.10e} eV")

    if "beta_v_over_c" in ctx:
        s1, s2, s3 = st.columns(3)
        s1.metric("Fixed speed β = v/c", f"{float(ctx['beta_v_over_c']):.10e}")
        s2.metric("Fixed speed v", f"{float(ctx['speed_m_per_s']):.10e} m/s")
        s3.metric("Corresponding γ", f"{float(ctx['gamma']):.10e}")

    st.markdown("### What the selected-point numbers mean")
    mdf = selected_point_metric_table(r)
    if not mdf.empty:
        st.dataframe(mdf, width="stretch", hide_index=True, column_config=_num_cfg(st, mdf))


def render_representative_comparison(st, representative_results, v11, scan_name="gamma"):
    """Compare several expensive full results at existing representative scan rows."""
    if not representative_results:
        return
    x_label = axis_title(scan_name)
    st.markdown("## Representative full-case comparison")
    st.caption(
        "These are full current-V11 calculations for a small set of existing scan rows. They are used "
        "only where calculating/storing a full trajectory, observer waveform and spectrum at every scan "
        "row would be unnecessarily expensive."
    )
    st.warning(
        "SPECIAL-POINT SUITE — each case below is fixed at the scan value printed in its title. "
        "Frequency, z, time, angle, and other local axes describe physics inside that one fixed case; "
        "they do not replace the full-scan trend charts above."
    )

    summary_rows = []
    for i, item in enumerate(representative_results):
        r = item["result"]
        scan_value = float(item.get("scan_value", item.get("gamma", np.nan)))
        context = scan_point_context(scan_name, scan_value, v11=v11)
        summary_rows.append({
            "case": f"Case {i+1}", "scan_variable": scan_name,
            "scan_axis_label": x_label, "scan_value": scan_value,
            "beta_v_over_c": context.get("beta_v_over_c", np.nan),
            "speed_m_per_s": context.get("speed_m_per_s", np.nan),
            "gamma": context.get("gamma", item.get("gamma", np.nan)),
            "f0_Hz": float(r["f0"]), "f_theory_Hz": float(r["f_expected"]),
            "lambda_m": float(r["lambda0"]), "power_W": float(r["P_larmor"]),
            "photon_energy_eV": float(r["photon_energy"]["eV"]),
            "P_circ": float(r["Stokes"]["P_circ"]), "P_lin": float(r["Stokes"]["P_lin"]),
            "relative_linewidth": float(r["relative_linewidth"]),
        })
    sdf = pd.DataFrame(summary_rows)
    st.dataframe(sdf, width="stretch", hide_index=True, column_config=_num_cfg(st, sdf))

    # Same kind of deep plot repeated case-by-case. We intentionally do not use
    # output-vs-output plots as the primary comparison.
    st.markdown("### Spectra — frequency is the independent axis")
    for i, item in enumerate(representative_results):
        r = item["result"]
        sv = float(item.get("scan_value", item.get("gamma", np.nan)))
        f = np.asarray(r["freq"], dtype=float)
        a = np.asarray(r["fft"], dtype=float)
        norm = a / np.nanmax(a) if len(a) and np.nanmax(a) > 0 else a
        fig = go.Figure(go.Scatter(x=f, y=norm, mode="lines", name="Spectrum"))
        fig.add_vline(x=float(r["f0"]), line_dash="dash", annotation_text="f_sim")
        fig.add_vline(x=float(r["f_expected"]), line_dash="dot", annotation_text="f_theory")
        fig.update_layout(
            title=f"FIXED SCAN POINT · Case {i+1} · {scan_point_context_text(scan_name, sv, v11=v11)}", height=460,
            xaxis_title="Frequency (Hz)", yaxis_title="Normalized FFT amplitude",
            margin=dict(l=80, r=30, t=85, b=70),
        )
        positive = f[np.isfinite(f) & (f > 0)]
        if len(positive) and positive.max()/positive.min() > 100:
            fig.update_xaxes(type="log")
        _plot(st, fig, key=f"rep_spectrum_{scan_name}_{i}")

    st.markdown("### Trajectory — z is the independent axis")
    for i, item in enumerate(representative_results):
        r = item["result"]
        sv = float(item.get("scan_value", item.get("gamma", np.nan)))
        pos = np.asarray(r["r"], dtype=float)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pos[:, 2], y=pos[:, 0]*1e3, mode="lines", name="x"))
        fig.add_trace(go.Scatter(x=pos[:, 2], y=pos[:, 1]*1e3, mode="lines", name="y"))
        fig.update_layout(
            title=f"FIXED SCAN POINT · Case {i+1} · {scan_point_context_text(scan_name, sv, v11=v11)}", height=460,
            xaxis_title="z (m)", yaxis_title="Transverse position (mm)",
            margin=dict(l=80, r=30, t=85, b=70), hovermode="x unified",
        )
        _plot(st, fig, key=f"rep_traj_{scan_name}_{i}")

    st.markdown("### 3-D electron trajectories — fixed representative scan points")
    st.caption(
        "These 3-D plots belong to individual fixed scan points. Transverse x/y are shown in µm "
        "and the display aspect ratio is magnified so the orbit shape remains visible; the full-scan "
        f"independent axis remains {x_label}."
    )
    for i, item in enumerate(representative_results):
        r = item["result"]
        sv = float(item.get("scan_value", item.get("gamma", np.nan)))
        pos = np.asarray(r["r"], dtype=float)
        progress = np.linspace(0.0, 1.0, len(pos))
        z_local = pos[:, 2]
        centre_basis = np.column_stack([np.ones_like(z_local), z_local - np.nanmean(z_local)])
        x_centreline = centre_basis @ np.linalg.lstsq(centre_basis, pos[:, 0], rcond=None)[0]
        y_centreline = centre_basis @ np.linalg.lstsq(centre_basis, pos[:, 1], rcond=None)[0]
        x_orbit = pos[:, 0] - x_centreline
        y_orbit = pos[:, 1] - y_centreline
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=x_orbit * 1e6, y=y_orbit * 1e6, z=pos[:, 2],
            mode="lines", name="centreline-subtracted electron orbit",
            line={"width": 7, "color": progress, "colorscale": "Turbo", "showscale": True,
                  "colorbar": {"title": "Path progress"}},
            hovertemplate="x=%{x:.10e} µm<br>y=%{y:.10e} µm<br>z=%{z:.10e} m<extra></extra>",
        ))
        fig.add_trace(go.Scatter3d(
            x=[x_orbit[0] * 1e6, x_orbit[-1] * 1e6],
            y=[y_orbit[0] * 1e6, y_orbit[-1] * 1e6],
            z=[pos[0, 2], pos[-1, 2]],
            mode="markers+text", name="entrance / exit", text=["entrance", "exit"],
            textposition="top center", marker={"size": 7, "color": ["#22c55e", "#ef4444"]},
            hovertemplate="%{text}<br>x=%{x:.10e} µm<br>y=%{y:.10e} µm<br>z=%{z:.10e} m<extra></extra>",
        ))
        fig.update_layout(
            title=(
                f"FIXED SCAN POINT · 3-D electron trajectory · Case {i+1} · "
                f"{scan_point_context_text(scan_name, sv, v11=v11)}"
            ),
            height=650, margin=dict(l=45, r=45, t=95, b=45),
            scene={
                "xaxis_title": "x (µm)", "yaxis_title": "y (µm)", "zaxis_title": "z (m)",
                "aspectmode": "manual", "aspectratio": {"x": 0.8, "y": 0.8, "z": 2.5},
            },
        )
        _plot(st, fig, key=f"rep_traj_3d_{scan_name}_{i}")

        orbit_radius_um = np.hypot(x_orbit, y_orbit) * 1e6
        design_axis_radius_um = np.hypot(pos[:, 0], pos[:, 1]) * 1e6
        radius_fig = go.Figure()
        radius_fig.add_trace(go.Scatter(
            x=pos[:, 2], y=orbit_radius_um, mode="lines",
            name="Orbit-centred transverse radius",
            hovertemplate="z=%{x:.10e} m<br>orbit-centred radius=%{y:.10e} µm<extra></extra>",
        ))
        radius_fig.add_trace(go.Scatter(
            x=pos[:, 2], y=design_axis_radius_um, mode="lines",
            name="Distance from design axis",
            hovertemplate="z=%{x:.10e} m<br>distance from design axis=%{y:.10e} µm<extra></extra>",
        ))
        radius_fig.add_hline(
            y=float(np.nanmean(orbit_radius_um)), line_dash="dash", line_color="#22c55e",
            annotation_text="mean orbit-centred radius",
        )
        radius_fig.update_layout(
            title=(
                f"FIXED SCAN POINT · transverse orbit radius · Case {i+1} · "
                f"{scan_point_context_text(scan_name, sv, v11=v11)}"
            ),
            height=500, xaxis_title="Longitudinal position z (m)",
            yaxis_title="Transverse radius (µm)",
            margin=dict(l=80, r=30, t=90, b=70), hovermode="x unified",
        )
        _plot(st, radius_fig, key=f"rep_traj_radius_{scan_name}_{i}")
        st.caption(
            f"Case {i+1}: mean/max orbit-centred radius = "
            f"{float(np.nanmean(orbit_radius_um)):.10e} / {float(np.nanmax(orbit_radius_um)):.10e} µm; "
            f"maximum distance from the design axis = {float(np.nanmax(design_axis_radius_um)):.10e} µm."
        )

    st.markdown("### Magnetic field sampled along each selected trajectory")
    for i, item in enumerate(representative_results):
        r = item["result"]
        dev = item.get("dev")
        if dev is None:
            continue
        sv = float(item.get("scan_value", item.get("gamma", np.nan)))
        pos = np.asarray(r["r"], dtype=float)
        B = np.asarray(dev.B(pos), dtype=float)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pos[:, 2], y=B[:, 0], mode="lines", name="Bx"))
        fig.add_trace(go.Scatter(x=pos[:, 2], y=B[:, 1], mode="lines", name="By"))
        fig.add_trace(go.Scatter(x=pos[:, 2], y=B[:, 2], mode="lines", name="Bz"))
        fig.update_layout(
            title=f"FIXED SCAN POINT · Case {i+1} · {scan_point_context_text(scan_name, sv, v11=v11)}", height=460,
            xaxis_title="z (m)", yaxis_title="B along trajectory (T)",
            margin=dict(l=80, r=30, t=85, b=70), hovermode="x unified",
        )
        _plot(st, fig, key=f"rep_field_{scan_name}_{i}")
