from __future__ import annotations
import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from integration.bridge import discover_saved_transfer_records
import plotly.graph_objects as go

import undulator_v11_radia_integrated_v9 as v11
from v11_field_quality_v6 import field_quality_metrics
from reporting.full_results import render_full_results
from reporting.scan_overview import (
    render_scan_overview, render_selected_point_summary, render_representative_comparison,
    axis_title, representative_indices,
)
from reporting.live_progress import LiveProgressTable, render_saved_progress

st.title("Stage 2 — V11 Trajectory, Radiation & Dynamic Parameter Scan")


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1920px;
        padding-left: 2.6rem;
        padding-right: 2.6rem;
        padding-top: 1.4rem;
        padding-bottom: 4rem;
    }
    [data-testid="stMetric"] {
        min-height: 118px;
        padding: 0.85rem 1.0rem;
    }
    [data-testid="stMetricLabel"] {
        white-space: normal !important;
        overflow: visible !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        line-height: 1.35 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: break-word !important;
    }
    [data-testid="stPlotlyChart"] {
        margin-top: 0.5rem;
        margin-bottom: 2.2rem;
    }
    h2 {
        margin-top: 2.8rem !important;
    }
    h3 {
        margin-top: 2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Generalized RADIA insertion-device field → end-field event tracking → "
    "retarded Liénard–Wiechert 1/R radiation term → compliance-gated plots / tables / exports"
)

st.info(
    "Workflow: choose the device and scan variable → run a complete scalar scan → read every trend as "
    "scan variable → observable → then run full V11 deep analysis only for selected/representative scan rows.",
    icon="ℹ️",
)

st.warning(
    "Model boundary: prescribed magnetostatic field and one relativistic electron. "
    "The observer calculation keeps the causal 1/R acceleration (radiation) term, not the 1/R² near/velocity field. "
    "Bunch emittance, energy spread, coherent effects, beamline optics and detector response are outside this model. "
    "Every completed deep run now shows a physics-compliance table before the plots.",
    icon="⚠️",
)

if not st.session_state.get("stage2_disk_archive_restored"):
    history = list(st.session_state.get("magnet_model_history", []))
    known_paths = {str(item.get("saved_path")) for item in history if isinstance(item, dict)}
    for record in discover_saved_transfer_records():
        if str(record.get("saved_path")) not in known_paths:
            history.append(record)
    st.session_state["magnet_model_history"] = history
    if history and st.session_state.get("selected_magnet_model_id") is None:
        st.session_state["selected_magnet_model_id"] = history[-1]["id"]
    st.session_state["stage2_disk_archive_restored"] = True

with st.sidebar:
    st.header("1 · Magnetic field")
    archived_models = [
        item for item in st.session_state.get("magnet_model_history", [])
        if isinstance(item, dict) and item.get("bridge") is not None
    ]
    if archived_models:
        archived_ids = [item["id"] for item in archived_models]
        active_id = st.session_state.get("selected_magnet_model_id", archived_ids[-1])
        if active_id not in archived_ids:
            active_id = archived_ids[-1]
        active_id = st.selectbox(
            "Saved Stage-1 model used by Stage 2",
            archived_ids,
            index=archived_ids.index(active_id),
            format_func=lambda rid: next(
                (f"{r['name']} · {r['created_utc']}" for r in archived_models if r["id"] == rid), rid
            ),
            key="stage2_saved_model_selector",
        )
        active_record = next(r for r in archived_models if r["id"] == active_id)
        st.session_state["selected_magnet_model_id"] = active_id
        st.session_state["magnet_scan_bridge"] = active_record["bridge"]
        st.caption(
            f"Active model: {active_record['name']} · "
            f"{active_record.get('parameters', {}).get('device', 'unknown device')} · "
            f"λu={active_record.get('parameters', {}).get('period_mm', 'n/a')} mm"
        )
    stage1_bridge = st.session_state.get("magnet_scan_bridge")
    field_choices = []
    if isinstance(stage1_bridge, dict) and stage1_bridge.get("field_map_csv"):
        field_choices.append("Magnet Studio Stage-1 realized field")
    field_choices.extend([
        "RADIA generated 3D field",
        "RADIA 3D CSV field map",
        "V11 analytic field",
    ])
    field_label = st.selectbox(
        "Field model",
        field_choices,
        index=0,
    )
    field_model = {
        "Magnet Studio Stage-1 realized field": "studio_bridge",
        "RADIA generated 3D field": "radia_generated",
        "RADIA 3D CSV field map": "radia_csv",
        "V11 analytic field": "analytic",
    }[field_label]

    st.header("2 · Device")
    device_options = list(v11.list_device_presets())
    bridge_device_map = {
        "Planar": "planar", "Helical": "helical", "Elliptical": "elliptical",
        "APPLE-II": "apple2", "Wiggler": "wiggler",
    }
    bridge_params = stage1_bridge.get("parameters", {}) if isinstance(stage1_bridge, dict) else {}
    bridge_device = bridge_device_map.get(str(bridge_params.get("device")))
    device_default = bridge_device if bridge_device in device_options else (
        "helical" if "helical" in device_options else device_options[0]
    )
    device_preset = st.selectbox(
        "Insertion device",
        device_options,
        index=device_options.index(device_default),
        disabled=field_model == "studio_bridge",
    )

    with st.expander("Analytic magnetic harmonics", expanded=field_model == "analytic"):
        st.caption(
            "Applied only to the V11 analytic field. RADIA/imported-map harmonics are measured "
            "from the realized field and are never overwritten by these controls."
        )
        analytic_h3 = st.number_input(
            "Third-harmonic field coefficient H3/H1", min_value=-0.50, max_value=0.50,
            value=0.0, step=0.001, format="%.6f", disabled=field_model != "analytic",
        )
        analytic_h5 = st.number_input(
            "Fifth-harmonic field coefficient H5/H1", min_value=-0.50, max_value=0.50,
            value=0.0, step=0.001, format="%.6f", disabled=field_model != "analytic",
        )

    # Generated RADIA geometry is intentionally grouped in an expander so the
    # normal V11 workflow stays readable.
    if field_model == "radia_generated":
        with st.expander("Generated RADIA geometry / map", expanded=False):
            radia_period_mm = st.number_input("Period λu (mm)", min_value=1.0, value=50.0, step=1.0)
            preset_K = v11.get_device_preset(device_preset).get("wiggler_K")
            preset_B0 = (
                v11.B0_from_K(float(preset_K), float(radia_period_mm)*1e-3)
                if preset_K is not None else v11.RADIA_TARGET_B0_T
            )
            radia_target_mode = st.radio(
                "Generated field target",["Preset default","Manual B0"],horizontal=True
            )
            if radia_target_mode == "Preset default":
                radia_target_B0_T = float(preset_B0)
                if preset_K is not None:
                    st.caption(
                        f"Preset target: K={float(preset_K):.3g} → central B0≈{preset_B0:.6g} T "
                        f"for λu={float(radia_period_mm):.6g} mm."
                    )
                else:
                    st.caption(f"Preset central B0 target: {preset_B0:.6g} T")
            else:
                radia_target_B0_T = st.number_input(
                    "Manual target central B0 (T)", min_value=0.001,
                    value=float(preset_B0), step=0.01, format="%.6g",
                    key=f"manual_generated_b0_{device_preset}",
                )
            radia_gap_mm = st.number_input("Gap (mm)", min_value=0.5, value=12.0, step=0.5)
            cross_bank = device_preset in {"helical","left_helical","elliptical"}
            width_default = 10.0 if cross_bank else 40.0
            radia_block_width_mm = st.number_input(
                "Block tangential width (mm)" if cross_bank else "Block width (mm)",
                min_value=0.1, value=width_default, step=1.0,
                key=f"radia_block_width_{device_preset}",
                help=("For the rectangular four-bank helical/elliptical prototype, "
                      "tangential width must not exceed the magnetic gap." if cross_bank else None),
            )
            radia_block_height_mm = st.number_input("Block height / radial thickness (mm)", min_value=0.1, value=15.0, step=1.0)
            radia_ellipticity = st.slider(
                "Ellipticity",0.0,1.0,0.50,0.01,disabled=device_preset!="elliptical"
            )
            radia_apple_phase_deg = st.slider(
                "APPLE-II row phase (deg)",-180.0,180.0,90.0,1.0,
                disabled=device_preset not in {"apple2","variable_polarization"}
            )
            radia_apple_shift_mode = st.selectbox(
                "APPLE-II shift mode",["Antiparallel","Parallel"],
                disabled=device_preset not in {"apple2","variable_polarization"}
            )
            radia_material_mode = st.selectbox(
                "RADIA magnet model",["Fixed remanence","Linear NdFeB + relaxation"]
            )
            radia_mu_parallel = st.number_input("μr parallel",min_value=1.0,value=1.05,step=0.01)
            radia_mu_perpendicular = st.number_input("μr perpendicular",min_value=1.0,value=1.05,step=0.01)
            radia_seg = st.selectbox("Magnet subdivision",[1,2,3],index=0)
            radia_map_half_mm = st.number_input("Map transverse half-width (mm)",min_value=0.2,value=3.0,step=0.5)
            radia_map_nxy = st.select_slider("Map Nx = Ny",options=[3,5,7,9,11],value=7)
            radia_samples_per_period = st.select_slider("Map z samples / period",options=[12,18,24,32,48],value=24)
            radia_field_margin_periods = st.number_input("Fringe-field margin (periods)",min_value=0.0,value=1.0,step=0.5)

    st.header("3 · Main run")
    gamma0 = st.number_input(
        "Electron γ",
        min_value=1.01,
        max_value=60000.0,
        value=100.0,
        step=1.0,
        format="%.6g",
    )
    n_periods = st.number_input(
        "Undulator periods",
        min_value=2,
        max_value=100,
        value=max(2, min(100, int(bridge_params.get("periods", 20)))),
        step=1,
    )
    observer_distance = st.number_input(
        "Observer distance (m)",
        min_value=1.0,
        max_value=1000.0,
        value=100.0,
        step=10.0,
    )
    theta_x_mrad = st.number_input(
        "Observer θx (mrad)", value=0.0, step=0.05, format="%.4f"
    )
    theta_y_mrad = st.number_input(
        "Observer θy (mrad)", value=0.0, step=0.05, format="%.4f"
    )

    st.header("4 · Scan design")
    def _clear_scan_results_when_axis_changes():
        for key in (
            "scan_df", "scan_name", "representative_gamma_results",
            "representative_full_results", "selected_scan_deep",
        ):
            st.session_state.pop(key, None)
    scan_preset = st.selectbox(
        "Independent scan variable",
        ["gamma", "velocity", "K", "N_periods", "observer_distance", "angle"],
        index=0,
        format_func=lambda x: {
            "gamma": "Lorentz factor γ",
            "velocity": "Electron speed β = v/c",
            "K": "Undulator parameter K",
            "N_periods": "Number of periods N",
            "observer_distance": "Observer distance R",
            "angle": "Observer angle θx",
        }[x],
        on_change=_clear_scan_results_when_axis_changes,
        help="Whichever quantity is selected here becomes the actual x-axis of every primary scan-trend figure.",
    )
    scan_points = st.slider(
        "Scalar scan points across selected range", 7, 81, 50, 1,
        help=("Every successful scalar scan row is retained. Full trajectories/waveforms are expensive, "
              "so those are calculated only for selected representative rows unless you explicitly request more."),
    )
    representative_count = st.slider(
        "Representative full-analysis rows", 3, 6, 4, 1,
        help="Number of existing scan rows proposed for expensive full V11 analysis.",
    )
    representative_strategy = st.selectbox(
        "Representative-row strategy",
        ["Coverage", "Feature-aware"],
        index=1,
        help=("Coverage spreads rows over the scan. Feature-aware keeps endpoints and proposes rows near the "
              "largest normalized response changes; it never invents/interpolates displayed data."),
    )
    scan_view_mode = st.radio(
        "Scan plot set",
        ["Focused research trends", "Comprehensive available metrics"],
        index=0,
        help="Every chart still uses the scan variable on x; this only changes how many dependent observables are displayed.",
    )

    st.header("5 · Numerical")
    points_per_period = st.select_slider(
        "Tracking resolution",
        options=[48, 64, 96, 128],
        value=64,
    )


# ---------------- Error-isolation controls ----------------
with st.sidebar:
    st.header("6 · Error isolation")
    error_mode = st.radio(
        "Error mode",
        ["Selected errors", "All errors", "Ideal (no errors)"],
        index=0,
        help="Use ideal as a baseline; selected errors lets you isolate one or several sources.",
    )

    if field_model == "radia_generated":
        st.markdown("#### Error strengths")
        st.caption("Each value below is the physical error strength used when that error source is enabled.")
        radia_error_seed = st.number_input("Manufacturing-error seed", min_value=0, value=20260820, step=1)
        radia_field_sigma_pct = st.number_input("Field-amplitude σ (%)", min_value=0.0, value=0.2, step=0.05)
        radia_z_sigma_um = st.number_input("Longitudinal-position σ (µm)", min_value=0.0, value=20.0, step=1.0)
        radia_xy_sigma_um = st.number_input("Transverse-position σ (µm)", min_value=0.0, value=10.0, step=1.0)
        radia_angle_sigma_mrad = st.number_input("Magnetization-angle σ (mrad)", min_value=0.0, value=0.5, step=0.1)
        radia_gap_asym_um = st.number_input("Gap asymmetry (µm)", min_value=0.0, value=10.0, step=1.0)
        radia_bank_sigma_pct = st.number_input("Bank imbalance (%)", min_value=0.0, value=0.1, step=0.05)

    st.caption("RADIA prototype error sources")
    err_field = st.checkbox(
        "Field amplitude error",
        value=True,
        help="Random block magnetization-strength variation.",
    )
    err_z = st.checkbox(
        "Longitudinal position / phase error",
        value=True,
        help="Random block z-position displacement.",
    )
    err_xy = st.checkbox(
        "Transverse placement error",
        value=True,
        help="Random block x/y placement displacement.",
    )
    err_angle = st.checkbox(
        "Magnetization angle error",
        value=True,
        help="Small random magnetization-direction rotation.",
    )
    err_gap = st.checkbox(
        "Gap / bank asymmetry",
        value=True,
        help="Small local normal-direction bank displacement.",
    )
    err_bank = st.checkbox(
        "Bank strength imbalance",
        value=True,
        help="Systematic pair/bank strength imbalance.",
    )

    if field_model == "radia_generated":
        st.markdown("#### Error-strength response scan")
        error_sweep_source = st.selectbox(
            "Error source to sweep",
            [
                "field_amplitude", "longitudinal_position", "transverse_position",
                "magnetization_angle", "gap_asymmetry", "bank_strength_imbalance",
                "all_selected",
            ],
            format_func=lambda x: {
                "field_amplitude":"Field amplitude σ",
                "longitudinal_position":"Longitudinal position σ",
                "transverse_position":"Transverse position σ",
                "magnetization_angle":"Magnetization angle σ",
                "gap_asymmetry":"Gap asymmetry",
                "bank_strength_imbalance":"Bank strength imbalance",
                "all_selected":"All selected errors × nominal",
            }[x],
        )
        error_sweep_points = st.slider("Error-strength scan points", 3, 11, 6, 1)
        error_sweep_multiplier = st.slider(
            "Maximum strength relative to nominal", 0.5, 5.0, 3.0, 0.5,
            help="Single-source sweeps are plotted in the source's physical units. All-selected uses × nominal on x.",
        )

    compare_ideal = st.checkbox(
        "Also run ideal baseline",
        value=True,
        help="Runs the same point with every error source disabled for direct comparison.",
    )

def selected_error_switches():
    keys = {
        "field_amplitude": bool(err_field),
        "longitudinal_position": bool(err_z),
        "transverse_position": bool(err_xy),
        "magnetization_angle": bool(err_angle),
        "gap_asymmetry": bool(err_gap),
        "bank_strength_imbalance": bool(err_bank),
    }
    if field_model != "radia_generated":
        return {k: False for k in keys}
    if error_mode == "All errors":
        return {k: True for k in keys}
    if error_mode == "Ideal (no errors)":
        return {k: False for k in keys}
    return keys


# ---------------- Analysis selection ----------------
with st.sidebar:
    st.header("7 · Analysis selection")
    st.caption("The comprehensive result report is always shown after a full run. These switches add legacy/heavier analyses.")

    show_core = st.checkbox("Core radiation summary", value=True)
    show_spectrum = st.checkbox("Spectrum + linewidth / Q", value=True)
    show_polarization = st.checkbox("Stokes polarization", value=True)
    show_harmonics = st.checkbox("Radiation harmonics H3/H1, H5/H1", value=True)
    show_trajectory = st.checkbox("3D electron trajectory", value=True)
    show_phase = st.checkbox("Trajectory / phase diagnostics", value=True)
    show_field_quality = st.checkbox("Field quality diagnostics", value=True)

    st.caption("Optional / heavier")
    show_angular_1d = st.checkbox("1D angular scan", value=True)
    show_angular_2d = st.checkbox("2D angular map", value=False)
    show_error_ranking = st.checkbox("One-error-at-a-time sensitivity", value=False)
    show_convergence = st.checkbox("Numerical convergence", value=False)
    show_farfield = st.checkbox("Observer-distance validation", value=False)
    show_energy = st.checkbox("Energy accounting", value=True)
    show_quantum = st.checkbox("Quantum χ monitor", value=True)
    show_chaos = st.checkbox("Advanced chaos / MLE", value=False)

def selected_analyses():
    return {
        "core": show_core,
        "spectrum": show_spectrum,
        "polarization": show_polarization,
        "harmonics": show_harmonics,
        "trajectory": show_trajectory,
        "phase": show_phase,
        "field_quality": show_field_quality,
        "angular_1d": show_angular_1d,
        "angular_2d": show_angular_2d,
        "error_ranking": show_error_ranking,
        "convergence": show_convergence,
        "farfield": show_farfield,
        "energy": show_energy,
        "quantum": show_quantum,
        "chaos": show_chaos,
    }

uploaded_map = None
if field_model == "studio_bridge":
    if not isinstance(stage1_bridge, dict) or not stage1_bridge.get("field_map_csv"):
        st.error("Stage 1 has not published a usable 3-D field map.")
        csv_period_mm = 50.0
    else:
        csv_period_mm = float(stage1_bridge.get("parameters", {}).get("period_mm", 50.0))
        st.success(
            f"Connected to Stage 1 realized field · {stage1_bridge.get('parameters', {}).get('device', 'device')} · "
            f"λu={csv_period_mm:.6g} mm"
        )
elif field_model == "radia_csv":
    uploaded_map = st.file_uploader(
        "Upload a 3-D RADIA field map CSV",
        type=["csv"],
        help="Required columns: x_m, y_m, z_m, Bx_T, By_T, Bz_T",
    )
    csv_period_mm = st.number_input(
        "CSV magnetic period λu (mm)",
        min_value=0.1,
        value=50.0,
        step=1.0,
        help="Used for K, resonance, magnetic harmonics and radiation theory.",
    )

def nominal_error_config():
    if field_model != "radia_generated":
        return None
    return {
        "field_amplitude":{"enabled":True,"rms_fraction":float(radia_field_sigma_pct)/100.0},
        "longitudinal_position":{"enabled":True,"rms_m":float(radia_z_sigma_um)*1e-6},
        "transverse_position":{"enabled":True,"rms_m":float(radia_xy_sigma_um)*1e-6},
        "magnetization_angle":{"enabled":True,"rms_rad":float(radia_angle_sigma_mrad)*1e-3},
        "gap_asymmetry":{"enabled":True,"rms_m":float(radia_gap_asym_um)*1e-6},
        "bank_strength_imbalance":{"enabled":True,"rms_fraction":float(radia_bank_sigma_pct)/100.0},
    }


def current_radia_options(target_B0_override=None, error_config_override=None):
    if field_model != "radia_generated":
        return None
    cfg = nominal_error_config() if error_config_override is None else error_config_override
    return {
        "lambda_u_m":float(radia_period_mm)*1e-3,
        "target_B0_T":float(radia_target_B0_T if target_B0_override is None else target_B0_override),
        "gap_m":float(radia_gap_mm)*1e-3,
        "block_width_m":float(radia_block_width_mm)*1e-3,
        "block_height_m":float(radia_block_height_mm)*1e-3,
        "ellipticity":float(radia_ellipticity),
        "apple_phase_deg":float(radia_apple_phase_deg),
        "apple_shift_mode":radia_apple_shift_mode,
        "material_mode":radia_material_mode,
        "mu_parallel":float(radia_mu_parallel),
        "mu_perpendicular":float(radia_mu_perpendicular),
        "segmentation":(int(radia_seg),int(radia_seg),int(radia_seg)),
        "error_seed":int(radia_error_seed),
        "error_config":cfg,
        "x_half_m":float(radia_map_half_mm)*1e-3,
        "y_half_m":float(radia_map_half_mm)*1e-3,
        "nx":int(radia_map_nxy),
        "ny":int(radia_map_nxy),
        "samples_per_period":int(radia_samples_per_period),
        "field_margin_periods":float(radia_field_margin_periods),
    }


def build_selected_device(nper: int, target_B0_T=None, error_switches=None, error_config_override=None):
    if field_model in {"radia_csv", "studio_bridge"}:
        if field_model == "studio_bridge":
            data = stage1_bridge.get("field_map_csv") if isinstance(stage1_bridge, dict) else None
            if not data:
                raise ValueError("Complete Stage 1 or import a Magnet Studio transfer package first.")
        else:
            if uploaded_map is None:
                raise ValueError("Upload a RADIA 3-D CSV field map first.")
            data = uploaded_map.getvalue()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            dev = v11.make_default_undulator(
                preset=device_preset,
                field_model="radia_csv",
                n_periods=nper,
                radia_csv_path=path,
                radia_csv_lambda_u=float(csv_period_mm)*1e-3,
            )
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return dev

    if field_model == "radia_generated":
        return v11.make_default_undulator(
            preset=device_preset,
            field_model="radia_generated",
            n_periods=nper,
            error_switches=(selected_error_switches() if error_switches is None else error_switches),
            radia_options=current_radia_options(target_B0_T, error_config_override=error_config_override),
        )

    dev = v11.make_default_undulator(
        preset=device_preset,
        field_model="analytic",
        n_periods=nper,
        analytic_h3=float(analytic_h3),
        analytic_h5=float(analytic_h5),
    )
    if target_B0_T is not None:
        dev.B0 = float(target_B0_T)
    return dev

def observer_vector(R, tx_mrad=0.0, ty_mrad=0.0):
    tx = float(tx_mrad) * 1e-3
    ty = float(ty_mrad) * 1e-3
    return np.array([
        float(R) * np.tan(tx),
        float(R) * np.tan(ty),
        float(R),
    ])

def run_scalar(dev, gamma, nper, R, tx=0.0, ty=0.0):
    span = v11.simulation_span_for_device(float(gamma), dev, n_periods=int(nper))
    nbase = v11.samples_for_periods(
        int(nper),
        pts_per_period=int(points_per_period),
        min_pts=max(1000, int(nper)*int(points_per_period)),
        max_pts=max(4000, int(nper)*int(points_per_period)+1),
    )
    return v11.run_sim_scalar(
        dev,
        None,
        span,
        observer_vector(R, tx, ty),
        n_base=nbase,
        gamma0_input=float(gamma),
    )



def enrich_scan_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns that are useful for scan-level plotting only."""
    if not isinstance(df, pd.DataFrame):
        return df
    out = df.copy()
    if "v_z_avg" in out.columns and "v_z_avg_over_c" not in out.columns:
        out["v_z_avg_over_c"] = pd.to_numeric(out["v_z_avg"], errors="coerce") / float(v11.c0)
    return out


def _deep_case_parameters(scan_name: str, scan_value: float):
    case_gamma = float(gamma0)
    case_nperiods = int(n_periods)
    case_R = float(observer_distance)
    case_tx = float(theta_x_mrad)
    case_ty = float(theta_y_mrad)
    target_B = None

    if scan_name == "velocity":
        if not 0.0 < float(scan_value) < 1.0:
            raise ValueError("Velocity scan value must satisfy 0 < β=v/c < 1.")
        case_gamma = float(v11.gamma_from_beta(float(scan_value)))
    elif scan_name == "gamma":
        case_gamma = float(scan_value)
    elif scan_name == "K":
        if field_model in {"radia_csv", "studio_bridge"}:
            raise ValueError("K deep analysis is unavailable for a fixed uploaded field map.")
        lambda_u = float(radia_period_mm)*1e-3 if field_model == "radia_generated" else 0.05
        target_B = float(scan_value) * (2*np.pi*v11.me*v11.c0) / (v11.qe * lambda_u)
    elif scan_name == "N_periods":
        case_nperiods = int(round(scan_value))
    elif scan_name == "observer_distance":
        case_R = float(scan_value)
    elif scan_name == "angle":
        case_tx = float(scan_value)
    else:
        raise ValueError(f"Unsupported scan variable for full-case analysis: {scan_name}")
    return case_gamma, case_nperiods, case_R, case_tx, case_ty, target_B


def run_full_scan_case(scan_name: str, scan_value: float):
    """Run the latest full V11 engine at one existing scan row."""
    case_gamma, case_nperiods, case_R, case_tx, case_ty, target_B = _deep_case_parameters(scan_name, scan_value)
    dev_case = build_selected_device(
        case_nperiods,
        target_B0_T=target_B,
        error_switches=selected_error_switches(),
    )
    span = v11.simulation_span_for_device(case_gamma, dev_case, n_periods=case_nperiods)
    nbase = v11.samples_for_periods(
        case_nperiods,
        pts_per_period=int(points_per_period),
        min_pts=max(1200, case_nperiods*int(points_per_period)),
        max_pts=max(5000, case_nperiods*int(points_per_period)+1),
    )
    rr = v11.run_sim(
        dev_case,
        None,
        span,
        observer_vector(case_R, case_tx, case_ty),
        n_base=nbase,
        gamma0_input=case_gamma,
    )
    return {
        "scan_name": scan_name,
        "scan_value": float(scan_value),
        "result": rr,
        "dev": dev_case,
        "gamma": case_gamma,
        "n_periods": case_nperiods,
        "observer_distance": case_R,
        "theta_x_mrad": case_tx,
        "theta_y_mrad": case_ty,
    }


ERROR_SWEEP_META = {
    "field_amplitude": ("error_field_amplitude", "rms_fraction", 100.0),
    "longitudinal_position": ("error_longitudinal_position", "rms_m", 1e6),
    "transverse_position": ("error_transverse_position", "rms_m", 1e6),
    "magnetization_angle": ("error_magnetization_angle", "rms_rad", 1e3),
    "gap_asymmetry": ("error_gap_asymmetry", "rms_m", 1e6),
    "bank_strength_imbalance": ("error_bank_strength_imbalance", "rms_fraction", 100.0),
}


def run_error_strength_scan(source: str, npts: int, max_multiplier: float):
    """One-error-at-a-time strength scan using actual error strength as x.

    For a single source, all other error switches are disabled. The same random
    seed is kept across the strength scan so the change is primarily the error
    amplitude, not a different random realization.
    """
    if field_model != "radia_generated":
        raise ValueError("Error-strength scans require RADIA generated 3-D fields.")
    base_cfg = nominal_error_config()
    if not base_cfg:
        raise ValueError("No RADIA error configuration is available.")
    multipliers = np.linspace(0.0, float(max_multiplier), int(npts))
    rows = []

    if source == "all_selected":
        switches = selected_error_switches()
        if not any(switches.values()):
            raise ValueError("Enable at least one error source before scanning all selected errors.")
        scan_name = "error_all_multiplier"
        for mult in multipliers:
            cfg = {k: dict(v) for k, v in base_cfg.items()}
            for key, enabled in switches.items():
                if not enabled:
                    continue
                if key in {"field_amplitude", "bank_strength_imbalance"}:
                    cfg[key]["rms_fraction"] = float(base_cfg[key]["rms_fraction"]) * float(mult)
                elif key == "magnetization_angle":
                    cfg[key]["rms_rad"] = float(base_cfg[key]["rms_rad"]) * float(mult)
                else:
                    cfg[key]["rms_m"] = float(base_cfg[key]["rms_m"]) * float(mult)
            dev = build_selected_device(
                int(n_periods), error_switches=switches, error_config_override=cfg
            )
            rr = run_scalar(dev, gamma0, int(n_periods), observer_distance, theta_x_mrad, theta_y_mrad)
            if rr:
                rr["scan_x"] = float(mult)
                rows.append(rr)
        return enrich_scan_dataframe(pd.DataFrame(rows)), scan_name

    if source not in ERROR_SWEEP_META:
        raise ValueError(f"Unknown error source: {source}")
    scan_name, strength_field, unit_factor = ERROR_SWEEP_META[source]
    base_strength = float(base_cfg[source][strength_field])
    if base_strength <= 0:
        raise ValueError("The selected error has zero nominal strength; set a nonzero strength first.")
    switches = {k: (k == source) for k in selected_error_switches()}
    for mult in multipliers:
        cfg = {k: dict(v) for k, v in base_cfg.items()}
        cfg[source][strength_field] = base_strength * float(mult)
        dev = build_selected_device(
            int(n_periods), error_switches=switches, error_config_override=cfg
        )
        rr = run_scalar(dev, gamma0, int(n_periods), observer_distance, theta_x_mrad, theta_y_mrad)
        if rr:
            rr["scan_x"] = base_strength * float(mult) * unit_factor
            rows.append(rr)
    return enrich_scan_dataframe(pd.DataFrame(rows)), scan_name


def result_summary_row(r):
    return {
        "photon_energy_eV": float(r["photon_energy"]["eV"]),
        "fundamental_Hz": float(r["f0"]),
        "wavelength_m": float(r["lambda0"]),
        "average_power_W": float(r["P_larmor"]),
        "relative_linewidth": float(r["relative_linewidth"]),
        "quality_factor": float(r["spectral_quality_factor"]),
        "P_circ": float(r["Stokes"]["P_circ"]),
        "P_lin": float(r["Stokes"]["P_lin"]),
        "radiation_H3_over_H1": float(r["harmonic_ratios"]["H3_over_H1"]),
        "radiation_H5_over_H1": float(r["harmonic_ratios"]["H5_over_H1"]),
        "frequency_relative_residual": float(r["theory_residuals"]["frequency_relative_residual"]),
    }


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def result_json_bytes(result):
    import json
    return json.dumps(
        _json_safe(result),
        ensure_ascii=False,
        indent=2,
        allow_nan=True,
    ).encode("utf-8")

top1, top2 = st.columns([1, 1])
with top1:
    st.subheader("Current configuration")
    config_text = (
        'FIELD_MODEL   = "' + field_model + '"\n'
        'DEVICE_PRESET = "' + device_preset + '"\n'
        'SCAN_PRESET   = "' + scan_preset + '"\n'
        f'gamma         = {gamma0}\n'
        f'N_periods     = {n_periods}\n'
        f'observer_R    = {observer_distance} m\n'
        f'error_mode    = "{error_mode}"\n'
        f'errors        = {selected_error_switches()}\n'
        f'analyses      = {selected_analyses()}'
    )
    st.code(config_text, language="python")

with top2:
    st.subheader("RADIA connection")
    if field_model.startswith("radia"):
        try:
            rad = v11.load_radia_module()
            st.success(
                f"Connected to RADIA: {getattr(rad, '__file__', 'loaded module')}"
            )
            st.caption(
                "Generated mode now uses the strict Magnet Studio backend: "
                "ideal-only B0 calibration, actual geometry-derived map bounds, "
                "manufacturing errors, and full fringe-field tracking."
            )
        except Exception as exc:
            st.error(str(exc))
    else:
        st.info("Analytic V11 field selected; RADIA is not required for this run.")

st.divider()
st.subheader("Magnetic field preview")
st.caption(
    "This optional preview is for one magnetic device, so its internal coordinate is z. "
    "It is not a parameter-scan result."
)

if st.button("Build magnetic device / refresh field", type="primary"):
    try:
        with st.spinner("Building field model…"):
            st.session_state.dev = build_selected_device(int(n_periods), error_switches=selected_error_switches())
        st.session_state.pop("full_result", None)
    except Exception as exc:
        st.exception(exc)

show_field_preview = st.toggle(
    "Show single-device z-axis field preview",
    value=False,
    help="Leave this off when you want cross-scan plots whose x-axis is the selected scan quantity.",
)

if "dev" in st.session_state and show_field_preview:
    dev = st.session_state.dev
    if getattr(dev, "uses_real_end_fields", False) and hasattr(dev, "z_grid"):
        md = dict(getattr(dev, "metadata", {}) or {})
        z0 = float(md.get("tracking_z_start_m", dev.z_grid[0]))
        z1 = float(md.get("tracking_z_end_m", dev.z_grid[-1]))
        zs = np.linspace(z0, z1, max(101, int(n_periods)*30+1))
    else:
        L = int(n_periods) * dev.lambda_u
        zs = np.linspace(0.0, L, max(101, int(n_periods)*30+1))
    rr = np.column_stack([np.zeros_like(zs), np.zeros_like(zs), zs])
    BB = dev.B(rr)
    field_fig = go.Figure()
    field_fig.add_trace(go.Scatter(x=zs, y=BB[:,0], name="Bx"))
    field_fig.add_trace(go.Scatter(x=zs, y=BB[:,1], name="By"))
    field_fig.add_trace(go.Scatter(x=zs, y=BB[:,2], name="Bz"))
    field_fig.update_layout(
        xaxis_title="z (m)",
        yaxis_title="B (T)",
        hovermode="x unified",
        height=430,
    )
    st.plotly_chart(field_fig, width="stretch")
    if hasattr(dev, "error_summary"):
        st.json(dev.error_summary(), expanded=False)

    if selected_analyses()["field_quality"]:
        fq = field_quality_metrics(dev, int(n_periods))
        st.markdown("#### Field quality")
        st.caption("Central periodic region → K/harmonics; full map including fringe fields → field integrals.")
        q1,q2,q3 = st.columns(3)
        q1.metric("Central H3/H1", f"{fq['magnetic_H3_over_H1']:.4g}")
        q2.metric("Central H5/H1", f"{fq['magnetic_H5_over_H1']:.4g}")
        q3.metric("Central |B⊥| peak", f"{fq['Btrans_peak_central_T']:.6g} T")
        q4,q5,q6 = st.columns(3)
        q4.metric("Full ∫Bx dz", f"{fq['first_integral_Bx_Tm']:.3e} T·m")
        q5.metric("Full ∫By dz", f"{fq['first_integral_By_Tm']:.3e} T·m")
        q6.metric("Global |B⊥| peak", f"{fq['Btrans_peak_global_T']:.6g} T")
        with st.expander("Field integral details", expanded=False):
            st.write({
                "I2x_Tm2":fq["second_integral_Bx_Tm2"],
                "I2y_Tm2":fq["second_integral_By_Tm2"],
                "central_z_range_m":[fq["central_z_start_m"],fq["central_z_end_m"]],
                "full_z_range_m":[fq["full_z_start_m"],fq["full_z_end_m"]],
            })

st.divider()
st.subheader("Results workspace")
st.caption(
    "The primary research output is the parameter scan below. Single-point z-axis diagnostics are optional and hidden by default."
)
results_status = st.empty()

st.subheader("Optional single operating-point analysis")
st.caption(
    "This is not a scan. It follows one electron through one device, so trajectory and field charts here legitimately use z. "
    "For speed/K/γ trends, use the primary parameter scan below."
)

if st.button("Run one operating point (creates z-axis diagnostics)"):
    progress = st.progress(0, text="Preparing simulation…")
    run_stage = st.empty()
    full_tracker = LiveProgressTable(
        st,
        [
            ("Build magnetic field", field_label),
            ("Prepare integration", f"γ={float(gamma0):.6g}, N={int(n_periods)}"),
            ("Track electron + radiation", f"R={float(observer_distance):.6g} m"),
            ("Ideal comparison", "Conditional"),
            ("Finalize outputs", "Tables, plots, exports"),
        ],
        title="Live calculation progress — one operating point",
        session_key="last_single_point_progress",
    )

    try:
        active_errors = selected_error_switches()

        full_tracker.start(0, "Building selected field/device model")
        run_stage.info("Stage 1/5 — Building magnetic field model…")
        progress.progress(10, text="Building magnetic field model…")
        dev = build_selected_device(
            int(n_periods), error_switches=active_errors
        )
        full_tracker.complete(0, "Magnetic field/device ready")

        full_tracker.start(1, "Computing span and integration sample count")
        run_stage.info("Stage 2/5 — Preparing electron trajectory integration…")
        progress.progress(25, text="Preparing trajectory integration…")
        span = v11.simulation_span_for_device(
            float(gamma0), dev, n_periods=int(n_periods)
        )
        nbase = v11.samples_for_periods(
            int(n_periods),
            pts_per_period=int(points_per_period),
            min_pts=max(1200, int(n_periods)*int(points_per_period)),
            max_pts=max(5000, int(n_periods)*int(points_per_period)+1),
        )
        full_tracker.complete(1, f"Integration prepared with n_base={int(nbase)}")

        full_tracker.start(2, "Integrating trajectory and Liénard–Wiechert radiation")
        run_stage.info("Stage 3/5 — Tracking electron and calculating radiation…")
        progress.progress(40, text="Tracking electron and calculating radiation…")
        st.session_state.dev = dev
        res = v11.run_sim(
            dev,
            None,
            span,
            observer_vector(
                observer_distance, theta_x_mrad, theta_y_mrad
            ),
            n_base=nbase,
            gamma0_input=float(gamma0),
        )

        st.session_state.full_result = res
        st.session_state.full_errors = dict(active_errors)
        full_tracker.complete(2, "Core trajectory/radiation result complete")

        progress.progress(75, text="Core V11 analysis complete…")
        run_stage.info("Stage 4/5 — Preparing selected diagnostics and comparisons…")

        if compare_ideal and field_model == "radia_generated" and any(active_errors.values()):
            full_tracker.start(3, "Computing ideal reference case")
            progress.progress(80, text="Running ideal baseline for comparison…")
            ideal_errors = {k: False for k in active_errors}
            ideal_dev = build_selected_device(
                int(n_periods), error_switches=ideal_errors
            )
            ideal_span = v11.simulation_span_for_device(
                float(gamma0), ideal_dev, n_periods=int(n_periods)
            )
            ideal_res = v11.run_sim(
                ideal_dev,
                None,
                ideal_span,
                observer_vector(
                    observer_distance, theta_x_mrad, theta_y_mrad
                ),
                n_base=nbase,
                gamma0_input=float(gamma0),
            )
            st.session_state.ideal_result = ideal_res
            full_tracker.complete(3, "Ideal reference complete")
        else:
            st.session_state.pop("ideal_result", None)
            full_tracker.skip(3, "Ideal comparison not requested for this configuration")

        full_tracker.start(4, "Publishing result tables and visualization data")
        progress.progress(100, text="Simulation complete.")
        run_stage.success(
            "Stage 5/5 — Complete. Scroll just below this button: your selected results are displayed there.",
            icon="✅",
        )
        results_status.success(
            "Results are ready. The selected output sections are shown below.",
            icon="✅",
        )
        full_tracker.complete(4, "All selected outputs ready")
        full_tracker.finish("Single-point calculation complete")

    except Exception as exc:
        full_tracker.fail(detail=str(exc))
        progress.empty()
        run_stage.error("Simulation stopped because an error occurred.")
        results_status.error("No result was produced. See the error details below.")
        st.exception(exc)

render_saved_progress(st, "last_single_point_progress", "Last single-point progress table")

if "full_result" in st.session_state:
    st.success(
        "V11 run complete — the 3-D particle trajectory and all selected results are displayed below.",
        icon="✅",
    )
    r = st.session_state.full_result

    st.divider()
    st.markdown("## Main-run single operating point")
    fixed_beta = float(v11.beta_from_gamma(float(gamma0)))
    st.warning(
        "FIXED OPERATING POINT — this is not a full-axis scan. "
        f"γ={float(gamma0):.10e} · β=v/c={fixed_beta:.10e} · "
        f"v={fixed_beta*float(v11.c0):.10e} m/s. "
        "Plots below may therefore use position, time, frequency, angle, or another natural local coordinate."
    )
    st.caption(
        "This is not a parameter-scan trend. It is a deep analysis at the single Main run configuration above. "
        "Use the Preset scan section below when you want the scanned variable on the horizontal axis."
    )
    render_full_results(
        st, r, v11,
        dev=st.session_state.get("dev"),
        field_quality_fn=field_quality_metrics,
    )
    st.divider()
    st.markdown("## Additional / legacy analysis panels")

    required_keys = {
        "photon_energy", "f0", "P_larmor", "Stokes", "freq", "fft",
        "relative_linewidth", "spectral_quality_factor", "harmonic_ratios",
        "r", "trajectory_phase", "theory_residuals"
    }
    missing = sorted(required_keys.difference(r.keys()))
    if missing:
        st.error(
            "Result schema mismatch. Missing keys: " + ", ".join(missing)
        )
        st.stop()
    analyses = selected_analyses()
    with st.expander("Result structure / debug", expanded=False):
        st.write("Available result keys:")
        st.code(", ".join(sorted(map(str, r.keys()))))

    if analyses["core"]:
        st.markdown("### Core radiation summary")
        c1,c2,c3 = st.columns(3)
        c1.metric("Photon energy", f"{r['photon_energy']['eV']:.6g} eV")
        c2.metric("Fundamental frequency", f"{r['f0']:.6e} Hz")
        c3.metric("Fundamental wavelength", f"{r['lambda0']:.6e} m")

        c4,c5,c6 = st.columns(3)
        c4.metric("Average radiated power", f"{r['P_larmor']:.6e} W")
        c5.metric("Relative linewidth", f"{r['relative_linewidth']:.6g}")
        c6.metric(
            "Theory frequency residual",
            f"{r['theory_residuals']['frequency_relative_residual']:.3e}"
        )

        with st.expander("Exact core values", expanded=False):
            st.dataframe(
                pd.DataFrame([
                    {"quantity":"Photon energy","value":float(r["photon_energy"]["eV"]),"unit":"eV"},
                    {"quantity":"Fundamental frequency","value":float(r["f0"]),"unit":"Hz"},
                    {"quantity":"Fundamental wavelength","value":float(r["lambda0"]),"unit":"m"},
                    {"quantity":"Average radiated power","value":float(r["P_larmor"]),"unit":"W"},
                    {"quantity":"Relative linewidth","value":float(r["relative_linewidth"]),"unit":"dimensionless"},
                    {"quantity":"Quality factor Q","value":float(r["spectral_quality_factor"]),"unit":"dimensionless"},
                ]),
                width="stretch",
                hide_index=True,
            )

    if analyses["polarization"]:
        st.markdown("### Stokes polarization")
        p1,p2 = st.columns(2)
        p1.metric("Circular polarization", f"{r['Stokes']['P_circ']:.6f}")
        p2.metric("Linear polarization", f"{r['Stokes']['P_lin']:.6f}")

    if analyses["spectrum"]:
        st.markdown("### Spectrum / linewidth")
        s1,s2 = st.columns(2)
        s1.metric("Relative linewidth", f"{r['relative_linewidth']:.5g}")
        s2.metric("Quality factor Q", f"{r['spectral_quality_factor']:.6g}")
        spec = {"fp": r["freq"], "fft": r["fft"]}
        sf = go.Figure()
        sf.add_trace(go.Scatter(x=spec["fp"], y=np.asarray(spec["fft"])**2, name="spectral power proxy"))
        sf.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="|FFT|²", xaxis_type="log", yaxis_type="log", height=500)
        st.plotly_chart(sf, width="stretch")

    if analyses["harmonics"]:
        st.markdown("### Radiation harmonics")
        h1,h2 = st.columns(2)
        h1.metric("Radiation H3/H1", f"{r['harmonic_ratios']['H3_over_H1']:.5g}")
        h2.metric("Radiation H5/H1", f"{r['harmonic_ratios']['H5_over_H1']:.5g}")

    if "ideal_result" in st.session_state:
        ideal = st.session_state.ideal_result
        st.markdown("### Selected errors vs ideal")
        rows = []
        for name,actual,baseline in [
            ("Photon energy (eV)", r["photon_energy"]["eV"], ideal["photon_energy"]["eV"]),
            ("Relative linewidth", r["relative_linewidth"], ideal["relative_linewidth"]),
            ("Circular polarization", r["Stokes"]["P_circ"], ideal["Stokes"]["P_circ"]),
            ("Linear polarization", r["Stokes"]["P_lin"], ideal["Stokes"]["P_lin"]),
            ("Radiation H3/H1", r["harmonic_ratios"]["H3_over_H1"], ideal["harmonic_ratios"]["H3_over_H1"]),
            ("Radiation H5/H1", r["harmonic_ratios"]["H5_over_H1"], ideal["harmonic_ratios"]["H5_over_H1"]),
            ("Average power (W)", r["P_larmor"], ideal["P_larmor"]),
        ]:
            actual, baseline = float(actual), float(baseline)
            delta = actual-baseline
            rel = np.nan if abs(baseline)<1e-30 else 100.0*delta/baseline
            rows.append({"metric":name,"with_selected_errors":actual,"ideal":baseline,"absolute_delta":delta,"relative_delta_percent":rel})
        cdf = pd.DataFrame(rows)
        st.dataframe(cdf, width="stretch")
        st.download_button("Download ideal-vs-error CSV", cdf.to_csv(index=False).encode(), file_name="v11_error_vs_ideal.csv", mime="text/csv")

    if analyses["trajectory"]:
        st.markdown("### 3D electron trajectory")
        pos = r["r"]
        tf = go.Figure(go.Scatter3d(x=pos[:,0], y=pos[:,1], z=pos[:,2], mode="lines", name="electron"))
        tf.update_layout(scene={"xaxis_title":"x (m)","yaxis_title":"y (m)","zaxis_title":"z (m)","aspectmode":"data"}, height=560)
        st.plotly_chart(tf, width="stretch")

    if analyses["phase"]:
        st.markdown("### Trajectory / phase diagnostics")
        st.json(r.get("trajectory_phase", {}), expanded=True)

    if analyses["field_quality"]:
        try:
            fq_dev = st.session_state.get("dev")
            if fq_dev is None:
                fq_dev = build_selected_device(int(n_periods), error_switches=selected_error_switches())
            st.markdown("### Field quality diagnostics")
            st.json(field_quality_metrics(fq_dev, int(n_periods)), expanded=True)
        except Exception as exc:
            st.warning(f"Field-quality analysis could not complete: {exc}")

    if analyses["energy"]:
        st.markdown("### Energy accounting")
        st.json(r.get("energy_accounting", {}), expanded=True)

    if analyses["quantum"]:
        st.markdown("### Quantum χ monitor")
        q = r.get("quantum", {})
        st.json({k:v for k,v in q.items() if k not in {"chi_array","g_array"}}, expanded=True)

    if analyses["angular_1d"]:
        st.markdown("### 1D angular scan")
        try:
            ar = np.asarray(
                v11.angle_scan(r, np.linspace(-2e-3, 2e-3, 21), n_obs=3500),
                dtype=float,
            )
            theta = ar[:, 0]
            fluence = ar[:, 2]
            adf = pd.DataFrame({
                "theta_rad": ar[:, 0],
                "frequency_Hz": ar[:, 1],
                "fluence_J_m2": ar[:, 2],
                "P_circ": ar[:, 3],
                "P_lin": ar[:, 4],
            })
            st.dataframe(adf, width="stretch")
            af = go.Figure(go.Scatter(x=theta*1e3, y=fluence, mode="lines+markers"))
            af.update_layout(xaxis_title="Observation angle (mrad)", yaxis_title="Fluence (J/m²)", height=440)
            st.plotly_chart(af, width="stretch")
        except Exception as exc:
            st.warning(f"1D angular scan could not complete: {exc}")

    if analyses["angular_2d"]:
        st.markdown("### 2D angular map")
        try:
            amap = v11.angular_map_2d(r, gamma_for_grid=float(gamma0), grid_points=21, extent_gamma_theta=4.0, observer_distance=float(observer_distance), n_obs=3000)
            hf = go.Figure(go.Heatmap(
                x=np.asarray(amap["theta_x"])*1e3,
                y=np.asarray(amap["theta_y"])*1e3,
                z=np.asarray(amap["fluence_J_m2"]),
                colorbar={"title":"J/m²"},
            ))
            if int(amap.get("failure_count",0)):
                st.warning(
                    f"Angular-map solver failed at {int(amap['failure_count'])} pixel(s); "
                    "those pixels are shown as missing/NaN, not zero."
                )
                with st.expander("Angular-map failed pixels", expanded=False):
                    st.dataframe(pd.DataFrame(amap.get("failures",[])), width="stretch", hide_index=True)
            hf.update_layout(xaxis_title="θx (mrad)", yaxis_title="θy (mrad)", height=520)
            st.plotly_chart(hf, width="stretch")
            d1,d2 = st.columns(2)
            d1.metric("RMS divergence x", f"{amap['rms_divergence_x_rad']*1e3:.5g} mrad")
            d2.metric("RMS divergence y", f"{amap['rms_divergence_y_rad']*1e3:.5g} mrad")
        except Exception as exc:
            st.warning(f"2D angular map could not complete: {exc}")

    if analyses["convergence"]:
        st.markdown("### Numerical convergence")
        rows = []
        try:
            for ppp in [48,96,192]:
                devc = build_selected_device(int(n_periods), error_switches=selected_error_switches())
                span = v11.simulation_span_for_device(float(gamma0), devc, n_periods=int(n_periods))
                nbase = v11.samples_for_periods(int(n_periods), pts_per_period=ppp, min_pts=max(1200,int(n_periods)*ppp), max_pts=max(8000,int(n_periods)*ppp+1))
                rr = v11.run_sim_scalar(devc, None, span, observer_vector(observer_distance,theta_x_mrad,theta_y_mrad), n_base=nbase, gamma0_input=float(gamma0))
                if rr:
                    rows.append({"points_per_period":ppp,"photon_energy_eV":rr["photon_energy"]["eV"],"P_larmor":rr["P_larmor"],"relative_linewidth":rr["relative_linewidth"],"P_circ":rr["P_circ"]})
            st.dataframe(pd.DataFrame(rows), width="stretch")
        except Exception as exc:
            st.warning(f"Convergence analysis could not complete: {exc}")

    if analyses["farfield"]:
        st.markdown("### Observer-distance validation")
        rows = []
        try:
            devf = build_selected_device(int(n_periods), error_switches=selected_error_switches())
            for RR in [0.5*float(observer_distance),float(observer_distance),2.0*float(observer_distance)]:
                rr = run_scalar(devf,gamma0,int(n_periods),RR,theta_x_mrad,theta_y_mrad)
                if rr:
                    rows.append({"distance_m":RR,"photon_energy_eV":rr["photon_energy"]["eV"],"P_larmor":rr["P_larmor"],"P_circ":rr["P_circ"]})
            st.dataframe(pd.DataFrame(rows), width="stretch")
        except Exception as exc:
            st.warning(f"Observer-distance validation could not complete: {exc}")

    if analyses["error_ranking"] and field_model=="radia_generated":
        st.markdown("### One-error-at-a-time sensitivity")
        try:
            active=selected_error_switches()
            keys=list(active.keys())
            cases=[("ideal",{k:False for k in keys})]
            for key in keys:
                one={k:False for k in keys}; one[key]=True; cases.append((key,one))
            cases.append(("selected_combination",active))
            rows=[]
            for name,switches in cases:
                devx=build_selected_device(int(n_periods),error_switches=switches)
                span=v11.simulation_span_for_device(float(gamma0),devx,n_periods=int(n_periods))
                nbase=v11.samples_for_periods(int(n_periods),pts_per_period=int(points_per_period),min_pts=max(1200,int(n_periods)*int(points_per_period)),max_pts=max(5000,int(n_periods)*int(points_per_period)+1))
                rr=v11.run_sim(devx,None,span,observer_vector(observer_distance,theta_x_mrad,theta_y_mrad),n_base=nbase,gamma0_input=float(gamma0))
                row=result_summary_row(rr); row["case"]=name; rows.append(row)
            edf=pd.DataFrame(rows)
            base=edf.iloc[0]
            for col in ["photon_energy_eV","average_power_W","relative_linewidth","P_circ","P_lin","radiation_H3_over_H1","radiation_H5_over_H1"]:
                b=float(base[col]); edf[col+"_delta"]=edf[col]-b; edf[col+"_rel_pct"]=np.nan if abs(b)<1e-30 else 100.0*(edf[col]-b)/b
            st.dataframe(edf,width="stretch")
            st.download_button("Download error-sensitivity CSV",edf.to_csv(index=False).encode(),file_name="v11_error_sensitivity.csv",mime="text/csv")
        except Exception as exc:
            st.warning(f"Error sensitivity could not complete: {exc}")

    if analyses["chaos"]:
        st.markdown("### Advanced chaos / MLE")
        st.info("This stays in Advanced because it is not a core insertion-device quality metric.")

    st.markdown("### Export current result")
    summary_df = pd.DataFrame([result_summary_row(r)])
    st.dataframe(summary_df, width="stretch")
    e1,e2 = st.columns(2)
    e1.download_button(
        "Download result summary CSV",
        summary_df.to_csv(index=False).encode(),
        file_name="v11_current_result_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
    e2.download_button(
        "Download complete result JSON",
        result_json_bytes(r),
        file_name="v11_complete_result.json",
        mime="application/json",
        use_container_width=True,
    )

st.divider()
st.subheader(f"Primary parameter scan — x = {axis_title(scan_preset)}")
st.caption(
    "The selected independent variable is scanned across the full requested range. Each successful row receives scalar V11 observables, "
    "and every primary trend plot uses that scan variable on the horizontal axis. RADIA scans run sequentially to avoid duplicating large field maps."
)

if scan_preset == "velocity":
    beta_min = st.number_input(
        "Speed minimum β = v/c", min_value=0.001, max_value=0.999999999,
        value=0.60, step=0.01, format="%.9f", key="beta_min",
    )
    beta_max = st.number_input(
        "Speed maximum β = v/c", min_value=0.001, max_value=0.999999999,
        value=0.95, step=0.01, format="%.9f", key="beta_max",
    )
    st.caption(
        "Points are evenly spaced in speed β=v/c. These exact values become plot x-values and exported "
        "scan_x values; γ is derived only for the relativistic solver."
    )
elif scan_preset == "gamma":
    gmin = st.number_input("γ min", 1.01, 60000.0, 1.25, key="gmin")
    gmax = st.number_input("γ max", 1.01, 60000.0, 60000.0, key="gmax")
    st.caption("Default γ scan spans the full research range; every successful scan point is kept in the output table.")
elif scan_preset == "K":
    kmin = st.number_input("K min", 0.05, 10.0, 0.3)
    kmax = st.number_input("K max", 0.05, 10.0, 1.2)
    if field_model in {"radia_csv", "studio_bridge"}:
        st.warning("K scan cannot regenerate an uploaded fixed field map.")
elif scan_preset == "N_periods":
    nmin = st.number_input("N min", 2, 100, 5)
    nmax = st.number_input("N max", 2, 100, min(100, max(10,int(n_periods))))
elif scan_preset == "observer_distance":
    rmin = st.number_input("R min (m)", 1.0, 1000.0, 20.0)
    rmax = st.number_input("R max (m)", 1.0, 1000.0, 200.0)
else:
    amin = st.number_input("θx min (mrad)", -20.0, 20.0, -2.0)
    amax = st.number_input("θx max (mrad)", -20.0, 20.0, 2.0)

if st.button(f"Run scan across {axis_title(scan_preset)}", type="primary"):
    rows = []
    scan_tracker = None
    active_scan_progress_index = None
    try:
        def tracked_values(values):
            nonlocal_values = list(values)
            tracker = LiveProgressTable(
                st,
                [
                    (f"Scan point {i+1}/{len(nonlocal_values)}", f"{axis_title(scan_preset)} = {float(value):.10e}")
                    for i, value in enumerate(nonlocal_values)
                ],
                title=f"Live calculation progress — {axis_title(scan_preset)} scan",
                session_key="last_primary_scan_progress",
            )
            return nonlocal_values, tracker

        def begin_point(index, value):
            nonlocal_detail = f"Solving {axis_title(scan_preset)}={float(value):.10e}"
            scan_tracker.start(index, nonlocal_detail)

        def finish_point(index, result):
            if result:
                scan_tracker.complete(index, "Scalar trajectory/radiation observables complete")
            else:
                scan_tracker.fail(index, "Solver returned no valid result")

        if scan_preset == "velocity":
            if not float(beta_min) < float(beta_max):
                raise ValueError("Speed minimum must be smaller than speed maximum.")
            vals, scan_tracker = tracked_values(np.linspace(float(beta_min), float(beta_max), int(scan_points)))
            dev = build_selected_device(int(n_periods), error_switches=selected_error_switches())
            for scan_i, beta in enumerate(vals):
                active_scan_progress_index = scan_i
                begin_point(scan_i, beta)
                gamma_for_solver = float(v11.gamma_from_beta(float(beta)))
                rr = run_scalar(
                    dev, gamma_for_solver, int(n_periods), observer_distance,
                    theta_x_mrad, theta_y_mrad
                )
                if rr:
                    rr["scan_x"] = float(beta)
                    rr["beta_v_over_c"] = float(beta)
                    rr["speed_m_per_s"] = float(beta) * float(v11.c0)
                    rr["gamma_from_scan_speed"] = gamma_for_solver
                    rows.append(rr)
                finish_point(scan_i, rr)
        elif scan_preset == "gamma":
            vals, scan_tracker = tracked_values(np.geomspace(float(gmin), float(gmax), int(scan_points)))
            dev = build_selected_device(int(n_periods), error_switches=selected_error_switches())
            for scan_i, val in enumerate(vals):
                active_scan_progress_index = scan_i
                begin_point(scan_i, val)
                rr = run_scalar(
                    dev, val, int(n_periods), observer_distance,
                    theta_x_mrad, theta_y_mrad
                )
                if rr:
                    rr["scan_x"] = val
                    rows.append(rr)
                finish_point(scan_i, rr)

        elif scan_preset == "K":
            if field_model in {"radia_csv", "studio_bridge"}:
                raise ValueError("K scan is unavailable for a fixed uploaded field map.")
            vals, scan_tracker = tracked_values(np.linspace(float(kmin), float(kmax), int(scan_points)))
            lambda_u = (float(radia_period_mm)*1e-3 if field_model == "radia_generated" else 0.05)
            for scan_i, kval in enumerate(vals):
                active_scan_progress_index = scan_i
                begin_point(scan_i, kval)
                target_B = kval * (2*np.pi*v11.me*v11.c0) / (
                    v11.qe * lambda_u
                )
                dev = build_selected_device(int(n_periods), target_B0_T=target_B)
                rr = run_scalar(
                    dev, gamma0, int(n_periods), observer_distance,
                    theta_x_mrad, theta_y_mrad
                )
                if rr:
                    rr["scan_x"] = kval
                    rows.append(rr)
                finish_point(scan_i, rr)

        elif scan_preset == "N_periods":
            vals = np.unique(np.rint(np.linspace(
                int(nmin), int(nmax), int(scan_points)
            )).astype(int))
            vals, scan_tracker = tracked_values(vals)
            for scan_i, val in enumerate(vals):
                active_scan_progress_index = scan_i
                begin_point(scan_i, val)
                dev = build_selected_device(int(val), error_switches=selected_error_switches())
                rr = run_scalar(
                    dev, gamma0, int(val), observer_distance,
                    theta_x_mrad, theta_y_mrad
                )
                if rr:
                    rr["scan_x"] = int(val)
                    rows.append(rr)
                finish_point(scan_i, rr)

        elif scan_preset == "observer_distance":
            vals, scan_tracker = tracked_values(np.geomspace(float(rmin), float(rmax), int(scan_points)))
            dev = build_selected_device(int(n_periods), error_switches=selected_error_switches())
            for scan_i, val in enumerate(vals):
                active_scan_progress_index = scan_i
                begin_point(scan_i, val)
                rr = run_scalar(
                    dev, gamma0, int(n_periods), val,
                    theta_x_mrad, theta_y_mrad
                )
                if rr:
                    rr["scan_x"] = val
                    rows.append(rr)
                finish_point(scan_i, rr)

        else:
            vals, scan_tracker = tracked_values(np.linspace(float(amin), float(amax), int(scan_points)))
            dev = build_selected_device(int(n_periods), error_switches=selected_error_switches())
            for scan_i, val in enumerate(vals):
                active_scan_progress_index = scan_i
                begin_point(scan_i, val)
                rr = run_scalar(
                    dev, gamma0, int(n_periods), observer_distance,
                    val, theta_y_mrad
                )
                if rr:
                    rr["scan_x"] = val
                    rows.append(rr)
                finish_point(scan_i, rr)

        scan_df = enrich_scan_dataframe(pd.DataFrame(rows))
        if scan_preset == "gamma" and len(scan_df):
            scan_df.insert(
                1, "beta_v_over_c",
                [float(v11.beta_from_gamma(float(g))) for g in scan_df["scan_x"]],
            )
        st.session_state.scan_df = scan_df
        st.session_state.scan_name = scan_preset
        st.session_state.pop("representative_gamma_results", None)
        st.session_state.pop("representative_full_results", None)
        st.session_state.pop("selected_scan_deep", None)
        if scan_tracker is not None:
            scan_tracker.finish(
                f"Scan complete — {len(scan_df)} valid result(s) from {len(scan_tracker.rows)} point(s)"
            )
        st.success(
            f"Completed {len(scan_df)} points. Every cross-scan chart below uses "
            f"{axis_title(scan_preset)} as its numeric horizontal axis."
        )
    except Exception as exc:
        if scan_tracker is not None:
            scan_tracker.fail(active_scan_progress_index, str(exc))
        st.exception(exc)

render_saved_progress(st, "last_primary_scan_progress", "Last primary-scan progress table")

if "scan_df" in st.session_state and len(st.session_state.scan_df):
    df = st.session_state.scan_df
    scan_name = st.session_state.scan_name
    rep_idx = render_scan_overview(
        st, df, scan_name, v11=v11,
        representative_count=int(representative_count),
        representative_strategy=representative_strategy,
        comprehensive=(scan_view_mode == "Comprehensive available metrics"),
    )

    st.markdown("### Choose one existing scan point for deep analysis")
    st.caption(
        "The scan plots above use the selected scan variable as the horizontal axis. "
        "A full trajectory / field / waveform / spectrum analysis is run only after you choose "
        "one actual row from that scan. This keeps scan trends separate from single-case diagnostics."
    )

    rep_set = {int(i) for i in np.asarray(rep_idx, dtype=int)}
    scan_axis = axis_title(scan_name)
    options = list(range(len(df)))
    default_pos = int(rep_idx[len(rep_idx)//2]) if len(rep_idx) else len(df)//2

    def _scan_point_label(i):
        val = float(df.iloc[int(i)]["scan_x"])
        star = "★ representative" if int(i) in rep_set else "scan point"
        return f"#{int(i):03d} · {scan_axis}={val:.10e} · {star}"

    selected_scan_index = st.selectbox(
        "Deep-analysis scan point",
        options=options,
        index=max(0, min(default_pos, len(options)-1)),
        format_func=_scan_point_label,
        key=f"deep_scan_point_{scan_name}",
    )

    st.markdown("### Representative full-case analysis suite")
    st.caption(
        "If a full result at every scalar scan row is too expensive, run the proposed representative rows instead. "
        "Each case gets its own spectrum, z-based trajectory and z-based magnetic-field plots; the full scalar scan remains the trend backbone."
    )
    if st.button("Run all representative full cases", key=f"run_representatives_{scan_name}"):
        representative_tracker = LiveProgressTable(
            st,
            [
                (
                    f"Representative case {j+1}",
                    f"{scan_axis}={float(df.iloc[int(ridx)]['scan_x']):.10e}",
                )
                for j, ridx in enumerate(np.asarray(rep_idx, dtype=int))
            ],
            title="Live calculation progress — representative full cases",
            session_key="last_representative_progress",
        )
        try:
            representative_full = []
            progress = st.progress(0.0, text="Running representative full cases…")
            for j, ridx in enumerate(np.asarray(rep_idx, dtype=int)):
                scan_value = float(df.iloc[int(ridx)]["scan_x"])
                representative_tracker.start(j, "Running full trajectory/radiation analysis")
                item = run_full_scan_case(scan_name, scan_value)
                item["scan_index"] = int(ridx)
                representative_full.append(item)
                representative_tracker.complete(j, "Full case complete")
                progress.progress((j+1)/max(len(rep_idx),1), text=f"Representative case {j+1}/{len(rep_idx)} complete")
            progress.empty()
            representative_tracker.finish("All representative full cases complete")
            st.session_state.representative_full_results = {
                "scan_name": scan_name,
                "indices": [int(i) for i in np.asarray(rep_idx, dtype=int)],
                "items": representative_full,
            }
        except Exception as exc:
            representative_tracker.fail(detail=str(exc))
            st.exception(exc)

    render_saved_progress(st, "last_representative_progress", "Last representative-case progress table")

    rep_full_state = st.session_state.get("representative_full_results")
    if rep_full_state and rep_full_state.get("scan_name") == scan_name:
        render_representative_comparison(st, rep_full_state.get("items", []), v11, scan_name=scan_name)

    if st.button("Run selected scan-point deep analysis", key=f"run_selected_deep_{scan_name}"):
        try:
            i = int(selected_scan_index)
            scan_value = float(df.iloc[i]["scan_x"])
            deep_item = run_full_scan_case(scan_name, scan_value)
            deep_item["scan_index"] = i
            st.session_state.selected_scan_deep = deep_item
        except Exception as exc:
            st.exception(exc)

    deep = st.session_state.get("selected_scan_deep")
    if deep and deep.get("scan_name") == scan_name:
        render_selected_point_summary(
            st,
            scan_name,
            int(deep["scan_index"]),
            float(deep["scan_value"]),
            deep["result"],
            v11,
        )
        st.code(
            "\n".join([
                f"scan_variable = {scan_name!r}",
                f"scan_point_index = {int(deep['scan_index'])}",
                f"scan_value = {float(deep['scan_value']):.12e}",
                f"gamma = {float(deep['gamma']):.12e}",
                f"N_periods = {int(deep['n_periods'])}",
                f"observer_distance_m = {float(deep['observer_distance']):.12e}",
                f"theta_x_mrad = {float(deep['theta_x_mrad']):.12e}",
                f"theta_y_mrad = {float(deep['theta_y_mrad']):.12e}",
            ]),
            language="text",
        )
        render_full_results(
            st,
            deep["result"],
            v11,
            dev=deep.get("dev"),
            field_quality_fn=field_quality_metrics,
        )

if field_model == "radia_generated":
    st.divider()
    st.subheader("Error-strength response scan")
    st.caption(
        "This is a second, controlled scan. The x-axis is the physical strength of one selected manufacturing error "
        "(or a multiplier of all currently selected errors), and every plotted y-value is a response observable from the latest V11/RADIA calculation."
    )
    if st.button("Run error-strength scan", key="run_error_strength_scan"):
        try:
            edf, ename = run_error_strength_scan(
                error_sweep_source, int(error_sweep_points), float(error_sweep_multiplier)
            )
            st.session_state.error_strength_scan_df = edf
            st.session_state.error_strength_scan_name = ename
        except Exception as exc:
            st.exception(exc)

    if "error_strength_scan_df" in st.session_state and len(st.session_state.error_strength_scan_df):
        render_scan_overview(
            st,
            st.session_state.error_strength_scan_df,
            st.session_state.error_strength_scan_name,
            v11=v11,
            representative_count=min(int(representative_count), len(st.session_state.error_strength_scan_df)),
            representative_strategy=representative_strategy,
            comprehensive=(scan_view_mode == "Comprehensive available metrics"),
        )

st.divider()
st.caption(
    "RADIA-generated mode uses the official local RADIA solver to generate "
    "B(x,y,z), then V11 interpolates that map during the ODE. The GUI is only "
    "a control/visualization layer; it does not replace the V11 physics engine."
)
