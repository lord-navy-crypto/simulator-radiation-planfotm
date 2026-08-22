from __future__ import annotations
import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import undulator_v11_radia_integrated_v9 as v11
from v11_field_quality_v6 import field_quality_metrics
from reporting.full_results import render_full_results

st.set_page_config(
    page_title="V11 RADIA Radiation Studio",
    page_icon="⚛️",
    layout="wide",
)

st.title("V11 RADIA Radiation Studio — v9 Full Visualization Edition")


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
    "Generalized RADIA insertion-device field → exact end-field V11 tracking → "
    "Liénard–Wiechert radiation → comprehensive scrollable plots / tables / exports"
)

st.info(
    "Workflow: choose settings in the left sidebar → build/preview the magnetic field → "
    "click Run full V11 analysis. Results appear in the Results section below.",
    icon="ℹ️",
)

with st.sidebar:
    st.header("1 · Magnetic field")
    field_label = st.selectbox(
        "Field model",
        [
            "RADIA generated 3D field",
            "RADIA 3D CSV field map",
            "V11 analytic field",
        ],
        index=0,
    )
    field_model = {
        "RADIA generated 3D field": "radia_generated",
        "RADIA 3D CSV field map": "radia_csv",
        "V11 analytic field": "analytic",
    }[field_label]

    st.header("2 · Device")
    device_options = list(v11.list_device_presets())
    device_preset = st.selectbox(
        "Insertion device",
        device_options,
        index=device_options.index("helical") if "helical" in device_options else 0,
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
            radia_error_seed = st.number_input("Manufacturing-error seed",min_value=0,value=20260820,step=1)
            radia_field_sigma_pct = st.number_input("Field-amplitude σ (%)",min_value=0.0,value=0.2,step=0.05)
            radia_z_sigma_um = st.number_input("Longitudinal-position σ (µm)",min_value=0.0,value=20.0,step=1.0)
            radia_xy_sigma_um = st.number_input("Transverse-position σ (µm)",min_value=0.0,value=10.0,step=1.0)
            radia_angle_sigma_mrad = st.number_input("Magnetization-angle σ (mrad)",min_value=0.0,value=0.5,step=0.1)
            radia_gap_asym_um = st.number_input("Gap asymmetry (µm)",min_value=0.0,value=10.0,step=1.0)
            radia_bank_sigma_pct = st.number_input("Bank imbalance (%)",min_value=0.0,value=0.1,step=0.05)
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
        value=20,
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

    st.header("4 · Scan")
    scan_preset = st.selectbox(
        "Scan variable",
        ["gamma", "K", "N_periods", "observer_distance", "angle"],
        index=0,
    )
    scan_points = st.slider("Scan points", 3, 21, 7, 2)

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
if field_model == "radia_csv":
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

def current_radia_options(target_B0_override=None):
    if field_model != "radia_generated":
        return None
    cfg = {
        "field_amplitude":{"enabled":True,"rms_fraction":float(radia_field_sigma_pct)/100.0},
        "longitudinal_position":{"enabled":True,"rms_m":float(radia_z_sigma_um)*1e-6},
        "transverse_position":{"enabled":True,"rms_m":float(radia_xy_sigma_um)*1e-6},
        "magnetization_angle":{"enabled":True,"rms_rad":float(radia_angle_sigma_mrad)*1e-3},
        "gap_asymmetry":{"enabled":True,"rms_m":float(radia_gap_asym_um)*1e-6},
        "bank_strength_imbalance":{"enabled":True,"rms_fraction":float(radia_bank_sigma_pct)/100.0},
    }
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


def build_selected_device(nper: int, target_B0_T=None, error_switches=None):
    if field_model == "radia_csv":
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
            radia_options=current_radia_options(target_B0_T),
        )

    dev = v11.make_default_undulator(
        preset=device_preset,
        field_model="analytic",
        n_periods=nper,
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

if st.button("Build magnetic device / refresh field", type="primary"):
    try:
        with st.spinner("Building field model…"):
            st.session_state.dev = build_selected_device(int(n_periods), error_switches=selected_error_switches())
        st.session_state.pop("full_result", None)
    except Exception as exc:
        st.exception(exc)

if "dev" in st.session_state:
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
st.subheader("Results")
st.caption(
    "This is the main output area. After you click Run full V11 analysis, "
    "all selected plots, metrics and tables will appear here."
)
results_status = st.empty()

st.subheader("Full V11 analysis — one selected point")
st.caption(
    "Runs the V11 trajectory, radiation field, FFT/Stokes, linewidth, "
    "harmonics, energy accounting, quantum monitor and trajectory/phase diagnostics."
)

if st.button("Run full V11 analysis", type="primary"):
    progress = st.progress(0, text="Preparing simulation…")
    run_stage = st.empty()

    try:
        active_errors = selected_error_switches()

        run_stage.info("Stage 1/5 — Building magnetic field model…")
        progress.progress(10, text="Building magnetic field model…")
        dev = build_selected_device(
            int(n_periods), error_switches=active_errors
        )

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

        progress.progress(75, text="Core V11 analysis complete…")
        run_stage.info("Stage 4/5 — Preparing selected diagnostics and comparisons…")

        if compare_ideal and field_model == "radia_generated" and any(active_errors.values()):
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
        else:
            st.session_state.pop("ideal_result", None)

        progress.progress(100, text="Simulation complete.")
        run_stage.success(
            "Stage 5/5 — Complete. Scroll just below this button: your selected results are displayed there.",
            icon="✅",
        )
        results_status.success(
            "Results are ready. The selected output sections are shown below.",
            icon="✅",
        )

    except Exception as exc:
        progress.empty()
        run_stage.error("Simulation stopped because an error occurred.")
        results_status.error("No result was produced. See the error details below.")
        st.exception(exc)

if "full_result" in st.session_state:
    st.success(
        "V11 run complete — selected results are displayed in this section.",
        icon="✅",
    )
    r = st.session_state.full_result

    st.divider()
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
st.subheader("Preset scan")
st.caption(
    "The scan uses V11 scalar analysis for each point. RADIA scans run "
    "sequentially to avoid duplicating large field maps."
)

if scan_preset == "gamma":
    gmin = st.number_input("γ min", 1.01, 60000.0, max(1.25, gamma0/2), key="gmin")
    gmax = st.number_input("γ max", 1.01, 60000.0, min(60000.0, gamma0*2), key="gmax")
elif scan_preset == "K":
    kmin = st.number_input("K min", 0.05, 10.0, 0.3)
    kmax = st.number_input("K max", 0.05, 10.0, 1.2)
    if field_model == "radia_csv":
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

if st.button("Run selected scan"):
    rows = []
    try:
        if scan_preset == "gamma":
            vals = np.geomspace(float(gmin), float(gmax), int(scan_points))
            dev = build_selected_device(int(n_periods), error_switches=selected_error_switches())
            for val in vals:
                rr = run_scalar(
                    dev, val, int(n_periods), observer_distance,
                    theta_x_mrad, theta_y_mrad
                )
                if rr:
                    rr["scan_x"] = val
                    rows.append(rr)

        elif scan_preset == "K":
            if field_model == "radia_csv":
                raise ValueError("K scan is unavailable for a fixed uploaded field map.")
            vals = np.linspace(float(kmin), float(kmax), int(scan_points))
            lambda_u = (float(radia_period_mm)*1e-3 if field_model == "radia_generated" else 0.05)
            for kval in vals:
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

        elif scan_preset == "N_periods":
            vals = np.unique(np.rint(np.linspace(
                int(nmin), int(nmax), int(scan_points)
            )).astype(int))
            for val in vals:
                dev = build_selected_device(int(val), error_switches=selected_error_switches())
                rr = run_scalar(
                    dev, gamma0, int(val), observer_distance,
                    theta_x_mrad, theta_y_mrad
                )
                if rr:
                    rr["scan_x"] = int(val)
                    rows.append(rr)

        elif scan_preset == "observer_distance":
            vals = np.geomspace(float(rmin), float(rmax), int(scan_points))
            dev = build_selected_device(int(n_periods), error_switches=selected_error_switches())
            for val in vals:
                rr = run_scalar(
                    dev, gamma0, int(n_periods), val,
                    theta_x_mrad, theta_y_mrad
                )
                if rr:
                    rr["scan_x"] = val
                    rows.append(rr)

        else:
            vals = np.linspace(float(amin), float(amax), int(scan_points))
            dev = build_selected_device(int(n_periods), error_switches=selected_error_switches())
            for val in vals:
                rr = run_scalar(
                    dev, gamma0, int(n_periods), observer_distance,
                    val, theta_y_mrad
                )
                if rr:
                    rr["scan_x"] = val
                    rows.append(rr)

        st.session_state.scan_df = pd.DataFrame(rows)
        st.session_state.scan_name = scan_preset
    except Exception as exc:
        st.exception(exc)

if "scan_df" in st.session_state and len(st.session_state.scan_df):
    df = st.session_state.scan_df
    scan_name = st.session_state.scan_name
    metric = st.selectbox(
        "Plot scan output",
        [
            "photon_energy_eV",
            "P_larmor",
            "P_circ",
            "P_lin",
            "relative_linewidth",
            "radiation_H3_over_H1",
            "radiation_H5_over_H1",
            "frequency_relative_residual",
        ],
    )
    if metric in df.columns:
        fig = go.Figure(go.Scatter(
            x=df["scan_x"],
            y=df[metric],
            mode="lines+markers",
            name=metric,
        ))
        fig.update_layout(
            xaxis_title=scan_name,
            yaxis_title=metric,
            height=470,
        )
        if scan_name in {"gamma","observer_distance"}:
            fig.update_xaxes(type="log")
        st.plotly_chart(fig, width="stretch")

    st.dataframe(df, width="stretch", height=360)
    st.download_button(
        "Download scan CSV",
        df.to_csv(index=False).encode(),
        file_name=f"v11_{field_model}_{device_preset}_{scan_name}_scan.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "RADIA-generated mode uses the official local RADIA solver to generate "
    "B(x,y,z), then V11 interpolates that map during the ODE. The GUI is only "
    "a control/visualization layer; it does not replace the V11 physics engine."
)
