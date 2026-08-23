from __future__ import annotations
import math
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from magnet_studio.radia_support import load_radia
from magnet_studio.devices.factory import build_device
from magnet_studio.solver.pipeline import solve_model
from magnet_studio.solver.pipeline import sample_on_axis, sample_slice_xz, sample_slice_yz, sample_3d
from magnet_studio.analysis.metrics import analyze, compare_metrics, classify_k
from magnet_studio.analysis.geometry_bounds import union_field_range
from magnet_studio.calibration.target_b0 import calibrate_br
from magnet_studio.visualization.plots import (
    field_lines, slice_heatmap, field_cones, trajectory_plot,
    geometry_view, ideal_error_field_plot, electron_phase_plot
)
from magnet_studio.export.exporters import (
    csv_bytes, json_bytes, hdf5_bytes, pdf_bytes, fieldmap3d_csv_bytes,
    research_package_bytes, validate_research_package_bytes,
)
from magnet_studio.presets import (
    BUILTIN_PRESETS, build_preset, parse_preset, preset_json_bytes,
    runtime_to_widget_state,
)

from integration.bridge import load_transfer_package, discover_saved_transfer_records
from reporting.live_progress import LiveProgressTable, render_saved_progress

st.title("Stage 1 — RADIA Magnet Design, Field Generation & Inspection")
st.caption(
    "Build and inspect RADIA magnetic devices; solve and sample on-axis, 2D, and 3D fields; "
    "analyze trajectories, field integrals, harmonics, phase error, and polarization-related metrics; "
    "then export the same results for downstream trajectory and radiation tools."
)


def _model_history():
    history = st.session_state.setdefault("magnet_model_history", [])
    return history if isinstance(history, list) else []


def _restore_disk_archive_once():
    if st.session_state.get("magnet_disk_archive_restored"):
        return
    history = _model_history()
    known_paths = {str(item.get("saved_path")) for item in history if isinstance(item, dict)}
    for record in discover_saved_transfer_records():
        if str(record.get("saved_path")) not in known_paths:
            history.append(record)
    st.session_state["magnet_model_history"] = history
    if history and st.session_state.get("selected_magnet_model_id") is None:
        st.session_state["selected_magnet_model_id"] = history[-1]["id"]
    st.session_state["magnet_disk_archive_restored"] = True


def _safe_record_stem(name):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip()).strip("._")
    return stem or "radia_magnet_model"


def _save_transfer_file(package, record_name, created_utc):
    """Save every successful build as a separate local transfer ZIP."""
    archive_dir = Path(__file__).resolve().parents[2] / "saved_models"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = created_utc.replace("-", "").replace(":", "").replace("+00:00", "Z").replace(".", "")
    path = archive_dir / f"{_safe_record_stem(record_name)}_{stamp}.zip"
    serial = 2
    while path.exists():
        path = archive_dir / f"{_safe_record_stem(record_name)}_{stamp}_{serial}.zip"
        serial += 1
    path.write_bytes(bytes(package))
    return str(path)


def _register_model_record(record):
    history = _model_history()
    history.append(record)
    st.session_state["magnet_model_history"] = history
    st.session_state["selected_magnet_model_id"] = record["id"]
    st.session_state["stage1_archive_selector"] = record["id"]
    st.session_state["stage2_saved_model_selector"] = record["id"]
    if record.get("bridge") is not None:
        st.session_state["magnet_scan_bridge"] = record["bridge"]
    if record.get("visuals") is not None:
        st.session_state["magnet_completed_visuals"] = record["visuals"]


def render_model_archive():
    history = _model_history()
    st.subheader("Generated magnet-model archive")
    if not history:
        st.info("No completed models are archived in this session yet. Every successful build will be saved separately here.")
        return
    options = [item["id"] for item in history]
    selected = st.session_state.get("selected_magnet_model_id", options[-1])
    if selected not in options:
        selected = options[-1]
    selected_id = st.selectbox(
        "Active generated model",
        options,
        index=options.index(selected),
        format_func=lambda rid: next(
            (f"{r['name']} · {r['created_utc']}" for r in history if r["id"] == rid), rid
        ),
        key="stage1_archive_selector",
    )
    record = next(r for r in history if r["id"] == selected_id)
    st.session_state["selected_magnet_model_id"] = selected_id
    if record.get("bridge") is not None:
        st.session_state["magnet_scan_bridge"] = record["bridge"]
    if record.get("visuals") is not None:
        st.session_state["magnet_completed_visuals"] = record["visuals"]
    st.dataframe(
        pd.DataFrame([
            {
                "name": r["name"], "created_utc": r["created_utc"],
                "device": r.get("parameters", {}).get("device"),
                "period_mm": r.get("parameters", {}).get("period_mm"),
                "periods": r.get("parameters", {}).get("periods"),
                "Stage_2_ready": r.get("bridge") is not None,
                "saved_file": r.get("saved_path", "session only"),
            }
            for r in history
        ]),
        width="stretch", hide_index=True,
    )
    st.download_button(
        "Download selected model transfer ZIP",
        data=record["package"],
        file_name=record["file_name"],
        mime="application/zip",
        key=f"download_archived_model_{selected_id}",
        width="stretch",
    )


_restore_disk_archive_once()


def render_completed_entity_gallery(payload):
    """Restore the last successful Stage-1 visual result after page reruns.

    Streamlit reruns a page when the user navigates away and returns.  Raw
    arrays/geometry are therefore kept in session state and figures are rebuilt
    here instead of relying on the one-time state of the Build button.
    """
    if not isinstance(payload, dict):
        return
    z = np.asarray(payload["z_mm"], dtype=float)
    B = np.asarray(payload["B_T"], dtype=float)
    metrics = dict(payload["metrics"])
    blocks = list(payload.get("blocks", []))
    params = dict(payload.get("params", {}))
    geometry_limit_saved = int(payload.get("geometry_limit", 600))
    slice_data_saved = payload.get("slice_data")
    slice_axis_saved = payload.get("slice_axis")
    slice_plane_saved = str(payload.get("slice_plane", "XZ"))
    grid3_saved = payload.get("grid3")
    field3_saved = payload.get("field3")

    st.divider()
    st.subheader("Restored completed magnet entity & field gallery")
    st.success(
        "The last successfully solved Stage-1 entity is restored from this session. "
        "These figures describe the completed result named below, not unsolved sidebar edits."
    )
    st.caption(
        f"Completed device: {params.get('device', 'unknown')} · "
        f"period λu={float(params.get('period_mm', np.nan)):.6g} mm · "
        f"periods={params.get('periods', 'unknown')} · "
        f"magnet blocks={len(blocks)}"
    )

    cards = st.columns(3)
    cards[0].metric("Peak transverse field |B⊥|", f"{float(metrics['Bperp_peak_T']):.6g} T")
    cards[1].metric("Undulator K (peak)", f"{float(metrics['K_peak']):.6g}")
    cards[2].metric("Magnetic regime", str(payload.get("classification", "unclassified")))

    tabs = st.tabs([
        "3D magnet entity", "On-axis field", "2D field slice",
        "3D vector field", "Electron trajectory", "Electron phase",
    ])
    with tabs[0]:
        if blocks:
            st.plotly_chart(
                geometry_view(blocks, geometry_limit_saved, True),
                width="stretch", key="restored_stage1_geometry",
            )
            st.caption(
                "Restored solved geometry: cuboids are the actual block centres/dimensions supplied "
                "to RADIA; cones indicate magnetization/easy-axis directions."
            )
            with st.expander("Completed magnetic-block entity table", expanded=False):
                st.dataframe(
                    pd.DataFrame(blocks[:min(1000, len(blocks))]),
                    width="stretch", hide_index=True,
                )
        else:
            st.info("The completed result does not contain magnetic-block geometry records.")

    with tabs[1]:
        st.plotly_chart(
            field_lines(z, B), width="stretch", key="restored_stage1_axis_field"
        )

    with tabs[2]:
        if slice_data_saved is None or slice_axis_saved is None:
            st.info("The 2D field slice was disabled for this completed calculation.")
        else:
            comp = 1 if abs(metrics["By_peak_T"]) >= abs(metrics["Bx_peak_T"]) else 0
            axis_saved, z2_saved = slice_axis_saved
            st.plotly_chart(
                slice_heatmap(
                    np.asarray(axis_saved), np.asarray(z2_saved),
                    np.asarray(slice_data_saved), comp,
                    "x" if slice_plane_saved == "XZ" else "y",
                ),
                width="stretch", key="restored_stage1_2d_field",
            )

    with tabs[3]:
        if field3_saved is None or grid3_saved is None:
            st.info("The sparse 3D field map was disabled for this completed calculation.")
        else:
            gx, gy, gz = grid3_saved
            st.plotly_chart(
                field_cones(np.asarray(gx), np.asarray(gy), np.asarray(gz), np.asarray(field3_saved)),
                width="stretch", key="restored_stage1_3d_field",
            )

    with tabs[4]:
        st.plotly_chart(
            trajectory_plot(z, metrics["trajectory"]),
            width="stretch", key="restored_stage1_trajectory",
        )

    with tabs[5]:
        st.plotly_chart(
            electron_phase_plot(metrics["electron_phase"]),
            width="stretch", key="restored_stage1_electron_phase",
        )
        eph = float(metrics.get("electron_phase_error_rms_deg", np.nan))
        st.metric("Trajectory-derived electron phase error RMS", "n/a" if not math.isfinite(eph) else f"{eph:.5g}°")

with st.sidebar:
    st.header("Presets")
    builtin_name = st.selectbox("Built-in preset", list(BUILTIN_PRESETS), key="preset_builtin_name")
    if st.button("Load built-in preset", width="stretch"):
        runtime = parse_preset(BUILTIN_PRESETS[builtin_name])
        st.session_state.update(runtime_to_widget_state(runtime))
        st.session_state["preset_extensions"] = runtime["extensions"]
        st.session_state["preset_study_defaults"] = runtime["study_defaults"]
        st.rerun()
    uploaded_preset = st.file_uploader("Import preset (.json)", type=["json"], key="preset_upload")
    if uploaded_preset is not None:
        try:
            imported_runtime = parse_preset(uploaded_preset.getvalue())
            st.caption(
                f"Validated: {imported_runtime['metadata'].get('name', 'Unnamed preset')} "
                f"({imported_runtime['parameters']['device']})"
            )
            for warning in imported_runtime["warnings"]:
                st.warning(warning)
            if st.button("Apply imported preset", type="primary", width="stretch"):
                st.session_state.update(runtime_to_widget_state(imported_runtime))
                st.session_state["preset_extensions"] = imported_runtime["extensions"]
                st.session_state["preset_study_defaults"] = imported_runtime["study_defaults"]
                st.rerun()
        except Exception as exc:
            st.error(f"Preset rejected: {exc}")
    uploaded_transfer = st.file_uploader(
        "Import completed Magnet Studio transfer (.zip)", type=["zip"],
        key="completed_transfer_upload",
        help="Loads a previously generated 3-D field directly into Stage 2 without solving again.",
    )
    if uploaded_transfer is not None and st.button(
        "Use imported field in Stage 2", width="stretch", key="apply_completed_transfer"
    ):
        try:
            imported_bytes = uploaded_transfer.getvalue()
            imported_bridge = load_transfer_package(imported_bytes)
            created_utc = datetime.now(timezone.utc).isoformat()
            imported_name = Path(uploaded_transfer.name).stem
            saved_path = _save_transfer_file(imported_bytes, imported_name, created_utc)
            _register_model_record({
                "id": f"imported_{created_utc}_{len(_model_history())+1}",
                "name": imported_name,
                "created_utc": created_utc,
                "file_name": Path(saved_path).name,
                "saved_path": saved_path,
                "package": bytes(imported_bytes),
                "bridge": imported_bridge,
                "visuals": None,
                "parameters": dict(imported_bridge.get("parameters", {})),
                "metrics": dict(imported_bridge.get("metrics", {})),
            })
            st.success("Completed magnetic field loaded. Open Stage 2 from the application navigation.")
        except Exception as exc:
            st.error(f"Transfer package rejected: {exc}")
    st.divider()
    st.header("Device")
    device = st.selectbox("Type", ["Planar", "Helical", "Elliptical", "APPLE-II", "Wiggler"], key="cfg_device")
    model_record_name = st.text_input(
        "Saved model name", value=f"{device} model", key="cfg_model_record_name",
        help="Every successful build is stored as a separate transfer ZIP and archive entry under this name.",
    )
    period_mm = st.number_input("Period λu (mm)", min_value=1.0, value=50.0, step=1.0, key="cfg_period_mm")
    periods = st.number_input("Number of periods", min_value=1, value=20, step=1, key="cfg_periods")
    gap_mm = st.number_input("Magnetic gap (mm)", min_value=0.5, value=12.0, step=0.5, key="cfg_gap_mm")
    blocks_per_period = st.selectbox(
        "Blocks per period", [4, 8, 12, 16],
        index=1 if device in ("Helical", "Elliptical") else 0, key="cfg_blocks_per_period"
    )

    st.header("Magnet blocks")
    block_width_mm = st.number_input("Block width x (mm)", min_value=0.1, value=10.0, key="cfg_block_width_mm")
    block_height_mm = st.number_input("Block height / radial thickness (mm)", min_value=0.1, value=10.0, key="cfg_block_height_mm")
    longitudinal_fill = st.slider("Longitudinal fill factor", 0.50, 0.99, 0.90, 0.01, key="cfg_longitudinal_fill")
    br_t = st.number_input("Remanent induction Br (T)", min_value=0.01, value=1.20, step=0.05, key="cfg_br_t")

    st.header("Target B0 calibration")
    target_b0_enabled = st.checkbox("Calibrate Br to target B0", value=False, key="cfg_target_b0_enabled")
    target_b0_t = st.number_input(
        "Target B0 (T)", min_value=0.001, value=0.15, step=0.01,
        disabled=not target_b0_enabled, key="cfg_target_b0_t"
    )
    b0_mode = st.selectbox(
        "B0 definition",
        ["Central-period peak B⊥", "Central 3-period peak B⊥", "Global peak B⊥"],
        disabled=not target_b0_enabled, key="cfg_b0_definition"
    )

    st.header("Material / solve")
    material_mode = st.selectbox("Magnet model", ["Fixed remanence", "Linear NdFeB + relaxation"], key="cfg_material_mode")
    mu_parallel = st.number_input("μr parallel", min_value=1.0, value=1.05, step=0.01, key="cfg_mu_parallel")
    mu_perpendicular = st.number_input("μr perpendicular", min_value=1.0, value=1.05, step=0.01, key="cfg_mu_perpendicular")
    seg_n = st.selectbox("Magnet subdivision", [1, 2, 3], index=0, key="cfg_seg_n")
    relax = (material_mode == "Linear NdFeB + relaxation") and st.checkbox(
        "Run RADIA relaxation", value=True, key="cfg_relax"
    )
    precision = st.number_input("Relaxation precision (T)", min_value=1e-7, value=1e-4, format="%.1e", key="cfg_precision")
    max_iter = st.number_input("Relaxation max iterations", min_value=1, value=1000, step=100, key="cfg_max_iter")

    st.header("Device-specific")
    ellipticity = st.slider("Ellipticity", 0.0, 1.0, 0.5, 0.01, disabled=device != "Elliptical", key="cfg_ellipticity")
    apple_phase_deg = st.slider(
        "APPLE-II magnetic row phase (deg)", -180.0, 180.0, 90.0, 1.0,
        disabled=device != "APPLE-II", key="cfg_apple_phase_deg"
    )
    apple_shift_mode = st.selectbox(
        "APPLE-II shift mode", ["Antiparallel", "Parallel"],
        disabled=device != "APPLE-II", key="cfg_apple_shift_mode"
    )
    if device == "APPLE-II":
        st.caption(
            "Prototype four-array geometry; phase is implemented as real longitudinal "
            "array displacement Δz = φ λu / 360°."
        )

    st.header("Manufacturing error model")
    errors_enabled = st.checkbox("Enable manufacturing errors", value=False, key="cfg_errors_enabled")
    field_error_pct = st.number_input("Field amplitude error σ (%)", min_value=0.0, value=1.0, step=0.1, disabled=not errors_enabled, key="cfg_field_error_pct")
    longitudinal_error_mm = st.number_input("Longitudinal position error σ (mm)", min_value=0.0, value=0.05, step=0.01, disabled=not errors_enabled, key="cfg_longitudinal_error_mm")
    transverse_error_mm = st.number_input("Transverse position error σ (mm)", min_value=0.0, value=0.05, step=0.01, disabled=not errors_enabled, key="cfg_transverse_error_mm")
    angle_error_deg = st.number_input("Magnetization angle error σ (deg)", min_value=0.0, value=0.5, step=0.1, disabled=not errors_enabled, key="cfg_angle_error_deg")
    gap_asymmetry_mm = st.number_input("Gap asymmetry (mm)", value=0.0, step=0.01, disabled=not errors_enabled, key="cfg_gap_asymmetry_mm")
    bank_imbalance_pct = st.number_input("Bank strength imbalance (%)", value=0.0, step=0.1, disabled=not errors_enabled, key="cfg_bank_imbalance_pct")
    error_seed = st.number_input("Random seed", min_value=0, value=12345, step=1, disabled=not errors_enabled, key="cfg_error_seed")
    compare_ideal = st.checkbox("Compute ideal-vs-error comparison", value=True, disabled=not errors_enabled, key="cfg_compare_ideal")

    st.header("Field sampling")
    axis_samples = st.slider("On-axis samples", 100, 4000, 1000, 100, key="cfg_axis_samples")
    field_margin_periods = st.number_input(
        "Longitudinal field margin (periods)", min_value=0.0, value=1.0, step=0.5,
        help="Added beyond the actual outer magnet-block edges for fringe-field integrals.", key="cfg_field_margin_periods"
    )
    electron_energy_GeV = st.number_input("Electron energy (GeV)", min_value=0.01, value=3.0, step=0.1, key="cfg_electron_energy_GeV")
    make_2d = st.checkbox("Calculate 2D field slice", value=True, key="cfg_make_2d")
    make_3d = st.checkbox("Calculate sparse 3D field map", value=True, key="cfg_make_3d")
    transverse_half_mm = st.number_input("Transverse map half-width (mm)", min_value=0.1, value=5.0, step=0.5, key="cfg_transverse_half_mm")
    geometry_limit = st.slider("Maximum blocks in 3D geometry viewer", 100, 1200, 600, 100, key="cfg_geometry_limit")

current_params = {
    "device": device, "period_mm": float(period_mm), "periods": int(periods),
    "gap_mm": float(gap_mm), "blocks_per_period": int(blocks_per_period),
    "block_width_mm": float(block_width_mm), "block_height_mm": float(block_height_mm),
    "longitudinal_fill": float(longitudinal_fill), "br_t": float(br_t),
    "material_mode": material_mode, "mu_parallel": float(mu_parallel),
    "mu_perpendicular": float(mu_perpendicular),
    "segmentation": (int(seg_n), int(seg_n), int(seg_n)),
    "ellipticity": float(ellipticity), "apple_phase_deg": float(apple_phase_deg),
    "apple_shift_mode": apple_shift_mode, "errors_enabled": bool(errors_enabled),
    "field_error_pct": float(field_error_pct),
    "longitudinal_error_mm": float(longitudinal_error_mm),
    "transverse_error_mm": float(transverse_error_mm),
    "angle_error_deg": float(angle_error_deg), "gap_asymmetry_mm": float(gap_asymmetry_mm),
    "bank_imbalance_pct": float(bank_imbalance_pct), "error_seed": int(error_seed),
    "target_b0_enabled": bool(target_b0_enabled), "target_b0_t": float(target_b0_t),
    "b0_definition": b0_mode,
}
current_settings = {
    "axis_samples": int(axis_samples), "field_margin_periods": float(field_margin_periods),
    "electron_energy_GeV": float(electron_energy_GeV), "relax": bool(relax),
    "precision": float(precision), "max_iter": int(max_iter), "method": 4,
    "calculate_2d": bool(make_2d), "calculate_3d": bool(make_3d),
    "transverse_half_width_mm": float(transverse_half_mm),
}
current_ui = {"compare_ideal": bool(compare_ideal), "geometry_limit": int(geometry_limit)}

st.download_button(
    "Export current preset (.json)",
    preset_json_bytes(
        current_params, current_settings, name=f"{device} requested configuration",
        ui_settings=current_ui,
        study_defaults=st.session_state.get("preset_study_defaults"),
        extensions=st.session_state.get("preset_extensions"),
    ),
    "radia_magnet_preset_v1.json", "application/json",
    on_click="ignore", width="stretch",
)

run = st.button("Build + Solve + Analyze", type="primary", width="stretch")

if run:
    magnet_tracker = LiveProgressTable(
        st,
        [
            ("Initialize RADIA", "Load solver and clear previous objects"),
            ("Target B0 calibration", "Calibrate Br when enabled"),
            ("Build magnet geometry", f"{device}, {int(periods)} periods"),
            ("Solve magnetic model", material_mode),
            ("Sample on-axis field", f"{int(axis_samples)} samples"),
            ("Calculate 2D field slice", "Enabled" if make_2d else "Disabled"),
            ("Calculate 3D field map", "Enabled" if make_3d else "Disabled"),
            ("Publish Stage-2 field", "Session bridge"),
        ],
        title="Live calculation progress — magnet build and field solve",
        session_key="last_magnet_build_progress",
    )
    try:
        st.session_state.pop("magnet_scan_bridge", None)
        magnet_tracker.start(0, "Loading RADIA module")
        rad = load_radia()
        if hasattr(rad, "UtiDelAll"):
            rad.UtiDelAll()
        magnet_tracker.complete(0, "RADIA ready")

        params = dict(current_params)
        created_utc = datetime.now(timezone.utc).isoformat()
        record_name = str(model_record_name).strip() or f"{device} model"
        params.update({
            "relax_enabled": bool(relax), "relax_precision_t": float(precision),
            "relax_max_iterations": int(max_iter), "axis_samples": int(axis_samples),
            "field_margin_periods": float(field_margin_periods),
            "electron_energy_GeV": float(electron_energy_GeV),
            "calculate_2d_slice": bool(make_2d),
            "calculate_3d_field_map": bool(make_3d),
            "transverse_map_half_width_mm": float(transverse_half_mm),
            "model_record_name": record_name,
            "model_record_created_utc": created_utc,
        })
        progress = st.progress(0, text="Preparing model…")

        calibration_history = []
        if target_b0_enabled:
            magnet_tracker.start(1, "Iterating Br toward requested B0")
            calibrated_br, calibration_history = calibrate_br(
                rad, device, params, float(target_b0_t),
                mode=b0_mode,
                relax=bool(relax), precision=float(precision), max_iter=int(max_iter)
            )
            params["br_t"] = float(calibrated_br)
            progress.progress(12, text=f"B0 calibration complete: Br={calibrated_br:.6g} T")
            if hasattr(rad, "UtiDelAll"):
                rad.UtiDelAll()
            magnet_tracker.complete(1, f"Calibrated Br={calibrated_br:.6g} T")
        else:
            magnet_tracker.skip(1, "Target-B0 calibration disabled")

        # Build models first. Sampling range is derived from actual generated geometry,
        # including APPLE-II row shifts and random longitudinal block errors.
        ideal_model = None
        magnet_tracker.start(2, "Generating RADIA block geometry")
        if errors_enabled and compare_ideal:
            p_ideal = dict(params)
            p_ideal["errors_enabled"] = False
            ideal_model = build_device(rad, device, p_ideal)

        model = build_device(rad, device, params)
        progress.progress(25, text=f"Generated {len(model['blocks'])} magnetic blocks.")
        magnet_tracker.complete(2, f"Generated {len(model['blocks'])} magnetic blocks")

        magnet_tracker.start(3, "Solving magnetization / relaxation")
        ideal_rlx = None
        if ideal_model is not None:
            ideal_rlx = solve_model(
                rad, ideal_model, relax=bool(relax),
                precision=float(precision), max_iter=int(max_iter), method=4
            )

        rlx = solve_model(
            rad, model, relax=bool(relax),
            precision=float(precision), max_iter=int(max_iter), method=4
        )
        progress.progress(40, text="RADIA solve stage converged / completed.")
        magnet_tracker.complete(3, "RADIA solve completed")

        magnet_tracker.start(4, "Sampling on-axis Bx, By, Bz and calculating metrics")
        range_models = [model] + ([ideal_model] if ideal_model is not None else [])
        z_lo, z_hi = union_field_range(
            range_models, float(period_mm), float(field_margin_periods)
        )
        z = np.linspace(z_lo, z_hi, int(axis_samples))
        params["field_range_mm"] = [float(z_lo), float(z_hi)]
        B = sample_on_axis(rad, model["obj"], z)
        metrics = analyze(z, B, float(period_mm), float(electron_energy_GeV))

        Bideal = None
        ideal_metrics = None
        if ideal_model is not None:
            Bideal = sample_on_axis(rad, ideal_model["obj"], z)
            ideal_metrics = analyze(z, Bideal, float(period_mm), float(electron_energy_GeV))
        progress.progress(58, text="Geometry-derived on-axis field range sampled.")
        magnet_tracker.complete(4, f"Sampled {len(z)} longitudinal points")

        slice_data = None
        slice_axis = None
        slice_plane = "XZ"
        if make_2d:
            magnet_tracker.start(5, "Sampling transverse-longitudinal field slice")
            t = np.linspace(-float(transverse_half_mm), float(transverse_half_mm), 31)
            z2 = np.linspace(z_lo, z_hi, min(181, max(61, int(axis_samples)//5)))
            if abs(metrics["By_peak_T"]) >= abs(metrics["Bx_peak_T"]):
                slice_data = sample_slice_xz(rad, model["obj"], t, z2, 0.0)
                slice_plane = "XZ"
            else:
                slice_data = sample_slice_yz(rad, model["obj"], t, z2, 0.0)
                slice_plane = "YZ"
            slice_axis = (t, z2)
            magnet_tracker.complete(5, f"2D {slice_plane} slice complete")
        else:
            magnet_tracker.skip(5, "2D field slice disabled")
        progress.progress(72, text="2D map complete." if make_2d else "2D map skipped.")

        field3 = None
        grid3 = None
        if make_3d:
            magnet_tracker.start(6, "Sampling sparse 3D vector field")
            x3 = np.linspace(-float(transverse_half_mm), float(transverse_half_mm), 5)
            y3 = np.linspace(-float(transverse_half_mm), float(transverse_half_mm), 5)
            z3 = np.linspace(z_lo, z_hi, 17)
            field3 = sample_3d(rad, model["obj"], x3, y3, z3)
            grid3 = (x3, y3, z3)
            magnet_tracker.complete(6, f"3D map complete: {len(x3)}×{len(y3)}×{len(z3)}")
        else:
            magnet_tracker.skip(6, "3D field map disabled")
        progress.progress(90, text="3D field map complete." if make_3d else "3D map skipped.")

        classification = classify_k(metrics["K_peak"])
        completed_visuals = {
            "z_mm": np.asarray(z, dtype=float).copy(),
            "B_T": np.asarray(B, dtype=float).copy(),
            "metrics": metrics,
            "blocks": [dict(block) for block in model.get("blocks", [])],
            "params": dict(params),
            "classification": str(classification),
            "geometry_limit": int(geometry_limit),
            "slice_data": None if slice_data is None else np.asarray(slice_data, dtype=float).copy(),
            "slice_axis": None if slice_axis is None else tuple(np.asarray(v, dtype=float).copy() for v in slice_axis),
            "slice_plane": str(slice_plane),
            "grid3": None if grid3 is None else tuple(np.asarray(v, dtype=float).copy() for v in grid3),
            "field3": None if field3 is None else np.asarray(field3, dtype=float).copy(),
        }
        comparison = compare_metrics(ideal_metrics, metrics) if ideal_metrics is not None else None
        run_metadata = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "radia_module": getattr(rad, "__file__", None),
            "radia_relaxation_result": rlx,
            "radia_relaxation_converged": rlx is not None,
            "calibration_history": calibration_history,
        }
        package = research_package_bytes(
            params, metrics, z, B,
            grid3=grid3, field3=field3,
            blocks=model.get("blocks"), comparison=comparison,
            run_metadata=run_metadata,
        )
        package_status = validate_research_package_bytes(package)
        bridge = load_transfer_package(package) if field3 is not None and grid3 is not None else None
        saved_path = _save_transfer_file(package, record_name, created_utc)
        record_id = f"generated_{created_utc}_{len(_model_history())+1}"
        _register_model_record({
            "id": record_id,
            "name": record_name,
            "created_utc": created_utc,
            "file_name": Path(saved_path).name,
            "saved_path": saved_path,
            "package": bytes(package),
            "bridge": bridge,
            "visuals": completed_visuals,
            "parameters": dict(params),
            "metrics": {k: v for k, v in metrics.items() if k not in ("trajectory", "electron_phase")},
        })

        magnet_tracker.start(7, "Publishing realized field and metadata")
        if bridge is not None:
            st.session_state["magnet_scan_bridge"] = bridge
            st.success(
                f"Stage 1 complete: ‘{record_name}’ was saved separately and selected for Stage 2. "
                "You can generate more models without losing this record."
            )
            magnet_tracker.complete(7, "Stage-2 field bridge ready")
        else:
            st.warning(
                "Stage 1 analysis completed, but the 3-D field map was disabled. Enable "
                "‘Calculate sparse 3D field map’ and rerun before continuing to Stage 2."
            )
            magnet_tracker.skip(7, "No 3D map available to publish")
        progress.progress(100, text="Complete.")
        magnet_tracker.finish("Magnet build, solve, and field analysis complete")
        # Publish the baseline only after the magnetic model, field sampling, and
        # primary analysis have all completed successfully. Advanced analysis below
        # consumes this exact realized configuration (including calibrated Br).
        st.session_state["magnet_study_source"] = {
            "base_parameters": dict(params),
            "settings": dict(current_settings),
            "study_defaults": dict(st.session_state.get("preset_study_defaults", {})),
            "extensions": dict(st.session_state.get("preset_extensions", {})),
        }

        eph = metrics["electron_phase_error_rms_deg"]

        # Persist raw completed-result data, not Plotly figure objects.  When
        # Streamlit reruns after page navigation the figures can be rebuilt
        # exactly and the completed entity gallery no longer disappears.
        st.session_state["magnet_completed_visuals"] = completed_visuals

        st.subheader("Key results")

        # Two rows of three cards are much more readable than six compressed
        # cards on laptops / narrow browser windows.
        r1 = st.columns(3)
        r1[0].metric(
            "Peak transverse field |B⊥|",
            f"{metrics['Bperp_peak_T']:.6g} T",
            help="Maximum sqrt(Bx² + By²) over the sampled on-axis field range."
        )
        r1[1].metric(
            "Undulator K (peak)",
            f"{metrics['K_peak']:.6g}",
            help="Kpeak = 0.934 × max(|B⊥|)[T] × λu[cm]."
        )
        r1[2].metric(
            "Remanent induction Br",
            f"{params['br_t']:.6g} T",
            help="Actual Br used after optional target-B0 calibration."
        )

        r2 = st.columns(3)
        r2[0].metric(
            "Kx / Ky",
            f"{metrics['Kx_peak']:.6g} / {metrics['Ky_peak']:.6g}",
            help="Component K amplitudes from By and Bx respectively."
        )
        r2[1].metric(
            "3rd harmonic H3/H1",
            f"{metrics['H3_over_H1']:.6e}",
            help="FFT amplitude ratio of the dominant transverse field component."
        )
        r2[2].metric(
            "Electron phase error RMS",
            "n/a" if not math.isfinite(eph) else f"{eph:.6g}°",
            help="Trajectory/slippage-derived electron phase-error RMS."
        )

        # Exact-value table: preserves the same computed results while making
        # units and definitions visible in one place.
        key_rows = [
            {
                "Quantity": "Peak transverse field |B⊥|",
                "Value": float(metrics["Bperp_peak_T"]),
                "Unit": "T",
            },
            {
                "Quantity": "K peak",
                "Value": float(metrics["K_peak"]),
                "Unit": "dimensionless",
            },
            {
                "Quantity": "Kx peak",
                "Value": float(metrics["Kx_peak"]),
                "Unit": "dimensionless",
            },
            {
                "Quantity": "Ky peak",
                "Value": float(metrics["Ky_peak"]),
                "Unit": "dimensionless",
            },
            {
                "Quantity": "K vector norm",
                "Value": float(metrics["K_vector_norm"]),
                "Unit": "dimensionless",
            },
            {
                "Quantity": "H3/H1",
                "Value": float(metrics["H3_over_H1"]),
                "Unit": "ratio",
            },
            {
                "Quantity": "H5/H1",
                "Value": float(metrics["H5_over_H1"]),
                "Unit": "ratio",
            },
            {
                "Quantity": "Electron phase error RMS",
                "Value": float(eph) if math.isfinite(eph) else None,
                "Unit": "deg",
            },
            {
                "Quantity": "Zero-crossing field phase RMS",
                "Value": (
                    float(metrics["zero_crossing_field_phase_rms_deg"])
                    if math.isfinite(metrics["zero_crossing_field_phase_rms_deg"])
                    else None
                ),
                "Unit": "deg",
            },
            {
                "Quantity": "I1x / I1y",
                "Value": f"{metrics['I1x_Tm']:.9g} / {metrics['I1y_Tm']:.9g}",
                "Unit": "T·m",
            },
            {
                "Quantity": "I2x / I2y",
                "Value": f"{metrics['I2x_Tm2']:.9g} / {metrics['I2y_Tm2']:.9g}",
                "Unit": "T·m²",
            },
        ]

        with st.expander("Detailed numerical results", expanded=False):
            st.dataframe(
                pd.DataFrame(key_rows),
                width="stretch",
                hide_index=True,
            )

        st.info(
            f"**Field integration range:** {z_lo:.3f} → {z_hi:.3f} mm  "
            f"\n\n**Fringe-field margin:** {float(field_margin_periods):.2f} period(s)  "
            f"\n\n**Magnetic regime:** {classification}"
        )

        if device == "Wiggler":
            if metrics["K_peak"] >= 3:
                st.success(f"Wiggler mode — Kpeak={metrics['K_peak']:.3g}; {classification}.")
            else:
                st.warning(
                    f"Wiggler mode selected, but Kpeak={metrics['K_peak']:.3g}; "
                    f"{classification}. Increase field/period if a high-K wiggler is intended."
                )
        else:
            st.info(f"Computed magnetic regime: {classification}.")

        tabs = st.tabs([
            "On-axis field", "2D map", "3D field map", "3D magnet geometry",
            "Trajectory", "Electron phase", "Ideal comparison", "Metrics & export"
        ])

        with tabs[0]:
            st.plotly_chart(field_lines(z, B), width="stretch")

        with tabs[1]:
            if slice_data is None:
                st.info("2D map was disabled.")
            else:
                comp = 1 if abs(metrics["By_peak_T"]) >= abs(metrics["Bx_peak_T"]) else 0
                axis, z2 = slice_axis
                st.plotly_chart(
                    slice_heatmap(axis, z2, slice_data, comp, "x" if slice_plane == "XZ" else "y"),
                    width="stretch"
                )

        with tabs[2]:
            if field3 is None:
                st.info("3D field map was disabled.")
            else:
                st.plotly_chart(field_cones(*grid3, field3), width="stretch")
                st.download_button(
                    "Export V11-compatible 3D field map CSV",
                    fieldmap3d_csv_bytes(*grid3, field3),
                    "radia_3d_field_map.csv", "text/csv", on_click="ignore"
                )

        with tabs[3]:
            st.plotly_chart(
                geometry_view(model["blocks"], int(geometry_limit), True),
                width="stretch"
            )
            st.caption(
                "Cuboids use the actual centres/dimensions passed to RADIA; cones show "
                "easy-axis/magnetization directions."
            )
            st.dataframe(pd.DataFrame(model["blocks"][:min(1000, len(model["blocks"]))]), width="stretch")

        with tabs[4]:
            st.plotly_chart(trajectory_plot(z, metrics["trajectory"]), width="stretch")
            st.caption("Ultra-relativistic small-angle trajectory from the sampled transverse field.")

        with tabs[5]:
            st.plotly_chart(electron_phase_plot(metrics["electron_phase"]), width="stretch")
            st.metric(
                "Trajectory-derived electron phase error RMS",
                "n/a" if not math.isfinite(eph) else f"{eph:.5g}°"
            )
            zc = metrics["zero_crossing_field_phase_rms_deg"]
            st.metric(
                "Zero-crossing field phase RMS (diagnostic only)",
                "n/a" if not math.isfinite(zc) else f"{zc:.5g}°"
            )
            st.caption(
                "Electron phase uses longitudinal slippage from x′ and y′ and removes the "
                "best-fit linear slippage. The zero-crossing metric is retained only as a "
                "field-shape diagnostic."
            )

        with tabs[6]:
            if ideal_metrics is None:
                st.info("Enable Manufacturing errors + Ideal-vs-error comparison to populate this tab.")
            else:
                st.plotly_chart(ideal_error_field_plot(z, Bideal, B), width="stretch")
                cmp = compare_metrics(ideal_metrics, metrics)
                rows = [
                    {"metric": key, "ideal": val["ideal"], "error_model": val["error"], "delta": val["delta"]}
                    for key, val in cmp.items()
                ]
                st.dataframe(pd.DataFrame(rows), width="stretch")

        with tabs[7]:
            shown = {k: v for k, v in metrics.items() if k not in ("trajectory", "electron_phase")}
            shown["classification"] = classification
            shown["field_range_mm"] = [float(z_lo), float(z_hi)]
            if rlx is not None:
                shown["RADIA_relaxation_result"] = rlx
                shown["RADIA_relaxation_converged"] = True
            st.json(shown)

            if calibration_history:
                st.subheader("Target B0 calibration")
                st.dataframe(pd.DataFrame(calibration_history), width="stretch")
                st.write(
                    f"Target: **{float(target_b0_t):.6g} T** using **{b0_mode}** "
                    f"→ calibrated Br: **{params['br_t']:.6g} T**"
                )

            st.subheader("Export")
            try:
                st.download_button(
                    "Download realized preset (.json)",
                    preset_json_bytes(
                        params, current_settings,
                        name=f"{device} realized configuration",
                        description="Exact successfully solved configuration, including calibrated Br.",
                        realized=True, calibration_history=calibration_history,
                        ui_settings=current_ui,
                        study_defaults=st.session_state.get("preset_study_defaults"),
                        extensions=st.session_state.get("preset_extensions"),
                    ),
                    "radia_magnet_realized_preset_v1.json", "application/json",
                    on_click="ignore", width="stretch",
                    help="Portable settings-only file for reproducing this solved model or loading it in another application.",
                )
            except Exception as exc:
                st.warning(f"Realized-preset export unavailable: {exc}")

            e1, e2 = st.columns(2)
            try:
                e1.download_button(
                    "Download on-axis field CSV",
                    csv_bytes(z, B),
                    "radia_on_axis_field.csv",
                    "text/csv",
                    on_click="ignore",
                    width="stretch",
                )
            except Exception as exc:
                e1.warning(f"CSV export unavailable: {exc}")

            try:
                e2.download_button(
                    "Download complete JSON",
                    json_bytes(params, metrics),
                    "radia_summary.json",
                    "application/json",
                    on_click="ignore",
                    width="stretch",
                )
            except Exception as exc:
                e2.warning(f"JSON export unavailable: {exc}")

            e3, e4 = st.columns(2)
            try:
                e3.download_button(
                    "Download HDF5",
                    hdf5_bytes(z, B, params, metrics),
                    "radia_field.h5",
                    "application/x-hdf5",
                    on_click="ignore",
                    width="stretch",
                )
            except Exception as exc:
                e3.warning(f"HDF5 export unavailable: {exc}")

            try:
                e4.download_button(
                    "Download PDF report",
                    pdf_bytes(z, B, params, metrics),
                    "radia_report.pdf",
                    "application/pdf",
                    on_click="ignore",
                    width="stretch",
                )
            except Exception as exc:
                e4.warning(f"PDF export unavailable: {exc}")

            try:
                st.caption(
                    f"Transfer package verified: schema {package_status['schema_version']}; "
                    f"{package_status['payload_files_checked']} payload files checked. "
                    f"Saved separately as {Path(saved_path).name}."
                )
                st.download_button(
                    "Download downstream research package (.zip)",
                    package,
                    Path(saved_path).name,
                    "application/zip",
                    on_click="ignore",
                    width="stretch",
                    help=(
                        "Versioned package containing device settings, units, coordinate definitions, "
                        "geometry, checksums, on-axis field data and the optional 3D field map."
                    ),
                )
            except Exception as exc:
                st.warning(f"Research-package export unavailable: {exc}")

            if device == "APPLE-II":
                st.warning(
                    "APPLE-II remains a physics-informed four-array research prototype, "
                    "not a facility/manufacturer-certified magnetic model."
                )

    except Exception as exc:
        magnet_tracker.fail(detail=str(exc))
        st.exception(exc)

# A successful result must survive navigation to Stage 2 and back.  During the
# original calculation the full tab set above is already visible, so avoid a
# duplicate gallery on that same run; rebuild it on later reruns instead.
render_model_archive()

if not run and st.session_state.get("magnet_completed_visuals") is not None:
    render_completed_entity_gallery(st.session_state["magnet_completed_visuals"])

render_saved_progress(st, "last_magnet_build_progress", "Last magnet-build progress table")

from magnet_studio.app.advanced_analysis import render_advanced_analysis

render_advanced_analysis()
