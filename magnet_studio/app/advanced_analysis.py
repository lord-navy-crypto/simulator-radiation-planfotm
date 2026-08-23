from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

from magnet_studio.studies.core import stable_hash

MAGNET_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / ".radia_studies" / "integrated"


def _numbers(text, cast=float):
    values = [cast(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Enter at least one comma-separated value.")
    return values


def _running(pid_path):
    if not pid_path.exists():
        return False
    try:
        os.kill(int(pid_path.read_text(encoding="utf-8")), 0)
        return True
    except (OSError, ValueError):
        return False


def _render_summary(output_dir):
    summary_path = output_dir / "study_summary.json"
    if not summary_path.exists():
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.warning(f"The analysis summary is being updated: {exc}")
        return
    st.subheader("Advanced-analysis results")
    if summary.get("best") is not None:
        best = summary["best"]
        st.success(
            f"Best {summary['objective']} case ({summary['goal']}): "
            f"{best['metrics'].get(summary['objective'])}"
        )
        st.json({"parameters": best.get("parameters", {}), "metrics": best.get("metrics", {})})
    elif "statistics" in summary:
        st.dataframe(
            [dict(metric=name, **values) for name, values in summary["statistics"].items()],
            width="stretch", hide_index=True,
        )
    elif "rows" in summary:
        st.caption(f"Relative convergence tolerance: {summary.get('relative_tolerance')}")
        st.dataframe(summary["rows"], width="stretch", hide_index=True)
    artifacts = output_dir / "artifacts"
    if artifacts.exists():
        for image_path in sorted(artifacts.glob("sensitivity_*.svg")):
            st.image(str(image_path), caption=image_path.stem.replace("_", " "))


@st.fragment
def render_advanced_analysis():
    st.divider()
    st.header("Advanced analysis of this generated magnetic field")
    baseline = st.session_state.get("magnet_study_source")
    if baseline is None:
        st.info(
            "Build the magnetic device above first. Its exact parameters and solver settings "
            "will automatically flow into this analysis section."
        )
        return

    st.success(
        "Connected to the magnetic field generated above. No export, page switch, or manual "
        "parameter transfer is required."
    )
    analysis_type = st.selectbox(
        "Next analysis",
        ["Parameter sensitivity and optimization", "Monte Carlo uncertainty", "Convergence verification"],
        key="integrated_analysis_type",
    )
    base = dict(baseline["base_parameters"])
    settings = dict(baseline["settings"])
    study_defaults = dict(baseline.get("study_defaults", {}))

    if analysis_type == "Parameter sensitivity and optimization":
        c1, c2 = st.columns(2)
        gap_values = c1.text_input("Gap values (mm)", f"{base['gap_mm'] * 0.9:.6g}, {base['gap_mm']:.6g}, {base['gap_mm'] * 1.1:.6g}")
        br_values = c2.text_input("Br values (T)", f"{base['br_t'] * 0.95:.6g}, {base['br_t']:.6g}, {base['br_t'] * 1.05:.6g}")
        c3, c4 = st.columns(2)
        period_values = c3.text_input("Period values (mm)", f"{base['period_mm']:.6g}")
        phase_values = c4.text_input("APPLE-II phase values (deg)", f"{base.get('apple_phase_deg', 90.0):.6g}")
        error_values = st.text_input("Field-amplitude error σ values (%)", f"{base.get('field_error_pct', 0.0):.6g}")
        objective = st.selectbox("Optimization objective", ["K_peak", "Bperp_peak_T", "electron_phase_error_rms_deg"])
        goal = st.selectbox("Optimization goal", ["maximize", "minimize"], index=0 if objective != "electron_phase_error_rms_deg" else 1)
        try:
            error_grid = _numbers(error_values)
            scan_base = dict(base)
            # A nonzero error-amplitude scan must actually activate the original
            # manufacturing-error model; zero remains the ideal reference case.
            if any(abs(value) > 0 for value in error_grid):
                scan_base["errors_enabled"] = True
            config = {
                "study_type": "parameter_scan", "base_parameters": scan_base, "settings": settings,
                "grid": {
                    "gap_mm": _numbers(gap_values), "br_t": _numbers(br_values),
                    "period_mm": _numbers(period_values), "apple_phase_deg": _numbers(phase_values),
                    "field_error_pct": error_grid,
                },
                "objective": objective, "goal": goal,
            }
        except ValueError as exc:
            st.error(str(exc))
            return
    elif analysis_type == "Monte Carlo uncertainty":
        c1, c2 = st.columns(2)
        samples = c1.number_input(
            "Random realizations", min_value=2,
            value=int(study_defaults.get("monte_carlo_samples", 20)), step=1,
        )
        seed_start = c2.number_input("First random seed", min_value=0, value=int(base.get("error_seed", 0)), step=1)
        config = {
            "study_type": "monte_carlo", "base_parameters": base, "settings": settings,
            "samples": int(samples), "seed_start": int(seed_start),
        }
    else:
        c1, c2, c3 = st.columns(3)
        segmentations = c1.text_input("Magnet subdivisions", "1, 2")
        sample_counts = c2.text_input("Axis sample counts", f"{max(100, settings['axis_samples']//2)}, {settings['axis_samples']}")
        margins = c3.text_input("Field margins (periods)", f"{max(0.0, settings['field_margin_periods']*0.5):.6g}, {settings['field_margin_periods']:.6g}")
        tolerance = st.number_input(
            "Relative convergence tolerance", min_value=1e-6,
            value=float(study_defaults.get("convergence_tolerance", 0.01)), format="%.6g",
        )
        try:
            config = {
                "study_type": "convergence", "base_parameters": base, "settings": settings,
                "segmentations": _numbers(segmentations, int),
                "axis_samples": _numbers(sample_counts, int),
                "margin_periods": _numbers(margins), "relative_tolerance": float(tolerance),
            }
        except ValueError as exc:
            st.error(str(exc))
            return

    worker_max = max(1, min(8, os.cpu_count() or 1))
    worker_default = max(1, min(worker_max, int(study_defaults.get("worker_processes", 2))))
    workers = st.slider("Parallel RADIA worker processes", 1, worker_max, worker_default)
    plan_id = stable_hash(config)[:16]
    output_dir = RUN_ROOT / plan_id
    config_path = output_dir / "config.json"
    cancel_path = output_dir / "CANCEL"
    log_path = output_dir / "study.log"
    pid_path = output_dir / "study.pid"

    left, middle, right = st.columns(3)
    if left.button("Continue analysis", type="primary", width="stretch"):
        if _running(pid_path):
            st.warning("This analysis is already running.")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config, indent=2, allow_nan=False), encoding="utf-8")
            if cancel_path.exists():
                cancel_path.unlink()
            log_handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, "-m", "magnet_studio.studies.cli", "--config", str(config_path),
                 "--output-dir", str(output_dir), "--workers", str(workers),
                 "--cancel-file", str(cancel_path)],
                cwd=PROJECT_ROOT, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True,
            )
            log_handle.close()
            pid_path.write_text(str(process.pid), encoding="utf-8")
            st.success(f"Advanced analysis started from the generated field (PID {process.pid}).")
    if middle.button("Stop safely", width="stretch"):
        output_dir.mkdir(parents=True, exist_ok=True)
        cancel_path.write_text("Stop requested.\n", encoding="utf-8")
        st.warning("Cancellation requested; completed cases remain checkpointed.")
    right.button("Refresh results", width="stretch")

    checkpoint = output_dir / "checkpoint.json"
    if checkpoint.exists():
        try:
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            st.write({
                "status": state.get("status"),
                "completed_cases": len(state.get("results", [])),
                "running": _running(pid_path),
            })
        except (OSError, json.JSONDecodeError):
            st.caption("Checkpoint is currently being updated.")
    _render_summary(output_dir)
    bundle = output_dir / "study_results.zip"
    if bundle.exists():
        st.download_button(
            "Download complete advanced-analysis results", bundle.read_bytes(),
            "radia_advanced_analysis.zip", "application/zip", width="stretch",
            on_click="ignore",
        )
