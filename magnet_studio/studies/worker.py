from __future__ import annotations

import math
import time

import numpy as np

from magnet_studio.analysis.geometry_bounds import geometry_field_range
from magnet_studio.analysis.metrics import analyze, classify_k
from magnet_studio.calibration.target_b0 import calibrate_br
from magnet_studio.devices.factory import build_device
from magnet_studio.radia_support import load_radia
from magnet_studio.solver.pipeline import sample_on_axis, solve_model


def evaluate_radia_task(task):
    """Evaluate one configuration in an isolated worker process."""
    started = time.perf_counter()
    params = dict(task["parameters"])
    settings = dict(task["settings"])
    rad = load_radia()
    if hasattr(rad, "UtiDelAll"):
        rad.UtiDelAll()
    try:
        kind = params.pop("device")
        calibration_history = []
        if params.get("target_b0_enabled"):
            calibrated_br, calibration_history = calibrate_br(
                rad, kind, params, float(params["target_b0_t"]),
                mode=params.get("b0_definition") or "Central-period peak B⊥",
                relax=bool(settings.get("relax", False)),
                precision=float(settings.get("precision", 1e-4)),
                max_iter=int(settings.get("max_iter", 1000)),
            )
            params["br_t"] = float(calibrated_br)
            if hasattr(rad, "UtiDelAll"):
                rad.UtiDelAll()
        model = build_device(rad, kind, params)
        relaxation = solve_model(
            rad, model,
            relax=bool(settings.get("relax", False)),
            precision=float(settings.get("precision", 1e-4)),
            max_iter=int(settings.get("max_iter", 1000)),
            method=int(settings.get("method", 4)),
        )
        period = float(params["period_mm"])
        lo, hi = geometry_field_range(
            model["blocks"], period,
            margin_periods=float(settings.get("field_margin_periods", 1.0)),
        )
        samples = int(settings.get("axis_samples", 1000))
        z = np.linspace(lo, hi, samples)
        field = sample_on_axis(rad, model["obj"], z)
        metrics = analyze(
            z, field, period,
            float(settings.get("electron_energy_GeV", 3.0)),
        )
        scalar_metrics = {
            key: float(value) for key, value in metrics.items()
            if isinstance(value, (int, float, np.number)) and math.isfinite(float(value))
        }
        scalar_metrics["classification"] = classify_k(metrics["K_peak"])
        return {
            "task_id": task["task_id"],
            "study_type": task["study_type"],
            "labels": task.get("labels", {}),
            "parameters": task["parameters"],
            "realized_parameters": {"device": kind, **params},
            "settings": task["settings"],
            "metrics": scalar_metrics,
            "relaxation_result": relaxation,
            "calibration_history": calibration_history,
            "elapsed_s": time.perf_counter() - started,
            "status": "success",
        }
    finally:
        if hasattr(rad, "UtiDelAll"):
            rad.UtiDelAll()
