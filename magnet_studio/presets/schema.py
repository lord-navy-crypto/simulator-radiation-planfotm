from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone

from magnet_studio.devices.factory import BUILDERS, validate_device_params

PRESET_SCHEMA = "radia-magnet-preset"
PRESET_VERSION = "1.0.0"

BASE_PARAMETERS = {
    "device": "Planar", "period_mm": 50.0, "periods": 20, "gap_mm": 12.0,
    "blocks_per_period": 4, "block_width_mm": 10.0, "block_height_mm": 10.0,
    "longitudinal_fill": 0.9, "br_t": 1.2, "material_mode": "Fixed remanence",
    "mu_parallel": 1.05, "mu_perpendicular": 1.05, "segmentation": [1, 1, 1],
    "ellipticity": 0.5, "apple_phase_deg": 90.0,
    "apple_shift_mode": "Antiparallel", "errors_enabled": False,
    "field_error_pct": 1.0, "longitudinal_error_mm": 0.05,
    "transverse_error_mm": 0.05, "angle_error_deg": 0.5,
    "gap_asymmetry_mm": 0.0, "bank_imbalance_pct": 0.0, "error_seed": 12345,
    "target_b0_enabled": False, "target_b0_t": 0.15,
    "b0_definition": "Central-period peak B⊥",
}
BASE_SETTINGS = {
    "axis_samples": 1000, "field_margin_periods": 1.0,
    "electron_energy_GeV": 3.0, "relax": False,
    "precision": 1e-4, "max_iter": 1000, "method": 4,
    "calculate_2d": True, "calculate_3d": True,
    "transverse_half_width_mm": 5.0,
}
BASE_UI = {"compare_ideal": True, "geometry_limit": 600}
BASE_STUDY = {"worker_processes": 2, "monte_carlo_samples": 20, "convergence_tolerance": 0.01}


def _finite(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON boolean.")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _fingerprint(payload):
    copy_payload = copy.deepcopy(payload)
    copy_payload.pop("fingerprint_sha256", None)
    return hashlib.sha256(_canonical(copy_payload)).hexdigest()


def build_preset(parameters, settings, *, name="Custom configuration", description="",
                 realized=False, calibration_history=None, study_defaults=None,
                 ui_settings=None, extensions=None, created_at=None):
    p = {**BASE_PARAMETERS, **dict(parameters)}
    s = {**BASE_SETTINGS, **dict(settings)}
    ui = {**BASE_UI, **dict(ui_settings or {})}
    study = {**BASE_STUDY, **dict(study_defaults or {})}
    payload = {
        "schema": {"name": PRESET_SCHEMA, "version": PRESET_VERSION},
        "metadata": {
            "name": str(name), "description": str(description),
            "created_by": "RADIA Magnet Studio", "producer_version": "3.1.0",
            "created_at_utc": created_at or datetime.now(timezone.utc).isoformat(),
            "kind": "realized" if realized else "requested",
        },
        "device": {
            "type": p["device"], "period_mm": p["period_mm"], "periods": p["periods"],
            "gap_mm": p["gap_mm"], "blocks_per_period": p["blocks_per_period"],
            "block_width_mm": p["block_width_mm"], "block_height_mm": p["block_height_mm"],
            "longitudinal_fill": p["longitudinal_fill"], "br_t": p["br_t"],
            "ellipticity": p["ellipticity"], "apple_phase_deg": p["apple_phase_deg"],
            "apple_shift_mode": p["apple_shift_mode"],
        },
        "material": {
            "mode": p["material_mode"], "mu_parallel": p["mu_parallel"],
            "mu_perpendicular": p["mu_perpendicular"],
            "segmentation": list(p["segmentation"]),
        },
        "manufacturing_errors": {
            "enabled": p["errors_enabled"], "field_error_pct": p["field_error_pct"],
            "longitudinal_error_mm": p["longitudinal_error_mm"],
            "transverse_error_mm": p["transverse_error_mm"],
            "angle_error_deg": p["angle_error_deg"],
            "gap_asymmetry_mm": p["gap_asymmetry_mm"],
            "bank_imbalance_pct": p["bank_imbalance_pct"], "seed": p["error_seed"],
        },
        "calibration": {
            "enabled": p["target_b0_enabled"], "target_b0_t": p["target_b0_t"],
            "definition": p["b0_definition"], "history": list(calibration_history or []),
        },
        "solver": {
            "relax": s["relax"], "precision_t": s["precision"],
            "max_iterations": s["max_iter"], "method": s["method"],
        },
        "sampling": {
            "axis_samples": s["axis_samples"],
            "field_margin_periods": s["field_margin_periods"],
            "electron_energy_GeV": s["electron_energy_GeV"],
            "calculate_2d": s["calculate_2d"], "calculate_3d": s["calculate_3d"],
            "transverse_half_width_mm": s["transverse_half_width_mm"],
        },
        "ui": ui,
        "study_defaults": study,
        "conventions": {
            "length_unit": "mm", "magnetic_field_unit": "T",
            "electron_energy_unit": "GeV", "longitudinal_axis": "z",
            "transverse_axes": ["x", "y"], "right_handed": True,
        },
        "extensions": copy.deepcopy(extensions or {}),
    }
    parse_preset(payload, verify_fingerprint=False)
    payload["fingerprint_sha256"] = _fingerprint(payload)
    return payload


def parse_preset(source, *, verify_fingerprint=True):
    payload = json.loads(source.decode("utf-8") if isinstance(source, bytes) else source) if isinstance(source, (str, bytes)) else copy.deepcopy(source)
    if not isinstance(payload, dict):
        raise ValueError("Preset root must be a JSON object.")
    schema = payload.get("schema", {})
    if schema.get("name") != PRESET_SCHEMA:
        raise ValueError(f"Unsupported preset schema: {schema.get('name')!r}.")
    version = str(schema.get("version", ""))
    if version.split(".")[0] != PRESET_VERSION.split(".")[0]:
        raise ValueError(f"Incompatible preset major version: {version!r}.")
    required = ("device", "material", "manufacturing_errors", "calibration", "solver", "sampling")
    missing = [name for name in required if not isinstance(payload.get(name), dict)]
    if missing:
        raise ValueError("Missing preset section(s): " + ", ".join(missing))
    d, m, e, c, so, sa = (payload[name] for name in required)
    p = {
        **BASE_PARAMETERS, "device": d.get("type"),
        **{key: d[key] for key in ("period_mm", "periods", "gap_mm", "blocks_per_period", "block_width_mm", "block_height_mm", "longitudinal_fill", "br_t", "ellipticity", "apple_phase_deg", "apple_shift_mode") if key in d},
        "material_mode": m.get("mode"), "mu_parallel": m.get("mu_parallel"),
        "mu_perpendicular": m.get("mu_perpendicular"), "segmentation": m.get("segmentation"),
        "errors_enabled": e.get("enabled"), "field_error_pct": e.get("field_error_pct"),
        "longitudinal_error_mm": e.get("longitudinal_error_mm"),
        "transverse_error_mm": e.get("transverse_error_mm"),
        "angle_error_deg": e.get("angle_error_deg"), "gap_asymmetry_mm": e.get("gap_asymmetry_mm"),
        "bank_imbalance_pct": e.get("bank_imbalance_pct"), "error_seed": e.get("seed"),
        "target_b0_enabled": c.get("enabled"), "target_b0_t": c.get("target_b0_t"),
        "b0_definition": c.get("definition"),
    }
    if p["device"] not in BUILDERS:
        raise ValueError(f"Unknown device type: {p['device']!r}.")
    validate_device_params(p)
    if int(p["blocks_per_period"]) not in (4, 8, 12, 16):
        raise ValueError("blocks_per_period must be one of 4, 8, 12, or 16.")
    segmentation = list(p["segmentation"])
    if len(set(int(value) for value in segmentation)) != 1 or int(segmentation[0]) not in (1, 2, 3):
        raise ValueError("The UI supports equal segmentation values of 1, 2, or 3.")
    if not 0 <= _finite(p["ellipticity"], "device.ellipticity") <= 1:
        raise ValueError("device.ellipticity must be between 0 and 1.")
    if not -180 <= _finite(p["apple_phase_deg"], "device.apple_phase_deg") <= 180:
        raise ValueError("device.apple_phase_deg must be between -180 and 180.")
    if p["apple_shift_mode"] not in ("Antiparallel", "Parallel"):
        raise ValueError("Unsupported APPLE-II shift mode.")
    p["errors_enabled"] = _boolean(p["errors_enabled"], "manufacturing_errors.enabled")
    p["target_b0_enabled"] = _boolean(p["target_b0_enabled"], "calibration.enabled")
    if int(p["error_seed"]) < 0:
        raise ValueError("manufacturing_errors.seed must be non-negative.")
    if p["target_b0_enabled"] and _finite(p["target_b0_t"], "calibration.target_b0_t") <= 0:
        raise ValueError("calibration.target_b0_t must be positive when calibration is enabled.")
    settings = {
        **BASE_SETTINGS, "relax": _boolean(so.get("relax"), "solver.relax"),
        "precision": _finite(so.get("precision_t"), "solver.precision_t"),
        "max_iter": int(so.get("max_iterations")), "method": int(so.get("method", 4)),
        "axis_samples": int(sa.get("axis_samples")),
        "field_margin_periods": _finite(sa.get("field_margin_periods"), "sampling.field_margin_periods"),
        "electron_energy_GeV": _finite(sa.get("electron_energy_GeV"), "sampling.electron_energy_GeV"),
        "calculate_2d": _boolean(sa.get("calculate_2d"), "sampling.calculate_2d"),
        "calculate_3d": _boolean(sa.get("calculate_3d"), "sampling.calculate_3d"),
        "transverse_half_width_mm": _finite(sa.get("transverse_half_width_mm"), "sampling.transverse_half_width_mm"),
    }
    if settings["precision"] <= 0 or settings["max_iter"] < 1 or settings["axis_samples"] < 100:
        raise ValueError("Invalid solver precision, iteration count, or sampling count.")
    if settings["field_margin_periods"] < 0 or settings["electron_energy_GeV"] <= 0 or settings["transverse_half_width_mm"] <= 0:
        raise ValueError("Invalid sampling range or electron energy.")
    conventions = payload.get("conventions", {})
    expected = {"length_unit": "mm", "magnetic_field_unit": "T", "electron_energy_unit": "GeV"}
    for key, value in expected.items():
        if conventions.get(key) != value:
            raise ValueError(f"Unsupported {key}: {conventions.get(key)!r}; expected {value!r}.")
    fingerprint = payload.get("fingerprint_sha256")
    if verify_fingerprint:
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("Preset is missing a valid SHA-256 fingerprint.")
        if fingerprint != _fingerprint(payload):
            raise ValueError("Preset SHA-256 fingerprint does not match its contents.")
    known = {"schema", "metadata", "device", "material", "manufacturing_errors", "calibration", "solver", "sampling", "ui", "study_defaults", "conventions", "extensions", "fingerprint_sha256"}
    warnings = [f"Unknown top-level field ignored: {key}" for key in payload if key not in known]
    ui_settings = {**BASE_UI, **payload.get("ui", {})}
    if not 100 <= int(ui_settings["geometry_limit"]) <= 1200:
        raise ValueError("ui.geometry_limit must be between 100 and 1200.")
    ui_settings["compare_ideal"] = _boolean(ui_settings["compare_ideal"], "ui.compare_ideal")
    study_defaults = {**BASE_STUDY, **payload.get("study_defaults", {})}
    if int(study_defaults["worker_processes"]) < 1 or int(study_defaults["monte_carlo_samples"]) < 2:
        raise ValueError("Invalid study worker or Monte Carlo defaults.")
    if _finite(study_defaults["convergence_tolerance"], "study_defaults.convergence_tolerance") <= 0:
        raise ValueError("Convergence tolerance must be positive.")
    return {
        "parameters": p, "settings": settings,
        "ui_settings": ui_settings, "study_defaults": study_defaults,
        "metadata": payload.get("metadata", {}), "extensions": payload.get("extensions", {}),
        "warnings": warnings,
    }


def preset_json_bytes(*args, **kwargs):
    return json.dumps(build_preset(*args, **kwargs), indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")


WIDGET_KEYS = {
    "device": "cfg_device", "period_mm": "cfg_period_mm", "periods": "cfg_periods",
    "gap_mm": "cfg_gap_mm", "blocks_per_period": "cfg_blocks_per_period",
    "block_width_mm": "cfg_block_width_mm", "block_height_mm": "cfg_block_height_mm",
    "longitudinal_fill": "cfg_longitudinal_fill", "br_t": "cfg_br_t",
    "target_b0_enabled": "cfg_target_b0_enabled", "target_b0_t": "cfg_target_b0_t",
    "b0_definition": "cfg_b0_definition", "material_mode": "cfg_material_mode",
    "mu_parallel": "cfg_mu_parallel", "mu_perpendicular": "cfg_mu_perpendicular",
    "ellipticity": "cfg_ellipticity", "apple_phase_deg": "cfg_apple_phase_deg",
    "apple_shift_mode": "cfg_apple_shift_mode", "errors_enabled": "cfg_errors_enabled",
    "field_error_pct": "cfg_field_error_pct", "longitudinal_error_mm": "cfg_longitudinal_error_mm",
    "transverse_error_mm": "cfg_transverse_error_mm", "angle_error_deg": "cfg_angle_error_deg",
    "gap_asymmetry_mm": "cfg_gap_asymmetry_mm", "bank_imbalance_pct": "cfg_bank_imbalance_pct",
    "error_seed": "cfg_error_seed",
}


def runtime_to_widget_state(runtime):
    p, s, ui = runtime["parameters"], runtime["settings"], runtime["ui_settings"]
    state = {WIDGET_KEYS[key]: value for key, value in p.items() if key in WIDGET_KEYS}
    state.update({
        "cfg_seg_n": int(list(p["segmentation"])[0]), "cfg_relax": bool(s["relax"]),
        "cfg_precision": float(s["precision"]), "cfg_max_iter": int(s["max_iter"]),
        "cfg_axis_samples": int(s["axis_samples"]),
        "cfg_field_margin_periods": float(s["field_margin_periods"]),
        "cfg_electron_energy_GeV": float(s["electron_energy_GeV"]),
        "cfg_make_2d": bool(s["calculate_2d"]), "cfg_make_3d": bool(s["calculate_3d"]),
        "cfg_transverse_half_mm": float(s["transverse_half_width_mm"]),
        "cfg_compare_ideal": bool(ui["compare_ideal"]), "cfg_geometry_limit": int(ui["geometry_limit"]),
    })
    return state


def _builtin(name, device, **changes):
    p = {**BASE_PARAMETERS, "device": device, **changes}
    if device in ("Helical", "Elliptical"):
        p["blocks_per_period"] = 8
    return build_preset(p, BASE_SETTINGS, name=name, description="Research example; not a facility-certified design.", created_at="2026-01-01T00:00:00+00:00")


BUILTIN_PRESETS = {
    "Planar baseline": _builtin("Planar baseline", "Planar"),
    "Helical baseline": _builtin("Helical baseline", "Helical"),
    "Elliptical baseline": _builtin("Elliptical baseline", "Elliptical"),
    "APPLE-II baseline": _builtin("APPLE-II baseline", "APPLE-II"),
    "High-K wiggler example": _builtin("High-K wiggler example", "Wiggler", period_mm=100.0, br_t=1.5),
    "Manufacturing-error demonstration": _builtin("Manufacturing-error demonstration", "Planar", errors_enabled=True),
}
