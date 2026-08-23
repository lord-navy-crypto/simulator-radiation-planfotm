from __future__ import annotations

import csv
import hashlib
import html
import io
import itertools
import json
import math
import os
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

STUDY_SCHEMA = "radia-magnet-studio-study"
STUDY_VERSION = "1.0.0"


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def stable_hash(value):
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _task(study_type, parameters, settings, labels=None):
    body = {
        "engine_version": STUDY_VERSION,
        "study_type": study_type,
        "parameters": dict(parameters),
        "settings": dict(settings),
        "labels": dict(labels or {}),
    }
    return {"task_id": stable_hash(body), **body}


def parameter_scan_tasks(base_parameters, settings, grid):
    if not grid:
        raise ValueError("Parameter scan grid cannot be empty.")
    keys = list(grid)
    values = []
    for key in keys:
        sequence = list(grid[key])
        if not sequence:
            raise ValueError(f"Parameter scan values for {key!r} cannot be empty.")
        values.append(sequence)
    tasks = []
    for combination in itertools.product(*values):
        params = dict(base_parameters)
        labels = {}
        for key, value in zip(keys, combination):
            params[key] = value
            labels[key] = value
        tasks.append(_task("parameter_scan", params, settings, labels))
    return tasks


def monte_carlo_tasks(base_parameters, settings, *, samples, seed_start=0):
    count = int(samples)
    if count < 2:
        raise ValueError("Monte Carlo analysis requires at least two samples.")
    tasks = []
    for seed in range(int(seed_start), int(seed_start) + count):
        params = dict(base_parameters)
        params["errors_enabled"] = True
        params["error_seed"] = seed
        tasks.append(_task("monte_carlo", params, settings, {"error_seed": seed}))
    return tasks


def convergence_tasks(
    base_parameters, settings, *, segmentations, axis_samples, margin_periods,
):
    if not segmentations or not axis_samples or not margin_periods:
        raise ValueError("Convergence grids cannot be empty.")
    tasks = []
    for segmentation, samples, margin in itertools.product(
        segmentations, axis_samples, margin_periods
    ):
        seg = int(segmentation)
        sample_count = int(samples)
        field_margin = float(margin)
        if seg < 1 or sample_count < 20 or field_margin < 0:
            raise ValueError("Invalid convergence setting.")
        params = dict(base_parameters)
        params["segmentation"] = [seg, seg, seg]
        task_settings = dict(settings)
        task_settings["axis_samples"] = sample_count
        task_settings["field_margin_periods"] = field_margin
        labels = {
            "segmentation": seg,
            "axis_samples": sample_count,
            "field_margin_periods": field_margin,
        }
        tasks.append(_task("convergence", params, task_settings, labels))
    return tasks


class JsonCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, task_id):
        return self.directory / f"{task_id}.json"

    def get(self, task_id):
        path = self.path_for(task_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if data.get("task_id") == task_id else None

    def put(self, result):
        _atomic_json(self.path_for(result["task_id"]), result)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_checkpoint(path, plan_hash):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if payload.get("plan_hash") != plan_hash:
        raise ValueError("Checkpoint belongs to a different study plan.")
    return {item["task_id"]: item for item in payload.get("results", [])}


def _save_checkpoint(path, plan_hash, results, status):
    _atomic_json(path, {
        "schema": {"name": STUDY_SCHEMA, "version": STUDY_VERSION},
        "plan_hash": plan_hash,
        "status": status,
        "results": list(results.values()),
    })


def run_tasks(
    tasks, evaluator, *, checkpoint_path, cache_directory, max_workers=1,
    cancel_requested=None, progress=None, continue_on_error=True,
    retry_failed=True,
):
    tasks = list(tasks)
    if not tasks:
        raise ValueError("Study contains no tasks.")
    plan_hash = stable_hash([{k: v for k, v in task.items() if k != "task_id"} for task in tasks])
    results = _load_checkpoint(checkpoint_path, plan_hash)
    if retry_failed:
        results = {
            task_id: result for task_id, result in results.items()
            if result.get("status") != "failed"
        }
    cache = JsonCache(cache_directory)
    valid_task_ids = {task["task_id"] for task in tasks}
    pending = []
    cache_hits = 0
    for task in tasks:
        if task["task_id"] in results:
            continue
        cached = cache.get(task["task_id"])
        if cached is not None:
            results[task["task_id"]] = cached
            cache_hits += 1
        else:
            pending.append(task)
    _save_checkpoint(checkpoint_path, plan_hash, results, "running")

    total = len(tasks)
    cancelled = False

    def record(result):
        if result.get("task_id") not in valid_task_ids:
            raise ValueError("Evaluator returned an unknown task_id.")
        result.setdefault("status", "success")
        if result["status"] == "success":
            cache.put(result)
        results[result["task_id"]] = result
        _save_checkpoint(checkpoint_path, plan_hash, results, "running")
        if progress:
            progress(len(results), total, result)

    if progress and results:
        progress(len(results), total, None)

    if int(max_workers) <= 1:
        try:
            for task in pending:
                if cancel_requested and cancel_requested():
                    cancelled = True
                    break
                try:
                    record(evaluator(task))
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    record({
                        "task_id": task["task_id"],
                        "study_type": task["study_type"],
                        "labels": task.get("labels", {}),
                        "parameters": task.get("parameters", {}),
                        "settings": task.get("settings", {}),
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
        except KeyboardInterrupt:
            cancelled = True
    else:
        with ProcessPoolExecutor(max_workers=int(max_workers)) as executor:
            futures = {executor.submit(evaluator, task): task for task in pending}
            try:
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        record(future.result())
                    except Exception as exc:
                        if not continue_on_error:
                            raise
                        record({
                            "task_id": task["task_id"],
                            "study_type": task["study_type"],
                            "labels": task.get("labels", {}),
                            "parameters": task.get("parameters", {}),
                            "settings": task.get("settings", {}),
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        })
                    if cancel_requested and cancel_requested():
                        cancelled = True
                        for other in futures:
                            other.cancel()
                        break
            except KeyboardInterrupt:
                cancelled = True
                for future in futures:
                    future.cancel()

    status = "cancelled" if cancelled else ("complete" if len(results) == total else "partial")
    _save_checkpoint(checkpoint_path, plan_hash, results, status)
    ordered = [results[task["task_id"]] for task in tasks if task["task_id"] in results]
    return {
        "schema": {"name": STUDY_SCHEMA, "version": STUDY_VERSION},
        "plan_hash": plan_hash,
        "status": status,
        "completed": len(ordered),
        "total": total,
        "cache_hits": cache_hits,
        "failures": sum(result.get("status") == "failed" for result in ordered),
        "results": ordered,
    }


def summarize_monte_carlo(results, metric_names):
    summary = {}
    for name in metric_names:
        values = np.asarray([
            result.get("metrics", {}).get(name) for result in results
            if isinstance(result.get("metrics", {}).get(name), (int, float))
            and math.isfinite(float(result["metrics"][name]))
        ], dtype=float)
        if values.size == 0:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        half = 1.96 * std / math.sqrt(values.size) if values.size > 1 else 0.0
        summary[name] = {
            "n": int(values.size), "mean": mean, "std": std,
            "min": float(np.min(values)), "max": float(np.max(values)),
            "ci95_low": mean - half, "ci95_high": mean + half,
            "ci_method": "normal approximation",
        }
    return summary


def rank_parameter_scan(results, objective, goal="minimize"):
    candidates = []
    for result in results:
        value = result.get("metrics", {}).get(objective)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            candidates.append(result)
    reverse = goal == "maximize"
    if goal not in ("minimize", "maximize"):
        raise ValueError("goal must be 'minimize' or 'maximize'.")
    return sorted(candidates, key=lambda item: float(item["metrics"][objective]), reverse=reverse)


def convergence_report(results, metric_names, relative_tolerance=0.01):
    if not results:
        return {"reference_task_id": None, "rows": []}
    def resolution(result):
        labels = result.get("labels", {})
        return (
            int(labels.get("segmentation", 1)) ** 3,
            int(labels.get("axis_samples", 0)),
            float(labels.get("field_margin_periods", 0.0)),
        )
    reference = max(results, key=resolution)
    rows = []
    for result in results:
        row = {"task_id": result["task_id"], **result.get("labels", {})}
        verdicts = []
        for name in metric_names:
            value = result.get("metrics", {}).get(name)
            target = reference.get("metrics", {}).get(name)
            if not isinstance(value, (int, float)) or not isinstance(target, (int, float)):
                continue
            absolute = abs(float(value) - float(target))
            relative = absolute / max(abs(float(target)), 1e-30)
            row[f"{name}_value"] = float(value)
            row[f"{name}_abs_error"] = absolute
            row[f"{name}_rel_error"] = relative
            row[f"{name}_converged"] = relative <= float(relative_tolerance)
            verdicts.append(row[f"{name}_converged"])
        row["all_metrics_converged"] = bool(verdicts) and all(verdicts)
        rows.append(row)
    return {
        "reference_task_id": reference["task_id"],
        "relative_tolerance": float(relative_tolerance),
        "rows": rows,
    }


def parameter_sensitivities(results, objective):
    grouped = {}
    for result in results:
        value = result.get("metrics", {}).get(objective)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        for name, parameter_value in result.get("labels", {}).items():
            if isinstance(parameter_value, (int, float)) and math.isfinite(float(parameter_value)):
                grouped.setdefault(name, {}).setdefault(float(parameter_value), []).append(float(value))
    output = {}
    for name, by_value in grouped.items():
        output[name] = [
            {
                "parameter_value": x, "n": len(values),
                "mean": float(np.mean(values)), "min": min(values), "max": max(values),
            }
            for x, values in sorted(by_value.items())
        ]
    return output


def sensitivity_svg_bytes(parameter_name, objective, points):
    width, height, pad = 720, 420, 58
    if not points:
        raise ValueError("Sensitivity plot has no points.")
    xs = [float(point["parameter_value"]) for point in points]
    ys = [float(point["mean"]) for point in points]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    if x_hi == x_lo:
        x_hi = x_lo + 1.0
    if y_hi == y_lo:
        y_hi = y_lo + max(abs(y_lo) * 0.05, 1.0)
    sx = lambda x: pad + (x - x_lo) * (width - 2 * pad) / (x_hi - x_lo)
    sy = lambda y: height - pad - (y - y_lo) * (height - 2 * pad) / (y_hi - y_lo)
    polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, ys))
    dots = "".join(
        f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="5" fill="#0b5cad"/>'
        for x, y in zip(xs, ys)
    )
    x_label = html.escape(str(parameter_name))
    y_label = html.escape(f"mean {objective}")
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333"/>
<polyline points="{polyline}" fill="none" stroke="#0b5cad" stroke-width="3"/>{dots}
<text x="{width/2}" y="{height-14}" text-anchor="middle" font-family="sans-serif">{x_label}</text>
<text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90 18 {height/2})" font-family="sans-serif">{y_label}</text>
<text x="{pad}" y="{pad-14}" font-family="sans-serif" font-weight="bold">Sensitivity: {x_label}</text>
<text x="{pad}" y="{height-pad+22}" font-family="sans-serif" font-size="11">{x_lo:.6g}</text>
<text x="{width-pad}" y="{height-pad+22}" text-anchor="end" font-family="sans-serif" font-size="11">{x_hi:.6g}</text>
<text x="{pad-8}" y="{height-pad}" text-anchor="end" font-family="sans-serif" font-size="11">{y_lo:.6g}</text>
<text x="{pad-8}" y="{pad}" text-anchor="end" font-family="sans-serif" font-size="11">{y_hi:.6g}</text>
</svg>'''
    return svg.encode("utf-8")


def results_csv_bytes(results):
    rows = []
    for result in results:
        row = {"task_id": result["task_id"], "study_type": result.get("study_type")}
        for prefix in ("labels", "parameters", "settings", "metrics"):
            for key, value in result.get(prefix, {}).items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    row[f"{prefix}.{key}"] = value
        row["elapsed_s"] = result.get("elapsed_s")
        rows.append(row)
    columns = sorted({key for row in rows for key in row})
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def results_bundle_bytes(report, *, summary=None, artifacts=None):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("study_report.json", json.dumps(report, indent=2, allow_nan=False))
        archive.writestr("study_results.csv", results_csv_bytes(report.get("results", [])))
        if summary is not None:
            archive.writestr("study_summary.json", json.dumps(summary, indent=2, allow_nan=False))
        for name, content in (artifacts or {}).items():
            archive.writestr(name, content)
    return output.getvalue()
