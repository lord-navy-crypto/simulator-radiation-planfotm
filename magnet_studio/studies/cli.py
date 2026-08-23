from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from magnet_studio.studies.core import (
    convergence_report,
    convergence_tasks,
    monte_carlo_tasks,
    parameter_scan_tasks,
    parameter_sensitivities,
    rank_parameter_scan,
    results_bundle_bytes,
    run_tasks,
    summarize_monte_carlo,
    sensitivity_svg_bytes,
)
from magnet_studio.studies.worker import evaluate_radia_task


DEFAULT_METRICS = [
    "Bperp_peak_T", "K_peak", "first_integral_x_T_mm",
    "first_integral_y_T_mm", "electron_phase_error_rms_deg",
]


def build_tasks(config):
    study_type = config["study_type"]
    base = config["base_parameters"]
    settings = config.get("settings", {})
    if study_type == "parameter_scan":
        return parameter_scan_tasks(base, settings, config["grid"])
    if study_type == "monte_carlo":
        return monte_carlo_tasks(
            base, settings, samples=config["samples"],
            seed_start=config.get("seed_start", 0),
        )
    if study_type == "convergence":
        return convergence_tasks(
            base, settings,
            segmentations=config["segmentations"],
            axis_samples=config["axis_samples"],
            margin_periods=config["margin_periods"],
        )
    raise ValueError(f"Unsupported study_type: {study_type!r}")


def summarize(config, report):
    successful = [r for r in report["results"] if r.get("status") == "success"]
    metrics = config.get("summary_metrics", DEFAULT_METRICS)
    if config["study_type"] == "monte_carlo":
        return {"statistics": summarize_monte_carlo(successful, metrics)}
    if config["study_type"] == "convergence":
        return convergence_report(
            successful, metrics,
            relative_tolerance=config.get("relative_tolerance", 0.01),
        )
    objective = config.get("objective", "K_peak")
    goal = config.get("goal", "maximize")
    ranked = rank_parameter_scan(successful, objective, goal)
    return {
        "objective": objective,
        "goal": goal,
        "best": ranked[0] if ranked else None,
        "ranking": [item["task_id"] for item in ranked],
        "sensitivities": parameter_sensitivities(successful, objective),
    }


def run_config(config, output_dir, *, workers=1, progress=None, cancel_file=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(config)
    cancel_path = Path(cancel_file) if cancel_file else output / "CANCEL"
    report = run_tasks(
        tasks, evaluate_radia_task,
        checkpoint_path=output / "checkpoint.json",
        cache_directory=output / "cache",
        max_workers=workers,
        cancel_requested=cancel_path.exists,
        progress=progress,
    )
    summary = summarize(config, report)
    artifacts = {}
    if config["study_type"] == "parameter_scan":
        for parameter, points in summary.get("sensitivities", {}).items():
            safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in parameter)
            artifacts[f"sensitivity_{safe_name}.svg"] = sensitivity_svg_bytes(
                parameter, summary["objective"], points,
            )
    report_path = output / "study_report.json"
    summary_path = output / "study_summary.json"
    bundle_path = output / "study_results.zip"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    artifacts_dir = output / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        (artifacts_dir / name).write_bytes(content)
    bundle_path.write_bytes(results_bundle_bytes(report, summary=summary, artifacts=artifacts))
    return report, summary, bundle_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run or resume a RADIA Magnet Studio study.")
    parser.add_argument("--config", required=True, help="Study configuration JSON file")
    parser.add_argument("--output-dir", required=True, help="Checkpoint/cache/results directory")
    parser.add_argument("--workers", type=int, default=max(1, min(2, os.cpu_count() or 1)))
    parser.add_argument("--cancel-file", help="Stop scheduling work when this file exists")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    def show_progress(done, total, result):
        suffix = ""
        if result is not None:
            suffix = f"  {result.get('status', 'success')}: {result['task_id'][:10]}"
        print(f"[{done}/{total}]{suffix}", flush=True)

    report, _, bundle = run_config(
        config, args.output_dir, workers=args.workers,
        progress=show_progress, cancel_file=args.cancel_file,
    )
    print(f"Status: {report['status']} ({report['completed']}/{report['total']})")
    print(f"Cache hits: {report['cache_hits']}; failures: {report['failures']}")
    print(f"Results: {bundle}")


if __name__ == "__main__":
    main()
