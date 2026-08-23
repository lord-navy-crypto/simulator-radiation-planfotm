# RADIA Magnet Studio Studies

Open `START_HERE_RADIA_MAGNET_STUDIO.command`, build and analyze the magnetic
device, then continue directly into the advanced-analysis section on the same
page. It automatically receives the realized magnetic-device and solver
settings. No separate graphical Study Center or manual transfer is used.

`RUN_RADIA_STUDY.command` remains available only for unattended terminal jobs
that can be interrupted with Control-C and resumed with the same command.

## Study types

- `parameter_scan`: Cartesian product of any parameter arrays in `grid`, then
  ranks successful cases using `objective` and `goal`.
- `monte_carlo`: forces manufacturing errors on, evaluates deterministic seeds,
  and reports n, mean, sample standard deviation, range and a normal-approximation
  95% confidence interval for every selected metric.
- `convergence`: varies cubic magnet segmentation, on-axis sample count and
  fringe-field margin. The highest-resolution case is the reference and every
  case receives absolute and relative errors.

## Performance and recovery

- Content-addressed JSON cache: identical cases are reused between compatible runs.
- Process parallelism: each worker imports and clears its own RADIA state.
- Atomic checkpoint: a temporary file is flushed and then replaced, avoiding a
  half-written checkpoint after interruption.
- Cooperative cancellation: the UI writes `CANCEL`; the terminal accepts
  Control-C. Completed cases remain available and a rerun resumes the plan.
- Failed cases are reported without discarding successful cases and are retried
  on the next run.

Outputs are `checkpoint.json`, `study_report.json`, `study_summary.json`, a flat
`study_results.csv`, and `study_results.zip`.
