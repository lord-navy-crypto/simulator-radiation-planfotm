# RADIA Magnet Studio 3.1

Open the complete unified application with:

`START_HERE_RADIA_MAGNET_STUDIO.command`

Run the dependency-aware automated checks with:

`RUN_SELF_CHECK.command`

The application is one continuous workflow. After `Build + Solve + Analyze`
finishes, the same page automatically connects the realized device and solver
settings to parameter sensitivity, Monte Carlo, or convergence analysis. No
workspace switch, export, or manual baseline transfer is required.

The Presets sidebar imports validated `radia-magnet-preset` v1 JSON files and
restores every relevant UI control. The current requested configuration can be
exported before solving; a successful run can export a realized preset with
the calibrated Br and calibration history. Built-in research examples cover
all device families and a manufacturing-error demonstration. See
`PRESET_SPEC.md` and `examples/read_preset.py` for the cross-application contract.

For resumable terminal execution use `RUN_RADIA_STUDY.command`. See `STUDIES.md`.

Expected local RADIA extension:

`~/Desktop/Radia-master/cpp/gcc/radia*.so`

## Correctness fixes in this build

1. **RADIA relaxation convergence gate**
   - `RlxPre -> RlxAuto`
   - requires `AvPrec < precision`
   - requires `Niter < max_iter`
   - otherwise raises a visible error and stops analysis.

2. **Geometry-derived longitudinal field range**
   - finds the true outer z-edges of generated blocks
   - includes APPLE-II row shifts and manufacturing position errors
   - adds configurable fringe-field margin before I1/I2/trajectory analysis.

3. **Explicit K definitions**
   - `Kx_peak = 0.934 * |By|max * lambda_u(cm)`
   - `Ky_peak = 0.934 * |Bx|max * lambda_u(cm)`
   - `K_peak = 0.934 * max(|B_perp|) * lambda_u(cm)`
   - `K_vector_norm = sqrt(Kx_peak^2 + Ky_peak^2)` is reported separately
   - resonance term is explicitly `(Kx_peak^2 + Ky_peak^2)/2`.

4. **Trajectory-derived electron phase error**
   - integrates slippage using x' and y'
   - removes best-fit linear slippage
   - evaluates phase residual at half-period positions
   - reports a separate RMS
   - the old zero-crossing phase is retained only as a diagnostic.

5. **HDF5 export**
   - no silent metric-write exception
   - unsupported metadata are recorded in `export_skipped_items`.

6. **Target B0 calibration modes**
   - Central-period peak B_perp (default)
   - Central 3-period peak B_perp
   - Global peak B_perp
   - default avoids calibrating to accidental fringe-field/end-field overshoot.

## Existing integrated features
- Planar / Helical / Elliptical / APPLE-II prototype / Wiggler
- manufacturing error model + deterministic seed
- ideal-vs-error comparison
- 3D magnet geometry
- 2D / 3D real field sampling
- trajectory, I1/I2, harmonics
- CSV / JSON / HDF5 / PDF / V11-compatible 3D map export

## 2.1 reliability and interoperability improvements

- strict finite-value and array-shape validation before analysis or export
- standards-compliant JSON (`NaN` and infinity become `null`)
- nested trajectory and electron-phase datasets preserved in HDF5
- chunked field sampling with explicit RADIA result validation
- versioned downstream research package with units, coordinate conventions,
  realized device geometry and SHA-256 checksums
- dependency-aware self-check runner that does not require pytest
- leakage-resistant joint sinusoidal harmonic fitting
- complete sampling, electron-energy, calibration and solver provenance
- transfer-package integrity validation before download
- reference downstream reader in `examples/read_transfer_package.py`
- download buttons no longer rerun and clear the calculated page

The new `radia_magnet_studio_transfer_v1.zip` export is the supported one-way
interface to a separate trajectory/radiation application. See
`INTERCHANGE_SPEC.md` for the file contract.

## 3.0 computational studies

- Cartesian batch scans over gap, Br, period, APPLE-II phase, error amplitudes,
  or any supported device parameter
- Monte Carlo ensembles with deterministic seeds, mean, sample standard
  deviation, range, and 95% confidence intervals
- automatic convergence grids over magnet subdivision, sampling density, and
  longitudinal fringe-field margin
- content-addressed result caching, process-isolated parallel workers, atomic
  checkpoints, cooperative cancellation, and resume
- Study Center GUI, terminal runner, CSV/JSON/ZIP results, and failure reporting

## APPLE-II scope
The APPLE-II backend is a physics-informed four-array research prototype with
real longitudinal row displacement. It is not a manufacturer- or facility-
certified replica of a specific installed device.


## NumPy compatibility
Field integrals no longer depend on the removed legacy `numpy.trapz` alias.
The code uses `numpy.trapezoid` when available, with an equivalent fallback for
older NumPy releases.


## Clear-results / JSON fix
- Numerical backend is unchanged from the NumPy-compatible strict-correctness build.
- The six compressed result cards were changed to two rows of three.
- An exact-value results table now shows values and units.
- Field-range/regime information is shown in a dedicated information panel.
- JSON export recursively converts NumPy arrays and NumPy scalar types to JSON-safe values.
- The complete trajectory and electron-phase arrays are preserved in JSON.
- Export buttons are arranged in a readable 2×2 layout and one failed exporter no longer blocks the others.
