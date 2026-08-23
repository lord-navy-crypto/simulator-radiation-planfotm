# V11 RADIA Radiation Studio — v9.3 Scan-Centric Research Visualization

## Unified two-stage application

The package now starts one Streamlit application with a continuous workflow:

1. **Magnet design and field generation.** Select/import a Magnet Studio preset, build and inspect the RADIA device, and generate its realized 3-D field map. A previously exported checksum-protected Magnet Studio transfer ZIP can also be imported here.
2. **Trajectory, radiation and scanning.** The completed Stage-1 field is transferred through shared application state and becomes the field source for the V11 solver. Then select the particle/radiation settings and the actual scan variable. No intermediate download/re-upload is required for a run completed in the same session.

The last successful Stage-1 visual result is also retained when navigating to
Stage 2 and back. A restored gallery places the completed 3-D magnetic-block
entity first and also rebuilds the on-axis field, optional 2-D/3-D field maps,
electron trajectory and electron-phase plots from saved raw data.

Start the unified interface with `START_HERE_RADIA_UNIFIED.command`.

Every completed fixed-point V11 report now opens with an interactive physical-
scale 3-D electron trajectory, a transverse-magnified 3-D orbit, and a
radius-versus-z diagnostic. The radius view distinguishes the orbit-centred
oscillation radius from distance to the design axis. These fixed-point local
plots do not alter the scan-level rule: the selected scan quantity remains the
horizontal axis for all cross-scan trends.

### Saved magnet-model archive and harmonic controls

Each successful Stage-1 solve now creates a separate timestamped transfer ZIP
inside `saved_models/` and a corresponding archive entry in the interface.
Stage 2 provides an explicit selector for choosing which saved realized field
drives the trajectory, radiation calculation, and scan. The saved packages are
rediscovered when the application restarts and each can also be downloaded or
re-imported independently.

For the analytic V11 field, third- and fifth-harmonic coefficients are explicit
user inputs and default to zero. Their field-integral contributions are included
in the matched helical initial condition. For RADIA-generated or imported field
maps, harmonic content is measured from the realized field instead of being
silently imposed by analytic defaults.

The launcher starts the root-level `unified_entry.py`. Both navigation page
paths are resolved and checked as absolute paths before Streamlit creates the
navigation, preventing failures caused by launching an entrypoint from the
`app/` subdirectory.

### Live calculation progress

Every main compute button now creates a live progress bar and progress table.
The table shows the current task or exact scan point, pending/running/complete/
failed/skipped status, overall completion, elapsed seconds, and a short solver
detail. Parameter scans update one row per selected scan value. Magnet solving,
single-point V11 analysis, and representative full-case analysis use stage or
case rows. The most recent table is retained in the session for later review.

### v10.3 scan-scope visualization rule

The interface now separates the two meanings of an x-axis:

- **Cross-scan result:** x is always the selected independent scan quantity
  (`β=v/c`, K, γ, period count, observer distance, or observation angle).
- **Single operating-point diagnostic:** z is allowed only for internal field,
  trajectory, and along-device diagnostics at one fixed scan point.

Single-point z-based plots are hidden by default. The primary scan button names
the selected x quantity, the result page lists every exact horizontal-axis
point, and a runtime invariant rejects any scan-level chart whose numeric x
values are not drawn from the selected scan array.

This rule applies to every natural coordinate, not only z. Position, observer
time, frequency, photon energy, angle, field-component loci, and phase-space
coordinates are permitted only in a clearly labelled fixed-point/internal
analysis. Every fixed-point section prints the selected scan value; velocity
and gamma points additionally print β, speed in m/s, and γ together.

A research-oriented insertion-device simulation environment that connects
**RADIA 3-D magnetostatics**, relativistic **single-electron trajectory
integration**, and **Lienard-Wiechert radiation analysis** in one Python /
Streamlit workflow.

> Status: research software. Internal regression tests are included. The
> project is not a facility-certified magnetic measurement pipeline and is not
> a substitute for experimental validation.

## What it does

The current V9 codebase includes:

- RADIA-generated 3-D field maps for planar, helical, left-helical,
  elliptical, APPLE-II research-prototype, and wiggler configurations.
- Manufacturing-error models for field amplitude, longitudinal/transverse
  placement, magnetization angle, gap asymmetry, and bank imbalance.
- Ideal-geometry B0 calibration with explicit convergence verification.
- Full entrance/exit fringe-field trajectory tracking.
- Generalized planar/helical/elliptical resonance handling.
- Causal retarded-time Lienard-Wiechert `1/R` radiation-term evaluation, with
  the omitted `1/R²` near/velocity field stated as a model boundary.
- On-axis and off-axis spectral analysis.
- Stokes polarization diagnostics.
- 1-D angle scans and 2-D angular maps using radiative fluence
  `ε0 c ∫|E|² dt`.
- Magnetic harmonics, field integrals, K scans, period-number scans, and
  device-preset comparison helpers.
- NumPy-safe JSON export and RADIA CSV field-map import with arbitrary period.
- Streamlit graphical interface.


## v10.6 physics-compliance and visualization audit

Every full result now opens with a machine-readable PASS / WARNING / FAIL
physics-compliance table. It checks relativistic state validity, generalized
resonance residual, energy balance, Stokes realizability, temporal sampling,
radiation-zone applicability, aperture loss, quantum-regime applicability and
RADIA calibration metadata when present.

The v10.6 audit also fixes total spectral power across orthogonal polarization
components: the spectrum now uses `|S1|² + |S2|²`, with no artificial cross
term. The UI explicitly identifies the observer field as the causal retarded
`1/R` acceleration/radiation term and no longer implies that the omitted
`1/R²` near/velocity field is included. See
[`docs/PHYSICS_COMPLIANCE_AUDIT_v10_6.md`](docs/PHYSICS_COMPLIANCE_AUDIT_v10_6.md).

### v10.7 3-D electron trajectory views

Each fixed-point full result now contains two complementary interactive 3-D
electron-trajectory plots: a consistent-unit physical-coordinate view and an
explicitly labelled transverse-magnified orbit view. Both show entrance, exit,
path direction/progress and high-precision hover values. Representative full
scan cases also receive their own 3-D orbit view; these remain fixed-point
diagnostics and never replace the selected scan quantity on full-scan x-axes.

## Comprehensive result visualization

After a full V11 run, the Streamlit page now renders a long scrollable report rather than only a small summary. Without rerunning the physics engine it displays magnetic-field curves and integrals, 3-D and projected trajectories, trajectory angles, gamma evolution, instantaneous radiation power, observer E-field waveforms, Poynting flux and cumulative fluence, polarization locus, full/log and fundamental-window spectra, harmonic bars, Stokes diagnostics, energy/photon-yield/theory tables, quantum χ/Gaunt curves, and a complete scalar-result inventory.

A single **complete result data bundle** download contains trajectory, observer-field, spectrum, quantum, field and scalar tables plus JSON diagnostics. Heavy analyses such as 2-D angular maps, convergence and error-sensitivity sweeps remain explicit controls because they require additional simulations.

## Repository layout

```text
analysis/                         magnetic-field analysis
calibration/                      B0 / Br calibration
devices/                          RADIA device geometry builders
errors/                           manufacturing-error model
solver/                           RADIA solve / sampling pipeline
tests/                            mock and physics regression tests
docs/                             setup, validation and GitHub documentation
scripts/                          local test and publication preflight tools
.github/                          CI, Dependabot and contribution templates
undulator_v11_radia_integrated_v9.py
undulator_v11_radia_gui_v9.py
v11_radia_backend_v8.py
v11_field_quality_v6.py
START_HERE_V11_RADIA_v9.command
```



## v9.3 scan-centric visualization workflow

The physics engine remains the validated V9/RADIA backend. v9.3 rebuilds the presentation around a strict experimental rule: **the selected scan variable is the independent x-axis of every primary scan-level trend**.

1. **One scan variable → one dependent observable per chart.** Frequency, wavelength, power, photon energy, polarization, linewidth, harmonics, trajectory quality, quantum monitor values and validation residuals each get their own scan-variable-based trend. A second line is permitted only when it is the theory/reference for the *same* observable.
2. **Scan-aware plot groups.** Gamma, K, period-count, observer-distance and angle scans open with physically relevant metric groups; a comprehensive mode can show every available scalar observable without changing the x-axis rule.
3. **Representative existing rows.** The user can choose 3–6 full-analysis rows. Coverage selection spreads them across the scan, while feature-aware selection keeps endpoints and proposes actual rows near the strongest normalized response changes. No displayed point is interpolated or invented.
4. **Representative full-case suite.** Expensive full V11 calculations can be run only for the representative rows. Each case receives frequency-based spectra, z-based trajectory plots and z-based magnetic-field plots, while the scalar scan remains the trend backbone.
5. **Any-row deep analysis.** Any existing scalar scan row can still be selected for the complete Full Visualization report. Once a row is fixed, detailed plots use natural single-case independent variables such as z, source/observer time, frequency and photon energy.
6. **Metric dictionary and interpretation table.** Scan results include a glossary of units/meaning, and selected deep cases expose an exact-value interpretation table before the long report.
7. **Error-strength scans.** Every RADIA manufacturing-error source has an explicit physical strength control. A dedicated one-error-at-a-time sweep plots actual error strength on x (%, µm or mrad) against V11 observables; all selected errors can also be swept together using a nominal-strength multiplier.
8. **Secondary output-vs-output loci stay secondary.** Bx–By, x–y, phase-space, polarization and Gaunt-vs-χ loci remain useful inside a fixed single-case report but are never presented as the primary parameter scan.

The selected scan quantity is always the true independent x-axis. Gamma, speed **β=v/c**, K, period count, observer distance and angle scans each use their own requested point arrays in every primary scan-level trend and in exported scan_x values. For a velocity scan only, γ is derived at the solver boundary. Scalar density defaults to 50 points and complete per-point tables/CSV exports are preserved.

## Full visualization output

The Streamlit result page is deliberately long and vertically scrollable.
Major plots are rendered one-per-row at large height, numerical hover values
use high precision, summary cards use two columns to prevent clipping, and
major raw data tables are shown full-width with CSV downloads.

The full report includes magnetic fields and field integrals, spatial field
spectra, 3-D trajectory, orbit and phase-space views, beta/gamma evolution,
instantaneous and cumulative radiation quantities, observer waveforms,
retarded-time diagnostics, linear/log/photon-energy spectra, harmonics,
Stokes diagnostics, quantum monitor plots, exact scalar inventories, and a
complete ZIP data export.

## Requirements

Python dependencies are listed in `requirements.txt`.

RADIA itself is a separate external dependency and is **not bundled** in this
repository. The default macOS launcher looks for a compiled RADIA Python
extension under:

```text
~/Desktop/Radia-master/cpp/gcc
```

Override that path by setting `RADIA_PYTHONPATH`.

Official RADIA project and documentation:

- https://github.com/ochubar/Radia
- https://www.esrf.fr/home/Accelerators/instrumentation--equipment/Software/Radia/Documentation.html

## Quick start on macOS

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Make sure the official RADIA Python extension can be imported, then run:

```bash
chmod +x START_HERE_V11_RADIA_v9.command
./START_HERE_V11_RADIA_v9.command
```

Or start Streamlit directly:

```bash
python3 -m streamlit run unified_entry.py
```

## Run the regression suite

```bash
python3 scripts/run_tests.py
```

The test suite uses `tests/fake_radia.py` for RADIA-independent regression
checks, so GitHub Actions can test the project without a platform-specific
compiled RADIA binary.

## Validation boundary

The repository distinguishes between:

- syntax / import / API regression tests;
- mock RADIA geometry and pipeline tests;
- analytic radiation checks;
- actual numerical runs using a locally compiled RADIA extension;
- facility/device benchmark validation.

See [`docs/VALIDATION.md`](docs/VALIDATION.md) for the exact current scope.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Physics changes should include a
regression test or quantitative benchmark whenever practical.

## Citation

GitHub can render [`CITATION.cff`](CITATION.cff) through its
**Cite this repository** interface. The default citation uses a project-level
contributor identity; the repository owner may replace it with their preferred
personal or institutional authorship metadata before publication.

## License

Repository-owned source and documentation are released under the MIT License.
RADIA and Python dependencies remain under their own licenses. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
