# V11 RADIA Radiation Studio — v9 Full Results Edition

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
- Causal retarded-time Lienard-Wiechert field evaluation.
- On-axis and off-axis spectral analysis.
- Stokes polarization diagnostics.
- 1-D angle scans and 2-D angular maps using radiative fluence
  `ε0 c ∫|E|² dt`.
- Magnetic harmonics, field integrals, K scans, period-number scans, and
  device-preset comparison helpers.
- NumPy-safe JSON export and RADIA CSV field-map import with arbitrary period.
- Streamlit graphical interface.


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
python3 -m streamlit run undulator_v11_radia_gui_v9.py
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
