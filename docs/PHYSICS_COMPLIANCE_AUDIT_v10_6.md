# Physics compliance audit — v10.6

## Audit conclusion

The implemented equations are internally consistent for the declared model:
a prescribed magnetostatic insertion-device field, one relativistic electron,
optional classical energy-loss drag, and the causal retarded `1/R` acceleration
field observed outside the source region. The software is research-grade, not a
facility-certified magnetic or radiation calculation.

This audit found and corrected one substantive spectrum-combination defect. The
old implementation formed `|S1| + |S2|` for two orthogonal observer-basis FFTs;
subsequent squaring introduced an unphysical cross term. v10.6 uses
`sqrt(|S1|^2 + |S2|^2)`, so every downstream amplitude-squared quantity is the
correct sum of orthogonal spectral powers.

The audit also corrected an overstatement of scope. The observer calculation
uses the retarded Liénard–Wiechert radiation term only. It does not include the
`1/R^2` velocity/Coulomb field and is no longer labelled as a complete or exact
near-field solution.

## Formula and convention checks

- Relativistic state: `gamma = sqrt(1 + |u|^2/(m^2 c^2))`, `v = u/(gamma m)`.
- Lorentz force: `du/dt = q v × B`; the electron charge is negative.
- General insertion-device resonance:
  `lambda_r = lambda_u [1 + (Kx^2 + Ky^2)/2 + gamma^2 theta^2]/(2 gamma^2)`.
  This reduces to `1 + K^2/2` for planar and `1 + K^2` for circular helical.
- Classical magnetic-field radiation power is evaluated from `|v × B|^2`.
- Radiation reaction is an energy-loss drag approximation, not the full
  Landau–Lifshitz equation.
- Stokes convention: right-handed `(e1, e2, n)` observer basis and
  `V = +2 Im(E1 E2*)`. Circular-polarization sign comparisons require matching
  the same basis and handedness convention.
- RADIA geometry coordinates are passed in millimetres and sampled magnetic
  induction is interpreted in tesla; the bridge converts coordinates to metres
  before trajectory integration.
- SI constants were updated to 2022 CODATA values where they are not exact.

## Runtime compliance gate

Every completed full calculation now records and displays:

1. relativistic state (`gamma >= 1`, `beta < 1`);
2. simulated/theoretical resonance residual;
3. particle-energy-loss versus integrated-radiation-energy balance;
4. Stokes realizability;
5. observer-time samples per fundamental cycle;
6. observer-distance/source-extent applicability warning for the radiation term;
7. aperture-loss status;
8. classical-versus-quantum regime through `chi_e`;
9. RADIA target-field calibration status when metadata are available.

The gate returns `PASS`, `WARNING`, or `FAIL`. A warning is intentionally not
hidden: it means the numerical result may be internally consistent while an
applicability assumption still needs review.

## Validation performed

- Added a regression proving that two equal orthogonal spectral components
  produce exactly twice, not four times, the one-component spectral power.
- Added PASS/FAIL regressions for the compliance gate and aperture loss.
- Re-ran all RADIA-independent physics, tracking, off-axis, field-quality,
  serialization, unified-workflow, progress-table and visualization-layout tests.
- The two Plotly-dependent scan presentation scripts could not execute in the
  audit container because Plotly was not installed there. Their source was not
  changed by this physics patch, and Plotly remains declared in `requirements.txt`.

## Remaining validation boundary

The code cannot guarantee agreement with a particular real magnet from input
parameters alone. Before publication or hardware decisions, perform a real local
RADIA run and document geometry/material assumptions, segmentation convergence,
solver tolerance, field-grid convergence, trajectory tolerance, observer sampling,
and comparison with a measured map or an independent solver. Bunch emittance,
energy spread, coherent radiation, beamline optics and detector response remain
outside this model.

## Primary references

- NIST, 2022 CODATA constants: https://physics.nist.gov/cuu/Constants/
- ESRF RADIA documentation and reference guide:
  https://www.esrf.fr/home/Accelerators/instrumentation--equipment/Software/Radia/Documentation.html
- CERN Accelerator School, *Synchrotron radiation*:
  https://cds.cern.ch/record/2928187/files
