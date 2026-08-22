# Architecture

```text
Device parameters
      |
      v
devices/* + errors/model.py
      |
      v
RADIA geometry / material model
      |
      v
solver/pipeline.py
      |
      +----> 3-D B(x,y,z) field map
      |
      v
undulator_v11_radia_integrated_v9.py
      |
      +----> relativistic trajectory
      |
      +----> retarded-time Lienard-Wiechert E field
      |
      +----> FFT / spectrum / harmonics
      |
      +----> Stokes polarization
      |
      +----> 1-D / 2-D angular fluence
      |
      v
undulator_v11_radia_gui_v9.py
```

## Magnetic-field characterization

Central periodic regions are used for resonance/K/harmonic quantities, while
the full map including fringe fields is used for field integrals and trajectory
effects.

## Radiation model

The radiation branch is a single-electron calculation. Observer-time windows
are constructed from causal source-to-observer arrival times, and the
retarded-time equation is solved with a bracketed root solver.

## Reproducibility

Manufacturing errors use a deterministic seed. Export or record the seed with
device parameters whenever comparing runs.
