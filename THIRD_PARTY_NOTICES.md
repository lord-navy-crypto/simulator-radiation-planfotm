# Third-party software and external dependencies

This repository contains the V11 RADIA Radiation Studio source code and test
mocks. It does **not** vendor the official RADIA source tree or compiled RADIA
binary.

## RADIA

The application loads an independently installed RADIA Python extension.
Official upstream project:

- https://github.com/ochubar/Radia
- https://www.esrf.fr/home/Accelerators/instrumentation--equipment/Software/Radia/Documentation.html

RADIA is distributed by the European Synchrotron Radiation Facility under its
own open-source license. Its copyright and license remain with its respective
authors/rightsholders.

## Python dependencies

The following packages are installed separately through `pip` and are not
vendored in this repository:

- NumPy
- SciPy
- pandas
- Matplotlib
- Streamlit
- Plotly

Each dependency remains subject to its own license. Installing a dependency
does not transfer its copyright to this repository.

## Repository license

The MIT `LICENSE` file in this repository applies to the repository's own
source code, documentation, tests, and project files, except where a file
explicitly states otherwise.
