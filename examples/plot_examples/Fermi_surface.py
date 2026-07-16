#!/usr/bin/env python
"""Example: interactive Fermi-surface plot from a PAOFLOW BXSF file.

This is a thin wrapper around the installed ``fermi-plotter`` console app
(:mod:`PAOFLOW.gen.fermi_plotter`).  Once PAOFLOW is installed you can
equivalently run::

    fermi-plotter FermiSurf_0.bxsf --interp 2

The example below renders the composite Fermi surface of the example08 QE run,
coloured by Fermi velocity, in an interactive Mayavi window.  Requires the
``fermisurface`` extra (``pip install "PAOFLOW[fermisurface]"``).
"""

import sys

from PAOFLOW.gen.fermi_plotter import main

if __name__ == '__main__':
    # Default to the bundled example file when no arguments are given.
    argv = sys.argv[1:] or [
        '../qe_examples/example08/Reference/FermiSurf_0.bxsf',
        '--interp',
        '2',
    ]
    raise SystemExit(main(argv))
