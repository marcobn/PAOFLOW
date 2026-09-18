#!/usr/bin/env python
"""Example: interactive Fermi-surface plot from a PAOFLOW BXSF file.

This is a thin wrapper around the installed ``fermi-plotter`` console app
(:mod:`PAOFLOW.gen.fermi_plotter`).  Once PAOFLOW is installed you can
equivalently run::

    fermi-plotter FermiSurf_0.bxsf --interp 2

The example below renders the composite Fermi surface of the example08 QE run,
coloured by Fermi velocity, in an interactive Mayavi window.  Requires the
``fermisurface`` extra (``pip install "PAOFLOW[fermisurface]"``).

To view several reciprocal cells with a Gamma point at the centre, and to
overlay the SKEAF field axis together with the plane it slices the surface
with (same option names as :meth:`PAOFLOW.PAOFLOW.pyskeaf`)::

    fermi-plotter FermiSurf_0.bxsf --supercell 2 --center \
        --b-field non_principal --azimuthal 30 --polar 45 \
        --field-plane --field-label --opacity 0.6
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
