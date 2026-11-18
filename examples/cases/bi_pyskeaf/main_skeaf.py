#!/usr/bin/env python3
"""pyskeaf alone on existing Fermi-surface BXSF files of the Bi example (no QE save needed).

Reads the single-band Fermi_surf_band_<n>.bxsf files of a folder, runs pyskeaf with the settings
of main.py (PYSKEAF) and writes qo_EF_0_freqvsangle_<n>.out and plot_frequencies.png there.

Run with:

    python main_skeaf.py            # the BXSF files that main.py wrote in output/
    python main_skeaf.py bxsf       # the BXSF files shipped with the example
    mpirun -np 4 python main_skeaf.py bxsf     # the field directions spread over the ranks
"""

import sys

from PAOFLOW import PAOFLOW

from main import HERE, OUTPUTDIR, PYSKEAF, RANK, colors, plot_frequencies


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else OUTPUTDIR
    paoflow = PAOFLOW.PAOFLOW(workpath=HERE, outputdir=folder, verbose=True)
    paoflow.pyskeaf(**PYSKEAF)
    paoflow.finish_execution()

    # Matplotlib is single-process; pyskeaf already ran collectively above.
    if RANK == 0:
        plot_frequencies(colors, folder)


if __name__ == '__main__':
    main()
