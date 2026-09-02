#!/usr/bin/env python3
"""SKEAF workflow driver for the Bi example.

Runs the whole de Haas-van Alphen (SKEAF) workflow in one script:

  1. PAOFLOW builds the PAO Hamiltonian and writes the Fermi-surface .bxsf files.
  2. The native PAOFLOW pyskeaf method runs SKEAF over those .bxsf files.
  3. The frequency-vs-angle results are plotted.

Run with:

    python main.py
    # or, in parallel:
    mpirun -np <N> python main.py
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from PAOFLOW import PAOFLOW
from PAOFLOW.basis_gen import generate_basis_for_pseudo
from PAOFLOW.basis_gen.driver import _default_shells
from PAOFLOW.inputs.read_upf import UPF as _UPFParser

try:
    from mpi4py import MPI

    RANK = MPI.COMM_WORLD.Get_rank()
except ImportError:
    RANK = 0


# ----------------------------------------------------------------------- #
# Configuration  (edit freely)                                            #
# ----------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
SAVEDIR = os.path.join(HERE, 'Bi2.save')
UPFS = [os.path.join(HERE, 'Bi2.save/Bi.upf')]
BASISPATH = os.path.join(HERE, 'BASIS_PS') + os.sep
OUTPUTDIR = 'output'

NPOOL = 1
SMEARING = 'gauss'
STD_BASIS = 'standard'  # basis configuration for standard properties
PTHR = 0.95  # projectability threshold

# Double-grid interpolation (denser FFT grid). Set to None to skip.
NFFT = (32, 32, 32)

colors = ['blue', 'green', 'yellow', 'red', 'black', 'magenta', 'orange', 'purple', 'brown']


def ensure_basis(preset='extended'):
    """Generate the pseudo-atom basis under BASISPATH for every species.

    The 'extended' preset is a superset of 'standard' and 'minimal', so
    generating it once is enough for every configuration used below.
    """
    if RANK != 0:
        return
    for upf_path in UPFS:
        upf = _UPFParser(upf_path)
        element = upf.element.strip()
        elem_dir = os.path.join(BASISPATH, element)
        expected = _default_shells(upf, preset=preset)
        missing = [s for s in expected if not os.path.exists(os.path.join(elem_dir, f'{s}.dat'))]
        if missing:
            print(f'Generating pseudo-atom basis for {element} under {BASISPATH} ...')
            generate_basis_for_pseudo(
                upf_path, BASISPATH.rstrip(os.sep), preset=preset, verbose=True
            )
        else:
            print(f'Using existing pseudo-atom basis for {element} under {BASISPATH}')


def run_properties():
    """Build the PAO Hamiltonian and write the Fermi-surface .bxsf files."""
    p = PAOFLOW.PAOFLOW(
        workpath=HERE,
        outputdir=OUTPUTDIR,
        savedir=SAVEDIR,
        smearing=SMEARING,
        npool=NPOOL,
        verbose=True,
    )

    p.projections(basispath=BASISPATH, configuration=STD_BASIS)
    p.projectability(pthr=PTHR)
    p.pao_hamiltonian()

    if NFFT is not None:
        p.interpolated_hamiltonian(nfft1=NFFT[0], nfft2=NFFT[1], nfft3=NFFT[2])
    p.pao_eigh()

    p.fermi_surface()

    p.pyskeaf(
        fermi_energy=0.0,
        num_interpolation=60,
        azimuthal=(0.0, 0.0),
        polar=(0.0, 90.0),
        num_angles=37,
        frequency_tolerance=0.01,
        orbit_tolerance=0.05,
        allow_wall_orbits=True,
        verbose=False,
    )

    p.finish_execution()


def plot_freq(file, col):
    if os.path.getsize(file) == 0:
        print(f'{file}: skipped empty file')
        return False
    with open(file, encoding='utf-8') as handle:
        next(handle, None)
        if not any(line.strip() for line in handle):
            print(f'{file}: skipped file with no data rows')
            return False

    freq = np.loadtxt(file, delimiter=',', skiprows=1)
    freq = np.atleast_2d(freq)
    if freq.shape[1] < 3:
        print(f'{file}: skipped malformed file with {freq.shape[1]} columns')
        return False

    y = freq[:, 2]
    x = freq[:, 1]

    plt.axvline(x=0, color='k', lw=1.00)
    plt.axhline(y=0, color='k', lw=1.00)

    plt.plot(x, y, color=col, linestyle='None', marker='o', markersize=3.5)

    plt.xlabel('angle ($°$)', fontsize=20)
    plt.ylabel(r'$\rm{B_F}$ ($10^3$ T)', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    return True


def plot_frequencies(colors):
    """Plot the SKEAF frequency-vs-angle results."""
    files_nz = glob.glob(os.path.join(OUTPUTDIR, 'qo_results_freqvsangle_*.out'))
    files_nz.sort()
    print(files_nz)

    for i, file in enumerate(files_nz):
        if i >= len(colors):
            print(f'{file}: skipped, no color configured')
            continue
        if not plot_freq(file, colors[i]):
            continue
        print(' ')
        print(file, 'CORRECT ! !')
        print(' ')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTDIR, 'plot_frequencies.png'), dpi=300)


def main():
    if not os.path.isdir(SAVEDIR):
        print(f'{SAVEDIR} not found. Run pw.x (scf then nscf) first.')
        sys.exit(1)

    ensure_basis(preset='extended')
    if 'MPI' in globals():
        MPI.COMM_WORLD.Barrier()

    run_properties()

    # Matplotlib is single-process; pyskeaf already ran collectively above.
    if RANK == 0:
        plot_frequencies(colors)


if __name__ == '__main__':
    main()
