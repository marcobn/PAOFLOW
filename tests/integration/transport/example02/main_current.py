"""Current-vs-bias calculation example.

This example demonstrates computing current as a function of applied bias
using transmission data from a prior conductor calculation.
"""

from PAOFLOW import PAOFLOW
from PAOFLOW.transport.current_pipeline import run_current_from_file


def main() -> None:
    # PAOFLOW setup
    paoflow = PAOFLOW.PAOFLOW(
        savedir='alh.save',
        outputdir='output',
        smearing='gauss',
        npool=1,
        verbose=True,
        save_overlaps=True,
    )

    # Read projections from QE output
    paoflow.read_atomic_proj_QE()

    # Projectability analysis
    paoflow.projectability()

    # Build PAO Hamiltonian
    paoflow.pao_hamiltonian(shift_type=1, expand_wedge=False)

    # Compute projections
    paoflow.projections()

    # Compute current from transmission data
    run_current_from_file(
        data={
            'fileout': './output/current.dat',
            'mu_L': -0.5,
            'mu_R': 0.5,
            'sigma': 0.05,
        },
        filein='./output/conductance_lcr.dat',
        bias_min=-1.0,
        bias_max=1.0,
        nbias=100,
    )


if __name__ == '__main__':
    main()
