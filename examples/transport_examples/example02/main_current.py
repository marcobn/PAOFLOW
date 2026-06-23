import sys

from mpi4py import MPI

from PAOFLOW import PAOFLOW
from PAOFLOW.Transport import Transport

comm = MPI.COMM_WORLD

_CURRENT_CONFIGS = {
    'current.yaml': {
        'filein': './output/paoflow/conductance_lcr.dat',
        'fileout': './output/paoflow/current.dat',
        'bias_min': -1.0,
        'bias_max': 1.0,
        'nbias': 1500,
        'sigma': 0.05,
        'mu_L': -0.5,
        'mu_R': 0.5,
    },
}


def main() -> None:
    yaml_file = sys.argv[1] if len(sys.argv) > 1 else 'current.yaml'
    if yaml_file not in _CURRENT_CONFIGS:
        raise ValueError(f'Unsupported current input selector: {yaml_file}')

    paoflow = PAOFLOW.PAOFLOW(
        savedir='output/qe/alh.save',
        outputdir='output/paoflow',
        smearing='gauss',
        npool=1,
        verbose=True,
        save_overlaps=True,
    )

    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian(shift_type=1, expand_wedge=False)
    paoflow.projections()

    transport = Transport(paoflow.data_controller)
    transport.current(**_CURRENT_CONFIGS[yaml_file])


if __name__ == '__main__':
    main()
