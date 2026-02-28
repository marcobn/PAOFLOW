from mpi4py import MPI

from PAOFLOW import PAOFLOW
from PAOFLOW.transport.Transport import ConductorRunner

comm = MPI.COMM_WORLD


def main():
    paoflow = PAOFLOW.PAOFLOW(
        savedir='al5.save',
        outputdir='output',
        smearing='gauss',
        npool=1,
        verbose=True,
        save_overlaps=True,
    )

    paoflow.read_atomic_proj_QE()
    paoflow.projectability(pthr=0.95)
    paoflow.pao_hamiltonian(
        shift_type=1,
        expand_wedge=False,
    )
    paoflow.projections()

    transport = ConductorRunner.from_yaml(
        yaml_file='conductor.yaml',
        data_controller=paoflow.data_controller,
    )
    transport.run()


if __name__ == '__main__':
    main()
