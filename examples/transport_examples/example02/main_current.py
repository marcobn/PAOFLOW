import sys
from PAOFLOW import PAOFLOW
from PAOFLOW.transport.Transport import CurrentRunner
from mpi4py import MPI

comm = MPI.COMM_WORLD


def main():
    yaml_file = sys.argv[1] if len(sys.argv) > 1 else "current.yaml"

    paoflow = PAOFLOW.PAOFLOW(
        savedir="output/qe/alh.save",
        outputdir="output/paoflow",
        smearing="gauss",
        npool=1,
        verbose=True,
        save_overlaps=True,
    )

    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian(shift_type=1, expand_wedge=False)
    paoflow.projections()

    transport = CurrentRunner.from_yaml(
        yaml_file=yaml_file,
        data_controller=paoflow.data_controller,
    )
    transport.run()


if __name__ == "__main__":
    main()
