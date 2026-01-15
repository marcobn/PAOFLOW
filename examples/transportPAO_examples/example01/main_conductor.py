import yaml

from PAOFLOW import PAOFLOW
from PAOFLOW.transport.parsers.parser_base import parse_args
from PAOFLOW.transport.Transport import ConductorRunner


def main():
    paoflow = PAOFLOW.PAOFLOW(
        savedir="output/qe/al5.save",
        outputdir="output/paoflow",
        smearing="gauss",
        npool=1,
        verbose=True,
    )

    paoflow.read_atomic_proj_QE()
    arry, attr = paoflow.data_controller.data_dicts()
    paoflow.projectability()
    paoflow.pao_hamiltonian()
    print(arry.keys())
    print(attr.keys())
    print(arry["Hks"][:, :, 0, 0, 0, 0])
    yaml_file = parse_args()
    # print(arry["HRs"].shape)
    #### calls to the transportPAO modules here


if __name__ == "__main__":
    main()
