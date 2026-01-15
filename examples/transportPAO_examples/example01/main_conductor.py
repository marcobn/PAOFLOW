import yaml

from PAOFLOW import PAOFLOW
from PAOFLOW.transportPAO.parsers.parser_base import parse_args
from PAOFLOW.transportPAO import ConductorRunner


def main():
    paoflow = PAOFLOW.PAOFLOW(
        savedir="al5.save",
        outputdir="output",
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
