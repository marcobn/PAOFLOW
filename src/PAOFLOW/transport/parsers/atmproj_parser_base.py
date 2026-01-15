from typing import Dict, Optional

import numpy as np

from PAOFLOW.DataController import DataController
from PAOFLOW.transport.utils.converters import crystal_to_cartesian


def parse_header(data_controller: DataController) -> Dict:
    _, attr = data_controller.data_dicts()
    return {
        "nbnds": attr["nbnds"],
        "nkpnts": attr["nkpnts"],
        "nspin": attr["nspin"],
        "nawf": attr["nawf"],
        "nelec": attr["nelec"],
        "efermi": attr["Efermi"],
        "energy_units": attr.get("energy_units", "eV"),
    }


def parse_kpoints(data_controller: DataController) -> Dict:
    arry, attr = data_controller.data_dicts()
    kpoints = arry["kpnts"].T
    wk = arry["kpnts_wght"]
    wk = wk / np.sum(wk)
    alat = attr["alat"]
    bvec = arry["b_vectors"] * (2.0 * np.pi / alat)
    vkpts_crystal = kpoints
    vkpts_cartesian = crystal_to_cartesian(vkpts_crystal, bvec)
    return {
        "kpts": kpoints,
        "wk": wk,
        "vkpts_cartesian": vkpts_cartesian,
        "vkpts_crystal": vkpts_crystal,
    }


def parse_eigenvalues(data_controller: DataController) -> np.ndarray:
    arry, _ = data_controller.data_dicts()
    eigvals = arry[
        "my_eigsmat"
    ]  # TODO This eigenvalue array may already be shifted by the Fermi energy unlike the implementation in paoflow-qtpy

    return eigvals


def parse_projections(data_controller: DataController) -> np.ndarray:
    arry, _ = data_controller.data_dicts()
    proj = arry["U"].swapaxes(0, 1)
    return proj


def parse_overlaps(data_controller: DataController) -> Optional[np.ndarray]:
    _, attr = data_controller.data_dicts()
    acbn0 = attr["acbn0"]
    if not acbn0:
        return None
    else:
        raise NotImplementedError(
            "Current implementaion requires overlap matrix to have dimensions (nawf, nawf, nkpnts, nspin). The overlap matrix in DataController has dimensions (nawf,nbnds,nkpnts)."
        )
