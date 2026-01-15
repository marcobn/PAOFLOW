import re
import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, Optional

from PAOFLOW.transport.io.write_data import iotk_index
from PAOFLOW.transport.utils.converters import cartesian_to_crystal


def parse_header(root: ET.Element) -> Dict:
    header = root.find("HEADER")
    return {
        "nbnd": int(header.findtext("NUMBER_OF_BANDS")),
        "nkpts": int(header.findtext("NUMBER_OF_K-POINTS")),
        "nspin": int(header.findtext("NUMBER_OF_SPIN_COMPONENTS")),
        "natomwfc": int(header.findtext("NUMBER_OF_ATOMIC_WFC")),
        "nelec": float(header.findtext("NUMBER_OF_ELECTRONS")),
        "efermi": float(header.findtext("FERMI_ENERGY")),
        "energy_units": header.find("UNITS_FOR_ENERGY").attrib["UNITS"],
    }


def parse_kpoints(root: ET.Element, lattice_data: Dict) -> Dict:
    kpoints = np.array(
        [
            [float(val) for val in line.strip().split()]
            for line in root.find("K-POINTS").text.strip().split("\n")
        ]
    ).T
    wk = np.array(
        [float(val) for val in root.find("WEIGHT_OF_K-POINTS").text.strip().split()]
    )
    wk = wk / np.sum(wk)
    vkpts = kpoints * 2 * np.pi / lattice_data["alat"]
    vkpts_crystal = cartesian_to_crystal(vkpts, lattice_data["bvec"])
    return {
        "kpts": kpoints,
        "wk": wk,
        "vkpts_cartesian": vkpts,
        "vkpts_crystal": vkpts_crystal,
    }


def parse_eigenvalues(
    root: ET.Element, nbnd: int, nkpts: int, nspin: int
) -> np.ndarray:
    eigvals = np.zeros((nbnd, nkpts, nspin))
    eig_section = root.find("EIGENVALUES")
    for ik, kpoint in enumerate(eig_section):
        for isp in range(nspin):
            spin_tag = f"EIG{iotk_index(isp + 1)}" if nspin > 1 else "EIG"
            eig_tag = kpoint.find(spin_tag)
            eigvals[:, ik, isp] = [float(x) for x in eig_tag.text.strip().split()]
    return eigvals


def parse_projections(
    root: ET.Element, nbnd: int, nkpts: int, nspin: int, natomwfc: int
) -> np.ndarray:
    proj = np.zeros((natomwfc, nbnd, nkpts, nspin), dtype=np.complex128)
    projections_section = root.find("PROJECTIONS")
    for ik, kpoint in enumerate(projections_section):
        for isp in range(nspin):
            spin_node = (
                kpoint.find(f"SPIN{iotk_index(isp + 1)}") if nspin == 2 else kpoint
            )
            for ias in range(natomwfc):
                tag = f"ATMWFC{iotk_index(ias + 1)}"
                data = re.split(r"[\s,]+", spin_node.find(tag).text.strip())
                for ib in range(nbnd):
                    real, im = float(data[2 * ib]), float(data[2 * ib + 1])
                    proj[ias, ib, ik, isp] = real + 1j * im
    return proj


def parse_overlaps(
    root: ET.Element, nkpts: int, nspin: int, natomwfc: int
) -> Optional[np.ndarray]:
    overlap_section = root.find("OVERLAPS")
    if overlap_section is None:
        return None
    overlap = np.zeros((natomwfc, natomwfc, nkpts, nspin), dtype=np.complex128)
    for ik, kpoint in enumerate(overlap_section):
        for isp in range(nspin):
            tag = f"OVERLAP{iotk_index(isp + 1)}"
            data = re.split(r"[\s,]+", kpoint.find(tag).text.strip())
            matrix = np.array(
                [
                    complex(float(data[i]), float(data[i + 1]))
                    for i in range(0, len(data), 2)
                ]
            )
            overlap[:, :, ik, isp] = matrix.reshape(natomwfc, natomwfc)
    return overlap
