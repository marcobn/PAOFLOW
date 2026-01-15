import os
from pathlib import Path
from typing import Dict

from PAOFLOW.DataController import DataController
import numpy as np

from PAOFLOW.transport.grid.rgrid import get_rgrid
from PAOFLOW.transport.io.input_parameters import AtomicProjData, ConductorData
import PAOFLOW.transport.io.log_module as log

from PAOFLOW.transport.io.write_data import (
    write_internal_format_files,
    write_projectability_files,
    write_overlap_files,
)
from PAOFLOW.transport.io.write_header import headered_function
from PAOFLOW.transport.parsers.atmproj_parser_base import (
    parse_eigenvalues,
    parse_header,
    parse_kpoints,
    parse_overlaps,
    parse_projections,
)
from PAOFLOW.transport.utils.timing import timed_function


def validate_proj_files(file_proj: str) -> str:
    """
    Ensure that both atomic_proj.xml and its companion data-file.xml exist.
    Returns the path to data-file-schema.xml if found.
    """
    savedir = os.path.dirname(file_proj)
    file_data = os.path.join(savedir, "data-file-schema.xml")
    if not os.path.exists(file_data):
        raise FileNotFoundError(f"Expected data-file-schema.xml at: {file_data}")
    return file_data


@timed_function("atmproj_to_internal")
@headered_function("Conductor Initialization")
def parse_atomic_proj(
    data: ConductorData, data_controller: DataController
) -> Dict[str, np.ndarray]:
    file_proj = data.file_names.datafile_C
    output_dir = data.file_names.output_dir
    opts = data.atomic_proj

    arry, _ = data_controller.data_dicts()

    proj_data = parse_atomic_proj_data(data, data_controller)

    log.log_proj_summary(
        proj_data,
        data,
    )

    hk_data = get_pao_hamiltonian(data_controller)

    nk = np.array([1, 1, 4], dtype=int)  # TODO: confirm hardcoded grid
    nr = nk
    ivr, wr = get_rgrid(nr)
    hk_data.update({"ivr": ivr, "wr": wr, "nk": nk, "nr": nr})
    arry.update(hk_data)
    name = Path(file_proj).name
    output_prefix = Path(output_dir) / name
    write_internal_format_files(
        Path(output_dir),
        str(output_prefix),
        data_controller,
        hk_data,
        proj_data,
        opts.do_overlap_transformation,
    )

    write_projectability_files(output_dir, proj_data, hk_data["Hk"])
    write_overlap_files(output_dir, hk_data["Sk"], opts.do_overlap_transformation)

    return hk_data


def parse_atomic_proj_data(
    data: ConductorData, data_controller: DataController
) -> AtomicProjData:
    header = parse_header(data_controller)
    kpt_data = parse_kpoints(data_controller)
    eigvals = parse_eigenvalues(data_controller)
    proj = parse_projections(data_controller)
    overlap = parse_overlaps(data, data_controller)

    return AtomicProjData(
        **header,
        **kpt_data,
        eigvals=eigvals,
        proj=proj,
        overlap=overlap,
    )


def get_pao_hamiltonian(data_controller: DataController) -> Dict[str, np.ndarray]:
    arry, attr = data_controller.data_dicts()
    Hks_raw = arry["Hks"]  # shape: (nawf, nawf, nk1, nk2, nk3, nspin)
    HRs_raw = arry["HRs"]  # shape: (nawf, nawf, nk1, nk2, nk3, nspin)
    nspin = attr["nspin"]
    nkpnts = attr["nkpnts"]
    nawf = attr["nawf"]

    # reshape to (nawf, nawf, nkpnts, nspin)
    Hks_reshaped = Hks_raw.reshape((nawf, nawf, nkpnts, nspin))
    # transpose to (nspin, nkpnts, nawf, nawf)
    Hk = np.transpose(Hks_reshaped, (3, 2, 0, 1))

    HRs_reshaped = HRs_raw.reshape((nawf, nawf, nkpnts, nspin))
    HR = np.transpose(HRs_reshaped, (3, 2, 0, 1))

    Sks_raw = (
        arry["Sks"] if "Sks" in arry else None
    )  # TODO Paoflow uses the acbn0 flag to transform Sk in case of non-orthogonality. Check if we need Sks computed with or without the acbn0 flag to get the right answer. In the current implementation, Sks is without the acbn0 flag.
    Sk = Sks_raw[:, :nawf, :] if Sks_raw is not None else None
    Sk = np.transpose(Sk, (2, 0, 1)) if Sk is not None else None

    SRs_raw = arry["SRs"] if "SRs" in arry else None
    SR = SRs_raw[:, :nawf, :] if SRs_raw is not None else None
    SR = np.transpose(SR, (2, 0, 1)) if SR is not None else None

    return {"Hk": Hk, "Sk": Sk, "HR": HR, "SR": SR}
