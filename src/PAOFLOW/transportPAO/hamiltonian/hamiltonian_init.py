from __future__ import annotations
from pathlib import Path

import numpy as np

from transportPAO.hamiltonian.hamiltonian import HamiltonianSystem
from transportPAO.io.input_parameters import CalculationType, ConductorData
from transportPAO.parsers.read_matrix import read_matrix
from transportPAO.utils.timing import timed_function


@timed_function("hamiltonian_init")
def initialize_hamiltonian_blocks(
    output_dir: str,
    ham_system: HamiltonianSystem,
    ivr_par3D: np.ndarray,
    wr_par: np.ndarray,
    table_par: np.ndarray,
    datafile_C: str,
    ispin: int,
    transport_direction: int,
    calculation_type: CalculationType,
    datafile_L: str = "",
    datafile_R: str = "",
    conductor_data: ConductorData | None = None,
) -> None:
    """
    Initialize all Hamiltonian and overlap matrix blocks for the transport system.

    Parameters
    ----------
    `ham_system` : HamiltonianSystem
        Container object holding all OperatorBlock instances for the device.
    `ivr_par3D` : np.ndarray
        (3, nrtot) array of integer lattice vectors for real-space blocks.
    `datafile_C` : str
        Path to the `.ham` file for the central conductor region.
    `datafile_L` : str
        Path to the `.ham` file for the left lead region.
    `datafile_R` : str
        Path to the `.ham` file for the right lead region.
    `ispin` : int
        Spin index (0-based) to select the spin channel to load.
    `transport_direction` : int
        Index (1-based) of the transport direction (1 = x, 2 = y, 3 = z).
    `calculation_type` : {"conductor", "bulk"}
        System configuration type.

    Returns
    -------
    `leads_are_identical` : bool
        Whether the left and right lead blocks are structurally and numerically identical.
    """

    def with_ham_suffix(path: str) -> str:
        name = Path(path).name
        if not name.endswith(".ham"):
            return f"{output_dir}/{name}.ham"
        else:
            return f"./{name}"

    def extract_2D_ivrs(ivr3D: np.ndarray, transport_direction: int) -> np.ndarray:
        if transport_direction == 1:
            return ivr3D[1:, :]
        elif transport_direction == 2:
            return ivr3D[[0, 2], :]
        elif transport_direction == 3:
            return ivr3D[:2, :]
        else:
            raise ValueError(f"Invalid transport_direction: {transport_direction}")

    ham_system.allocate(ivr_par3D, conductor_data.hamiltonian_tags)

    ivr_par2D = extract_2D_ivrs(ivr_par3D, transport_direction)

    for block in ham_system.blocks.values():
        block.ivr_par = ivr_par2D
        block.wr_par = wr_par
        block.table_par = table_par

    read_matrix(
        with_ham_suffix(datafile_C), ispin, transport_direction, ham_system.blc_00C
    )
    read_matrix(
        with_ham_suffix(datafile_C), ispin, transport_direction, ham_system.blc_CR
    )

    if calculation_type == "conductor":
        read_matrix(
            with_ham_suffix(datafile_C), ispin, transport_direction, ham_system.blc_LC
        )
        read_matrix(
            with_ham_suffix(datafile_L), ispin, transport_direction, ham_system.blc_00L
        )
        read_matrix(
            with_ham_suffix(datafile_L), ispin, transport_direction, ham_system.blc_01L
        )
        read_matrix(
            with_ham_suffix(datafile_R), ispin, transport_direction, ham_system.blc_00R
        )
        read_matrix(
            with_ham_suffix(datafile_R), ispin, transport_direction, ham_system.blc_01R
        )

    elif calculation_type == "bulk":
        ham_system.blc_00L = ham_system.blc_00C.copy()
        ham_system.blc_00R = ham_system.blc_00C.copy()
        ham_system.blc_01L = ham_system.blc_CR.copy()
        ham_system.blc_01R = ham_system.blc_CR.copy()
        ham_system.blc_LC = ham_system.blc_CR.copy()

        for block in (
            ham_system.blc_00L,
            ham_system.blc_00R,
            ham_system.blc_01L,
            ham_system.blc_01R,
            ham_system.blc_LC,
        ):
            block.tag = {
                "rows": "all",
                "cols": "all",
                "rows_sgm": "all",
                "cols_sgm": "all",
            }
            block.ivr_par = ivr_par2D

    else:
        raise ValueError(f"Invalid calculation_type: {calculation_type}")

    nk = ham_system.blc_00C.nkpts
    if nk != ham_system.blc_00L.nkpts or nk != ham_system.blc_00R.nkpts:
        raise RuntimeError("Mismatch in nkpts among C, L, R blocks")

    for ik in range(nk):
        for block in [ham_system.blc_00C, ham_system.blc_00L, ham_system.blc_00R]:
            if block.H is not None:
                H_k = block.H[..., ik]
                block.H[..., ik] = 0.5 * (H_k + H_k.T.conj())
            if block.S is not None:
                S_k = block.S[..., ik]
                block.S[..., ik] = 0.5 * (S_k + S_k.T.conj())


def check_leads_are_identical(
    ham_system: HamiltonianSystem,
    datafile_L: str = "",
    datafile_R: str = "",
    datafile_L_sgm: str = "",
    datafile_R_sgm: str = "",
) -> bool:
    """
    Determine if left and right leads are structurally and numerically identical.

    Parameters
    ----------
    `datafile_L` : str
        Path to the left lead file.
    `datafile_R` : str
        Path to the right lead file.
    `datafile_L_sgm` : str
        Path to the left lead self-energy file.
    `datafile_R_sgm` : str
        Path to the right lead self-energy file.
    `ham_system` : HamiltonianSystem
        Transport Hamiltonian system with all blocks loaded.

    Returns
    -------
    `identical` : bool
        True if the left and right leads are identical.
    """
    if datafile_L.strip() != datafile_R.strip():
        return False
    if datafile_L_sgm.strip() != datafile_R_sgm.strip():
        return False

    for key in ("irows", "icols", "irows_sgm", "icols_sgm"):
        if not np.array_equal(
            getattr(ham_system.blc_00L, key), getattr(ham_system.blc_00R, key)
        ):
            return False
        if not np.array_equal(
            getattr(ham_system.blc_01L, key), getattr(ham_system.blc_01R, key)
        ):
            return False

    return True
