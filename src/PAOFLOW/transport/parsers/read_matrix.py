from __future__ import annotations

import numpy as np
from pathlib import Path

from PAOFLOW.transport.hamiltonian.operator_blc import OperatorBlock
from PAOFLOW.transport.io.iotk_reader import IOTKReader
from PAOFLOW.transport.parsers.parser_base import parse_index_array
from PAOFLOW.transport.hamiltonian.fourier_par import fourier_transform_real_to_kspace
from PAOFLOW.transport.utils.timing import timed_function

# TODO check if fortran is also meant to skip block VR.1 and only read blocks  VR.2, VR.3 ... Initial tests seem to suggest that this is happening in the fortran version and hence implemented ths same way in python


@timed_function("read_matrix")
def read_matrix(
    filename: str,
    ispin: int,
    transport_direction: int,
    opr: OperatorBlock,
) -> None:
    """
    Read the Hamiltonian and overlap block from a .ham IOTK-formatted file
    and populate the given OperatorBlock with the k-space transformed data.

    Parameters
    ----------
    `filename` : str
        Path to the input .ham file.
    `ispin` : int
        Spin index (0-based) to select the spin component (if applicable).
    `transport_direction` : int
        Transport direction (1=x, 2=y, 3=z).
    `opr` : OperatorBlock
        Target operator block to populate.

    Notes
    -----
    This function reads real-space Hamiltonian and overlap data from a Quantum ESPRESSO
    `.ham` file in IOTK format, processes it, and populates the operator block `opr`
    in reciprocal space using a partial Fourier transform.

    The function supports spin-polarized and non-spin-polarized cases. For spin-polarized
    input, the appropriate spin channel is extracted from the nested `@SPINn` sections.

    Matrix blocks are selected by identifying matching 3D integer R-vectors `ivr_aux` that
    correspond to a fixed component along the transport direction (e.g., 0 for on-site blocks,
    1 for coupling blocks). These vectors are matched against the global `IVR` list from the file.

    For each matching R-vector:
        - A real-space Hamiltonian block `VRn` is read.
        - If present, the corresponding overlap block `OVERLAPn` is also read.
        - If overlap is not present in the file and the block label corresponds to an on-site
          block, then an identity matrix is used for the zero R-vector and zeros elsewhere.

    The selected block is sliced using index arrays parsed from the `opr.tag` metadata, and
    inserted into 3D tensors A (Hamiltonian) and S (overlap), indexed over the R-vector grid.

    Finally, a partial 2D Fourier transform is applied in the directions orthogonal to
    the transport axis to obtain the k-resolved operator block.

    """
    if not opr.allocated:
        raise RuntimeError("OperatorBlock is not allocated")

    attr = opr.tag
    label = opr.name.strip()

    # === Defaults and attribute parsing ===
    filein = filename
    cols = attr.get("cols", "all").lower()
    rows = attr.get("rows", "all").lower()
    cols_sgm = attr.get("cols_sgm", cols).lower()
    rows_sgm = attr.get("rows_sgm", rows).lower()
    ivr_input = int(attr.get("ivr", 0))
    ivr_from_input = "ivr" in attr

    dim1, dim2 = opr.dim1, opr.dim2

    # Convert "all" to full ranges
    if rows == "all":
        rows = f"1-{dim1}"
    if cols == "all":
        cols = f"1-{dim2}"
    if rows_sgm == "all":
        rows_sgm = f"1-{dim1}"
    if cols_sgm == "all":
        cols_sgm = f"1-{dim2}"

    # === File parsing ===
    reader = IOTKReader(Path(filein))
    header = reader.read_header()

    ldimwann = header["dimwann"]
    nspin = header.get("nspin", 1)
    nrtot = header["nrtot"]
    lhave_ovp = header.get("have_overlap", False)

    irows = parse_index_array(rows, ldimwann)
    icols = parse_index_array(cols, ldimwann)
    irows_sgm = parse_index_array(rows_sgm, ldimwann)
    icols_sgm = parse_index_array(cols_sgm, ldimwann)

    opr.irows = irows
    opr.icols = icols
    opr.irows_sgm = irows_sgm
    opr.icols_sgm = icols_sgm

    if nspin == 2 and ispin < 0:
        raise ValueError("Unspecified ispin for spin-polarized case")

    ivr = reader.read_array("IVR", shape=(nrtot, 3), dtype=int).T

    # Check grid dimensions
    nrtot_par = opr.H.shape[2]
    A = np.zeros((dim1, dim2, nrtot_par), dtype=complex)
    S = np.zeros((dim1, dim2, nrtot_par), dtype=complex)

    if nspin == 2:
        reader.find_section(f"SPIN{ispin + 1}")

    reader.find_section("RHAM")
    for ir_par in range(nrtot_par):
        ivr_aux = np.zeros(3, dtype=int)
        j = 0
        for i in range(3):
            if i + 1 == transport_direction:
                if label.lower() in {
                    "block_00c",
                    "block_00r",
                    "block_00l",
                    "block_t",
                    "block_e",
                    "block_b",
                    "block_eb",
                    "block_be",
                }:
                    ivr_aux[i] = 0
                elif label.lower() in {
                    "block_01r",
                    "block_01l",
                    "block_lc",
                    "block_cr",
                }:
                    ivr_aux[i] = 1
                else:
                    raise ValueError(f"Invalid block label {label}")
                if ivr_from_input:
                    ivr_aux[i] = ivr_input
            else:
                ivr_aux[i] = opr.ivr_par[j, ir_par]
                j += 1

        matches = [ir for ir in range(nrtot) if np.array_equal(ivr[:, ir], ivr_aux)]
        if not matches:
            raise ValueError(f"3D R-vector {ivr_aux} not found for ir_par={ir_par}")

        ind = matches[0] + 1
        A_loc = reader.read_array(
            f"VR.{ind}", shape=(ldimwann, ldimwann), dtype=complex
        )

        if lhave_ovp:
            S_loc = reader.read_array(
                f"OVERLAP{ind}", shape=(ldimwann, ldimwann), dtype=complex
            )
        else:
            S_loc = np.zeros_like(A_loc)
            if label.lower() in {
                "block_00c",
                "block_00r",
                "block_00l",
                "block_t",
                "block_e",
                "block_b",
                "block_eb",
                "block_be",
            } and np.all(ivr_aux == 0):
                S_loc[:] = np.eye(ldimwann)

        A_loc_T = A_loc.T
        S_loc_T = S_loc.T

        for j in range(dim2):
            for i in range(dim1):
                if icols[j] < 0 or irows[i] < 0:
                    continue
                A[i, j, ir_par] = A_loc_T[irows[i], icols[j]]
                S[i, j, ir_par] = S_loc_T[irows[i], icols[j]]

    opr.H = fourier_transform_real_to_kspace(A, opr.wr_par, opr.table_par)
    opr.S = fourier_transform_real_to_kspace(S, opr.wr_par, opr.table_par)
