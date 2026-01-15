from PAOFLOW.DataController import DataController
import numpy as np
import os
from pathlib import Path

from typing import Dict, Optional

import numpy.typing as npt

from PAOFLOW.transport.hamiltonian.compute_rham import compute_rham
from PAOFLOW.transport.io.input_parameters import AtomicProjData
from PAOFLOW.transport.utils.converters import crystal_to_cartesian
from PAOFLOW.transport.io.log_module import log_rank0


def write_data(
    egrid: npt.NDArray[np.float64],
    data: npt.NDArray[np.float64],
    label: str,
    output_dir: Path,
    prefix: str = "",
    postfix: str = "",
    precision: int = 9,
    verbose: bool = True,
) -> None:
    """
    Write general data (e.g., conductance or DOS) into a single text file.

    Parameters
    ----------
    `egrid` : (ne,) ndarray
        Energy grid.
    `data` : (dim, ne) or (ne,) ndarray
        Data to write.
    `label` : str
        Data type label used for header and filename (e.g., "conductance", "doscond").
    `output_dir` : Path
        Directory to store the output files.
    `prefix` : str
        Optional prefix to prepend to the filename.
    `postfix` : str
        Optional postfix to append to the filename.
    `precision` : int
        Number of decimal places to write.
    `verbose` : bool
        Whether to print output file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{label}_{postfix}.dat" if prefix else f"{label}{postfix}.dat"
    filepath = output_dir / filename

    width = 15
    fmt = f"{{:{width}.{precision}f}}"

    with filepath.open("w") as f:
        if data.ndim == 1:
            f.write(f"# E (eV)   {label}(E)\n")
            for e, val in zip(egrid, data):
                f.write(f"{fmt.format(e)}{fmt.format(val)}\n")
        else:
            dim, ne = data.shape
            if dim == 1:
                f.write(f"# E (eV)   {label}(E)\n")
                for ie in range(ne):
                    f.write(f"{fmt.format(egrid[ie])}{fmt.format(data[0, ie])}\n")
            else:
                header_channels = " ".join(f"channel_{i + 1}" for i in range(dim))
                f.write(f"# E (eV)   {label}_total {header_channels}\n")
                for ie in range(ne):
                    values = " ".join(fmt.format(data[i, ie]) for i in range(dim))
                    f.write(f"{fmt.format(egrid[ie])}{values}\n")

    if verbose:
        log_rank0(f"Writing {label} to {filepath}")


def write_eigenchannels(
    data: np.ndarray,
    ie: int,
    ik: int,
    vkpt: np.ndarray,
    transport_direction: int,
    output_dir: Path,
    prefix: str = "eigchn",
    overwrite: bool = True,
    verbose: bool = True,
) -> Path:
    """
    Write eigenchannel data to a compressed .npz file with metadata.

    Parameters
    ----------
    `data` : (n, m) complex ndarray
        Eigenchannel matrix. Columns correspond to eigenchannels.
    `ie` : int
        Energy index.
    `ik` : int
        k-point index.
    `vkpt` : (3,) float ndarray
        Coordinates of the k-point in crystal units.
    `transport_direction` : int
        Direction of transport (typically 1, 2, or 3).
    `output_dir` : Path
        Directory to write the output file.
    `prefix` : str
        Prefix for the filename (default: "eigchn").
    `overwrite` : bool
        If True, overwrite existing file.
    `verbose` : bool
        If True, print where the file was written.

    Returns
    -------
    `filepath` : Path
        Path to the written file.

    Notes
    -----
    This uses `.npz` to store:
        - eigenchannel data
        - metadata: ie, ik, vkpt, dims, transport_direction
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_ik{ik:04d}_ie{ie:04d}.npz"
    filepath = output_dir / filename

    if filepath.exists() and not overwrite:
        raise FileExistsError(f"File {filepath} already exists.")

    np.savez_compressed(
        filepath,
        eigenchannels=data,
        ie=ie,
        ik=ik,
        vkpt=vkpt,
        transport_direction=transport_direction,
        dim1=data.shape[0],
        dim2=data.shape[1],
    )

    if verbose:
        log.log_rank0(f"[INFO] Eigenchannels written to: {filepath}")

    return filepath


def iotk_index(n: int) -> str:
    """Return IOTK index tag used in XML labels (e.g., 1 → '.1')."""
    return f".{n}"


def write_internal_format_files(
    output_dir: Path,
    output_prefix: str,
    data_controller: DataController,
    hk_data: Dict[str, np.ndarray],
    proj_data: AtomicProjData,
    do_overlap_transformation: bool,
) -> None:
    """
    Write Hamiltonian and optional overlap matrices in a format that matches the legacy IOTK-style .ham file structure.

    The output includes:
    - Dimensional and symmetry metadata in a DATA tag
    - Real-space and reciprocal lattice vectors
    - K-point list and weights
    - R-vectors and their weights
    - Hamiltonian matrix blocks (VR.#)
    - Overlap matrix blocks (OVERLAP.#), if enabled

    Parameters
    ----------
    `output_prefix` : str
        Prefix for the output file (e.g., 'al5_bulk' → 'al5_bulk.ham').
    `hk_data` : Dict[str, np.ndarray]
        Dictionary containing:
            - "Hk": shape (nspin, nrtot, dim, dim), Hamiltonian matrices
            - "S" (optional): shape (nspin, nrtot, dim, dim), Overlap matrices
            - "ivr": shape (nrtot, 3), R-vectors
            - "wr": shape (nrtot,), R-vector weights
    `proj_data` : Dict[str, np.ndarray]
        Dictionary containing:
            - "kpts": shape (nkpnts, 3), list of k-points
            - "wk": shape (nkpnts,), k-point weights
    `lattice_data` : Dict[str, np.ndarray]
        Dictionary containing:
            - "avec": shape (3, 3), direct lattice vectors
            - "bvec": shape (3, 3), reciprocal lattice vectors
    `do_overlap_transformation` : bool
        If True and overlap matrices are provided, overlap blocks will be written to the output.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ham_file = (
        output_prefix if output_prefix.endswith(".ham") else output_prefix + ".ham"
    )
    arry, attr = data_controller.data_dicts()
    Hk = hk_data["Hk"]
    Sk = hk_data["Sk"] if "Sk" in hk_data else None
    ivr = hk_data["ivr"]
    wr = hk_data["wr"]

    avec = arry["a_vectors"] * attr["alat"]
    bvec = arry["b_vectors"] * (2.0 * np.pi / attr["alat"])
    kpts = proj_data.kpts
    vkpts_crystal = proj_data.vkpts_crystal
    vkpts_cartesian = proj_data.vkpts_cartesian
    wk = proj_data.wk
    spin_component = "all"
    shift = np.zeros(3, dtype=float)  # No shift in k-point grid for crystal coordinates
    nspin, _, dim, _ = Hk.shape
    nkpnts = kpts.shape[1]
    nrtot = ivr.shape[0]
    nk = hk_data["nk"]
    nr = hk_data["nr"]
    have_overlap = Sk is not None and do_overlap_transformation
    fermi_energy = 0.0

    vr_crystal = ivr.astype(np.float64).T
    rgrid_cart = crystal_to_cartesian(vr_crystal, avec).T  # (nrtot, 3)
    Hr = np.empty((nrtot, dim, dim), dtype=np.complex128)

    for ir in range(nrtot):
        Hr[ir] = compute_rham(rgrid_cart[ir], Hk[0], vkpts_cartesian, wk)

    if have_overlap:
        Sr = np.empty((nrtot, dim, dim), dtype=np.complex128)
        for ir in range(nrtot):
            Sr[ir] = compute_rham(rgrid_cart[ir], Sk, vkpts_cartesian, wk)

    arry["HRs"] = Hr

    if have_overlap:
        arry["SRs"] = Sr

    with open(ham_file, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<?iotk version="1.2.0"?>\n')
        f.write('<?iotk file_version="1.0"?>\n')
        f.write('<?iotk binary="F"?>\n')
        f.write("<Root>\n")
        f.write("  <HAMILTONIAN>\n")

        f.write(
            f'    <DATA dimwann="{dim}" nkpnts="{nkpnts}" nspin="{nspin}" spin_component="{spin_component}" '
        )
        f.write(
            f'nk="{nk[0]} {nk[1]} {nk[2]}" shift="{shift}" nrtot="{nrtot}" nr="{nr[0]} {nr[1]} {nr[2]}" '
        )
        f.write(f'have_overlap="{"T" if have_overlap else "F"}"\n')
        f.write(f'fermi_energy="{fermi_energy:.15E}"/>\n')

        f.write('    <DIRECT_LATTICE type="real" size="9" columns="3" units="bohr">\n')
        for row in avec.T:
            f.write(" " + "  ".join(f"{x:.15E}" for x in row) + "\n")
        f.write("    </DIRECT_LATTICE>\n")

        f.write(
            '    <RECIPROCAL_LATTICE type="real" size="9" columns="3" units="bohr^-1">\n'
        )
        for row in bvec.T:
            f.write(" " + "  ".join(f"{x:.15E}" for x in row) + "\n")
        f.write("    </RECIPROCAL_LATTICE>\n")

        f.write(
            f'    <VKPT type="real" size="{3 * nkpnts}" columns="3" units="crystal">\n'
        )
        for i in range(vkpts_crystal.shape[1]):
            f.write(
                " " + "  ".join(f"{vkpts_crystal[j, i]:.15E}" for j in range(3)) + "\n"
            )
        f.write("    </VKPT>\n")

        f.write(f'    <WK type="real" size="{nkpnts}">\n')
        for w in wk:
            f.write(f" {w:.15E}\n")
        f.write("    </WK>\n")
        f.write(
            f'    <IVR type="integer" size="{3 * nrtot}" columns="3" units="crystal">\n'
        )
        for row in ivr:
            f.write(" {:10d}{:10d}{:10d} \n".format(*row))
        f.write("    </IVR>\n")
        f.write(f'    <WR type="real" size="{nrtot}">\n')
        for w in wr:
            f.write(f" {w:.15E}\n")
        f.write("    </WR>\n")
        f.write("    <RHAM>\n")
        for ir in range(nrtot):
            tag = f"VR.{ir + 1}"
            f.write(f'      <{tag} type="complex" size="{dim * dim}">\n')
            for z in Hr[ir].flatten():
                f.write(f" {z.real:> .15E},{z.imag:> .15E}\n")
            f.write(f"      </{tag}>\n")

            if have_overlap:
                tag = f"OVERLAP.{ir + 1}"
                f.write(f'      <{tag} type="complex" size="{dim * dim}">\n')
                for z in Sr[ir].flatten():
                    f.write(f" {z.real:> .15E},{z.imag:> .15E}\n")
                f.write(f"      </{tag}>\n")
        f.write("    </RHAM>\n")
        write_kham(Hk, f)

        f.write("  </HAMILTONIAN>\n")
        f.write("</Root>\n")


def write_kham(
    Hk: np.ndarray,
    f: object,
    spin_component: str = "all",
    tag: str = "KHAM",
    block_prefix: str = "KH",
) -> None:
    """
    Write Hk to an IOTK-style XML file.

    Parameters
    ----------
    `Hk` : (nspin, nkpnts, dim, dim) complex ndarray
        Hamiltonian matrices in k-space.
    `output_file` : Path
        Destination XML file.
    `spin_component` : str
        One of: "all", "up", "down".
    `tag` : str
        Name of the XML block (default: "KHAM").
    `block_prefix` : str
        Prefix for matrix block tags (default: "KH" → <KH.1>, <KH.2>, ...)
    """
    f.write("  <HAMILTONIAN>\n")
    nspin, nkpnts, _, _ = Hk.shape

    for isp in range(nspin):
        if spin_component == "up" and isp == 1:
            continue
        if spin_component == "down" and isp == 0:
            continue

        if spin_component == "all" and nspin == 2:
            f.write(f"    <SPIN.{isp + 1}>\n")

        f.write(f"      <{tag}>\n")
        for ik in range(nkpnts):
            tagname = f"{block_prefix}.{ik + 1}"
            mat = Hk[isp, ik]
            dim = mat.shape[0]
            f.write(f'        <{tagname} type="complex" size="{dim * dim}">\n')
            for i in range(dim):
                for j in range(dim):
                    z = mat[i, j]
                    f.write(f" {z.real: .15E},{z.imag: .15E}\n")
            f.write(f"        </{tagname}>\n")
        f.write(f"      </{tag}>\n")

        if spin_component == "all" and nspin == 2:
            f.write(f"    </SPIN.{isp + 1}>\n")

    f.write("  </HAMILTONIAN>\n")


def write_operator_xml(
    *,
    output_dir: Path,
    filename: str,
    operator_matrix: Optional[np.ndarray] = None,
    ivr: Optional[np.ndarray] = None,
    vr: Optional[np.ndarray] = None,
    grid: Optional[np.ndarray] = None,
    dimwann: int,
    dynamical: bool,
    analyticity: str = "",
    eunits: str = "eV",
    nomega: Optional[int] = None,
    iomg_s: Optional[int] = None,
    iomg_e: Optional[int] = None,
    nrtot: Optional[int] = None,
) -> None:
    """
    Write operator data to XML file in the exact format produced by Fortran iotk library.

    This function mimics the Fortran subroutine operator_write_aux exactly, including
    formatting, spacing, and element ordering.
    """
    if dynamical and grid is None:
        raise ValueError("grid must be present for dynamical operators")
    if dynamical and not analyticity:
        raise ValueError("analyticity must be present for dynamical operators")
    if vr is None and ivr is None:
        raise ValueError("both VR and IVR not present")
    if not dynamical and nomega is not None and nomega != 1:
        raise ValueError("invalid nomega for static operator")

    if operator_matrix is not None:
        if nomega is None:
            nomega = operator_matrix.shape[0]
        if nrtot is None:
            nrtot = operator_matrix.shape[1]
    else:
        if nomega is None:
            nomega = 1
        if nrtot is None:
            nrtot = len(ivr) if ivr is not None else len(vr)

    file = output_dir / filename
    with open(file, "w") as f:
        f.write('<?xml version="1.0"?>\n')

        f.write("<OPERATOR>\n")

        f.write("  <DATA")
        f.write(f' dimwann="{dimwann}"')
        f.write(f' nrtot="{nrtot}"')
        f.write(f' dynamical="{str(dynamical).upper()}"')
        f.write(f' nomega="{nomega}"')

        if iomg_s is not None:
            f.write(f' iomg_s="{iomg_s}"')
        if iomg_e is not None:
            f.write(f' iomg_e="{iomg_e}"')

        if dynamical:
            f.write(f' analyticity="{analyticity}"')

        f.write(" />\n")

        if vr is not None:
            f.write("  <VR>\n")

            rows, cols = vr.shape
            for i in range(rows):
                for j in range(cols):
                    val = vr[i, j]
                    f.write(f"    {val.real:18.15E},{val.imag:18.15E}\n")
            f.write("  </VR>\n")

        if ivr is not None:
            f.write("  <IVR>\n")
            rows, cols = ivr.shape
            for i in range(rows):
                f.write("    ")
                for j in range(cols):
                    if j > 0:
                        f.write(" ")
                    f.write(f"{ivr[i, j]:8d}")
                f.write("\n")
            f.write("  </IVR>\n")

        if grid is not None:
            f.write("  <GRID")
            if eunits:
                f.write(f' units="{eunits}"')
            f.write(">\n")

            grid_flat = np.array(grid).flatten()

            for i in range(len(grid_flat)):
                if i % 4 == 0:
                    if i > 0:
                        f.write(" \n")

                else:
                    f.write(" ")

                f.write(f"{grid_flat[i]:18.15E}")
            if len(grid_flat) > 0:
                f.write(" \n")
            f.write("  </GRID>\n")

        if operator_matrix is not None:
            for ie in range(nomega):
                f.write(f"  <OPR.{ie + 1}>\n")

                for ir in range(nrtot):
                    matrix = operator_matrix[ie, ir]
                    rows, cols = matrix.shape
                    total_elements = rows * cols

                    f.write(
                        f'    <VR.{ir + 1} type="complex" size="{total_elements}">\n'
                    )

                    for j in range(cols):
                        for i in range(rows):
                            val = matrix[i, j]
                            f.write(f"{val.real: .15E},{val.imag: .15E}\n")

                    f.write(f"    </VR.{ir + 1}>\n")

                f.write(f"  </OPR.{ie + 1}>\n")

            f.write("</OPERATOR>\n")


def write_projectability_files(
    output_dir: str, proj_data: AtomicProjData, Hk: np.ndarray
) -> None:
    proj = proj_data.proj
    eigvals = proj_data.eigvals
    nspin, nkpnts, _, _ = Hk.shape
    nbnds = proj_data.nbnds

    for isp in range(nspin):
        proj_file = (
            os.path.join(output_dir, f"projectability_{['up', 'dn'][isp]}.txt")
            if nspin == 2
            else os.path.join(output_dir, "projectability.txt")
        )
        with open(proj_file, "w") as f:
            f.write("# Energy (eV)        Projectability\n")
            for ik in range(nkpnts):
                for ib in range(nbnds):
                    proj_vec = proj[:, ib, ik, isp]
                    weight = np.vdot(proj_vec, proj_vec).real
                    energy = eigvals[ib, ik, isp]
                    f.write(f"{energy:20.13f}  {weight:20.13f}\n")
    log_rank0("Printed projectabilities to projectability.txt")


def write_overlap_files(
    output_dir: str, Sk: np.ndarray, do_overlap_transformation: bool
) -> None:
    if not do_overlap_transformation or Sk is None:
        return
    nR = Sk.shape[2]
    nawf = Sk.shape[0]
    kovp_file = os.path.join(output_dir, "kovp.txt")
    with open(kovp_file, "w") as f:
        f.write("# Overlap Real        Overlap Imag\n")
        for ik in range(nR):
            mat = Sk[:, :, ik]
            for i in range(nawf):
                for j in range(nawf):
                    f.write(f"{mat[i, j].real:20.13f}  {mat[i, j].imag:20.13f}\n")
    log_rank0("Printed overlap matrices to kovp.txt")
