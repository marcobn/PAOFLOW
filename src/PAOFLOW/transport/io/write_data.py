from pathlib import Path
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

import PAOFLOW.transport.io.log_module as log
from PAOFLOW.DataController import DataController
from PAOFLOW.transport.hamiltonian.compute_rham import compute_rham
from PAOFLOW.transport.io.log_module import log_rank0
from PAOFLOW.transport.utils.converters import crystal_to_cartesian


def write_data(
    egrid: npt.NDArray[np.float64],
    data: npt.NDArray[np.float64],
    label: str,
    output_dir: Path,
    prefix: str = '',
    postfix: str = '',
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
    filename = f'{prefix}_{label}_{postfix}.dat' if prefix else f'{label}{postfix}.dat'
    filepath = output_dir / filename

    width = 15
    fmt = f'{{:{width}.{precision}f}}'

    with filepath.open('w') as f:
        if data.ndim == 1:
            f.write(f'# E (eV)   {label}(E)\n')
            for e, val in zip(egrid, data):
                f.write(f'{fmt.format(e)}{fmt.format(val)}\n')
        else:
            dim, ne = data.shape
            if dim == 1:
                f.write(f'# E (eV)   {label}(E)\n')
                for ie in range(ne):
                    f.write(f'{fmt.format(egrid[ie])}{fmt.format(data[0, ie])}\n')
            else:
                header_channels = ' '.join(f'channel_{i + 1}' for i in range(dim))
                f.write(f'# E (eV)   {label}_total {header_channels}\n')
                for ie in range(ne):
                    values = ' '.join(fmt.format(data[i, ie]) for i in range(dim))
                    f.write(f'{fmt.format(egrid[ie])}{values}\n')

    if verbose:
        log_rank0(f'Writing {label} to {filepath}')


def write_eigenchannels(
    data: np.ndarray,
    ie: int,
    ik: int,
    vkpt: np.ndarray,
    transport_direction: int,
    output_dir: Path,
    prefix: str = 'eigchn',
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
    filename = f'{prefix}_ik{ik:04d}_ie{ie:04d}.npz'
    filepath = output_dir / filename

    if filepath.exists() and not overwrite:
        raise FileExistsError(f'File {filepath} already exists.')

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
        log_rank0(f'[INFO] Eigenchannels written to: {filepath}')

    return filepath


def populate_real_space_hamiltonian(
    data_controller: DataController,
    hk_data: Dict[str, np.ndarray],
    do_overlap_transformation: bool,
) -> None:
    """
    Compute the real-space Hamiltonian (and optional overlap) blocks and store them
    in the shared data store as ``HRs``/``SRs``.

    These in-memory arrays are what the transport pipeline consumes downstream.
    Serializing them to the ``.ham`` file (see :func:`write_internal_format_files`)
    is an independent, debug-only step that depends on the arrays populated here.

    Parameters
    ----------
    `hk_data` : Dict[str, np.ndarray]
        Dictionary containing:
            - "Hk": shape (nspin, nkpnts, dim, dim), Hamiltonian matrices
            - "Sk" (optional): shape (dim, dim, nkpnts), Overlap matrices
            - "ivr": shape (nrtot, 3), R-vectors
    `do_overlap_transformation` : bool
        If True and overlap matrices are provided, the overlap blocks are also computed.

    Notes
    -----
    The Cartesian k-points and weights used by the Fourier sum are read from the
    shared data store (``vkpts_cartesian``/``wk``), which is the single source of truth.
    """
    arry, attr = data_controller.data_dicts()
    Hk = hk_data['Hk']
    Sk = hk_data['Sk'] if 'Sk' in hk_data else None
    ivr = hk_data['ivr']

    avec = arry['a_vectors'] * attr['alat']
    vkpts_cartesian = arry['vkpts_cartesian']
    wk = arry['wk']
    _, _, dim, _ = Hk.shape
    nrtot = ivr.shape[0]
    have_overlap = Sk is not None and do_overlap_transformation

    vr_crystal = ivr.astype(np.float64).T
    rgrid_cart = crystal_to_cartesian(vr_crystal, avec).T  # (nrtot, 3)
    Hr = np.empty((nrtot, dim, dim), dtype=np.complex128)

    for ir in range(nrtot):
        Hr[ir] = compute_rham(rgrid_cart[ir], Hk[0], vkpts_cartesian, wk)

    arry['HRs'] = Hr

    if have_overlap:
        Sr = np.empty((nrtot, dim, dim), dtype=np.complex128)
        for ir in range(nrtot):
            Sr[ir] = compute_rham(rgrid_cart[ir], Sk, vkpts_cartesian, wk)
        arry['SRs'] = Sr


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
    analyticity: str = '',
    eunits: str = 'eV',
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
        raise ValueError('grid must be present for dynamical operators')
    if dynamical and not analyticity:
        raise ValueError('analyticity must be present for dynamical operators')
    if vr is None and ivr is None:
        raise ValueError('both VR and IVR not present')
    if not dynamical and nomega is not None and nomega != 1:
        raise ValueError('invalid nomega for static operator')

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
    with open(file, 'w') as f:
        f.write('<?xml version="1.0"?>\n')

        f.write('<OPERATOR>\n')

        f.write('  <DATA')
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

        f.write(' />\n')

        if vr is not None:
            f.write('  <VR>\n')

            rows, cols = vr.shape
            for i in range(rows):
                for j in range(cols):
                    val = vr[i, j]
                    f.write(f'    {val.real:18.15E},{val.imag:18.15E}\n')
            f.write('  </VR>\n')

        if ivr is not None:
            f.write('  <IVR>\n')
            rows, cols = ivr.shape
            for i in range(rows):
                f.write('    ')
                for j in range(cols):
                    if j > 0:
                        f.write(' ')
                    f.write(f'{ivr[i, j]:8d}')
                f.write('\n')
            f.write('  </IVR>\n')

        if grid is not None:
            f.write('  <GRID')
            if eunits:
                f.write(f' units="{eunits}"')
            f.write('>\n')

            grid_flat = np.array(grid).flatten()

            for i in range(len(grid_flat)):
                if i % 4 == 0:
                    if i > 0:
                        f.write(' \n')

                else:
                    f.write(' ')

                f.write(f'{grid_flat[i]:18.15E}')
            if len(grid_flat) > 0:
                f.write(' \n')
            f.write('  </GRID>\n')

        if operator_matrix is not None:
            for ie in range(nomega):
                f.write(f'  <OPR.{ie + 1}>\n')

                for ir in range(nrtot):
                    matrix = operator_matrix[ie, ir]
                    rows, cols = matrix.shape
                    total_elements = rows * cols

                    f.write(f'    <VR.{ir + 1} type="complex" size="{total_elements}">\n')

                    for j in range(cols):
                        for i in range(rows):
                            val = matrix[i, j]
                            f.write(f'{val.real: .15E},{val.imag: .15E}\n')

                    f.write(f'    </VR.{ir + 1}>\n')

                f.write(f'  </OPR.{ie + 1}>\n')

            f.write('</OPERATOR>\n')


def write_current_results(
    *,
    output_dir: str,
    bias_grid: NDArray[np.float64],
    currents: NDArray[np.float64],
) -> None:
    """Write current-vs-bias results to disk.

    Parameters
    ----------
    output_dir : str
        Directory where the output file is written.
    postfix : str
        String appended to the default file name ``current``.
    bias_grid : NDArray[np.float64]
        Bias voltages in V, shape ``(nbias,)``.
    currents : NDArray[np.float64]
        Current values aligned with ``bias_grid``, shape ``(nbias,)``.

    Returns
    -------
    None
        Writes ``current{postfix}.dat`` with two columns ``V I`` using
        :func:`numpy.savetxt`.
    """
    outpath = Path(output_dir) / 'current.dat'
    outpath.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(outpath, np.column_stack([bias_grid, currents]))
    log.log_rank0(f'Saved current vs bias to {outpath}')
