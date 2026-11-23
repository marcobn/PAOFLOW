from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from PAOFLOW.transport.data import ConductorData
from PAOFLOW.transport.io.log_module import log_rank0
from PAOFLOW.transport.io.write_data import (
    write_data,
    write_operator_xml,
)
from PAOFLOW.transport.io.write_header import headered_function
from PAOFLOW.utils.constants import AMCONV, RYDCM1


def write_conductor_operators(
    *,
    rank: int,
    data: ConductorData,
    gf_out: NDArray[np.complex128] | None,
    rsgmL_out: NDArray[np.complex128] | None,
    rsgmR_out: NDArray[np.complex128] | None,
    ivr_par3D: NDArray[np.int64],
    egrid: NDArray[np.float64],
    dimC: int,
) -> None:
    """Write real-space operators to XML files on rank 0.

    Parameters
    ----------
    rank : int
        MPI rank.
    data : ConductorData
        Validated transport input and runtime flags.
    gf_out : NDArray[np.complex128] or None
        Real-space Green's function output array.
    rsgmL_out : NDArray[np.complex128] or None
        Real-space left self-energy output array.
    rsgmR_out : NDArray[np.complex128] or None
        Real-space right self-energy output array.
    ivr_par3D : NDArray[np.int64]
        Integer real-space vectors associated with operator rows.
    egrid : NDArray[np.float64]
        Energy grid in eV, shape ``(ne,)``.
    dimC : int
        Conductor block dimension.

    Returns
    -------
    None
        Writes ``greenf.xml``, ``lead_L_sgm.xml``, and ``lead_R_sgm.xml``
        depending on enabled flags. Non-root ranks perform no writes.
    """
    if rank != 0:
        return

    if data.symmetry.write_gf and gf_out is not None:
        write_operator_xml(
            output_dir=Path(data.file_names.output_dir),
            filename='greenf.xml',
            operator_matrix=gf_out,
            ivr=ivr_par3D,
            grid=egrid,
            dimwann=dimC,
            dynamical=True,
            eunits='eV',
            analyticity='retarded',
        )
    if data.symmetry.write_lead_sgm and rsgmL_out is not None and rsgmR_out is not None:
        write_operator_xml(
            output_dir=Path(data.file_names.output_dir),
            filename='lead_L_sgm.xml',
            operator_matrix=rsgmL_out,
            ivr=ivr_par3D,
            grid=egrid,
            dimwann=dimC,
            dynamical=True,
            eunits='eV',
            analyticity='retarded',
        )
        write_operator_xml(
            output_dir=Path(data.file_names.output_dir),
            filename='lead_R_sgm.xml',
            operator_matrix=rsgmR_out,
            ivr=ivr_par3D,
            grid=egrid,
            dimwann=dimC,
            dynamical=True,
            eunits='eV',
            analyticity='retarded',
        )


@headered_function('Writing surface bands')
def write_surface_bands(
    *,
    rank: int,
    data: ConductorData,
    dos_k: NDArray[np.float64],
    egrid: NDArray[np.float64],
) -> None:
    r"""Write the surface-projected spectral map and its plotting axes.

    Parameters
    ----------
    rank : int
        MPI rank. Only rank 0 writes.
    data : ConductorData
        Validated transport input and runtime flags.
    dos_k : NDArray[np.float64]
        Surface spectral function :math:`A(k, E)`, shape ``(ne, nkpts)``.
    egrid : NDArray[np.float64]
        Energy grid in eV, shape ``(ne,)``.

    Returns
    -------
    None
        Writes three files under ``data.file_names.output_dir``:

        - ``surfband{postfix}.dat`` : the raw ``(ne, nkpts)`` matrix, one row per
          energy, directly loadable as a 2D heatmap.
        - ``surfband_egrid{postfix}.dat`` : the energy axis, one value per row.
        - ``surfband_kpath{postfix}.dat`` : the k axis as
          ``index  kdist  label``, with ``label`` set at high-symmetry points.

    Notes
    -----
    In surface mode the conductor Green's function excludes the right-lead
    self-energy, so

    .. math::

        A(k, E) = -\frac{1}{\pi}\,\mathrm{Im}\,\mathrm{Tr}\,
        \left[\left(E + i\delta\right)S - H_C - \Sigma_L\right]^{-1}

    is the surface-projected bulk band structure. Weights along the k-path are
    unity, so the values are reported raw rather than BZ-averaged.
    """
    if rank != 0:
        return

    output_dir = Path(data.file_names.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    postfix = data.file_names.postfix
    runtime = data.get_runtime_data()

    spectral_path = output_dir / f'surfband{postfix}.dat'
    np.savetxt(spectral_path, dos_k, fmt='%15.9f')

    egrid_path = output_dir / f'surfband_egrid{postfix}.dat'
    np.savetxt(egrid_path, egrid, fmt='%15.9f')

    nkpts = dos_k.shape[1]
    kdist = runtime.kpath_dist
    if kdist is None:
        kdist = np.arange(nkpts, dtype=np.float64)

    tick_labels: dict[int, str] = {}
    if runtime.kpath_ticks is not None and runtime.kpath_labels is not None:
        for index, label in zip(runtime.kpath_ticks, runtime.kpath_labels):
            if 0 <= int(index) < nkpts:
                tick_labels[int(index)] = label

    kpath_path = output_dir / f'surfband_kpath{postfix}.dat'
    with kpath_path.open('w') as f:
        f.write('# index   kdist            label\n')
        for ik in range(nkpts):
            f.write(f'{ik:6d} {kdist[ik]:15.9f}  {tick_labels.get(ik, "")}\n')

    log_rank0(f'Writing surface bands to {spectral_path}')
    log_rank0(f'  energy axis -> {egrid_path}')
    log_rank0(f'  k-path axis -> {kpath_path}')


@headered_function('Writing data')
def write_conductor_output(
    *,
    rank: int,
    data: ConductorData,
    conduct: NDArray[np.float64],
    dos: NDArray[np.float64],
    conduct_k: NDArray[np.float64],
    dos_k: NDArray[np.float64],
    egrid: NDArray[np.float64],
) -> None:
    """Write conductance and DOS data products for the conductor workflow.

    Parameters
    ----------
    rank : int
        MPI rank.
    data : ConductorData
        Validated transport input and runtime flags.
    conduct : NDArray[np.float64]
        Total conductance and optional eigenchannels, shape ``(1 + neigchn, ne)``.
    dos : NDArray[np.float64]
        Total DOS, shape ``(ne,)``.
    conduct_k : NDArray[np.float64]
        k-resolved conductance, shape ``(1 + neigchn, nkpts_par, ne)``.
    dos_k : NDArray[np.float64]
        k-resolved DOS, shape ``(ne, nkpts_par)``.
    egrid : NDArray[np.float64]
        Energy grid in eV, shape ``(ne,)``.

    Returns
    -------
    None
        Writes ``conductance*.dat`` and ``doscond*.dat`` under
        ``data.file_names.output_dir``. When ``write_kdata`` is enabled,
        also writes per-kpoint files. Non-root ranks perform no writes.
    """
    if rank != 0:
        return

    output_dir = Path(data.file_names.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    postfix = data.file_names.postfix

    if data.carriers == 'phonons':
        egrid_out = np.sqrt(egrid * RYDCM1**2 / AMCONV)
    else:
        egrid_out = egrid

    write_data(egrid_out, conduct, 'conductance', output_dir, postfix=postfix)
    write_data(egrid_out, dos, 'doscond', output_dir, postfix=postfix)

    if data.symmetry.write_kdata:
        nkpts_par = data.get_runtime_data().nkpts_par

        for ik in range(nkpts_par):
            ik_str = f'{ik + 1:04d}'
            filename_cond = f'cond{postfix}-{ik_str}.dat'
            filename_dos = f'doscond{postfix}-{ik_str}.dat'

            with (output_dir / filename_cond).open('w') as f:
                for ie in range(egrid.shape[0]):
                    values = ' '.join(
                        f'{conduct_k[ch, ik, ie]:15.9f}' for ch in range(conduct_k.shape[0])
                    )
                    f.write(f'{egrid[ie]:15.9f} {values}\n')

            with (output_dir / filename_dos).open('w') as f:
                for ie in range(egrid.shape[0]):
                    f.write(f'{egrid[ie]:15.9f} {dos_k[ie, ik]:15.9f}\n')
