from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from PAOFLOW.transport.calculators.transmittance import evaluate_transmittance
from PAOFLOW.transport.data import ConductorData
from PAOFLOW.transport.io.write_data import write_eigenchannels
from PAOFLOW.transport.observables.broadening import compute_broadening_matrix


def accumulate_dos(
    dos: NDArray[np.float64],
    dos_k: NDArray[np.float64],
    gC: NDArray[np.complex128],
    wk: NDArray[np.float64],
    ie_g: int,
    ik: int,
) -> None:
    """Accumulate DOS at one ``(E, k)`` point.

    Parameters
    ----------
    dos : NDArray[np.float64]
        Total DOS accumulator, shape ``(ne,)``.
    dos_k : NDArray[np.float64]
        k-resolved DOS accumulator, shape ``(ne, nkpts_par)``.
    gC : NDArray[np.complex128]
        Conductor retarded Green's function at the current ``(E, k)``,
        shape ``(dimC, dimC)``.
    wk : NDArray[np.float64]
        k-point weights, shape ``(nkpts_par,)``.
    ie_g : int
        Energy index.
    ik : int
        k-point index.

    Returns
    -------
    None
        Updates ``dos`` and ``dos_k`` in place for the selected energy and
        k-point.

    Notes
    -----
    The DOS contribution is evaluated as

    .. math::

        \\mathrm{DOS}(E, k) = -\\frac{w_k}{\\pi} \\operatorname{Im}
        \\operatorname{Tr}\\left[G_C^r(E, k)\\right].
    """
    diag_imag = np.imag(np.diagonal(gC))
    dos_k[ie_g, ik] = -wk[ik] * np.sum(diag_imag) / np.pi
    dos[ie_g] += dos_k[ie_g, ik]


def accumulate_transmission(
    conduct: NDArray[np.float64],
    conduct_k: NDArray[np.float64],
    gC: NDArray[np.complex128],
    sigma_L: NDArray[np.complex128],
    sigma_R: NDArray[np.complex128],
    wk: NDArray[np.float64],
    ie_g: int,
    ik: int,
    *,
    data: ConductorData,
    delta: float,
    rank: int,
    vkpt: NDArray[np.float64],
) -> None:
    """Accumulate transmission and eigenchannel contributions at one ``(E, k)``.

    Parameters
    ----------
    conduct : NDArray[np.float64]
        Total conductance and optional eigenchannel accumulators,
        shape ``(1 + neigchn, ne)``.
    conduct_k : NDArray[np.float64]
        k-resolved conductance accumulators, shape ``(1 + neigchn, nkpts_par, ne)``.
    gC : NDArray[np.complex128]
        Conductor retarded Green's function, shape ``(dimC, dimC)``.
    sigma_L : NDArray[np.complex128]
        Left lead self-energy, shape ``(dimC, dimC)``.
    sigma_R : NDArray[np.complex128]
        Right lead self-energy, shape ``(dimC, dimC)``.
    wk : NDArray[np.float64]
        k-point weights, shape ``(nkpts_par,)``.
    ie_g : int
        Energy index.
    ik : int
        k-point index.
    data : ConductorData
        Transport input and symmetry flags.
    delta : float
        Broadening parameter used by the transmission evaluator.
    rank : int
        MPI rank for guarded output.
    vkpt : NDArray[np.float64]
        Cartesian k-point coordinates, shape ``(3, nkpts_par)``.

    Returns
    -------
    None
        Updates ``conduct`` and ``conduct_k`` in place. If eigenchannel plotting
        is enabled at the selected ``(ie_g, ik)`` and ``rank == 0``, writes files
        under ``output/eigenchannels`` via ``write_eigenchannels``.

    Notes
    -----
    The transmission is evaluated from

    .. math::

        T(E, k) = \\operatorname{Tr}\\left[\\Gamma_L G_C^r \\Gamma_R G_C^a\\right],

    with :math:`\\Gamma_{L/R} = i\\left(\\Sigma_{L/R} - \\Sigma_{L/R}^\\dagger\\right)`.
    """
    gamma_L = compute_broadening_matrix(sigma_L)
    gamma_R = compute_broadening_matrix(sigma_R)

    do_eigplot_now = (
        data.symmetry.do_eigenchannels
        and data.symmetry.do_eigplot
        and ie_g == data.symmetry.ie_eigplot
        and ik == data.symmetry.ik_eigplot
    )

    cond_aux, z_eigplot = evaluate_transmittance(
        gamma_L=gamma_L,
        gamma_R=gamma_R,
        G_ret=gC,
        formula=data.conduct_formula,
        do_eigenchannels=data.symmetry.do_eigenchannels,
        do_eigplot=do_eigplot_now,
        sgm_corr=None,
        eta=delta,
        S_overlap=None,
    )

    conduct[0, ie_g] += wk[ik] * np.sum(cond_aux)
    conduct_k[0, ik, ie_g] += wk[ik] * np.sum(cond_aux)

    if data.symmetry.do_eigenchannels:
        nchan = min(conduct.shape[0] - 1, cond_aux.shape[0])
        conduct[1 : 1 + nchan, ie_g] += wk[ik] * cond_aux[:nchan]
        conduct_k[1 : 1 + nchan, ik, ie_g] += wk[ik] * cond_aux[:nchan]

    if do_eigplot_now and z_eigplot is not None and rank == 0:
        write_eigenchannels(
            data=z_eigplot,
            ie=ie_g,
            ik=ik,
            vkpt=vkpt[:, ik],
            transport_direction=data.transport_direction,
            output_dir=Path('output/eigenchannels'),
            prefix='eigchn',
            overwrite=True,
            verbose=True,
        )
