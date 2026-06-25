from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from PAOFLOW.DataController import DataController
from PAOFLOW.transport.calculators.green import compute_conductor_green_function
from PAOFLOW.transport.calculators.leads_self_energy import build_self_energies_from_blocks
from PAOFLOW.transport.calculators.transmittance import evaluate_transmittance
from PAOFLOW.transport.grid.egrid import initialize_energy_grid
from PAOFLOW.transport.hamiltonian.hamiltonian_setup import hamiltonian_setup
from PAOFLOW.transport.io.input_parameters import ConductorData
from PAOFLOW.transport.observables.broadening import compute_broadening_matrix
from PAOFLOW.transport.utils.memusage import MemoryTracker
from PAOFLOW.transport.workspace.prepare_data import (
    prepare_conductor_data,
    prepare_conductor_runtime,
)


@dataclass
class ConductorStepState:
    """State container for staged conductor computations.

    Attributes
    ----------
    data : ConductorData
        Validated conductor input model and runtime metadata.
    memory_tracker : MemoryTracker
        Memory profiler used by conductor preparation and setup stages.
    energy_grid : NDArray[np.float64]
        Energy grid in eV, shape ``(ne,)``.
    blc_blocks : dict[str, Any] or None, optional
        Hamiltonian and coupling block operators after ``build_conductor_blocks``.
    last_sigma_L : NDArray[np.complex128] or None, optional
        Last computed left lead self-energy for staged reuse.
    last_sigma_R : NDArray[np.complex128] or None, optional
        Last computed right lead self-energy for staged reuse.
    last_gC : NDArray[np.complex128] or None, optional
        Last computed conductor retarded Green's function for staged reuse.
    last_ik : int or None, optional
        k-point index associated with ``last_sigma_*`` and ``last_gC``.
    """

    data: ConductorData
    memory_tracker: MemoryTracker
    energy_grid: NDArray[np.float64]
    blc_blocks: dict[str, Any] | None = None
    last_sigma_L: NDArray[np.complex128] | None = None
    last_sigma_R: NDArray[np.complex128] | None = None
    last_gC: NDArray[np.complex128] | None = None
    last_ik: int | None = None


def build_conductor_input_values(
    *,
    datafile_C: str,
    dimC: int,
    dimL: int | None = None,
    dimR: int | None = None,
    datafile_L: str | None = None,
    datafile_R: str | None = None,
    emin: float,
    emax: float,
    ne: int,
    delta: float,
    nk: list[int] | tuple[int, int] = (0, 0),
    formula: str = 'landauer',
    transport_direction: int = 1,
    carriers: str = 'electrons',
    work_dir: str = './',
    output_dir: str = './',
    postfix: str = '',
    **kwargs: Any,
) -> dict[str, Any]:
    """Build ``ConductorData`` constructor inputs from direct arguments.

    Parameters
    ----------
    datafile_C : str
        Path to the conductor input file.
    dimC : int
        Conductor block dimension.
    dimL : int or None, optional
        Left lead block dimension for non-bulk calculations.
    dimR : int or None, optional
        Right lead block dimension for non-bulk calculations.
    datafile_L : str or None, optional
        Path to the left lead input file for non-bulk calculations.
    datafile_R : str or None, optional
        Path to the right lead input file for non-bulk calculations.
    emin : float
        Minimum energy in eV.
    emax : float
        Maximum energy in eV.
    ne : int
        Number of energy points.
    delta : float
        Broadening parameter.
    nk : list[int] or tuple[int, int], optional
        2D k-grid dimensions.
    formula : str, optional
        Conductance formula identifier.
    transport_direction : int, optional
        Transport direction index in ``{1, 2, 3}``.
    carriers : str, optional
        Carrier type (for example ``'electrons'`` or ``'phonons'``).
    work_dir : str, optional
        Working directory for transport assets.
    output_dir : str, optional
        Output directory for generated files.
    postfix : str, optional
        Output postfix appended to default file names.
    **kwargs : Any
        Additional optional ``ConductorData`` fields.

    Returns
    -------
    dict[str, Any]
        Keyword arguments ready for ``ConductorData(..., **input_values)``.
    """
    input_values: dict[str, Any] = {
        'datafile_C': datafile_C,
        'dimC': dimC,
        'emin': emin,
        'emax': emax,
        'ne': ne,
        'delta': delta,
        'nk': list(nk),
        'conduct_formula': formula,
        'transport_direction': transport_direction,
        'carriers': carriers,
        'work_dir': work_dir,
        'output_dir': output_dir,
        'postfix': postfix,
    }

    calculation_type = str(kwargs.get('calculation_type', '')).strip().lower()
    if calculation_type != 'bulk':
        input_values['dimL'] = 0 if dimL is None else dimL
        input_values['dimR'] = 0 if dimR is None else dimR
        input_values['datafile_L'] = '' if datafile_L is None else datafile_L
        input_values['datafile_R'] = '' if datafile_R is None else datafile_R

    input_values.update(kwargs)
    return input_values


def prepare_conductor_step_state(
    *,
    data_controller: DataController,
    input_values: dict[str, Any],
) -> ConductorStepState:
    """Create staged conductor state with validated data and runtime metadata.

    Parameters
    ----------
    data_controller : DataController
        Shared data controller used by transport preparation.
    input_values : dict[str, Any]
        Keyword arguments for ``ConductorData`` construction.

    Returns
    -------
    ConductorStepState
        Prepared state containing validated conductor input, initialized memory
        tracker, and the energy grid.

    Notes
    -----
    This routine mutates transport-related runtime containers through
    ``prepare_conductor_data`` as part of the existing preparation flow.
    """
    data = ConductorData(filename='<direct-arguments>', validate=True, **input_values)
    memory_tracker = MemoryTracker()
    prepare_conductor_data(data, data_controller)
    energy_grid = initialize_energy_grid(
        emin=data.energy.emin,
        emax=data.energy.emax,
        ne=data.energy.ne,
        carriers=data.carriers,
    )
    return ConductorStepState(data=data, memory_tracker=memory_tracker, energy_grid=energy_grid)


def build_conductor_blocks(
    *,
    state: ConductorStepState,
    data_controller: DataController,
) -> dict[str, Any]:
    """Build Hamiltonian blocks for staged conductor computations.

    Parameters
    ----------
    state : ConductorStepState
        Staged conductor state returned by ``prepare_conductor_step_state``.
    data_controller : DataController
        Shared data controller used by Hamiltonian setup routines.

    Returns
    -------
    dict[str, Any]
        Block-operator mapping used by conductor self-energy and Green-function
        calculations.

    Notes
    -----
    Updates ``state.blc_blocks`` with the returned operator dictionary.
    """
    hamiltonian_system = prepare_conductor_runtime(
        state.data, data_controller, state.memory_tracker
    )
    state.blc_blocks = hamiltonian_system.blocks
    return state.blc_blocks


def compute_conductor_self_energy(
    *,
    state: ConductorStepState,
    ie_g: int,
    ik: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], int]:
    """Compute lead self-energies for one ``(E, k)`` point.

    Parameters
    ----------
    state : ConductorStepState
        Staged conductor state with initialized block operators.
    ie_g : int
        Global energy index in ``state.energy_grid``.
    ik : int
        Local k-point index.

    Returns
    -------
    tuple[NDArray[np.complex128], NDArray[np.complex128], int]
        ``(sigma_L, sigma_R, total_iterations)``, where self-energies have
        shape ``(dimC, dimC)`` and ``total_iterations`` is the summed transfer
        matrix iteration count.

    Raises
    ------
    RuntimeError
        If Hamiltonian blocks were not built before calling this function.

    Notes
    -----
    Updates ``state.last_sigma_L``, ``state.last_sigma_R``, and ``state.last_ik``.
    """
    if state.blc_blocks is None:
        raise RuntimeError(
            'Hamiltonian blocks are unavailable. Call build_conductor_blocks(...) first.'
        )

    hamiltonian_setup(
        ik=ik,
        ie_g=ie_g,
        egrid=state.energy_grid,
        shift_L=state.data.shift_L,
        shift_C=state.data.shift_C,
        shift_R=state.data.shift_R,
        shift_C_corr=getattr(state.data, 'shift_corr', 0.0),
        blc_blocks=state.blc_blocks,
        ie_buff=1,
    )

    sigma_R, sigma_L, niter_R, niter_L = build_self_energies_from_blocks(
        blc_00R=state.blc_blocks['blc_00R'].at_k(ik),
        blc_01R=state.blc_blocks['blc_01R'].at_k(ik),
        blc_00L=state.blc_blocks['blc_00L'].at_k(ik),
        blc_01L=state.blc_blocks['blc_01L'].at_k(ik),
        blc_CR=state.blc_blocks['blc_CR'].at_k(ik),
        blc_LC=state.blc_blocks['blc_LC'].at_k(ik),
        leads_are_identical=state.data.advanced.leads_are_identical,
        delta=state.data.energy.delta,
        niterx=state.data.iteration.niterx,
        transfer_thr=state.data.iteration.transfer_thr,
        fail_counter=None,
        fail_limit=state.data.iteration.nfailx,
        verbose=False,
    )
    total_iterations = niter_R + (niter_L if not state.data.advanced.leads_are_identical else 0)
    state.last_sigma_L = sigma_L
    state.last_sigma_R = sigma_R
    state.last_ik = ik
    return sigma_L, sigma_R, total_iterations


def compute_conductor_green(
    *,
    state: ConductorStepState,
    ik: int,
    sigma_L: NDArray[np.complex128] | None = None,
    sigma_R: NDArray[np.complex128] | None = None,
) -> NDArray[np.complex128]:
    """Compute conductor retarded Green's function for one k-point.

    Parameters
    ----------
    state : ConductorStepState
        Staged conductor state with initialized block operators.
    ik : int
        Local k-point index.
    sigma_L : NDArray[np.complex128] or None, optional
        Left lead self-energy, shape ``(dimC, dimC)``. When ``None``, uses
        ``state.last_sigma_L``.
    sigma_R : NDArray[np.complex128] or None, optional
        Right lead self-energy, shape ``(dimC, dimC)``. When ``None``, uses
        ``state.last_sigma_R``.

    Returns
    -------
    NDArray[np.complex128]
        Retarded conductor Green's function, shape ``(dimC, dimC)``.

    Raises
    ------
    RuntimeError
        If Hamiltonian blocks are unavailable or ``sigma_L`` cannot be resolved.

    Notes
    -----
    Updates ``state.last_gC`` and ``state.last_ik``.
    """
    if state.blc_blocks is None:
        raise RuntimeError(
            'Hamiltonian blocks are unavailable. Call build_conductor_blocks(...) first.'
        )

    sigma_left = sigma_L if sigma_L is not None else state.last_sigma_L
    sigma_right = sigma_R if sigma_R is not None else state.last_sigma_R
    if sigma_left is None:
        raise RuntimeError(
            'sigma_L is required. Call compute_conductor_self_energy(...) first or pass sigma_L.'
        )

    g_c = compute_conductor_green_function(
        blc_00C=state.blc_blocks['blc_00C'].at_k(ik),
        sigma_l=sigma_left,
        sigma_r=sigma_right if not state.data.advanced.surface else None,
        delta=state.data.energy.delta,
        surface=state.data.advanced.surface,
    )
    state.last_gC = g_c
    state.last_ik = ik
    return g_c


def compute_conductor_transmission(
    *,
    state: ConductorStepState,
    gC: NDArray[np.complex128] | None = None,
    sigma_L: NDArray[np.complex128] | None = None,
    sigma_R: NDArray[np.complex128] | None = None,
    weighted: bool = False,
) -> NDArray[np.float64]:
    """Compute transmission channels from Green's function and self-energies.

    Parameters
    ----------
    state : ConductorStepState
        Staged conductor state containing the latest computed operators.
    gC : NDArray[np.complex128] or None, optional
        Retarded conductor Green's function, shape ``(dimC, dimC)``. When
        ``None``, uses ``state.last_gC``.
    sigma_L : NDArray[np.complex128] or None, optional
        Left lead self-energy, shape ``(dimC, dimC)``. When ``None``, uses
        ``state.last_sigma_L``.
    sigma_R : NDArray[np.complex128] or None, optional
        Right lead self-energy, shape ``(dimC, dimC)``. When ``None``, uses
        ``state.last_sigma_R``.
    weighted : bool, optional
        If ``True``, multiply channels by the k-point weight of ``state.last_ik``.

    Returns
    -------
    NDArray[np.float64]
        Total and optional eigenchannel transmission values.

    Raises
    ------
    RuntimeError
        If required operators are unavailable or ``weighted`` cannot resolve a
        previously used k-point index.
    """
    g_ret = gC if gC is not None else state.last_gC
    sigma_left = sigma_L if sigma_L is not None else state.last_sigma_L
    sigma_right = sigma_R if sigma_R is not None else state.last_sigma_R
    if g_ret is None or sigma_left is None or sigma_right is None:
        raise RuntimeError(
            'gC, sigma_L, and sigma_R are required. Compute self-energy and Green function first.'
        )

    gamma_L = compute_broadening_matrix(sigma_left)
    gamma_R = compute_broadening_matrix(sigma_right)
    channels, _ = evaluate_transmittance(
        gamma_L=gamma_L,
        gamma_R=gamma_R,
        G_ret=g_ret,
        formula=state.data.conduct_formula,
        do_eigenchannels=state.data.symmetry.do_eigenchannels,
        do_eigplot=False,
        sgm_corr=None,
        eta=state.data.energy.delta,
        S_overlap=None,
    )
    if weighted:
        if state.last_ik is None:
            raise RuntimeError('weighted=True requires a known k-point from previous calls.')
        return state.data._runtime.wk_par[state.last_ik] * channels
    return channels


def compute_conductor_dos(
    *,
    state: ConductorStepState,
    gC: NDArray[np.complex128] | None = None,
    weighted: bool = False,
) -> float:
    """Compute DOS contribution from a conductor Green's function.

    Parameters
    ----------
    state : ConductorStepState
        Staged conductor state containing the latest computed operators.
    gC : NDArray[np.complex128] or None, optional
        Retarded conductor Green's function, shape ``(dimC, dimC)``. When
        ``None``, uses ``state.last_gC``.
    weighted : bool, optional
        If ``True``, multiply DOS by the k-point weight of ``state.last_ik``.

    Returns
    -------
    float
        DOS contribution for the selected ``(E, k)`` state.

    Raises
    ------
    RuntimeError
        If ``gC`` is unavailable or ``weighted`` cannot resolve a previously
        used k-point index.
    """
    g_ret = gC if gC is not None else state.last_gC
    if g_ret is None:
        raise RuntimeError('gC is required. Call compute_conductor_green(...) first or pass gC.')

    dos_value = -np.sum(np.imag(np.diagonal(g_ret))) / np.pi
    if weighted:
        if state.last_ik is None:
            raise RuntimeError('weighted=True requires a known k-point from previous calls.')
        dos_value *= state.data._runtime.wk_par[state.last_ik]
    return float(dos_value)
