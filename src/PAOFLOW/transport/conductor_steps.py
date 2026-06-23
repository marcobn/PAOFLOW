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
from PAOFLOW.transport.utils.memusage import MemoryTracker
from PAOFLOW.transport.workspace.prepare_data import (
    prepare_conductor_data,
    prepare_conductor_runtime,
)


@dataclass
class ConductorStepState:
    """State container for staged conductor computations."""

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
    dimL: int,
    dimR: int,
    datafile_L: str,
    datafile_R: str,
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
    """Build ``ConductorData`` constructor inputs from direct arguments."""
    input_values: dict[str, Any] = {
        'datafile_C': datafile_C,
        'datafile_L': datafile_L,
        'datafile_R': datafile_R,
        'dimC': dimC,
        'dimL': dimL,
        'dimR': dimR,
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
    input_values.update(kwargs)
    return input_values


def prepare_conductor_step_state(
    *,
    data_controller: DataController,
    input_values: dict[str, Any],
) -> ConductorStepState:
    """Create staged conductor state with validated data and runtime metadata."""
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
    """Build Hamiltonian blocks for staged conductor computations."""
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
    """Compute lead self-energies for one ``(E, k)`` point."""
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
    """Compute conductor retarded Green's function for one k-point."""
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
    """Compute transmission channels from Green's function and self-energies."""
    g_ret = gC if gC is not None else state.last_gC
    sigma_left = sigma_L if sigma_L is not None else state.last_sigma_L
    sigma_right = sigma_R if sigma_R is not None else state.last_sigma_R
    if g_ret is None or sigma_left is None or sigma_right is None:
        raise RuntimeError(
            'gC, sigma_L, and sigma_R are required. Compute self-energy and Green function first.'
        )

    gamma_L = 1j * (sigma_left - sigma_left.conj().T)
    gamma_R = 1j * (sigma_right - sigma_right.conj().T)
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
    """Compute DOS contribution from a conductor Green's function."""
    g_ret = gC if gC is not None else state.last_gC
    if g_ret is None:
        raise RuntimeError('gC is required. Call compute_conductor_green(...) first or pass gC.')

    dos_value = -np.sum(np.imag(np.diagonal(g_ret))) / np.pi
    if weighted:
        if state.last_ik is None:
            raise RuntimeError('weighted=True requires a known k-point from previous calls.')
        dos_value *= state.data._runtime.wk_par[state.last_ik]
    return float(dos_value)
