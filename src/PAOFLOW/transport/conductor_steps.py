from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PAOFLOW.DataController import DataController
from PAOFLOW.transport.data import ConductorData, build_conductor_data
from PAOFLOW.transport.utils.memusage import MemoryTracker
from PAOFLOW.transport.validation import validate_conductor_data
from PAOFLOW.transport.workspace.prepare_data import (
    prepare_conductor_data,
    prepare_conductor_runtime,
)


@dataclass
class ConductorStepState:
    """State container for staged conductor block construction.

    Attributes
    ----------
    data : ConductorData
        Validated conductor input model and runtime metadata.
    memory_tracker : MemoryTracker
        Memory profiler used by conductor preparation and setup stages.
    blc_blocks : dict[str, Any] or None, optional
        Hamiltonian and coupling block operators after ``build_conductor_blocks``.
    """

    data: ConductorData
    memory_tracker: MemoryTracker
    blc_blocks: dict[str, Any] | None = None


def build_conductor_input_values(
    *,
    dimC: int,
    dimL: int | None = None,
    dimR: int | None = None,
    formula: str = 'landauer',
    transport_direction: int = 1,
    calculation_type: str = 'bulk',
    carriers: str = 'electrons',
    work_dir: str = './',
    output_dir: str = './',
    postfix: str = '',
    **kwargs: Any,
) -> dict[str, Any]:
    """Build ``ConductorData`` constructor inputs from direct arguments.

    Energy-grid parameters (``emin``, ``emax``, ``ne``, ``delta``, ``nk``) are
    not accepted here.  Pass them through ``kwargs`` when assembling inputs for a
    full-grid calculation, or rely on the defaults defined in ``EnergySettings``.

    Parameters
    ----------
    dimC : int
        Conductor block dimension.
    dimL : int or None, optional
        Left lead block dimension for non-bulk calculations.
    dimR : int or None, optional
        Right lead block dimension for non-bulk calculations.
    formula : str, optional
        Conductance formula identifier.
    transport_direction : int, optional
        Transport direction index in ``{1, 2, 3}``.
    calculation_type : str, optional
        Calculation mode: ``'bulk'`` or ``'conductor'``.
    carriers : str, optional
        Carrier type (for example ``'electrons'`` or ``'phonons'``).
    work_dir : str, optional
        Working directory for transport assets.
    output_dir : str, optional
        Output directory for generated files.
    postfix : str, optional
        Output postfix appended to default file names.
    **kwargs : Any
        Additional optional ``ConductorData`` fields, including energy-grid
        parameters when assembling a full calculation input.

    Returns
    -------
    dict[str, Any]
        Keyword arguments ready for ``build_conductor_data(**input_values)``.
    """
    input_values: dict[str, Any] = {
        'dimC': dimC,
        'conduct_formula': formula,
        'transport_direction': transport_direction,
        'calculation_type': calculation_type,
        'carriers': carriers,
        'work_dir': work_dir,
        'output_dir': output_dir,
        'postfix': postfix,
    }

    calculation_type_lower = calculation_type.strip().lower()
    if calculation_type_lower != 'bulk':
        input_values['dimL'] = 0 if dimL is None else dimL
        input_values['dimR'] = 0 if dimR is None else dimR

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
        Flat keyword arguments passed to ``build_conductor_data``.

    Returns
    -------
    ConductorStepState
        Prepared state containing validated conductor input and an initialized
        memory tracker.

    Notes
    -----
    This routine mutates transport-related runtime containers through
    ``prepare_conductor_data`` as part of the existing preparation flow.
    """
    data = build_conductor_data(**input_values)
    validate_conductor_data(data)
    memory_tracker = MemoryTracker()
    prepare_conductor_data(data, data_controller)
    return ConductorStepState(data=data, memory_tracker=memory_tracker)


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
