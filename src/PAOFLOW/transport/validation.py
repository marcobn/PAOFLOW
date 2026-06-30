from __future__ import annotations

import numpy as np

from PAOFLOW.transport.data import (
    AdvancedSettings,
    ConductorData,
    CurrentSettings,
    EnergySettings,
    FileNamesData,
    HamiltonianData,
    IterationConvergenceSettings,
    KPointGridSettings,
    SymmetryOutputOptions,
)
from PAOFLOW.utils.constants import AMCONV, RYDCM1


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f'{name} must be a positive integer.')


def _require_non_negative_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0.0:
        raise ValueError(f'{name} must be non-negative.')


def _require_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer.')


def validate_file_names(file_names: FileNamesData) -> None:
    if not file_names.datafile_C:
        raise ValueError('datafile_C must be specified.')


def validate_hamiltonian_selectors(hamiltonian: HamiltonianData) -> None:
    pass


def validate_kpoint_grid(grid: KPointGridSettings) -> None:
    for i, n in enumerate(grid.nk):
        _require_non_negative_int(f'kpoint_grid.nk[{i}]', n)
    for i, sv in enumerate(grid.s):
        if sv not in (0, 1):
            raise ValueError(f'kpoint_grid.s[{i}] must be 0 or 1.')
    _require_non_negative_int('kpoint_grid.nkpts_par', grid.nkpts_par)
    _require_non_negative_int('kpoint_grid.nrtot_par', grid.nrtot_par)


def validate_energy_settings(energy: EnergySettings, *, carriers: str = 'electrons') -> None:
    if energy.emax <= energy.emin:
        raise ValueError('energy.emax must be greater than energy.emin.')
    if energy.ne <= 1:
        raise ValueError('energy.ne must be greater than 1.')
    _require_positive_int('energy.ne_buffer', energy.ne_buffer)
    if energy.delta < 0.0:
        raise ValueError('energy.delta must be non-negative.')
    # Physics constraint: very large delta causes unphysical broadening
    if energy.delta > 0.3:
        raise ValueError('energy.delta must be <= 0.3 (physics constraint).')
    if energy.delta_ratio < 0.0:
        raise ValueError('energy.delta_ratio must be non-negative.')
    # Physics constraint
    if energy.delta_ratio > 0.1:
        raise ValueError('energy.delta_ratio must be <= 0.1 (physics constraint).')
    if energy.xmax < 10.0:
        raise ValueError('energy.xmax must be >= 10.')
    _require_non_negative_float('energy.energy_step', energy.energy_step)
    _require_non_negative_int('energy.nx_smear', energy.nx_smear)


def validate_symmetry_output_options(symmetry: SymmetryOutputOptions) -> None:
    if symmetry.ie_eigplot > 0 and not symmetry.do_eigplot:
        raise ValueError('symmetry.ie_eigplot > 0 requires symmetry.do_eigplot=True.')
    if symmetry.ik_eigplot > 0 and not symmetry.do_eigplot:
        raise ValueError('symmetry.ik_eigplot > 0 requires symmetry.do_eigplot=True.')
    if symmetry.do_eigplot and not symmetry.do_eigenchannels:
        raise ValueError('symmetry.do_eigplot=True requires symmetry.do_eigenchannels=True.')
    if symmetry.write_lead_sgm and symmetry.use_sym:
        raise ValueError('symmetry.write_lead_sgm=True is incompatible with symmetry.use_sym=True.')
    if symmetry.write_gf and symmetry.use_sym:
        raise ValueError('symmetry.write_gf=True is incompatible with symmetry.use_sym=True.')


def validate_iteration_settings(iteration: IterationConvergenceSettings) -> None:
    _require_positive_int('iteration.nprint', iteration.nprint)
    _require_positive_int('iteration.niterx', iteration.niterx)
    _require_positive_int('iteration.nfailx', iteration.nfailx)
    if iteration.transfer_thr <= 0.0:
        raise ValueError('iteration.transfer_thr must be positive.')


def validate_advanced_settings(advanced: AdvancedSettings) -> None:
    if advanced.ispin not in (0, 1, 2):
        raise ValueError('advanced.ispin must be 0, 1, or 2.')
    if advanced.efermi_bulk < 0.0:
        raise ValueError('advanced.efermi_bulk must be non-negative.')


def validate_conductor_data(data: ConductorData) -> None:
    """Validate all conductor input fields before numerical work begins.

    Raises ValueError or RuntimeError with a message naming the bad field
    and the violated rule.
    """
    validate_file_names(data.file_names)
    validate_hamiltonian_selectors(data.hamiltonian)
    validate_kpoint_grid(data.kpoint_grid)
    validate_energy_settings(data.energy, carriers=data.carriers)
    validate_symmetry_output_options(data.symmetry)
    validate_iteration_settings(data.iteration)
    validate_advanced_settings(data.advanced)

    if data.dimC <= 0:
        raise ValueError('dimC must be positive.')
    if data.transport_direction not in (1, 2, 3):
        raise ValueError('transport_direction must be 1, 2, or 3.')
    if data.calculation_type not in ('bulk', 'conductor'):
        raise ValueError("calculation_type must be 'bulk' or 'conductor'.")
    if data.conduct_formula not in ('landauer', 'generalized'):
        raise ValueError("conduct_formula must be 'landauer' or 'generalized'.")
    if data.carriers not in ('electrons', 'phonons'):
        raise ValueError("carriers must be 'electrons' or 'phonons'.")

    if data.calculation_type == 'conductor':
        if data.dimL <= 0:
            raise ValueError('dimL must be positive when calculation_type="conductor".')
        if data.dimR <= 0:
            raise ValueError('dimR must be positive when calculation_type="conductor".')
        if not data.file_names.datafile_L:
            raise ValueError('datafile_L must be specified when calculation_type="conductor".')
        if not data.file_names.datafile_R:
            raise ValueError('datafile_R must be specified when calculation_type="conductor".')

    if data.calculation_type == 'bulk':
        if data.file_names.datafile_L.strip():
            raise ValueError('datafile_L must not be specified when calculation_type="bulk".')
        if data.file_names.datafile_R.strip():
            raise ValueError('datafile_R must not be specified when calculation_type="bulk".')
        data.dimL = data.dimC
        data.dimR = data.dimC

    if (
        data.conduct_formula != 'landauer'
        and not data.file_names.datafile_sgm
        and not data.file_names.datafile_C_sgm
    ):
        raise ValueError('conduct_formula="generalized" requires datafile_sgm or datafile_C_sgm.')

    if data.carriers == 'phonons':
        scale = (RYDCM1 / np.sqrt(AMCONV)) ** 2
        data.energy.emin = data.energy.emin**2 / scale
        if data.energy.emin < 0.0:
            raise ValueError('energy.emin < 0.0 after phonon energy conversion.')
        data.energy.emax = data.energy.emax**2 / scale


def validate_current_settings(settings: CurrentSettings) -> None:
    if settings.Vmax <= settings.Vmin:
        raise ValueError('current.Vmax must be greater than current.Vmin.')
    if settings.nV <= 0:
        raise ValueError('current.nV must be positive.')
    if settings.sigma < 0.0:
        raise ValueError('current.sigma must be non-negative.')
