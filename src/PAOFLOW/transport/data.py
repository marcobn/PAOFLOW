from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

CalculationType = Literal['conductor', 'bulk']
ConductFormula = Literal['landauer', 'generalized']
Carriers = Literal['electrons', 'phonons']
SmearingType = Literal[
    'lorentzian',
    'gaussian',
    'fermi-dirac',
    'fd',
    'methfessel-paxton',
    'mp',
    'marzari-vanderbilt',
    'mv',
]
FileFormat = Literal['internal', 'atmproj']

_FILE_NAMES_FIELDS: frozenset[str] = frozenset(
    {
        'work_dir',
        'output_dir',
        'prefix',
        'postfix',
        'datafile_L',
        'datafile_C',
        'datafile_R',
        'datafile_sgm',
        'datafile_L_sgm',
        'datafile_C_sgm',
        'datafile_R_sgm',
    }
)
_HAMILTONIAN_FIELDS: frozenset[str] = frozenset(
    {
        'H00_C',
        'H_CR',
        'H_CL',
        'H_LC',
        'H00_L',
        'H01_L',
        'H00_R',
        'H01_R',
    }
)
_KPOINT_FIELDS: frozenset[str] = frozenset({'nk', 's', 'nkpts_par', 'nrtot_par'})
_ENERGY_FIELDS: frozenset[str] = frozenset(
    {
        'emin',
        'emax',
        'ne',
        'ne_buffer',
        'delta',
        'smearing_type',
        'delta_ratio',
        'xmax',
        'energy_step',
        'nx_smear',
    }
)
_SYMMETRY_FIELDS: frozenset[str] = frozenset(
    {
        'use_sym',
        'write_kdata',
        'write_lead_sgm',
        'write_gf',
        'do_eigenchannels',
        'neigchnx',
        'do_eigplot',
        'ie_eigplot',
        'ik_eigplot',
    }
)
_ITERATION_FIELDS: frozenset[str] = frozenset({'nprint', 'niterx', 'nfailx', 'transfer_thr'})
_ATOMIC_PROJ_FIELDS: frozenset[str] = frozenset({'do_overlap_transformation', 'write_intermediate'})
_ADVANCED_FIELDS: frozenset[str] = frozenset(
    {
        'debug_level',
        'ispin',
        'surface',
        'efermi_bulk',
        'lhave_corr',
        'ldynam_corr',
        'leads_are_identical',
        'shifting_scheme',
    }
)
_TOP_LEVEL_FIELDS: frozenset[str] = frozenset(
    {
        'dimL',
        'dimR',
        'dimC',
        'transport_direction',
        'calculation_type',
        'conduct_formula',
        'carriers',
        'bias',
        'shift_L',
        'shift_C',
        'shift_R',
        'shift_corr',
    }
)


@dataclass(slots=True)
class FileNamesData:
    work_dir: str = './'
    output_dir: str = './'
    prefix: str = ''
    postfix: str = ''
    datafile_L: str = ''
    datafile_C: str = ''
    datafile_R: str = ''
    datafile_sgm: str = ''
    datafile_L_sgm: str = ''
    datafile_C_sgm: str = ''
    datafile_R_sgm: str = ''


@dataclass(slots=True)
class HamiltonianData:
    H00_C: dict[str, Any] | None = None
    H_CR: dict[str, Any] | None = None
    H_CL: dict[str, Any] | None = None
    H_LC: dict[str, Any] | None = None
    H00_L: dict[str, Any] | None = None
    H01_L: dict[str, Any] | None = None
    H00_R: dict[str, Any] | None = None
    H01_R: dict[str, Any] | None = None


@dataclass(slots=True)
class KPointGridSettings:
    nk: list[int] = field(default_factory=lambda: [0, 0])
    s: list[int] = field(default_factory=lambda: [0, 0])
    nkpts_par: int = 1
    nrtot_par: int = 1


@dataclass(slots=True)
class EnergySettings:
    emin: float = -10.0
    emax: float = 10.0
    ne: int = 1000
    ne_buffer: int = 1
    delta: float = 1.0e-5
    smearing_type: SmearingType = 'lorentzian'
    delta_ratio: float = 5.0e-3
    xmax: float = 25.0
    energy_step: float = 0.001
    nx_smear: int = 20000


@dataclass(slots=True)
class SymmetryOutputOptions:
    use_sym: bool = True
    write_kdata: bool = False
    write_lead_sgm: bool = False
    write_gf: bool = False
    do_eigenchannels: bool = False
    neigchnx: int = 200000
    do_eigplot: bool = False
    ie_eigplot: int = 0
    ik_eigplot: int = 0


@dataclass(slots=True)
class IterationConvergenceSettings:
    nprint: int = 20
    niterx: int = 200
    nfailx: int = 5
    transfer_thr: float = 1.0e-7


@dataclass(slots=True)
class AtomicProjectionOverlapSettings:
    do_overlap_transformation: bool = False
    write_intermediate: bool = True


@dataclass(slots=True)
class AdvancedSettings:
    debug_level: int = 0
    ispin: int = 0
    surface: bool = False
    efermi_bulk: float = 0.0
    lhave_corr: bool = False
    ldynam_corr: bool = False
    leads_are_identical: bool = True
    shifting_scheme: int = 1


@dataclass
class RuntimeData:
    nproc: int
    prefix: str
    work_dir: str
    nk_par: list[int]
    s_par: list[int]
    nk_par3d: np.ndarray
    s_par3d: np.ndarray
    nr_par3d: np.ndarray
    vkpt_par3D: np.ndarray
    wk_par: np.ndarray
    ivr_par3D: np.ndarray
    wr_par: np.ndarray
    nkpts_par: int
    nrtot_par: int


@dataclass
class ConductorData:
    file_names: FileNamesData
    hamiltonian: HamiltonianData
    kpoint_grid: KPointGridSettings
    energy: EnergySettings
    symmetry: SymmetryOutputOptions
    iteration: IterationConvergenceSettings
    atomic_proj: AtomicProjectionOverlapSettings
    advanced: AdvancedSettings
    dimL: int = 0
    dimR: int = 0
    dimC: int = 0
    transport_direction: int = 1
    calculation_type: CalculationType = 'conductor'
    conduct_formula: ConductFormula = 'landauer'
    carriers: Carriers = 'electrons'
    bias: float = 0.0
    shift_L: float = 0.0
    shift_C: float = 0.0
    shift_R: float = 0.0
    shift_corr: float = 0.0
    _runtime: RuntimeData | None = field(default=None, init=False, repr=False)

    def set_runtime_data(self, runtime: RuntimeData) -> None:
        self._runtime = runtime

    def get_runtime_data(self) -> RuntimeData:
        if self._runtime is None:
            raise RuntimeError('Runtime data has not been initialized.')
        return self._runtime

    @property
    def hamiltonian_tags(self) -> dict[str, dict[str, str]]:
        tag_dict: dict[str, dict[str, str]] = {}
        name_map = {
            'H00_C': 'block_00C',
            'H_CR': 'block_CR',
            'H_LC': 'block_LC',
            'H00_L': 'block_00L',
            'H01_L': 'block_01L',
            'H00_R': 'block_00R',
            'H01_R': 'block_01R',
        }
        for field_name, block_name in name_map.items():
            entry = getattr(self.hamiltonian, field_name)
            if entry is not None:
                tag_dict[block_name] = {
                    'rows': entry.get('rows', 'all'),
                    'cols': entry.get('cols', 'all'),
                    'rows_sgm': entry.get('rows_sgm', entry.get('rows', 'all')),
                    'cols_sgm': entry.get('cols_sgm', entry.get('cols', 'all')),
                }
        return tag_dict


@dataclass(slots=True)
class CurrentSettings:
    filein: str
    fileout: str
    Vmin: float
    Vmax: float
    nV: int
    sigma: float
    mu_L: float
    mu_R: float


def build_conductor_data(**kwargs: Any) -> ConductorData:
    """Build a ConductorData from a flat keyword argument mapping.

    Distributes flat kwargs into nested sub-dataclass fields. Raises
    ValueError for any unrecognised keyword arguments.
    """
    remaining = dict(kwargs)

    def extract(keys: frozenset[str]) -> dict[str, Any]:
        return {k: remaining.pop(k) for k in keys if k in remaining}

    file_names = FileNamesData(**extract(_FILE_NAMES_FIELDS))
    hamiltonian = HamiltonianData(**extract(_HAMILTONIAN_FIELDS))
    kpoint_grid = KPointGridSettings(**extract(_KPOINT_FIELDS))
    energy = EnergySettings(**extract(_ENERGY_FIELDS))
    symmetry = SymmetryOutputOptions(**extract(_SYMMETRY_FIELDS))
    iteration = IterationConvergenceSettings(**extract(_ITERATION_FIELDS))
    atomic_proj = AtomicProjectionOverlapSettings(**extract(_ATOMIC_PROJ_FIELDS))
    advanced = AdvancedSettings(**extract(_ADVANCED_FIELDS))
    top_level = extract(_TOP_LEVEL_FIELDS)

    if remaining:
        raise ValueError(f'Unrecognised conductor parameters: {sorted(remaining)}')

    return ConductorData(
        file_names=file_names,
        hamiltonian=hamiltonian,
        kpoint_grid=kpoint_grid,
        energy=energy,
        symmetry=symmetry,
        iteration=iteration,
        atomic_proj=atomic_proj,
        advanced=advanced,
        **top_level,
    )
