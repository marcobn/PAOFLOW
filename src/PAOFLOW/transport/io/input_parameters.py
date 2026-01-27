from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    PrivateAttr,
    confloat,
    conint,
    field_validator,
)
from pydantic import BaseModel as PydanticBaseModel
from typing_extensions import Annotated
from yaml import SafeLoader, load

from PAOFLOW.transport.utils.constants import amconv, rydcm1

CalculationType = Literal["conductor", "bulk"]
ConductFormula = Literal["landauer", "generalized"]
Carriers = Literal["electrons", "phonons"]
SmearingType = Literal[
    "lorentzian",
    "gaussian",
    "fermi-dirac",
    "fd",
    "methfessel-paxton",
    "mp",
    "marzari-vanderbilt",
    "mv",
]
FileFormat = Literal["internal", "crystal", "wannier90", "cp2k", "atmproj"]


class FileNamesData(PydanticBaseModel):
    work_dir: str = "./"
    output_dir: str = "./"
    prefix: str = ""
    postfix: str = ""
    datafile_L: str = ""
    datafile_C: str = ""
    datafile_R: str = ""
    datafile_sgm: str = ""
    datafile_L_sgm: str = ""
    datafile_C_sgm: str = ""
    datafile_R_sgm: str = ""

    @field_validator("datafile_C")
    @classmethod
    def check_datafile_C(cls, value: str) -> str:
        if len(value) == 0:
            raise ValueError("datafile_C unspecified")
        return value


class HamiltonianData(PydanticBaseModel):
    H00_C: dict[str, Any] | None = None
    H_CR: dict[str, Any] | None = None
    H_LC: dict[str, Any] | None = None
    H00_L: dict[str, Any] | None = None
    H01_L: dict[str, Any] | None = None
    H00_R: dict[str, Any] | None = None
    H01_R: dict[str, Any] | None = None


class KPointGridSettings(PydanticBaseModel):
    nk: list[NonNegativeInt] = [0, 0]
    s: list[NonNegativeInt] = [0, 0]
    nkpts_par: NonNegativeInt = 1
    nrtot_par: NonNegativeInt = 1

    @field_validator("s")
    @classmethod
    def check_s(cls, value: list[int]) -> list[int]:
        if any(v < 0 or v > 1 for v in value):
            raise ValueError("Invalid s: all values must be 0 or 1")
        return value


class EnergySettings(PydanticBaseModel):
    emin: float = -10.0
    emax: float = 10.0
    ne: Annotated[PositiveInt, conint(gt=1)] = 1000
    ne_buffer: Annotated[PositiveInt, conint(gt=0)] = 1
    delta: Annotated[NonNegativeFloat, confloat(ge=0.0, le=0.3)] = 1e-5
    smearing_type: SmearingType = "lorentzian"
    delta_ratio: Annotated[NonNegativeFloat, confloat(ge=0.0, le=0.1)] = 5.0e-3
    xmax: Annotated[NonNegativeFloat, confloat(ge=10)] = 25.0
    energy_step: NonNegativeFloat = 0.001
    nx_smear: NonNegativeInt = 20000

    @field_validator("emax")
    @classmethod
    def check_emax(cls, value: float, info) -> float:
        emin = info.data.get("emin", None)
        if emin is not None and value <= emin:
            raise ValueError("emax has to be greater than emin")
        return value


class SymmetryOutputOptions(PydanticBaseModel):
    use_sym: bool = True
    write_kdata: bool = False
    write_lead_sgm: bool = False
    write_gf: bool = False
    do_eigenchannels: bool = False
    neigchnx: NonNegativeInt = 200000
    do_eigplot: bool = False
    ie_eigplot: NonNegativeInt = 0
    ik_eigplot: NonNegativeInt = 0


class IterationConvergenceSettings(PydanticBaseModel):
    nprint: PositiveInt = 20
    niterx: PositiveInt = 200
    nfailx: PositiveInt = 5
    transfer_thr: Annotated[float, confloat(gt=0.0)] = 1e-7


class AtomicProjectionOverlapSettings(PydanticBaseModel):
    do_overlap_transformation: bool = False
    # TODO : check if the normalization option is still needed and is responsible for different results
    write_intermediate: bool = True


class AtomicProjData(PydanticBaseModel):
    """
    Parsed atomic projection data from Quantum ESPRESSO's `atomic_proj.xml`.

    Attributes
    ----------
    nbnds : int
        Number of bands.
    nkpnts : int
        Number of k-points.
    nspin : int
        Number of spin components (1 = non-magnetic, 2 = collinear magnetic).
    nawf : int
        Number of atomic wavefunctions (projectors).
    nelec : float
        Total number of electrons.
    efermi : float
        Fermi energy, in units specified by `energy_units`.
    energy_units : str
        Energy units reported in XML (e.g. 'eV', 'Ha', 'Ry').
    kpts : (3, nkpnts) ndarray of float
        K-point coordinates in crystal units.
    wk : (nkpnts,) ndarray of float
        K-point weights.
    eigvals : (nbnds, nkpnts, nspin) ndarray of float
        Band eigenvalues at each k-point and spin.
    proj : (nawf, nbnds, nkpnts, nspin) ndarray of complex
        Projection matrix elements ⟨atomic_wfc | Bloch_state⟩.
    overlap : (nawf, nawf, nkpnts, nspin) ndarray of complex, optional
        Overlap matrices S_ij(k) = ⟨atomic_wfc_i | atomic_wfc_j⟩ if present.
    """

    nbnds: int
    nkpnts: int
    nspin: int
    nawf: int
    nelec: float
    efermi: float
    energy_units: str

    kpts: npt.NDArray[np.float64]
    wk: npt.NDArray[np.float64]
    eigvals: npt.NDArray[np.float64]
    proj: npt.NDArray[np.complex128]
    overlap: npt.NDArray[np.complex128] | None = None

    vkpts_crystal: npt.NDArray[np.float64] | None = None
    vkpts_cartesian: npt.NDArray[np.float64] | None = None

    efermi_raw: float | None = None
    eigvals_raw: npt.NDArray[np.float64] | None = None

    class Config:
        arbitrary_types_allowed = True


class AdvancedSettings(PydanticBaseModel):
    debug_level: int = 0
    ispin: int = 0
    surface: bool = False
    efermi_bulk: NonNegativeFloat = 0.0
    lhave_corr: bool = False
    ldynam_corr: bool = False
    leads_are_identical: bool = True
    shifting_scheme: NonNegativeInt = 1

    @field_validator("ispin")
    @classmethod
    def check_ispin(cls, value: int) -> int:
        if value < 0 or value > 2:
            raise ValueError("Invalid ispin")
        return value


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


class ConductorData(PydanticBaseModel):
    file_names: FileNamesData
    hamiltonian: HamiltonianData
    kpoint_grid: KPointGridSettings
    energy: EnergySettings
    symmetry: SymmetryOutputOptions
    iteration: IterationConvergenceSettings
    atomic_proj: AtomicProjectionOverlapSettings
    advanced: AdvancedSettings
    dimL: NonNegativeInt = 0
    dimR: NonNegativeInt = 0
    dimC: NonNegativeInt = 0
    transport_direction: Annotated[int, conint(ge=1, le=3)] = 0
    calculation_type: CalculationType = "conductor"
    conduct_formula: ConductFormula = "landauer"
    carriers: Carriers = "electrons"

    bias: NonNegativeFloat = 0.0
    shift_L: NonNegativeFloat = 0.0
    shift_C: NonNegativeFloat = 0.0
    shift_R: NonNegativeFloat = 0.0
    shift_corr: NonNegativeFloat = 0.0

    _runtime: RuntimeData = PrivateAttr(default=None)

    def set_runtime_data(self, runtime: RuntimeData) -> None:
        self._runtime = runtime

    def get_runtime_data(self) -> RuntimeData:
        return self._runtime

    def __init__(self, filename: str, *, validate: bool = True, **data: Any) -> None:
        invalid_fields: dict[str, list[str]] = {}
        validated_data = {}

        def extract_block(cls, fieldname: str) -> dict[str, Any]:
            block = {}
            valid_keys = cls.model_fields.keys()
            for key in valid_keys:
                if key in data:
                    block[key] = data.pop(key)
            try:
                validated_data[fieldname] = cls(**block)
            except Exception as e:
                raise ValueError(f"Error validating block '{fieldname}': {e}") from e

        extract_block(FileNamesData, "file_names")
        extract_block(HamiltonianData, "hamiltonian")
        extract_block(KPointGridSettings, "kpoint_grid")
        extract_block(EnergySettings, "energy")
        extract_block(SymmetryOutputOptions, "symmetry")
        extract_block(IterationConvergenceSettings, "iteration")
        extract_block(AtomicProjectionOverlapSettings, "atomic_proj")
        extract_block(AdvancedSettings, "advanced")

        top_level_keys = {
            "dimL",
            "dimR",
            "dimC",
            "transport_direction",
            "calculation_type",
            "conduct_formula",
            "carriers",
            "ne",
            "ne_buffer",
            "bias",
            "shift_L",
            "shift_C",
            "shift_R",
            "shift_corr",
        }

        unknown_keys = set(data) - top_level_keys
        if unknown_keys:
            invalid_fields["input_conductor"] = sorted(unknown_keys)

        for key in top_level_keys:
            if key in data:
                validated_data[key] = data[key]

        if invalid_fields:
            message = "Invalid fields found in input YAML:\n"
            for section, keys in invalid_fields.items():
                message += f"  - {section}: {keys}\n"
            raise ValueError(message.strip())

        super().__init__(**validated_data)

        if validate:
            self.validate_input()

    def validate_input(self) -> None:
        if self.file_names.datafile_C is None:
            raise ValueError(f"Unable to find {self.file_names.datafile_C}")

        if self.symmetry.ie_eigplot > 0.0 and not self.symmetry.do_eigplot:
            raise ValueError("ie_eigplot needs do_eigplot")

        if self.symmetry.ik_eigplot > 0.0 and not self.symmetry.do_eigplot:
            raise ValueError("ik_eigplot needs do_eigplot")

        if self.energy.emax <= self.energy.emin:
            raise ValueError("emax has to be greater than emin")

        if self.calculation_type == "conductor":
            if self.dimL <= 0:
                raise ValueError("dimL needs to be positive")
            if self.dimR <= 0:
                raise ValueError("dimR needs to be positive")
            if len(self.file_names.datafile_L) == 0:
                raise ValueError("datafile_L unspecified")
            if len(self.file_names.datafile_R) == 0:
                raise ValueError("datafile_R unspecified")
            if not self.file_names.datafile_L:
                raise ValueError(f"Unable to find {self.file_names.datafile_L}")
            if not self.file_names.datafile_R:
                raise ValueError(f"Unable to find {self.file_names.datafile_R}")

        if self.calculation_type == "bulk":
            user_provided_fields = set(self.model_fields_set)
            if "dimL" in user_provided_fields or "dimR" in user_provided_fields:
                raise ValueError("dimL and dimR should not be set in bulk mode")
            self.dimL = self.dimC
            self.dimR = self.dimC

            if len(self.file_names.datafile_L.strip()) != 0:
                raise ValueError("datafile_L should not be specified in bulk mode")
            if len(self.file_names.datafile_R.strip()) != 0:
                raise ValueError("datafile_R should not be specified in bulk mode")

            self.dimL = self.dimC
            self.dimR = self.dimC

        if (
            self.conduct_formula != "landauer"
            and len(self.file_names.datafile_sgm) == 0
            and len(self.file_names.datafile_C_sgm) == 0
        ):
            raise ValueError("Invalid conduct formula")

        if self.symmetry.do_eigplot and not self.symmetry.do_eigenchannels:
            raise ValueError("do_eigplot needs do_eigenchannels")

        if self.symmetry.write_lead_sgm and self.symmetry.use_sym:
            raise ValueError("use_sym and write_lead_sgm not implemented")

        if self.symmetry.write_gf and self.symmetry.use_sym:
            raise ValueError("use_sym and write_gf not implemented")

        if self.carriers == "phonons":
            self.energy.emin = self.energy.emin**2 / (rydcm1 / np.sqrt(amconv)) ** 2
            if self.energy.emin < 0.0:
                raise ValueError("emin < 0.0, invalid emin")
            self.energy.emax = self.energy.emax**2 / (rydcm1 / np.sqrt(amconv)) ** 2

    @field_validator("transport_direction")
    @classmethod
    def check_transport_direction(cls, value: int) -> int:
        if value < 1 or value > 3:
            raise ValueError(
                "Invalid value for transport_direction. Allowed values are 1,2 or 3"
            )
        return value

    @field_validator("dimC")
    @classmethod
    def check_dimC(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("dimC needs to be positive")
        return value

    @property
    def hamiltonian_tags(self) -> dict[str, dict[str, str]]:
        """
        Extracts the tag dictionary for each Hamiltonian block
        from the `hamiltonian_data` section.

        Returns
        -------
        `tag_dict` : dict
            Dictionary mapping OperatorBlock names to tag dicts.
        """
        tag_dict = {}

        hdata = self.hamiltonian
        name_map = {
            "H00_C": "block_00C",
            "H_CR": "block_CR",
            "H_LC": "block_LC",
            "H00_L": "block_00L",
            "H01_L": "block_01L",
            "H00_R": "block_00R",
            "H01_R": "block_01R",
        }

        for yaml_name, block_name in name_map.items():
            entry = getattr(hdata, yaml_name)
            if entry is not None:
                tag_dict[block_name] = {
                    "rows": entry.get("rows", "all"),
                    "cols": entry.get("cols", "all"),
                    "rows_sgm": entry.get("rows_sgm", entry.get("rows", "all")),
                    "cols_sgm": entry.get("cols_sgm", entry.get("cols", "all")),
                }

        return tag_dict


class CurrentData(PydanticBaseModel):
    filein: str
    fileout: str
    Vmin: float
    Vmax: float
    nV: PositiveInt
    sigma: NonNegativeFloat
    mu_L: float
    mu_R: float

    def __init__(self, filename: str, *, validate: bool = True, **data: Any) -> None:
        input_dict = self.read(filename)
        data.update(input_dict.get("input", {}))
        super().__init__(**data)
        if validate:
            self.validate_input()

    def read(self, filename: str) -> dict[str, Any]:
        with open(Path(filename).absolute()) as f:
            return load(f, SafeLoader)

    def validate_input(self) -> None:
        if self.Vmax <= self.Vmin:
            raise ValueError("Vmax must be greater than Vmin")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
