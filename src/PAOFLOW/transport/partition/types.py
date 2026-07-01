from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AtomOrbitals:
    atom_index: int
    element: str
    position: np.ndarray
    first_orbital: int
    last_orbital: int

    @property
    def dimension(self) -> int:
        return self.last_orbital - self.first_orbital + 1


@dataclass(frozen=True)
class HamiltonianBlockPartition:
    dim_c: int
    dim_l: int | None
    dim_r: int | None
    transport_direction: str
    selectors: dict[str, dict[str, str]]
