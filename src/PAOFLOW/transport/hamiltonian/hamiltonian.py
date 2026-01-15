from typing import Literal
import numpy as np
from PAOFLOW.transport.hamiltonian.operator_blc import OperatorBlock


class HamiltonianSystem:
    """
    Represents a transport system defined by block Hamiltonians, overlap matrices, and correlation self-energies.

    This class holds and manages all matrix blocks used in quantum transport setups including the left/right leads,
    the central conductor, and inter-region couplings. Each block is stored as an OperatorBlock instance.

    Methods are provided for memory allocation, deallocation, and estimating memory usage.

    Attributes
    ----------
    dimL : int
        Dimension of the left lead region.
    dimC : int
        Dimension of the central conductor region.
    dimR : int
        Dimension of the right lead region.
    nkpts_par : int
        Number of parallel k-points.
    shift_L : float
        Energy shift for the left region.
    shift_C : float
        Energy shift for the center region.
    shift_R : float
        Energy shift for the right region.
    shift_corr : float
        Energy shift for correlation self-energy.
    """

    def __init__(self, dimL: int, dimC: int, dimR: int, nkpts_par: int) -> None:
        self.dimL = dimL
        self.dimC = dimC
        self.dimR = dimR
        self.nkpts_par = nkpts_par

        self.shift_L = 0.0
        self.shift_C = 0.0
        self.shift_R = 0.0
        self.shift_corr = 0.0

        self.allocated = False
        self.dimx = max(dimL, dimC, dimR)
        self.dimx_lead = max(dimL, dimR)

        self.blc_00L = OperatorBlock("block_00L")
        self.blc_01L = OperatorBlock("block_01L")
        self.blc_00R = OperatorBlock("block_00R")
        self.blc_01R = OperatorBlock("block_01R")
        self.blc_00C = OperatorBlock("block_00C")
        self.blc_LC = OperatorBlock("block_LC")
        self.blc_CR = OperatorBlock("block_CR")

    def allocate(
        self,
        ivr_par: np.ndarray,
        tag_dict: dict[str, dict[str, str]],
    ) -> None:
        """
        Allocate memory for all matrix blocks and initialize their metadata.

        Parameters
        ----------
        ivr_par : np.ndarray
            2D array of shape (2, nrtot_par) specifying the parallel real-space R-vectors.
            This is used to match R-slices in the Hamiltonian during read-in.
        """
        if self.allocated:
            raise RuntimeError("Hamiltonian blocks already allocated.")
        if min(self.dimL, self.dimC, self.dimR, self.nkpts_par) <= 0:
            raise ValueError("Invalid dimensions for Hamiltonian allocation.")

        block_specs = [
            (self.blc_00L, self.dimL, self.dimL),
            (self.blc_01L, self.dimL, self.dimL),
            (self.blc_00R, self.dimR, self.dimR),
            (self.blc_01R, self.dimR, self.dimR),
            (self.blc_00C, self.dimC, self.dimC),
            (self.blc_LC, self.dimL, self.dimC),
            (self.blc_CR, self.dimC, self.dimR),
        ]

        for block, d1, d2 in block_specs:
            block.allocate(d1, d2, self.nkpts_par)
            block.tag = tag_dict.get(
                block.name,
                {
                    "rows": "all",
                    "cols": "all",
                    "rows_sgm": "all",
                    "cols_sgm": "all",
                },
            )
            block.ivr_par = ivr_par

        self.allocated = True

    def memusage(self, memtype: Literal["ham", "corr", "all"] = "all") -> float:
        """
        Estimate total memory usage of all blocks in MB.

        Parameters
        ----------
        memtype : {"ham", "corr", "all"}
            Type of memory to report.

        Returns
        -------
        usage_mb : float
            Estimated memory in megabytes.
        """
        usage = 0.0
        for block in self.blocks.values():
            if block.allocated:
                usage += block.memusage(memtype)
        return usage

    @property
    def blocks(self) -> dict[str, OperatorBlock]:
        return {
            "blc_00L": self.blc_00L,
            "blc_01L": self.blc_01L,
            "blc_00R": self.blc_00R,
            "blc_01R": self.blc_01R,
            "blc_00C": self.blc_00C,
            "blc_LC": self.blc_LC,
            "blc_CR": self.blc_CR,
        }
