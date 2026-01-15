import numpy as np
from typing import Optional


class Workspace:
    """
    Temporary workspace arrays used in quantum transport calculations.

    It includes arrays for storing static and real-space self-energies, Green's functions, and
    auxiliary working matrices. Allocation is conditional on runtime flags for whether
    to output lead self-energies or Green's functions.

    Attributes
    ----------
    `allocated` : bool
        Whether workspace arrays are currently allocated.
    """

    def __init__(self):
        self.allocated = False

        self.tsum: Optional[np.ndarray] = None
        self.tsumt: Optional[np.ndarray] = None
        self.work: Optional[np.ndarray] = None

        self.gamma_R: Optional[np.ndarray] = None
        self.gamma_L: Optional[np.ndarray] = None

        self.sgm_L: Optional[np.ndarray] = None
        self.sgm_R: Optional[np.ndarray] = None
        self.rsgm_L: Optional[np.ndarray] = None
        self.rsgm_R: Optional[np.ndarray] = None

        self.gL: Optional[np.ndarray] = None
        self.gR: Optional[np.ndarray] = None
        self.gC: Optional[np.ndarray] = None

        self.rgC: Optional[np.ndarray] = None
        self.kgC: Optional[np.ndarray] = None

    def allocate(
        self,
        dimL: int,
        dimC: int,
        dimR: int,
        dimx_lead: int,
        nkpts_par: int,
        nrtot_par: int,
        write_lead_sgm: bool,
        write_gf: bool,
    ) -> None:
        """
        Allocate workspace arrays based on problem dimensions and I/O flags.

        Parameters
        ----------
        dimL : int
            Left lead dimension.
        dimC : int
            Conductor dimension.
        dimR : int
            Right lead dimension.
        dimx_lead : int
            Max of dimL and dimR (used for tsum, tsumt).
        nkpts_par : int
            Number of parallel k-points.
        nrtot_par : int
            Number of parallel real-space vectors.
        write_lead_sgm : bool
            Whether to allocate self-energy storage arrays.
        write_gf : bool
            Whether to allocate Green's function storage arrays.
        """
        if self.allocated:
            raise RuntimeError("Workspace already allocated")

        self.tsum = np.zeros((dimx_lead, dimx_lead), dtype=np.complex128)
        self.tsumt = np.zeros((dimx_lead, dimx_lead), dtype=np.complex128)

        self.work = np.zeros((dimx_lead, dimx_lead), dtype=np.complex128)

        self.gamma_L = np.zeros((dimC, dimC), dtype=np.complex128)
        self.gamma_R = np.zeros((dimC, dimC), dtype=np.complex128)

        self.gL = np.zeros((dimL, dimL), dtype=np.complex128)
        self.gR = np.zeros((dimR, dimR), dtype=np.complex128)
        self.gC = np.zeros((dimC, dimC), dtype=np.complex128)

        if write_lead_sgm:
            self.sgm_L = np.zeros((dimC, dimC, nkpts_par), dtype=np.complex128)
            self.sgm_R = np.zeros((dimC, dimC, nkpts_par), dtype=np.complex128)
            self.rsgm_L = np.zeros((dimC, dimC, nrtot_par), dtype=np.complex128)
            self.rsgm_R = np.zeros((dimC, dimC, nrtot_par), dtype=np.complex128)

        if write_gf:
            self.rgC = np.zeros((dimC, dimC, nrtot_par), dtype=np.complex128)
            self.kgC = np.zeros((dimC, dimC, nkpts_par), dtype=np.complex128)

        self.allocated = True

    def deallocate(self) -> None:
        """
        Deallocate all workspace arrays if currently allocated.
        """
        if not self.allocated:
            return

        self.tsum = None
        self.tsumt = None
        self.work = None

        self.gamma_L = None
        self.gamma_R = None

        self.gL = None
        self.gR = None
        self.gC = None

        self.sgm_L = None
        self.sgm_R = None
        self.rsgm_L = None
        self.rsgm_R = None

        self.rgC = None
        self.kgC = None

        self.allocated = False

    def memusage(self) -> float:
        """
        Compute total memory usage of all currently allocated workspace arrays.

        Returns
        -------
        usage_mb : float
            Memory used in megabytes.
        """
        cost = 0
        for arr in [
            self.tsum,
            self.tsumt,
            self.work,
            self.gamma_L,
            self.gamma_R,
            self.gL,
            self.gR,
            self.gC,
            self.sgm_L,
            self.sgm_R,
            self.rsgm_L,
            self.rsgm_R,
            self.rgC,
            self.kgC,
        ]:
            if arr is not None:
                cost += arr.nbytes
        return cost / 1_000_000.0
