from __future__ import annotations

from typing import Literal

import numpy as np

from transportPAO.grid.kpoints import compute_ivr_par, kpoints_mask


class OperatorBlockView:
    def __init__(self, parent: OperatorBlock, ik: int):
        self.ik = ik
        self.parent = parent

        self.H = parent.H[..., ik] if parent.H is not None else None
        self.S = parent.S[..., ik] if parent.S is not None else None
        self.aux = parent.aux[..., ik] if parent.aux is not None else None
        self.sgm = parent.sgm[..., ik, :] if parent.sgm is not None else None
        self.name = parent.name
        self.dim1 = parent.dim1
        self.dim2 = parent.dim2


class OperatorBlock:
    """
    Container for block-structured matrix data used in quantum transport calculations.

    This class manages related matrix data and metadata associated with a specific Hamiltonian
    or overlap block, including:

    - Hamiltonian matrices ``H`` and overlap matrices ``S`` defined over k-points
    - Optional self-energy data ``sgm`` and auxiliary workspaces ``aux``, ``sgm_aux``
    - Dimensions, indexing information, and optional lattice vector metadata
    - Flags that control which types of data are active

    It supports explicit allocation and cleanup of memory for these arrays, based on the
    simulation's dimensionality and configuration.

    Parameters
    ----------
    `name` : str
        Descriptive name of the operator block.
    """

    def __init__(self, name: str = ""):
        self.name: str = name
        self.dim1: int = 0
        self.dim2: int = 0
        self.nkpts: int = 0
        self.ne_sgm: int = 1

        self.nrtot: int = 0
        self.nrtot_sgm: int = 0

        self.iunit_sgm: int = -1
        self.iunit_sgm_opened: bool = False

        self.ie: int = 0
        self.ie_buff: int = 0
        self.ik: int = 0
        self.dimx_sgm: int = 0
        self.tag: dict[str, str] = {}

        self.H: np.ndarray | None = None
        self.S: np.ndarray | None = None
        self.sgm: np.ndarray | None = None
        self.aux: np.ndarray | None = None
        self.sgm_aux: np.ndarray | None = None

        self.icols: np.ndarray | None = None
        self.irows: np.ndarray | None = None
        self.icols_sgm: np.ndarray | None = None
        self.irows_sgm: np.ndarray | None = None

        self.ivr: np.ndarray | None = None
        self.ivr_sgm: np.ndarray | None = None

        self.lhave_aux = False
        self.lhave_sgm_aux = False
        self.lhave_ham = False
        self.lhave_ovp = False
        self.lhave_corr = False
        self.ldynam_corr = False

        self.allocated: bool = False

    def allocate(
        self,
        dim1: int,
        dim2: int,
        nkpts: int,
        ne_sgm: int = 1,
        nrtot: int = 0,
        nrtot_sgm: int = 0,
        lhave_aux: bool = True,
        lhave_sgm_aux: bool = False,
        lhave_ham: bool = True,
        lhave_ovp: bool = True,
        lhave_corr: bool = False,
    ):
        """
        Allocate memory for matrix blocks and auxiliary data.

        Parameters
        ----------
        `dim1` : int
            Number of rows in the block.
        `dim2` : int
            Number of columns in the block.
        `nkpts` : int
            Number of k-points.
        `ne_sgm` : int
            Number of energy grid points for self-energy.
        `nrtot` : int
            Number of real-space lattice vectors for H and S.
        `nrtot_sgm` : int
            Number of real-space lattice vectors for Sigma.
        `lhave_aux` : bool
            Whether to allocate auxiliary matrix.
        `lhave_sgm_aux` : bool
            Whether to allocate auxiliary Sigma matrix.
        `lhave_ham` : bool
            Whether to allocate the Hamiltonian matrix H.
        `lhave_ovp` : bool
            Whether to allocate the overlap matrix S.
        `lhave_corr` : bool
            Whether to allocate correlation self-energy Sigma.
        """
        if self.allocated:
            raise RuntimeError(f"Block '{self.name}' already allocated.")

        self.dim1 = dim1
        self.dim2 = dim2
        self.nkpts = nkpts
        self.ne_sgm = ne_sgm
        self.nrtot = nrtot
        self.nrtot_sgm = nrtot_sgm

        self.icols = np.zeros(dim2, dtype=int)
        self.irows = np.zeros(dim1, dtype=int)
        self.icols_sgm = np.zeros(dim2, dtype=int)
        self.irows_sgm = np.zeros(dim1, dtype=int)

        if lhave_aux:
            self.aux = np.zeros((dim1, dim2, nkpts), dtype=np.complex128)
            self.lhave_aux = True
        if lhave_sgm_aux:
            self.sgm_aux = np.zeros((dim1, dim2), dtype=np.complex128)
            self.lhave_sgm_aux = True
        if lhave_ham:
            self.H = np.zeros((dim1, dim2, nkpts), dtype=np.complex128)
            self.lhave_ham = True
        if lhave_ovp:
            self.S = np.zeros((dim1, dim2, nkpts), dtype=np.complex128)
            self.lhave_ovp = True
        if lhave_corr:
            self.sgm = np.zeros((dim1, dim2, nkpts, ne_sgm), dtype=np.complex128)
            self.lhave_corr = True
        if nrtot > 0:
            self.ivr = np.zeros((3, nrtot), dtype=int)
        if nrtot_sgm > 0:
            self.ivr_sgm = np.zeros((3, nrtot_sgm), dtype=int)

        self.allocated = True

    def at_k(self, ik: int) -> OperatorBlockView:
        if not self.allocated:
            raise RuntimeError(f"Cannot slice unallocated block '{self.name}'")
        if ik >= self.nkpts:
            raise IndexError(f"k-point index {ik} out of range (nkpts = {self.nkpts})")
        return OperatorBlockView(self, ik)

    def deallocate(self) -> None:
        """
        Deallocate all internal arrays and reset the OperatorBlock state.

        This releases memory and resets allocation flags, mirroring the behavior
        of the Fortran `operator_blc_deallocate` routine.
        """
        self.irows = None
        self.icols = None
        self.irows_sgm = None
        self.icols_sgm = None
        self.aux = None
        self.sgm_aux = None
        self.H = None
        self.S = None
        self.sgm = None
        self.ivr = None
        self.ivr_sgm = None

        self.iunit_sgm_opened = False
        self.allocated = False

        self.tag = {}

    def clear(self):
        """Release all allocated memory for this block."""
        self.__init__(name=self.name)

    def update(
        self,
        nrtot: int | None = None,
        nrtot_sgm: int | None = None,
        ie: int | None = None,
        ie_buff: int | None = None,
        ik: int | None = None,
        ldynam_corr: bool | None = None,
        tag: str | None = None,
        name: str | None = None,
    ):
        """
        Update metadata associated with the operator block.

        Parameters
        ----------
        `nrtot` : int, optional
        `nrtot_sgm` : int, optional
        `ie` : int, optional
        `ie_buff` : int, optional
        `ik` : int, optional
        `ldynam_corr` : bool, optional
        `tag` : str, optional
        `name` : str, optional
        """
        if not self.allocated:
            raise RuntimeError(f"Cannot update unallocated block '{self.name}'.")

        if nrtot is not None:
            self.nrtot = nrtot
        if nrtot_sgm is not None:
            self.nrtot_sgm = nrtot_sgm
        if ie is not None:
            self.ie = ie
        if ie_buff is not None:
            self.ie_buff = ie_buff
        if ik is not None:
            self.ik = ik
        if ldynam_corr is not None:
            self.ldynam_corr = ldynam_corr
        if tag is not None:
            self.tag = tag
        if name is not None:
            self.name = name

    def copy_from(self, other: OperatorBlock) -> None:
        """
        Copy all data and metadata from another OperatorBlock instance.

        Parameters
        ----------
        `other` : OperatorBlock
            The block to copy from.
        """
        if not other.allocated:
            raise ValueError("Source block is not allocated.")

        self.clear()
        self.allocate(
            dim1=other.dim1,
            dim2=other.dim2,
            nkpts=other.nkpts,
            ne_sgm=other.ne_sgm,
            nrtot=other.nrtot,
            nrtot_sgm=other.nrtot_sgm,
            lhave_aux=other.lhave_aux,
            lhave_sgm_aux=other.lhave_sgm_aux,
            lhave_ham=other.lhave_ham,
            lhave_ovp=other.lhave_ovp,
            lhave_corr=other.lhave_corr,
        )

        self.ie = other.ie
        self.ie_buff = other.ie_buff
        self.ik = other.ik
        self.tag = str(other.tag)
        self.ldynam_corr = other.ldynam_corr
        self.iunit_sgm = other.iunit_sgm
        self.iunit_sgm_opened = other.iunit_sgm_opened
        self.dimx_sgm = other.dimx_sgm

        self.irows = other.irows.copy() if other.irows is not None else None
        self.icols = other.icols.copy() if other.icols is not None else None
        self.irows_sgm = other.irows_sgm.copy() if other.irows_sgm is not None else None
        self.icols_sgm = other.icols_sgm.copy() if other.icols_sgm is not None else None
        self.ivr = other.ivr.copy() if other.ivr is not None else None
        self.ivr_sgm = other.ivr_sgm.copy() if other.ivr_sgm is not None else None

        if self.H is not None:
            self.H[:] = other.H
        if self.S is not None:
            self.S[:] = other.S
        if self.sgm is not None:
            self.sgm[:] = other.sgm
        if self.aux is not None:
            self.aux[:] = other.aux
        if self.sgm_aux is not None:
            self.sgm_aux[:] = other.sgm_aux

    def copy(self) -> OperatorBlock:
        new = OperatorBlock(self.name)
        new.copy_from(self)
        return new

    def set_ivr_par_from_nr(
        self,
        nr_par: tuple[int, int],
        transport_direction: int,
    ) -> None:
        """
        Set the `ivr_par` attribute from a 2D R-vector mesh orthogonal to the transport direction.

        Parameters
        ----------
        `nr_par` : tuple of int
            Mesh dimensions (n1, n2) orthogonal to the transport axis.
        `transport_direction` : int
            Direction of transport (1=x, 2=y, 3=z).
        """
        ivr_par_2d, _ = compute_ivr_par(nr_par)
        self.ivr_par = ivr_par_2d

        # Compute full 3D ivr (needed by read_matrix)
        ivr_par3D = np.array(
            [
                kpoints_mask(ivr_par_2d[:, i], 0, transport_direction)
                for i in range(ivr_par_2d.shape[1])
            ]
        ).T
        self.ivr = ivr_par3D

    def memusage(self, memtype: Literal["ham", "corr", "all"] = "all") -> float:
        """
        Estimate memory usage in MB for the specified memory type.

        Parameters
        ----------
        `memtype` : {"ham", "corr", "all"}
            Type of memory usage to compute.

        Returns
        -------
        `usage_mb` : float
            Estimated memory usage in megabytes.
        """
        memtype = memtype.lower()
        cost = 0.0

        if memtype not in {"ham", "corr", "all"}:
            raise ValueError(f"Invalid memtype: {memtype}")

        if memtype in {"ham", "all"}:
            if self.icols is not None:
                cost += self.icols.nbytes
            if self.irows is not None:
                cost += self.irows.nbytes
            if self.ivr is not None:
                cost += self.ivr.nbytes
            if self.H is not None:
                cost += self.H.nbytes
            if self.S is not None:
                cost += self.S.nbytes
            if self.aux is not None:
                cost += self.aux.nbytes

        if memtype in {"corr", "all"}:
            if self.icols_sgm is not None:
                cost += self.icols_sgm.nbytes
            if self.irows_sgm is not None:
                cost += self.irows_sgm.nbytes
            if self.ivr_sgm is not None:
                cost += self.ivr_sgm.nbytes
            if self.sgm is not None:
                cost += self.sgm.nbytes
            if self.sgm_aux is not None:
                cost += self.sgm_aux.nbytes

        return cost / 1e6

    def summary(self) -> str:
        """
        Return a human-readable summary of the block's shape and configuration.

        Returns
        -------
        `summary` : str
            Multi-line string describing the operator block state.
        """
        return (
            f"\n  OperatorBlock Summary: {self.name}\n"
            f"    dim1, dim2           : {self.dim1}, {self.dim2}\n"
            f"    nkpts                : {self.nkpts}\n"
            f"    dimx_sgm             : {self.dimx_sgm}\n"
            f"    nrtot, nrtot_sgm     : {self.nrtot}, {self.nrtot_sgm}\n"
            f"    ne_sgm               : {self.ne_sgm}\n"
            f"    have_aux             : {self.lhave_aux}\n"
            f"    have_sgm_aux         : {self.lhave_sgm_aux}\n"
            f"    have_ham             : {self.lhave_ham}\n"
            f"    have_ovp             : {self.lhave_ovp}\n"
            f"    have_corr            : {self.lhave_corr}\n"
            f"    dyn_corr             : {self.ldynam_corr}\n"
            f"    energy index (ie)    : {self.ie}\n"
            f"    energy buffer index  : {self.ie_buff}\n"
            f"    k-point index        : {self.ik}\n"
            f"    tag                  : {self.tag}\n"
        )
