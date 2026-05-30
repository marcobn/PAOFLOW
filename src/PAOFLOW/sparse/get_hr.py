from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy import sparse

SparseCooCache = tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]


@dataclass(frozen=True)
class SparseHRs:
    """Store the real-space Hamiltonian as sparse blocks on the FFT grid.

    Parameters
    ----------
    nawf : int
        Number of atomic-like basis functions, so each block has shape
        ``(nawf, nawf)``.
    nk1, nk2, nk3 : int
        Dimensions of the FFT real-space grid.
    nspin : int
        Number of spin channels.
    blocks : dict[tuple[int, int], scipy.sparse.csr_matrix]
        Sparse real-space blocks keyed by ``(ir, ispin)``, where ``ir`` is the
        flattened FFT-grid index.

    Notes
    -----
    Mathematically this object stores the matrices ``H(R)`` entering the Fourier
    relation

    ``H(k) = sum_R exp(-2 pi i k . R) H(R)``.

    Keeping ``H(R)`` blockwise and sparse allows later steps to evaluate only
    the pieces of the Fourier sum that are actually needed, rather than holding
    a dense six-dimensional tensor in memory.
    """

    nawf: int
    nk1: int
    nk2: int
    nk3: int
    nspin: int
    blocks: dict[tuple[int, int], sparse.csr_matrix]

    def to_dense(self) -> np.ndarray:
        """Materialize the dense real-space Hamiltonian tensor.

        Returns
        -------
        numpy.ndarray
            Dense tensor ``HRs(i, j, r1, r2, r3, s)`` with shape
            ``(nawf, nawf, nk1, nk2, nk3, nspin)``.

        Notes
        -----
        This is an explicit dense bridge. It is useful only for compatibility
        with parts of PAOFLOW that still require the legacy dense ``HRs``
        tensor, and it should be avoided in the main sparse workflow whenever
        possible.
        """
        nrtot = self.nk1 * self.nk2 * self.nk3
        dense_flat = np.zeros((self.nawf, self.nawf, nrtot, self.nspin), dtype=np.complex128)

        for (ir, ispin), block in self.blocks.items():
            dense_flat[:, :, ir, ispin] = block.toarray()

        return np.reshape(
            dense_flat,
            (self.nawf, self.nawf, self.nk1, self.nk2, self.nk3, self.nspin),
            order='C',
        )

    def compute_R_cart(self, a_vectors: np.ndarray) -> np.ndarray:
        """Return the FFT-grid lattice vectors in Cartesian coordinates.

        Parameters
        ----------
        a_vectors : numpy.ndarray
            Direct-lattice vectors with shape ``(3, 3)``.

        Returns
        -------
        numpy.ndarray
            Cartesian real-space vectors ``R`` with shape ``(nk1 * nk2 * nk3, 3)``.

        Notes
        -----
        These vectors define the phase factors entering the Fourier sums for
        ``H(k)``, ``dH/dk``, and higher derivatives. The vectors are returned in
        the same flattened FFT ordering used by the stored sparse blocks.
        """
        nrtot = self.nk1 * self.nk2 * self.nk3
        r_cart = np.zeros((nrtot, 3), dtype=float)

        for i in range(self.nk1):
            for j in range(self.nk2):
                for k in range(self.nk3):
                    ir = k + j * self.nk3 + i * self.nk2 * self.nk3

                    rx = float(i) / float(self.nk1)
                    ry = float(j) / float(self.nk2)
                    rz = float(k) / float(self.nk3)

                    if rx >= 0.5:
                        rx -= 1.0
                    if ry >= 0.5:
                        ry -= 1.0
                    if rz >= 0.5:
                        rz -= 1.0

                    rx -= int(rx)
                    ry -= int(ry)
                    rz -= int(rz)

                    r_cart[ir, :] = (
                        rx * self.nk1 * a_vectors[0, :]
                        + ry * self.nk2 * a_vectors[1, :]
                        + rz * self.nk3 * a_vectors[2, :]
                    )

        return r_cart

    @staticmethod
    def _normalise_kgrid(kgrid: np.ndarray) -> np.ndarray:
        """Return the k-point list in the standard ``(nkpnts, 3)`` form."""
        if kgrid.ndim != 2:
            raise ValueError('kgrid must be a rank-2 array')
        if kgrid.shape[0] == 3:
            return np.asarray(kgrid.T, dtype=float)
        if kgrid.shape[1] == 3:
            return np.asarray(kgrid, dtype=float)
        raise ValueError('kgrid must have one dimension of length 3')

    @staticmethod
    def _hermitianize_dense(matrix: np.ndarray) -> np.ndarray:
        """Return the Hermitian part of one matrix or a matrix batch."""
        return 0.5 * (matrix + np.swapaxes(np.conj(matrix), -1, -2))

    @staticmethod
    def _apply_threshold(
        matrix: sparse.csr_matrix,
        threshold: float,
    ) -> sparse.csr_matrix:
        """Remove matrix elements whose magnitude is below the chosen threshold."""
        matrix = matrix.tocsr(copy=True)
        matrix.sum_duplicates()

        if threshold > 0.0 and matrix.nnz > 0:
            matrix.data[np.abs(matrix.data) < threshold] = 0.0
            matrix.eliminate_zeros()

        return matrix

    def _assemble_weighted_block(
        self,
        weights: np.ndarray,
        ispin: int,
        threshold: float,
        coo_cache: SparseCooCache | None = None,
    ) -> sparse.csr_matrix:
        """Assemble a sparse weighted sum of the real-space Hamiltonian blocks.

        Parameters
        ----------
        weights : numpy.ndarray
            Complex weights defined on the real-space FFT grid, with shape
            ``(nk1 * nk2 * nk3,)``.
        ispin : int
            Spin-channel index.
        threshold : float
            Magnitude threshold applied after the weighted sum is assembled.
        coo_cache : tuple or None, optional
            Optional cached sparse COO payload for one spin channel returned by
            ``collect_spin_coo_cache``. When provided, repeated COO conversion
            and full-grid sparse block scanning are avoided.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse weighted block with shape ``(nawf, nawf)``.

        Notes
        -----
        This helper generalizes the Fourier assembly of ``H(k)`` by allowing an
        arbitrary weight on each real-space block. With suitable weights it can
        produce ``H(k)``, ``dH/dk``, or other Fourier-derived operators directly
        from ``H(R)`` without first stacking all real-space blocks into a dense
        array.
        """
        nrtot = self.nk1 * self.nk2 * self.nk3
        if weights.shape != (nrtot,):
            raise ValueError(f'weights must have shape ({nrtot},)')
        if ispin < 0 or ispin >= self.nspin:
            raise ValueError(f'ispin must be in [0, {self.nspin})')

        row_chunks: list[np.ndarray] = []
        col_chunks: list[np.ndarray] = []
        val_chunks: list[np.ndarray] = []

        if coo_cache is not None:
            active_ir, active_rows, active_cols, active_vals = coo_cache
            for idx, ir in enumerate(active_ir):
                weight = np.complex128(weights[int(ir)])
                if weight == 0.0:
                    continue
                row_chunks.append(active_rows[idx])
                col_chunks.append(active_cols[idx])
                val_chunks.append(weight * active_vals[idx])
        else:
            for ir in range(nrtot):
                block = self.blocks.get((ir, ispin))
                if block is None or block.nnz == 0:
                    continue

                weight = np.complex128(weights[ir])
                if weight == 0.0:
                    continue

                block_coo = block.tocoo(copy=False)
                row_chunks.append(np.asarray(block_coo.row, dtype=np.int32))
                col_chunks.append(np.asarray(block_coo.col, dtype=np.int32))
                val_chunks.append(np.asarray(weight * block_coo.data, dtype=np.complex128))

        if not val_chunks:
            return sparse.csr_matrix((self.nawf, self.nawf), dtype=np.complex128)

        rows = np.concatenate(row_chunks)
        cols = np.concatenate(col_chunks)
        vals = np.concatenate(val_chunks)

        weighted_block = sparse.coo_matrix(
            (
                vals,
                (
                    rows,
                    cols,
                ),
            ),
            shape=(self.nawf, self.nawf),
            dtype=np.complex128,
        ).tocsr()
        weighted_block.sum_duplicates()

        return self._apply_threshold(sparse.csr_matrix(weighted_block), threshold)

    def collect_spin_coo_cache(self, ispin: int) -> SparseCooCache:
        """Collect one-spin sparse COO payload for repeated weighted assembly.

        Parameters
        ----------
        ispin : int
            Spin-channel index.

        Returns
        -------
        tuple[numpy.ndarray, list[numpy.ndarray], list[numpy.ndarray], list[numpy.ndarray]]
            ``(active_ir, rows, cols, vals)`` where each list entry is one
            real-space block in COO triplet form. The payload remains sparse and
            can be reused across many k-points without dense materialization.
        """
        if ispin < 0 or ispin >= self.nspin:
            raise ValueError(f'ispin must be in [0, {self.nspin})')

        active_ir: list[int] = []
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []

        nrtot = self.nk1 * self.nk2 * self.nk3
        for ir in range(nrtot):
            block = self.blocks.get((ir, ispin))
            if block is None or block.nnz == 0:
                continue
            block_coo = block.tocoo(copy=False)
            active_ir.append(ir)
            rows.append(np.asarray(block_coo.row, dtype=np.int32))
            cols.append(np.asarray(block_coo.col, dtype=np.int32))
            vals.append(np.asarray(block_coo.data, dtype=np.complex128))

        return np.asarray(active_ir, dtype=np.int32), rows, cols, vals

    def _collect_spin_hr_blocks(
        self,
        ispin: int,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect the nonzero real-space blocks for one spin channel.

        Parameters
        ----------
        ispin : int
            Spin-channel index.
        threshold : float
            Magnitude threshold used when pruning very small matrix elements.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(active_ir, hr_dense)`` where ``active_ir`` contains the real-space
            grid points with nonzero blocks and ``hr_dense`` has shape
            ``(n_active_r, nawf, nawf)``.

        Notes
        -----
        Some streamed Fourier contractions are most efficient when the active
        sparse blocks are packed into a dense batch over the nonzero real-space
        points only. This helper provides exactly that compact batch without
        rebuilding the full dense ``HRs`` tensor.
        """
        active_ir: list[int] = []
        hr_dense_blocks: list[np.ndarray] = []

        nrtot = self.nk1 * self.nk2 * self.nk3
        for ir in range(nrtot):
            block = self.blocks.get((ir, ispin))
            if block is None or block.nnz == 0:
                continue

            block_dense = block.toarray()
            if threshold > 0.0:
                mask = np.abs(block_dense) >= threshold
                if not np.any(mask):
                    continue
                block_dense = np.where(mask, block_dense, 0.0)

            active_ir.append(ir)
            hr_dense_blocks.append(block_dense)

        if not active_ir:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, self.nawf, self.nawf), dtype=np.complex128),
            )

        return (
            np.asarray(active_ir, dtype=np.int32),
            np.asarray(hr_dense_blocks, dtype=np.complex128),
        )

    def iter_local_dHdk_batches(
        self,
        kgrid: np.ndarray,
        r_cart: np.ndarray,
        alat: float,
        dnm: np.ndarray,
        *,
        start_kpoint: int,
        stop_kpoint: int,
        directions: tuple[int, ...] | None = None,
        phase_sign: float = -1.0,
        hermitianize: bool = True,
        batch_size: int = 256,
    ) -> Iterator[tuple[int, int, int, np.ndarray]]:
        """Stream local first-derivative batches from the sparse real-space data.

        Parameters
        ----------
        kgrid : numpy.ndarray
            k-point list with shape ``(nkpnts, 3)`` or ``(3, nkpnts)``.
        r_cart : numpy.ndarray
            Real-space FFT vectors in Cartesian coordinates with shape
            ``(nk1 * nk2 * nk3, 3)``.
        alat : float
            Lattice scaling factor used in the derivative convention.
        dnm : numpy.ndarray
            Correction tensor ``Dnm`` with shape ``(nawf, nawf, 3)``.
        start_kpoint : int
            First global k-point index of the local slice.
        stop_kpoint : int
            First global k-point index after the local slice.
        directions : tuple[int, ...] | None, optional
            Cartesian derivative directions to evaluate. If omitted, all three
            directions are built.
        phase_sign : float, optional
            Sign used in the Fourier phase ``exp(phase_sign * 2 pi i k . R)``.
        hermitianize : bool, optional
            If ``True``, return the Hermitian part of the streamed matrices.
        batch_size : int, optional
            Number of local k-points processed together.

        Yields
        ------
        tuple[int, int, int, numpy.ndarray]
            ``(batch_start, batch_stop, ispin, dh_batch)`` where ``dh_batch``
            has shape ``(batch_stop - batch_start, 3, nawf, nawf)``.

        Notes
        -----
        The first derivative is built from the same Fourier representation as
        the Hamiltonian,

        ``dH/dk_l = sum_R (i alat R_l) exp(phase_sign * 2 pi i k . R) H(R) + i H(k) D_l``.

        The routine evaluates this only for a bounded batch of local k-points at
        a time. That keeps the fast vectorized contraction while avoiding the
        storage of a full dense derivative tensor over the entire local slice.
        """
        kpts = self._normalise_kgrid(kgrid)
        nrtot = self.nk1 * self.nk2 * self.nk3
        if r_cart.shape != (nrtot, 3):
            raise ValueError(f'r_cart must have shape ({nrtot}, 3)')
        if dnm.shape != (self.nawf, self.nawf, 3):
            raise ValueError(f'dnm must have shape ({self.nawf}, {self.nawf}, 3)')

        requested_directions = (
            (0, 1, 2)
            if directions is None
            else tuple(dict.fromkeys(int(direction) for direction in directions))
        )
        if not requested_directions:
            raise ValueError('directions must contain at least one Cartesian axis')
        invalid_directions = [
            direction for direction in requested_directions if direction < 0 or direction > 2
        ]
        if invalid_directions:
            raise ValueError(f'invalid Cartesian directions requested: {invalid_directions}')

        nkpnts = int(kpts.shape[0])
        if start_kpoint < 0 or start_kpoint > nkpnts:
            raise ValueError(f'start_kpoint {start_kpoint} outside valid range [0, {nkpnts}]')
        if stop_kpoint < start_kpoint or stop_kpoint > nkpnts:
            raise ValueError(
                f'stop_kpoint {stop_kpoint} outside valid range [{start_kpoint}, {nkpnts}]'
            )

        local_kpts = kpts[start_kpoint:stop_kpoint, :]
        local_nkpnts = int(local_kpts.shape[0])
        if local_nkpnts == 0:
            return

        for ispin in range(self.nspin):
            active_ir, hr_dense = self._collect_spin_hr_blocks(ispin, threshold=0.0)
            if active_ir.size == 0:
                for batch_start in range(0, local_nkpnts, batch_size):
                    batch_stop = min(batch_start + batch_size, local_nkpnts)
                    yield (
                        batch_start,
                        batch_stop,
                        ispin,
                        np.zeros(
                            (batch_stop - batch_start, 3, self.nawf, self.nawf),
                            dtype=np.complex128,
                        ),
                    )
                continue

            r_active = r_cart[active_ir, :]
            phase_prefactors = 1.0j * float(alat) * r_active

            for batch_start in range(0, local_nkpnts, batch_size):
                batch_stop = min(batch_start + batch_size, local_nkpnts)
                k_batch = local_kpts[batch_start:batch_stop, :]
                phase_batch = np.exp(phase_sign * 2.0j * np.pi * np.dot(k_batch, r_active.T))
                hk_batch = np.einsum('kr,rij->kij', phase_batch, hr_dense, optimize=True)
                if hermitianize:
                    hk_batch = self._hermitianize_dense(hk_batch)
                dh_batch = np.empty(
                    (batch_stop - batch_start, 3, self.nawf, self.nawf),
                    dtype=np.complex128,
                )
                dh_batch.fill(0.0)

                for l in requested_directions:
                    weighted_phase = phase_batch * phase_prefactors[:, l][None, :]
                    direction_batch = np.einsum(
                        'kr,rij->kij', weighted_phase, hr_dense, optimize=True
                    )
                    direction_batch += 1.0j * hk_batch * dnm[:, :, l][None, :, :]
                    if hermitianize:
                        direction_batch = self._hermitianize_dense(direction_batch)
                    dh_batch[:, l, :, :] = direction_batch

                yield batch_start, batch_stop, ispin, dh_batch

    def iter_local_d2Hdk2_batches(
        self,
        kgrid: np.ndarray,
        r_cart: np.ndarray,
        alat: float,
        *,
        start_kpoint: int,
        stop_kpoint: int,
        direction_pairs: tuple[tuple[int, int], ...],
        phase_sign: float = -1.0,
        hermitianize: bool = True,
        batch_size: int = 256,
    ) -> Iterator[tuple[int, int, int, tuple[tuple[int, int], ...], np.ndarray]]:
        """Stream local second-derivative batches from ``SparseHRs``.

        Parameters
        ----------
        kgrid : numpy.ndarray
            k-point list with shape ``(nkpnts, 3)`` or ``(3, nkpnts)``.
        r_cart : numpy.ndarray
            Real-space FFT vectors in Cartesian coordinates with shape
            ``(nk1 * nk2 * nk3, 3)``.
        alat : float
            Lattice scaling factor used in the second-derivative convention.
        start_kpoint : int
            First global k-point index of the local slice.
        stop_kpoint : int
            First global k-point index after the local slice.
        direction_pairs : tuple[tuple[int, int], ...]
            Cartesian direction pairs ``(l, lp)`` to evaluate.
        phase_sign : float, optional
            Sign used in the Fourier phase.
        hermitianize : bool, optional
            If ``True``, return the Hermitian part of the streamed matrices.
        batch_size : int, optional
            Number of local k-points processed together.

        Yields
        ------
        tuple[int, int, int, tuple[tuple[int, int], ...], numpy.ndarray]
            ``(batch_start, batch_stop, ispin, direction_pairs, d2h_batch)`` where
            ``d2h_batch`` has shape ``(batch_stop - batch_start, len(direction_pairs), nawf, nawf)``.

        Notes
        -----
        The second derivative follows from the same Fourier expansion,

        ``d^2H/dk_l dk_p = -alat^2 sum_R R_l R_p exp(phase_sign * 2 pi i k . R) H(R)``.

        This helper evaluates only the requested Cartesian pairs and only for a
        bounded local k-point batch, which is sufficient for the effective-mass
        calculation while avoiding dense global second-derivative storage.

        Parallelization strategy:
            Callers provide the local k-point window and receive only that local
            slice. No dense second-derivative gather is introduced here.
        """
        kpts = self._normalise_kgrid(kgrid)
        nrtot = self.nk1 * self.nk2 * self.nk3
        if r_cart.shape != (nrtot, 3):
            raise ValueError(f'r_cart must have shape ({nrtot}, 3)')

        nkpnts = int(kpts.shape[0])
        if start_kpoint < 0 or start_kpoint > nkpnts:
            raise ValueError(f'start_kpoint {start_kpoint} outside valid range [0, {nkpnts}]')
        if stop_kpoint < start_kpoint or stop_kpoint > nkpnts:
            raise ValueError(
                f'stop_kpoint {stop_kpoint} outside valid range [{start_kpoint}, {nkpnts}]'
            )
        if not direction_pairs:
            raise ValueError('direction_pairs must contain at least one Cartesian pair.')

        for direction_pair in direction_pairs:
            if len(direction_pair) != 2:
                raise ValueError('Each direction pair must contain exactly two indices.')
            if direction_pair[0] not in (0, 1, 2) or direction_pair[1] not in (0, 1, 2):
                raise ValueError('Cartesian derivative directions must be in {0, 1, 2}.')

        local_kpts = kpts[start_kpoint:stop_kpoint, :]
        local_nkpnts = int(local_kpts.shape[0])
        if local_nkpnts == 0:
            return

        pair_array = np.asarray(direction_pairs, dtype=int)

        for ispin in range(self.nspin):
            active_ir, hr_dense = self._collect_spin_hr_blocks(ispin, threshold=0.0)
            if active_ir.size == 0:
                for batch_start in range(0, local_nkpnts, batch_size):
                    batch_stop = min(batch_start + batch_size, local_nkpnts)
                    yield (
                        batch_start,
                        batch_stop,
                        ispin,
                        direction_pairs,
                        np.zeros(
                            (
                                batch_stop - batch_start,
                                len(direction_pairs),
                                self.nawf,
                                self.nawf,
                            ),
                            dtype=np.complex128,
                        ),
                    )
                continue

            r_active = r_cart[active_ir, :]
            pair_prefactors = -(float(alat) ** 2) * (
                r_active[:, pair_array[:, 0]] * r_active[:, pair_array[:, 1]]
            )

            for batch_start in range(0, local_nkpnts, batch_size):
                batch_stop = min(batch_start + batch_size, local_nkpnts)
                k_batch = local_kpts[batch_start:batch_stop, :]
                phase_batch = np.exp(phase_sign * 2.0j * np.pi * np.dot(k_batch, r_active.T))
                d2h_batch = np.empty(
                    (batch_stop - batch_start, len(direction_pairs), self.nawf, self.nawf),
                    dtype=np.complex128,
                )

                for pair_index in range(len(direction_pairs)):
                    weighted_phase = phase_batch * pair_prefactors[:, pair_index][None, :]
                    pair_batch = np.einsum('kr,rij->kij', weighted_phase, hr_dense, optimize=True)
                    if hermitianize:
                        pair_batch = self._hermitianize_dense(pair_batch)
                    d2h_batch[:, pair_index, :, :] = pair_batch

                yield batch_start, batch_stop, ispin, direction_pairs, d2h_batch

    def build_local_dHdk_blocks(
        self,
        kgrid: np.ndarray,
        r_cart: np.ndarray,
        alat: float,
        dnm: np.ndarray,
        *,
        start_kpoint: int,
        stop_kpoint: int,
        threshold: float = 0.0,
    ) -> dict[tuple[int, int], tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]]:
        """Build sparse first-derivative blocks for the local k-point window.

        Parameters
        ----------
        kgrid : numpy.ndarray
            k-point list with shape ``(nkpnts, 3)`` or ``(3, nkpnts)``.
        r_cart : numpy.ndarray
            Real-space FFT vectors in Cartesian coordinates with shape
            ``(nk1 * nk2 * nk3, 3)``.
        alat : float
            Lattice scaling factor used in the derivative convention.
        dnm : numpy.ndarray
            Correction tensor ``Dnm`` with shape ``(nawf, nawf, 3)``.
        start_kpoint : int
            First global k-point index of the local slice.
        stop_kpoint : int
            First global k-point index after the local slice.
        threshold : float, optional
            Magnitude threshold applied after each derivative block is built.

        Returns
        -------
        dict[tuple[int, int], tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]]
            Sparse derivative blocks keyed by ``(ik_local, ispin)``. Each value
            is the three-direction tuple ``(dH/dkx, dH/dky, dH/dkz)``.

        Notes
        -----
        This is the sparse-block analogue of the dense local derivative builder.
        The derivative is accumulated directly from ``H(R)`` into sparse matrix
        coordinates, and the same elementwise ``Dnm`` correction as in the dense
        path is then added. Because the result is stored block by block, the
        workflow avoids a dense local ``dHksp`` tensor entirely.

        Parallelization strategy:
        Callers provide the local contiguous k-point window and receive only the
        matching local derivative blocks. No dense global derivative tensor is
        introduced by this API.
        """
        kpts = self._normalise_kgrid(kgrid)
        nrtot = self.nk1 * self.nk2 * self.nk3
        if r_cart.shape != (nrtot, 3):
            raise ValueError(f'r_cart must have shape ({nrtot}, 3)')
        if dnm.shape != (self.nawf, self.nawf, 3):
            raise ValueError(f'dnm must have shape ({self.nawf}, {self.nawf}, 3)')

        nkpnts = int(kpts.shape[0])
        if start_kpoint < 0 or start_kpoint > nkpnts:
            raise ValueError(f'start_kpoint {start_kpoint} outside valid range [0, {nkpnts}]')
        if stop_kpoint < start_kpoint or stop_kpoint > nkpnts:
            raise ValueError(
                f'stop_kpoint {stop_kpoint} outside valid range [{start_kpoint}, {nkpnts}]'
            )

        local_kpts = kpts[start_kpoint:stop_kpoint, :]
        local_blocks: dict[
            tuple[int, int],
            tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix],
        ] = {}
        prefactors = 1.0j * float(alat) * r_cart

        for ik_local, kpoint in enumerate(local_kpts):
            phase = np.exp(-2.0j * np.pi * np.dot(r_cart, kpoint))
            for ispin in range(self.nspin):
                hk_sparse = self._assemble_weighted_block(phase, ispin, threshold=0.0)
                direction_blocks: list[sparse.csr_matrix] = []

                for direction in range(3):
                    dh_sparse = self._assemble_weighted_block(
                        phase * prefactors[:, direction],
                        ispin,
                        threshold=0.0,
                    )

                    if hk_sparse.nnz > 0:
                        correction = hk_sparse.multiply(1.0j * dnm[:, :, direction])
                        correction = correction.tocsr()
                        correction.sum_duplicates()
                        if correction.nnz > 0:
                            dh_sparse = (dh_sparse + correction).tocsr()

                    dh_sparse = 0.5 * (dh_sparse + dh_sparse.getH())
                    direction_blocks.append(self._apply_threshold(dh_sparse.tocsr(), threshold))

                local_blocks[(ik_local, ispin)] = (
                    direction_blocks[0],
                    direction_blocks[1],
                    direction_blocks[2],
                )

        return local_blocks
