"""Plane-wave blocks of a QE nscf run and their transforms to and from the FFT grid.

The coarse electron-phonon vertex of :mod:`PAOFLOW.elphon.dfpt_vertex` works with the
raw plane-wave coefficients of ``wfc<ik>.dat`` (no Loewdin step, unlike the projection
reader :func:`PAOFLOW.projection.do_atwfc_proj.read_QE_wfc`), the periodic parts of the
Bloch functions on the smooth FFT grid and the gather of a real-space product back onto
the ``k + q + G`` sphere.

Conventions (QE): ``psi_nk(r) = Omega^-1/2 sum_G c_n(G) exp(i (k+G).r)`` with
``sum_G |c_n(G)|^2 = 1``; the periodic part on the grid is
``u_n(r) = sum_G c_n(G) exp(i G.r)`` (inverse FFT without scaling); the forward
transform carries the ``1/N`` factor.  Grid index ``[i, j, k]`` is the point
``i/nr1 a1 + j/nr2 a2 + k/nr3 a3``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.io import FortranFile


@dataclass(frozen=True)
class PlaneWaveBlock:
    """Plane-wave coefficients of all bands at one k-point.

    Attributes
    ----------
    xk_cart : ndarray, shape ``(3,)``
        k-point (Cartesian, ``2 pi / alat``).
    miller : ndarray, shape ``(3, npw)``, int
        Miller indices of the ``k + G`` sphere (QE ``igk`` order).
    coefficients : ndarray, shape ``(nbnd, npol * npw)``, complex
        Raw coefficients ``c_n(G)`` (spinor components concatenated for ``npol = 2``).
    npol : int
        Spinor components.
    """

    xk_cart: NDArray[np.float64]
    miller: NDArray[np.int_]
    coefficients: NDArray[np.complex128]
    npol: int

    @property
    def npw(self) -> int:
        """Number of plane waves."""
        return int(self.miller.shape[1])

    @property
    def nbnd(self) -> int:
        """Number of bands."""
        return int(self.coefficients.shape[0])


def read_plane_wave_block(
    save_dir: str, ik: int, alat: float, nspin: int = 1, ispin: int = 0, bands: slice | None = None
) -> PlaneWaveBlock:
    """Read ``wfc<ik+1>.dat`` (or the HDF5 variant) of a QE save directory.

    Parameters
    ----------
    save_dir : str
        QE ``outdir/prefix.save`` directory.
    ik : int
        Zero-based k-point index in the nscf order.
    alat : float
        Lattice parameter (bohr): the file stores ``k`` in bohr^-1, the block in ``2 pi / alat``.
    nspin : int, optional
        1 (or 4 for noncollinear) reads ``wfc<ik>``; 2 reads ``wfcup``/``wfcdw``.
    ispin : int, optional
        Spin channel for ``nspin = 2`` (0 = down, 1 = up, as the projection reader).
    bands : slice, optional
        Subset of bands to keep.

    Returns
    -------
    PlaneWaveBlock
    """
    if nspin == 2:
        name = 'wfc%s%d' % ('up' if ispin == 1 else 'dw', ik + 1)
    else:
        name = 'wfc%d' % (ik + 1)
    dat = os.path.join(save_dir, name + '.dat')
    if os.path.exists(dat):
        with FortranFile(dat, 'r') as fh:
            head = fh.read_ints(np.int32)
            if len(head) != 11 or head[0] != ik + 1:
                raise ValueError('%s: unexpected header %s' % (dat, head))
            xk = np.frombuffer(head[1:7].tobytes(), np.float64).copy()
            gamma_only = head[8] != 0
            _ngw, igwx, npol, nbnd = (int(x) for x in fh.read_ints(np.int32))
            fh.read_reals(np.float64)  # bg (3, 3)
            mill = fh.read_ints(np.int32).reshape(3, igwx, order='F')
            coef = np.array([fh.read_reals(np.complex128) for _ in range(nbnd)])
    else:
        import h5py

        with h5py.File(os.path.join(save_dir, name + '.hdf5'), 'r') as h5:
            xk = np.array(h5.attrs['xk'], dtype=float)
            gamma_only = 'TRUE' in h5.attrs['gamma_only'].decode()
            igwx, npol, nbnd = (int(h5.attrs[a]) for a in ('igwx', 'npol', 'nbnd'))
            mill = np.transpose(np.array(h5['MillerIndices'], dtype=np.int64))
            evc = np.array(h5['evc'], dtype=np.float64)
            coef = evc[:, 0::2] + 1j * evc[:, 1::2]
    if gamma_only:
        raise NotImplementedError('gamma_only wavefunctions are not supported by the DFPT vertex')
    if bands is not None:
        coef = coef[bands]
    xk = xk * float(alat) / (2.0 * np.pi)
    return PlaneWaveBlock(
        xk, np.ascontiguousarray(mill.astype(int)), np.ascontiguousarray(coef), npol
    )


def k_plus_g_cart(block: PlaneWaveBlock, bg: NDArray[np.float64]) -> NDArray[np.float64]:
    """``k + G`` vectors of a block (Cartesian, ``2 pi / alat``), shape ``(npw, 3)``.

    Parameters
    ----------
    block : PlaneWaveBlock
    bg : ndarray, shape ``(3, 3)``
        Reciprocal lattice vectors (rows, ``2 pi / alat``).
    """
    return block.miller.T @ np.asarray(bg, dtype=float) + block.xk_cart


def box_indices(
    miller: NDArray[np.int_], fft_grid: tuple[int, int, int]
) -> tuple[NDArray[np.int_], ...]:
    """FFT-box indices ``(i1, i2, i3)`` of Miller indices (negative frequencies wrap)."""
    return tuple(np.mod(miller[d], int(fft_grid[d])) for d in range(3))


def periodic_part_on_grid(
    coefficients: NDArray[np.complex128], miller: NDArray[np.int_], fft_grid: tuple[int, int, int]
) -> NDArray[np.complex128]:
    """Periodic parts ``u_n(r) = sum_G c_n(G) exp(i G.r)`` on the FFT grid.

    Parameters
    ----------
    coefficients : ndarray, shape ``(nb, npw)``, complex
        Plane-wave coefficients (one spinor component).
    miller : ndarray, shape ``(3, npw)``, int
        Their Miller indices.
    fft_grid : tuple of int
        ``(nr1, nr2, nr3)``.

    Returns
    -------
    ndarray, shape ``(nb, nr1, nr2, nr3)``, complex
    """
    nb = coefficients.shape[0]
    box = np.zeros((nb,) + tuple(int(n) for n in fft_grid), dtype=np.complex128)
    i1, i2, i3 = box_indices(miller, fft_grid)
    box[:, i1, i2, i3] = coefficients
    return np.fft.ifftn(box, axes=(1, 2, 3)) * (box.shape[1] * box.shape[2] * box.shape[3])


def project_on_plane_waves(
    fields: NDArray[np.complex128],
    miller: NDArray[np.int_],
    fft_grid: tuple[int, int, int],
    umklapp: NDArray[np.int_] | tuple[int, int, int] = (0, 0, 0),
) -> NDArray[np.complex128]:
    """Fourier components of periodic fields on a ``k + q + G`` sphere.

    Parameters
    ----------
    fields : ndarray, shape ``(..., nr1, nr2, nr3)``, complex
        Periodic functions on the grid (for instance ``dv(r) u_n(r)``).
    miller : ndarray, shape ``(3, npw)``, int
        Miller indices of the stored block at ``k' = k + q - G0``.
    fft_grid : tuple of int
        ``(nr1, nr2, nr3)``.
    umklapp : array_like, shape ``(3,)``, int, optional
        ``G0``: the component at ``G' = G'' - G0`` is returned for every stored ``G''``.

    Returns
    -------
    ndarray, shape ``(..., npw)``, complex
        ``(1/N) sum_r f(r) exp(-i (G'' - G0).r)``.
    """
    n = int(np.prod(fft_grid))
    spectrum = np.fft.fftn(fields, axes=(-3, -2, -1)) / n
    shifted = miller - np.asarray(umklapp, dtype=int).reshape(3, 1)
    i1, i2, i3 = box_indices(shifted, fft_grid)
    return spectrum[..., i1, i2, i3]


def interpolate_to_grid(
    field: NDArray[np.complex128],
    fft_out: tuple[int, int, int],
    gcut: NDArray[np.bool_] | None = None,
) -> NDArray[np.complex128]:
    """Fourier-interpolate a periodic field from one FFT grid to another (QE ``fft_interpolate``).

    Parameters
    ----------
    field : ndarray, shape ``(..., nr1, nr2, nr3)``, complex
    fft_out : tuple of int
        Target grid.
    gcut : ndarray of bool, optional
        Mask over the *target* box selecting the G vectors to keep (``None``: every
        component representable on both grids).

    Returns
    -------
    ndarray, shape ``(..., *fft_out)``, complex
    """
    fft_in = field.shape[-3:]
    if tuple(fft_in) == tuple(fft_out) and gcut is None:
        return np.array(field, copy=True)
    spectrum = np.fft.fftn(field, axes=(-3, -2, -1)) / int(np.prod(fft_in))
    out = np.zeros(field.shape[:-3] + tuple(int(n) for n in fft_out), dtype=np.complex128)
    freqs = [np.fft.fftfreq(int(n), 1.0 / int(n)).astype(int) for n in fft_out]
    m1, m2, m3 = np.meshgrid(*freqs, indexing='ij')
    keep = np.ones(m1.shape, dtype=bool)
    for d, (m, n_in) in enumerate(zip((m1, m2, m3), fft_in)):
        keep &= (m >= -(int(n_in) - 1) // 2) & (m <= (int(n_in) - 1) // 2)
    if gcut is not None:
        keep &= gcut
    src = (np.mod(m1[keep], fft_in[0]), np.mod(m2[keep], fft_in[1]), np.mod(m3[keep], fft_in[2]))
    out[..., keep] = spectrum[..., src[0], src[1], src[2]]
    return np.fft.ifftn(out, axes=(-3, -2, -1)) * int(np.prod(fft_out))


def g_sphere(
    fft_grid: tuple[int, int, int], bg: NDArray[np.float64], gcut_tpiba2: float
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Reciprocal vectors of the FFT box inside ``|G|^2 <= gcut`` (QE ``ggen``).

    Parameters
    ----------
    fft_grid : tuple of int
    bg : ndarray, shape ``(3, 3)``
        Reciprocal lattice vectors (rows, ``2 pi / alat``).
    gcut_tpiba2 : float
        Cutoff ``ecut / tpiba^2`` in ``(2 pi / alat)^2`` units.

    Returns
    -------
    miller : ndarray, shape ``(3, ng)``, int
    g_cart : ndarray, shape ``(ng, 3)``
        Cartesian ``G`` (``2 pi / alat``).
    """
    freqs = [np.fft.fftfreq(int(n), 1.0 / int(n)).astype(int) for n in fft_grid]
    m = np.stack(np.meshgrid(*freqs, indexing='ij'), axis=0).reshape(3, -1)
    g = m.T @ np.asarray(bg, dtype=float)
    keep = np.einsum('gi,gi->g', g, g) <= gcut_tpiba2 * (1.0 + 1e-10)
    return np.ascontiguousarray(m[:, keep]), np.ascontiguousarray(g[keep])
