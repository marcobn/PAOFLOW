"""Sparsity and memory reporting helpers for the sparse PAOFLOW backend.

All sparse modules funnel their status lines through this module so that the
sparse logs are formatted consistently and remain directly comparable to the
dense PAOFLOW output.  Each helper returns a short, single-line string; the
caller is responsible for gating on ``verbose`` and choosing the output rank.
"""

# Bytes per stored entry of a complex128 CSR matrix: one complex value (16) plus
# one column index (4, int32) plus, amortised, the row-pointer overhead.
_BYTES_PER_NNZ = 16 + 4


def human_bytes(nbytes):
    """Format a byte count as a compact human-readable string.

    Parameters
    ----------
    nbytes : float
        Number of bytes.

    Returns
    -------
    str
        e.g. ``'294 MB'`` or ``'1.4 GB'``.
    """
    nbytes = float(nbytes)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if nbytes < 1024.0 or unit == 'TB':
            if unit in ('B', 'KB'):
                return f'{nbytes:.0f} {unit}'
            return f'{nbytes:.1f} {unit}'
        nbytes /= 1024.0


def hamiltonian_stats(sparse_h):
    """One-line sparsity summary for a :class:`SparseHamiltonian`.

    Parameters
    ----------
    sparse_h : SparseHamiltonian
        The sparse real-space Hamiltonian container.

    Returns
    -------
    str
        A line such as
        ``Sparse H(R): csr nawf=144, nR=1728, nnz=0.42M, density=1.1e-02,
        mem=8 MB, dense avoided=5.3 GB``.
    """
    nawf = sparse_h.nawf
    nR = sparse_h.nR
    nspin = sparse_h.nspin
    nnz = sparse_h.nnz
    dense_elems = nawf * nawf * nR * nspin
    density = nnz / dense_elems if dense_elems else 0.0
    sparse_mem = nnz * _BYTES_PER_NNZ + sparse_h.R.nbytes
    dense_mem = dense_elems * 16  # complex128
    return (
        f'Sparse H(R): csr nawf={nawf}, nR={nR}, nnz={nnz / 1e6:.2f}M, '
        f'density={density:.1e}, mem={human_bytes(sparse_mem)}, '
        f'dense avoided={human_bytes(dense_mem)}'
    )


def eigensolver_stats(name, window, n_eigs, nkpnts, converged, solver, iterations=None):
    """One-line summary of a sparse eigensolver stage.

    Parameters
    ----------
    name : str
        Stage label, e.g. ``'bands'`` or ``'pao_eigh'``.
    window : tuple of float or None
        ``(emin, emax)`` energy window in eV, or ``None`` when a fixed band
        count was requested rather than an energy window.
    n_eigs : int
        Number of eigenpairs computed per k-point.
    nkpnts : int
        Number of k-points solved.
    converged : int
        Number of k-points whose solve converged.
    solver : str
        Solver identifier, e.g. ``'eigsh(SA)'``, ``'eigsh(shift-invert)'`` or
        ``'dense(gated)'``.
    iterations : int, optional
        Total iteration count when available.

    Returns
    -------
    str
    """
    if window is not None:
        wtxt = f'window=[{window[0]:.2f}, {window[1]:.2f}] eV'
    else:
        wtxt = 'window=full'
    line = (
        f'Sparse {name}: {solver} {wtxt}, eigenpairs={n_eigs}, '
        f'k-points={nkpnts}, converged={converged}/{nkpnts}'
    )
    if iterations is not None:
        line += f', iters={iterations}'
    return line


def velocity_stats(nkpnts, n_sel, nspin):
    """One-line summary of the sparse band-velocity stage.

    Parameters
    ----------
    nkpnts : int
        Number of k-points.
    n_sel : int
        Number of selected bands carried through the sparse pipeline.
    nspin : int
        Number of spin channels.

    Returns
    -------
    str
    """
    # velkp is the only stored velocity object: (nkpnts, 3, n_sel, nspin) real.
    mem = nkpnts * 3 * n_sel * nspin * 8
    dense_mem = nkpnts * 3 * n_sel * n_sel * nspin * 16  # a dense pksp would be n_sel^2
    return (
        f'Sparse velocities: band-diagonal velkp shape=({nkpnts}, 3, {n_sel}, '
        f'{nspin}), mem={human_bytes(mem)}, dense pksp avoided='
        f'{human_bytes(dense_mem)}'
    )


def estimate_dense_grid_bytes(nawf, nkpnts, nspin, factor=1):
    """Estimate the bytes a dense ``(nawf, nawf, nkpnts, nspin)`` tensor would use.

    Used to size-gate the bounded coarse-grid operations and to report the dense
    memory the sparse path avoids.

    Parameters
    ----------
    nawf, nkpnts, nspin : int
        Tensor dimensions.
    factor : int
        Multiplicity (e.g. ``3`` for a gradient tensor).

    Returns
    -------
    float
        Estimated size in bytes (complex128).
    """
    return float(nawf) * nawf * nkpnts * nspin * factor * 16
