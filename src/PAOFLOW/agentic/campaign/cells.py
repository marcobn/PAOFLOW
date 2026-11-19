"""Supercells and k-grids for pw.x: FFT boxes, reduced bases, k-point sampling, and the rewrite of an input.

Large supercells built as multiples of a skewed primitive cell carry the skew into their basis. pw.x sizes its FFT
box from the lengths of the basis vectors, so the box (and the cost) can be many times what the volume needs;
a reduced basis of the same lattice fixes it without changing a single atom. The k-grid of the rewritten cell is
chosen by the shortest vector of its superlattice, a basis-independent measure of the sampling.
"""

from __future__ import annotations

import itertools
import re

import numpy as np
from numpy.typing import NDArray


def good_fft_order(size: int) -> int:
    """Smallest integer at least ``size`` whose prime factors are 2, 3 and 5 (the FFT orders of pw.x)."""
    while True:
        remainder = size
        for factor in (2, 3, 5):
            while remainder % factor == 0:
                remainder //= factor
        if remainder == 1:
            return size
        size += 1


def fft_box(cell_bohr: NDArray[np.float64], ecutrho: float) -> list[int]:
    """Dense FFT box pw.x uses for a cell at a charge-density cutoff.

    Parameters
    ----------
    cell_bohr : numpy.ndarray, shape (3, 3)
        Lattice vectors (rows, bohr).
    ecutrho : float
        Charge-density cutoff (Ry).

    Returns
    -------
    list of int
        FFT points along each lattice vector.

    Notes
    -----
    pw.x needs the largest Miller index :math:`n_i = \\sqrt{E_\\rho}\\,|\\mathbf{a}_i|/2\\pi` along each vector
    and takes :math:`2 n_i + 1` points, rounded up to a product of 2, 3 and 5. The box therefore grows with the
    lengths of the basis vectors, not with the volume.
    """
    return [
        good_fft_order(2 * int(np.sqrt(ecutrho) * np.linalg.norm(vector) / (2 * np.pi)) + 1)
        for vector in cell_bohr
    ]


def lll_reduce(basis: NDArray[np.float64], delta: float = 0.75) -> NDArray[np.float64]:
    """Lenstra–Lenstra–Lovász reduced basis (rows) of the lattice spanned by ``basis``.

    Parameters
    ----------
    basis : numpy.ndarray, shape (3, 3)
        Lattice vectors (rows).
    delta : float
        Lovász parameter.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        A nearly orthogonal basis of the same lattice.
    """
    reduced = basis.astype(float).copy()
    dimension = len(reduced)

    def gram_schmidt(
        vectors: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        orthogonal = np.zeros_like(vectors)
        projections = np.zeros((dimension, dimension))
        for i in range(dimension):
            orthogonal[i] = vectors[i]
            for j in range(i):
                projections[i, j] = vectors[i] @ orthogonal[j] / (orthogonal[j] @ orthogonal[j])
                orthogonal[i] -= projections[i, j] * orthogonal[j]
        return orthogonal, projections

    k = 1
    while k < dimension:
        for j in range(k - 1, -1, -1):
            _, projections = gram_schmidt(reduced)
            step = round(projections[k, j])
            if step:
                reduced[k] -= step * reduced[j]
        orthogonal, projections = gram_schmidt(reduced)
        if orthogonal[k] @ orthogonal[k] >= (delta - projections[k, k - 1] ** 2) * (
            orthogonal[k - 1] @ orthogonal[k - 1]
        ):
            k += 1
        else:
            reduced[[k, k - 1]] = reduced[[k - 1, k]]
            k = max(k - 1, 1)
    return reduced


def reduced_basis(cell_bohr: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Reduced, right-handed basis of the lattice of ``cell_bohr``, sorted by length.

    Parameters
    ----------
    cell_bohr : numpy.ndarray, shape (3, 3)
        Lattice vectors (rows, bohr).

    Returns
    -------
    reduced : numpy.ndarray, shape (3, 3)
        The new basis.
    transform : numpy.ndarray of int, shape (3, 3)
        Unimodular integer matrix with ``transform @ cell_bohr == reduced``.

    Raises
    ------
    RuntimeError
        If the result is not a basis of the same lattice (never expected).
    """
    reduced = lll_reduce(cell_bohr)
    reduced = reduced[np.argsort(np.linalg.norm(reduced, axis=1), kind='stable')]
    if np.linalg.det(reduced) < 0:
        reduced[2] = -reduced[2]
    transform = np.rint(reduced @ np.linalg.inv(cell_bohr)).astype(int)
    if abs(abs(round(np.linalg.det(transform))) - 1) != 0 or not np.allclose(
        transform @ cell_bohr, reduced, atol=1e-8
    ):
        raise RuntimeError('the reduced basis is not a basis of the same lattice')
    return reduced, transform


def fft_inflation(cell_bohr: NDArray[np.float64], ecutrho: float) -> float:
    """Ratio of the FFT points of a cell to those of the reduced basis of its lattice (1 for a compact basis)."""
    reduced, _ = reduced_basis(cell_bohr)
    return float(np.prod(fft_box(cell_bohr, ecutrho)) / np.prod(fft_box(reduced, ecutrho)))


def time_reversal_kpoints(kgrid: tuple[int, int, int] | list[int]) -> int:
    """k-points of a Gamma-centred grid reduced by time reversal only, as pw.x has them with ``nosym = .true.``.

    Parameters
    ----------
    kgrid : sequence of int
        Monkhorst–Pack divisions.

    Returns
    -------
    int
        :math:`(N + N_\\mathrm{TRIM})/2`, with :math:`N_\\mathrm{TRIM}` the grid points equal to their inverse.
    """
    invariant_points = int(np.prod([2 if divisions % 2 == 0 else 1 for divisions in kgrid]))
    return (int(np.prod(kgrid)) + invariant_points) // 2


def shortest_superlattice_vector(
    cell_bohr: NDArray[np.float64], kgrid: tuple[int, int, int] | list[int]
) -> float:
    """Length (bohr) of the shortest vector of the superlattice ``diag(kgrid) @ cell_bohr``.

    Parameters
    ----------
    cell_bohr : numpy.ndarray, shape (3, 3)
        Lattice vectors (rows, bohr).
    kgrid : sequence of int
        Gamma-centred grid in the basis of ``cell_bohr``.

    Returns
    -------
    float
        Shortest superlattice vector.

    Notes
    -----
    A Gamma-centred grid integrates exactly every Fourier component :math:`e^{i\\mathbf{k}\\cdot\\mathbf{R}}`
    except those with :math:`\\mathbf{R}` on this superlattice, so its shortest vector measures the sampling
    independently of the basis (Moreno and Soler, Phys. Rev. B 45, 13891 (1992)).
    """
    superlattice = lll_reduce(np.diag(np.asarray(kgrid, dtype=float)) @ cell_bohr)
    return float(min(
        np.linalg.norm(np.array(combination) @ superlattice)
        for combination in itertools.product(range(-2, 3), repeat=3)
        if any(combination)
    ))  # fmt: skip


def lattice_rotations(cell_bohr: NDArray[np.float64]) -> NDArray[np.int64]:
    """Point-group rotations of the bare lattice, integer matrices acting on fractional coordinates (needs spglib)."""
    import spglib

    symmetry = spglib.get_symmetry((cell_bohr, [[0.0, 0.0, 0.0]], [1]), symprec=1e-5)
    return np.unique(symmetry['rotations'], axis=0)


def grid_is_invariant(
    kgrid: tuple[int, int, int] | list[int], rotations: NDArray[np.int64]
) -> bool:
    """True if the Gamma-centred grid maps onto itself under every rotation.

    A grid that is not invariant under the lattice point group makes pw.x reduce it with the lattice group and
    unfold it with the crystal group, which adds off-grid points (three times the k-points of a displaced cell
    without symmetry in the Ta0.8Hf0.2C campaign).
    """
    divisions = np.array(kgrid, dtype=float)
    for rotation in rotations:
        reciprocal = np.linalg.inv(rotation).T
        for axis in range(3):
            image = reciprocal[:, axis] / divisions[axis] * divisions
            if not np.allclose(image, np.rint(image), atol=1e-8):
                return False
    return True


def choose_kgrid(
    cell_bohr: NDArray[np.float64],
    min_points: int = 3,
    floor_bohr: float = 48.0,
    max_points: int = 8,
) -> dict:
    """Gamma-centred grid of a cell: at least ``min_points`` per axis and a superlattice no shorter than ``floor_bohr``.

    Parameters
    ----------
    cell_bohr : numpy.ndarray, shape (3, 3)
        Lattice vectors (rows, bohr), preferably a reduced basis.
    min_points : int
        Points per axis at least. PAOFLOW's Fourier interpolation splits the Nyquist term between +R and -R, so
        with two points along an axis the odd part of the hoppings across the cell boundary is lost.
    floor_bohr : float
        Shortest superlattice vector at least (48 bohr = 25 Å, the cells of the TaC–HfC campaigns).
    max_points : int
        Largest number of points per axis tried.

    Returns
    -------
    dict
        ``kgrid``, ``kpoints`` (with time reversal only), ``shortest_vector_bohr`` and ``invariant`` (under the
        lattice point group; True when spglib is missing).

    Notes
    -----
    Among the grids that pass, the fewest k-points wins, then the longest shortest vector, then lattice-group
    invariance.
    """
    try:
        rotations = lattice_rotations(cell_bohr)
    except ImportError:
        rotations = np.eye(3, dtype=int)[None]
    best = None
    for kgrid in itertools.product(range(min_points, max_points + 1), repeat=3):
        length = shortest_superlattice_vector(cell_bohr, kgrid)
        if length < floor_bohr - 1e-6:
            continue
        key = (
            time_reversal_kpoints(kgrid),
            -round(length, 3),
            0 if grid_is_invariant(kgrid, rotations) else 1,
        )
        if best is None or key < best[0]:
            best = (key, list(kgrid), length)
    if best is None:
        raise ValueError(
            'no grid with at most %d points per axis reaches %.1f bohr' % (max_points, floor_bohr)
        )
    _, kgrid, length = best
    return {'kgrid': kgrid, 'kpoints': time_reversal_kpoints(kgrid), 'shortest_vector_bohr': round(length, 2),
            'invariant': grid_is_invariant(kgrid, rotations)}  # fmt: skip


# ------------------------------------------------------------------------------------------------ pw.x inputs
def read_cell(text: str) -> NDArray[np.float64]:
    """Lattice vectors (rows, bohr) of a pw.x input with ``CELL_PARAMETERS bohr`` or ``ibrav`` + ``celldm``."""
    if 'CELL_PARAMETERS' in text:
        rest = text.split('CELL_PARAMETERS', 1)[1]
        unit = rest.splitlines()[0].strip().lower()
        if 'bohr' not in unit:
            raise ValueError('only CELL_PARAMETERS bohr is handled, got %r' % unit)
        return np.array([[float(value) for value in row.split()] for row in rest.splitlines()[1:4]])
    from ...inputs.lattice_format import lattice_format_QE

    ibrav = int(re.search(r'\bibrav\s*=\s*(-?\d+)', text).group(1))
    celldm = {
        int(index): float(value.replace('d', 'e').replace('D', 'e'))
        for index, value in re.findall(r'celldm\((\d)\)\s*=\s*([-0-9.eEdD+]+)', text)
    }
    return np.array(lattice_format_QE(ibrav, [celldm.get(i, 0.0) for i in range(1, 7)]))


def read_positions(text: str) -> tuple[list[str], NDArray[np.float64]]:
    """Species and crystal coordinates of ``ATOMIC_POSITIONS crystal``."""
    block = text.split('ATOMIC_POSITIONS', 1)[1]
    unit = block.splitlines()[0].strip().lower().strip('{}() ')
    if unit != 'crystal':
        raise ValueError('only ATOMIC_POSITIONS crystal is handled, got %r' % unit)
    atoms = int(re.search(r'\bnat\s*=\s*(\d+)', text).group(1))
    rows = block.splitlines()[1 : 1 + atoms]
    return [row.split()[0] for row in rows], np.array(
        [[float(v) for v in row.split()[1:4]] for row in rows]
    )


def set_namelist_value(text: str, namelist: str, key: str, value: str) -> str:
    """Set ``key = value`` in a namelist of a pw.x input, replacing an existing assignment.

    Only the assignment of ``key`` is removed from its line, so lines with several comma-separated assignments
    (``ibrav = 0, nat = 2, ntyp = 2``) keep the others.
    """
    pattern = re.compile(r'(&%s\b.*?\n)(.*?)(^\s*/\s*$)' % namelist, re.S | re.M | re.I)
    match = pattern.search(text)
    if match is None:
        raise ValueError('namelist &%s not found' % namelist)
    assignment = re.compile(
        r'(^[ \t]*|,[ \t]*)%s[ \t]*=[ \t]*(\'[^\']*\'|"[^"]*"|[^,\n]*)' % re.escape(key),
        re.M | re.I,
    )
    body = assignment.sub(
        lambda m: '' if m.group(1).startswith(',') else m.group(1), match.group(2)
    )
    body = re.sub(
        r'(?m)^([ \t]*),[ \t]*', r'\1', body
    )  # a line that started with the removed assignment
    body = re.sub(r'(?m)^[ \t]*\n', '', body)  # a line that held only it
    body += '    %s = %s\n' % (key, value)
    return text[: match.start(2)] + body + text[match.end(2) :]


def reduce_cell_input(
    text: str,
    nosym: bool = True,
    damp_mixing: bool = True,
    start_from_density: bool = False,
    floor_bohr: float = 48.0,
) -> tuple[str, dict]:
    """A pw.x input rewritten in the reduced basis of its lattice, with a k-grid chosen for that basis.

    Parameters
    ----------
    text : str
        pw.x input with ``ATOMIC_POSITIONS crystal``, ``K_POINTS automatic`` and an ``ecutrho``.
    nosym : bool
        Add ``nosym = .true.`` (cells without symmetry: displaced supercells; also their undisplaced reference,
        so that both sample the same k-points).
    damp_mixing : bool
        ``mixing_mode = 'local-TF'``, ``mixing_beta = 0.3``, ``mixing_ndim = 12`` (large metallic cells slosh with
        plain mixing at β = 0.7).
    start_from_density : bool
        ``startingpot = 'file'``: the run starts from a density copied into its save beforehand (e.g. from the
        undisplaced reference of the same basis).
    floor_bohr : float
        Shortest superlattice vector of the new grid at least (:func:`choose_kgrid`).

    Returns
    -------
    new_text : str
        The rewritten input: same atoms (Cartesian positions unchanged), ``ibrav = 0``, ``CELL_PARAMETERS bohr``.
    summary : dict
        Old and new vector lengths, transform, FFT boxes, old and new grids with their k-points and shortest
        superlattice vectors.

    Raises
    ------
    RuntimeError
        If an atom moved in the rewrite (never expected).
    """
    ecutrho = float(re.search(r'ecutrho\s*=\s*([0-9.dDeE+]+)', text).group(1).replace('d', 'e'))
    old_cell = read_cell(text)
    new_cell, transform = reduced_basis(old_cell)
    species, fractional = read_positions(text)
    cartesian = fractional @ old_cell
    new_fractional = cartesian @ np.linalg.inv(new_cell)
    new_fractional -= np.floor(new_fractional + 1e-10)
    old_kgrid = [int(v) for v in text.split('K_POINTS', 1)[1].split('\n', 1)[1].split()[:3]]
    choice = choose_kgrid(new_cell, floor_bohr=floor_bohr)

    out = re.sub(r'^\s*celldm\(\d\)\s*=.*$\n?', '', text, flags=re.M)
    out = set_namelist_value(out, 'system', 'ibrav', '0')
    if nosym:
        out = set_namelist_value(out, 'system', 'nosym', '.true.')
    if damp_mixing:
        out = set_namelist_value(out, 'electrons', 'mixing_mode', "'local-TF'")
        out = set_namelist_value(out, 'electrons', 'mixing_beta', '0.3')
        out = set_namelist_value(out, 'electrons', 'mixing_ndim', '12')
    if start_from_density:
        out = set_namelist_value(out, 'electrons', 'startingpot', "'file'")

    cell_card = 'CELL_PARAMETERS bohr\n' + ''.join(
        '  %22.16f %22.16f %22.16f\n' % tuple(v) for v in new_cell
    )
    positions_card = 'ATOMIC_POSITIONS crystal\n' + ''.join(
        '  %-3s %20.16f %20.16f %20.16f\n' % (name, *position)
        for name, position in zip(species, new_fractional)
    )
    kpoints_card = 'K_POINTS automatic\n %d %d %d 0 0 0\n' % tuple(choice['kgrid'])
    head = (
        re.sub(r'CELL_PARAMETERS.*', '', out.split('ATOMIC_SPECIES', 1)[0], flags=re.S).rstrip()
        + '\n'
    )
    species_card = 'ATOMIC_SPECIES' + out.split('ATOMIC_SPECIES', 1)[1].split('ATOMIC_POSITIONS')[0]
    species_card = re.sub(r'CELL_PARAMETERS.*', '', species_card, flags=re.S).rstrip() + '\n'
    new_text = head + cell_card + species_card + positions_card + kpoints_card

    moved = (new_fractional @ new_cell - cartesian) @ np.linalg.inv(old_cell)
    if not np.allclose(moved - np.rint(moved), 0.0, atol=1e-9):
        raise RuntimeError('positions changed in the rewrite')
    summary = {
        'old_lengths_bohr': np.round(np.linalg.norm(old_cell, axis=1), 3).tolist(),
        'new_lengths_bohr': np.round(np.linalg.norm(new_cell, axis=1), 3).tolist(),
        'transform': transform.tolist(),
        'fft_old': fft_box(old_cell, ecutrho),
        'fft_new': fft_box(new_cell, ecutrho),
        'old_kgrid': old_kgrid,
        'old_shortest_vector_bohr': round(shortest_superlattice_vector(old_cell, old_kgrid), 2),
        **choice,
    }
    return new_text, summary
