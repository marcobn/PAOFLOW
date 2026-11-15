"""Orbital descriptors of the PAO basis for photoemission matrix elements."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from .atomic_radial import radial_table
from .harmonics import qe_harmonic, spinor_components
from .radial import BOHR_ANGSTROM


@dataclass
class Orbital:
    """One PAO basis function.

    Parameters
    ----------
    index : int
        Position in the basis (row of the Hamiltonian).
    atom : int
        Atom index in the cell.
    element : str
        Chemical symbol (a species label such as ``'Si1'`` is accepted).
    position : ndarray ``(3,)``
        Cartesian position (Å).
    l : int
        Angular momentum.
    m_qe : int
        Harmonic index ``1 .. 2l+1`` in QE order (:func:`~PAOFLOW.arpes.harmonics.qe_harmonic`).
    n : int, optional
        Principal quantum number (needed for the atomic radial fallbacks).
    label : str, optional
        Shell label of the basis table (``'3P'``).
    radial : tuple ``(r_bohr, chi)``, optional
        Radial table ``chi = r R(r)`` on ``r`` (bohr); ``None`` until a fallback is assigned.
    spin : int, optional
        ``0`` for a spinless basis, ``+1`` / ``-1`` for the spin component of a spinor basis.
    cross_section : float, optional
        Multiplicative photoionisation cross-section factor (chinook ``sigma``).
    depth : float, optional
        Depth below the surface (Å) for the escape-depth attenuation of slabs.
    components : list of ``(m_qe, spin, coefficient)``, optional
        A basis function that is a combination of real harmonics of the same shell and spin
        components (spinor PAO functions, :func:`~PAOFLOW.arpes.harmonics.spinor_components`);
        ``None`` means the single real harmonic ``m_qe`` with spin ``spin``.
    j : float, optional
        Total angular momentum of a spinor function (selects its radial table).
    """

    index: int
    atom: int
    element: str
    position: NDArray[np.float64]
    l: int
    m_qe: int
    n: int | None = None
    label: str = ''
    radial: tuple[NDArray[np.float64], NDArray[np.float64]] | None = field(default=None, repr=False)
    spin: int = 0
    cross_section: float = 1.0
    depth: float = 0.0
    components: list | None = field(default=None, repr=False)
    j: float | None = None

    @property
    def m(self) -> int:
        """Standard real-harmonic index ``-l .. l``."""
        return qe_harmonic(self.l, self.m_qe)[0]

    @property
    def sign(self) -> float:
        """Sign of the QE harmonic relative to the standard one."""
        return qe_harmonic(self.l, self.m_qe)[1]

    @property
    def radial_key(self) -> tuple:
        """Key shared by the orbitals of one shell of one species: ``(element, label, l[, j])``."""
        key = (self.element, self.label or 'n%s' % self.n, self.l)
        return key if self.j is None else key + (float(self.j),)

    @property
    def terms(self) -> list:
        """``[(m_qe, spin, coefficient), ...]`` of the function (one term for a real harmonic)."""
        if self.components is None:
            return [(self.m_qe, self.spin, 1.0)]
        return list(self.components)


class PhotoemissionBasis:
    """The orbitals of a PAO basis with everything the matrix elements need.

    Parameters
    ----------
    orbitals : sequence of :class:`Orbital`
        In the order of the Hamiltonian rows.
    """

    def __init__(self, orbitals) -> None:
        self.orbitals: list[Orbital] = list(orbitals)
        for i, o in enumerate(self.orbitals):
            if o.index != i:
                raise ValueError('orbital %d carries index %d' % (i, o.index))

    @classmethod
    def from_data_controller(cls, data_controller) -> PhotoemissionBasis:
        """From a PAOFLOW run after ``projections`` (``arrays['basis']``).

        Raises
        ------
        NotImplementedError
            For fully-relativistic (spinor) PAO bases, not yet supported.
        ValueError
            If the basis length differs from ``nawf``.
        """
        arry, attr = data_controller.data_dicts()
        entries = arry['basis']
        if 'nawf' in attr and len(entries) != attr['nawf']:
            raise ValueError('basis has %d orbitals, nawf = %d' % (len(entries), attr['nawf']))
        tau = np.asarray(arry['tau'], dtype=float)  # bohr
        spinor = bool(attr.get('dftSO', False))
        orbitals = []
        i = 0
        while i < len(entries):
            b = entries[i]
            pos = np.asarray(b['tau'], dtype=float)
            atom = int(np.argmin(np.linalg.norm(tau - pos[None, :], axis=1)))
            label = str(b.get('label', ''))
            n = int(label[0]) if label[:1].isdigit() else None
            l = int(b['l'])
            if not spinor:
                radial = None
                if 'r' in b and 'wfc' in b:
                    radial = (np.asarray(b['r'], dtype=float), np.asarray(b['wfc'], dtype=float))
                orbitals.append(
                    Orbital(i, atom, str(b['atom']), pos * BOHR_ANGSTROM, l, int(b['m']),
                            n=n, label=label, radial=radial)
                )  # fmt: skip
                i += 1
                continue
            # spinor shell: 2(2l+1) consecutive entries in assign_jm order; the radial tables of
            # the two j are those of the first (j = l - 1/2) and last (j = l + 1/2) entries
            size = 2 * (2 * l + 1)
            block = entries[i : i + size]
            if len(block) != size or any(int(e['l']) != l for e in block):
                raise ValueError(
                    'spinor basis: shell of l=%d at entry %d is not a %d-block' % (l, i, size)
                )
            comps = spinor_components(l)  # (jm, sigma, m_qe)
            radial_low = (np.asarray(block[0]['r'], float), np.asarray(block[0]['wfc'], float))
            radial_high = (np.asarray(block[-1]['r'], float), np.asarray(block[-1]['wfc'], float))
            for p in range(size):
                j = l - 0.5 if (l > 0 and p < 2 * l) else l + 0.5
                terms = [
                    (q + 1, +1 if sigma == 0 else -1, complex(comps[p, sigma, q]))
                    for sigma in range(2)
                    for q in range(2 * l + 1)
                    if abs(comps[p, sigma, q]) > 1e-12
                ]
                orbitals.append(
                    Orbital(i + p, atom, str(b['atom']), pos * BOHR_ANGSTROM, l, 1, n=n, label=label,
                            radial=radial_low if j < l else radial_high, spin=1,
                            components=terms, j=j)
                )  # fmt: skip
            i += size
        return cls(orbitals)

    @property
    def size(self) -> int:
        """Number of orbitals."""
        return len(self.orbitals)

    @property
    def positions(self) -> NDArray[np.float64]:
        """``(size, 3)`` positions (Å)."""
        return np.array([o.position for o in self.orbitals], dtype=float)

    @property
    def spins(self) -> NDArray[np.int_]:
        """``(size,)`` spin labels."""
        return np.array([o.spin for o in self.orbitals], dtype=int)

    @property
    def is_spinor(self) -> bool:
        """Whether the orbitals carry spin components."""
        return bool(np.any(self.spins != 0)) or any(o.components is not None for o in self.orbitals)

    def radial_functions(self) -> dict[tuple[str, str, int], tuple[NDArray, NDArray, int]]:
        """``key -> (r_bohr, chi, l)`` of the distinct radial tables.

        Raises
        ------
        ValueError
            If an orbital has no radial table (use :meth:`with_atomic_radial`).
        """
        out = {}
        for o in self.orbitals:
            if o.radial is None:
                raise ValueError(
                    'orbital %d (%s %s) has no radial function' % (o.index, o.element, o.label)
                )
            out.setdefault(o.radial_key, (o.radial[0], o.radial[1], o.l))
        return out

    def with_atomic_radial(self, kind: str, only_missing: bool = False) -> PhotoemissionBasis:
        """Copy with Slater or hydrogenic radial tables (:func:`~PAOFLOW.arpes.atomic_radial.radial_table`).

        Parameters
        ----------
        kind : {'slater', 'hydrogenic'}
        only_missing : bool, optional
            Keep the PAO tables of the orbitals that have one.
        """
        orbitals = []
        for o in self.orbitals:
            if only_missing and o.radial is not None:
                orbitals.append(o)
                continue
            if o.n is None:
                raise ValueError(
                    'orbital %d needs a principal quantum number for %s radials' % (o.index, kind)
                )
            orbitals.append(replace(o, radial=radial_table(kind, o.element, o.n, o.l),
                                    label=o.label or '%d%s' % (o.n, 'spdf'[o.l])))  # fmt: skip
        return PhotoemissionBasis(orbitals)

    def with_depths(
        self, normal: NDArray[np.float64], surface_position: float | None = None
    ) -> PhotoemissionBasis:
        """Copy with ``depth`` set from the projection on ``normal`` (Å), the topmost orbital at 0."""
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        heights = self.positions @ n
        top = heights.max() if surface_position is None else float(surface_position)
        return PhotoemissionBasis(
            [replace(o, depth=float(top - h)) for o, h in zip(self.orbitals, heights)]
        )
