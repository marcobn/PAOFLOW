r"""Operators of the Kubo-Bastin linear response: observables, components, units.

The field is an electric field, coupled through the velocity. The responding observable is one of:

=================== ==================== ============================ ===============================
observable          component            operator                     unit (bulk; sheet)
=================== ==================== ============================ ===============================
``charge_current``  (current, field)     :math:`\hbar v_a`            S/cm; e^2/hbar
``spin_current``    (spin, current,      :math:`\{S_s, \hbar v_a\}/2` (hbar/e) S/cm; (hbar/e) e^2/hbar
                    field)
``orbital_current`` (l, current, field)  :math:`\{L_l, \hbar v_a\}/2` (hbar/e) S/cm; (hbar/e) e^2/hbar
``spin_density``    (spin, field)        :math:`S_s`                  hbar cm/V per cell
``orbital_density`` (l, field)           :math:`L_l`                  hbar cm/V per cell
``spin_torque``     (spin, field)        :math:`i[H, S_s]/\hbar`      e a0 per cell (torkance)
``custom``          (field,)             a user matrix :math:`O`      [O] a0/V per cell
=================== ==================== ============================ ===============================

:math:`S = \sigma/2` and :math:`L` are in units of :math:`\hbar`. Every response but the charge
current carries the sign of the electron charge (:func:`charge_sign`). The velocity is
:math:`\hbar v = \partial H/\partial k`; PAOFLOW's ``dHksp`` is :math:`-\partial H/\partial k`
(the :math:`e^{-ik\cdot R}` convention of ``HRs``, verified on the models), so :math:`\hbar v =`
``-dHksp``. An orbital projector :math:`P` (``projector``) acts as :math:`(PO + OP)/2`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from numpy.typing import NDArray

from ..utils.perturb_split import perturb_split
from .kubo_units import BOHR_TO_CM, conductivity_conversion

if TYPE_CHECKING:
    from ..DataController import DataController

AXES = 'xyz'


@dataclass(frozen=True)
class Observable:
    """How an observable enters the Kubo-Bastin response.

    Attributes
    ----------
    name : str
    indices : tuple of str
        Meaning of the component indices, the field direction last.
    base : str or None
        DataController array of the angular-momentum operator (``'Sj'``, ``'Lj'``), if any.
    current : bool
        The operator is a current (anticommutator with the velocity).
    unit : str
        Bulk unit of the output.
    sheet_unit : str
        Unit of the sheet (two-dimensional) output.
    """

    name: str
    indices: tuple[str, ...]
    base: str | None
    current: bool
    unit: str
    sheet_unit: str


OBSERVABLES: dict[str, Observable] = {
    o.name: o
    for o in (
        Observable('charge_current', ('current', 'field'), None, True, 'S/cm', 'e^2/hbar, sheet'),
        Observable(
            'spin_current',
            ('spin', 'current', 'field'),
            'Sj',
            True,
            '(hbar/e) S/cm',
            '(hbar/e) e^2/hbar, sheet',
        ),  # fmt: skip
        Observable(
            'orbital_current',
            ('orbital', 'current', 'field'),
            'Lj',
            True,
            '(hbar/e) S/cm',
            '(hbar/e) e^2/hbar, sheet',
        ),  # fmt: skip
        Observable(
            'spin_density',
            ('spin', 'field'),
            'Sj',
            False,
            'hbar cm/V per cell',
            'hbar cm/V per cell',
        ),  # fmt: skip
        Observable(
            'orbital_density',
            ('orbital', 'field'),
            'Lj',
            False,
            'hbar cm/V per cell',
            'hbar cm/V per cell',
        ),  # fmt: skip
        Observable('spin_torque', ('spin', 'field'), 'Sj', False, 'e a0 per cell', 'e a0 per cell'),
        Observable('custom', ('field',), None, False, '[O] a0/V per cell', '[O] a0/V per cell'),
    )
}


def parse_component(
    observable: str, component: str | tuple[int, ...] | list[int]
) -> tuple[int, ...]:
    """Component indices of an observable from ``'zxy'``-like strings or integer tuples.

    Parameters
    ----------
    observable : str
        Key of :data:`OBSERVABLES`.
    component : str or sequence of int
        ``'xy'`` / ``(0, 1)`` for the charge current, ``'zxy'`` / ``(2, 0, 1)`` for the spin current
        (spin, current, field), ``'yx'`` for a spin density (spin, field), ``'x'`` for ``custom``.

    Returns
    -------
    tuple of int

    Raises
    ------
    KeyError
        Unknown observable.
    ValueError
        Wrong number of indices or an index outside x, y, z.
    """
    spec = OBSERVABLES[observable]
    if isinstance(component, str):
        if any(c not in AXES for c in component):
            raise ValueError(f'component {component!r}: use the letters x, y, z')
        indices = tuple(AXES.index(c) for c in component)
    else:
        indices = tuple(int(i) for i in component)
    if len(indices) != len(spec.indices) or any(i not in (0, 1, 2) for i in indices):
        raise ValueError(
            f'{observable} needs {len(spec.indices)} indices ({", ".join(spec.indices)}), got {component!r}'
        )
    return indices


def component_label(observable: str, indices: tuple[int, ...]) -> str:
    """File label of a component: ``'xy'``, ``'z_xy'`` (spin, current, field), ``'yx'``, ``'x'``."""
    letters = ''.join(AXES[i] for i in indices)
    if OBSERVABLES[observable].current and len(indices) == 3:
        return letters[0] + '_' + letters[1:]
    return letters


def charge_sign(observable: str) -> float:
    """Sign of the carrier charge in the response: -1 for every observable but the charge current.

    Parameters
    ----------
    observable : str

    Returns
    -------
    float
        +1 for ``charge_current`` (:math:`\\sigma = q^2K`), -1 otherwise.

    Notes
    -----
    The field couples to the charge :math:`q = -|e|` of the electrons, so
    :math:`\\delta\\langle A\\rangle = (q/|e|)\\,K(A,\\hbar v)\\,|e|E`: a response linear in the charge
    (spin and orbital currents and densities, torques, user operators) carries the sign of
    :math:`q`; the charge current :math:`qv` carries :math:`q^2`. This is the convention of
    G. Y. Guo et al., PRL 100, 096401 (2008), Eq. (1) with :math:`e < 0` (the spin Hall conductivity
    of Pt is positive), and of the tutorial03 spin Hall routes.
    """
    return 1.0 if observable == 'charge_current' else -1.0


def physical_bands(data_controller: 'DataController', bands: int | None = None) -> int:
    """Number of bands, from the lowest, that enter the band sums of the Kubo engines.

    Parameters
    ----------
    data_controller : DataController
        Reads ``E_k``, ``bnd`` and ``shift``.
    bands : int, optional
        Explicit number of bands; ``None`` selects the default.

    Returns
    -------
    int
        ``bands`` if given; otherwise, for a DFT run (finite projectability ``shift``), ``bnd``, the bands
        below the shift; for a model (``shift = inf``), every band.

    Notes
    -----
    The PAO Hamiltonian places the complement of the projected subspace at the energy ``shift``: the bands
    above ``bnd`` are not physical states. Their interband matrix elements with the occupied bands are
    artefacts. With them, the static dielectric constant of Si changes by 20 % and the spin Hall
    conductivity of Pt by 2.5 % (November 2026). ``dielectric_tensor`` (``do_epsilon``) uses the same limit.
    """
    arry, attr = data_controller.data_dicts()
    total = arry['E_k'].shape[1]
    if bands is not None:
        return max(1, min(int(bands), total))
    if np.isfinite(attr.get('shift', np.inf)) and attr.get('bnd'):
        return max(1, min(int(attr['bnd']), total))
    return total


def unit_factor(data_controller: 'DataController', observable: str, two_dimensional: bool) -> float:
    """Factor from the k-averaged kernel sum (eV, bohr) to the unit of :data:`OBSERVABLES`.

    Parameters
    ----------
    data_controller : DataController
    observable : str
    two_dimensional : bool
        Sheet values for the currents.

    Returns
    -------
    float
        Includes the sign of the electron charge (:func:`charge_sign`).
    """
    spec = OBSERVABLES[observable]
    if spec.current:
        factor = conductivity_conversion(data_controller, two_dimensional)
    elif observable in ('spin_density', 'orbital_density'):
        factor = BOHR_TO_CM
    else:
        factor = 1.0
    return charge_sign(observable) * factor


def band_matrices(
    data_controller: 'DataController',
    observable: str,
    indices: tuple[int, ...],
    ik: int,
    ispin: int,
    projector: NDArray[np.complex128] | None = None,
    operator: NDArray[np.complex128] | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Response and field operators in the band basis at one local k-point.

    Parameters
    ----------
    data_controller : DataController
        Reads ``dHksp``, ``v_k``, ``degen``, ``E_k`` and ``Sj`` / ``Lj``.
    observable : str
        Key of :data:`OBSERVABLES`.
    indices : tuple of int
        From :func:`parse_component`.
    ik, ispin : int
        Local k-point and spin channel.
    projector : ndarray, shape ``(nawf, nawf)``, optional
        Orbital projector applied as :math:`(PO + OP)/2` to the response operator.
    operator : ndarray, shape ``(nawf, nawf)``, optional
        The matrix of ``observable='custom'`` (PAO basis, k-independent).

    Returns
    -------
    response : ndarray, shape ``(nb, nb)``, complex
        :math:`A_{nm}`.
    field : ndarray, shape ``(nb, nb)``, complex
        :math:`B_{nm} = \\hbar v^{b}_{nm}` (eV bohr).

    Notes
    -----
    Degenerate subspaces are rotated to diagonalise the response operator (:func:`perturb_split`, as
    the tutorial03 routes do); the Kubo-Bastin terms do not depend on that choice, the split into
    intraband and interband parts does.
    """
    arry, _ = data_controller.data_dicts()
    return band_matrices_k(
        observable,
        indices,
        arry['dHksp'][ik, :, :, :, ispin],
        arry['v_k'][ik, :, :, ispin],
        arry['degen'][ispin][ik],
        arry['E_k'][ik, :, ispin],
        arry.get(OBSERVABLES[observable].base),
        projector,
        operator,
    )


def band_matrices_k(
    observable: str,
    indices: tuple[int, ...],
    gradient: Sequence[Any],
    eigenvectors: NDArray[np.complex128],
    degeneracies: Sequence[Any],
    energies: NDArray[np.float64],
    base_operator: Sequence[Any] | None = None,
    projector: Any | None = None,
    operator: NDArray[np.complex128] | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Response and field operators in the band basis from the data of one k-point.

    The body of :func:`band_matrices`, shared with the sparse backend
    (:mod:`PAOFLOW.sparse.properties.kubo`); matrices may be ndarrays or scipy sparse matrices.

    Parameters
    ----------
    observable : str
        Key of :data:`OBSERVABLES`.
    indices : tuple of int
        From :func:`parse_component`.
    gradient : sequence of 3 matrices, shape ``(nawf, nawf)``
        :math:`\\partial H/\\partial k_l` in the PAO basis (``dHksp`` of this k-point, eV bohr).
    eigenvectors : ndarray, shape ``(nawf, m)``, complex
        Eigenvectors of :math:`H(\\mathbf k)` (columns).
    degeneracies : sequence of index arrays
        Degenerate groups of this k-point (``degen[ispin][ik]``).
    energies : ndarray, shape ``(m,)``
        Band energies (eV); read for ``spin_torque`` only.
    base_operator : sequence of 3 matrices, optional
        ``Sj`` or ``Lj``, for the observables built on one.
    projector : matrix, shape ``(nawf, nawf)``, optional
        Orbital projector applied as :math:`(PO + OP)/2` to the response operator.
    operator : ndarray, shape ``(nawf, nawf)``, optional
        The matrix of ``observable='custom'``.

    Returns
    -------
    response, field : ndarray, shape ``(m, m)``, complex
        As :func:`band_matrices`.

    Raises
    ------
    ValueError
        ``observable='custom'`` without ``operator``.
    KeyError
        An observable built on ``Sj`` / ``Lj`` without ``base_operator``.
    """
    spec = OBSERVABLES[observable]
    # The operators are built from dHksp (= -hbar v) and rotated in the degenerate subspaces as the
    # tutorial03 / do_Hall routes do (same eigh ordering, hence the same gauge); the signs of the
    # true velocity are applied afterwards.
    field = gradient[indices[-1]]
    if observable == 'custom':
        if operator is None:
            raise ValueError("observable='custom' needs operator")
        response, response_sign = np.asarray(operator, dtype=complex), 1.0
    elif spec.base is None:
        response, response_sign = gradient[indices[0]], -1.0
    else:
        if base_operator is None:
            raise KeyError(
                f'{observable} needs {spec.base} (PAOFLOW.spin_operator / orbital_operator)'
            )
        momentum = base_operator[indices[0]]
        if spec.current:
            direction = gradient[indices[1]]
            response, response_sign = 0.5 * (momentum @ direction + direction @ momentum), -1.0
        else:
            response, response_sign = momentum, 1.0
    if projector is not None:
        response = 0.5 * (projector @ response + response @ projector)
    a, b = perturb_split(response, field, eigenvectors, degeneracies)
    if observable == 'spin_torque':
        a = 1j * (energies[:, None] - energies[None, :]) * a
    return response_sign * a, -b
