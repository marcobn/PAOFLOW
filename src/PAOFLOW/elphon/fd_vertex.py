r"""Finite-difference supercell vertex as a coarse coupling source (``'fd'``).

The displaced-cell route (:mod:`PAOFLOW.elphon.dvscf_fd`) gives the Cartesian
derivatives :math:`\partial_{\kappa\alpha} V` of the PAO Hamiltonian of a cell in
the fixed basis of its own reference structure.  For a supercell that is its own
unit cell (a thermally displaced cell, an alloy supercell: ``supercell_matrix=1``)
the dense-q drivers consume them directly: the cell's k-grid is the coarse grid,
the only phonon wavevector is :math:`\Gamma` (the :math:`3N` modes of the cell,
from the force constants of the same displaced runs) and the dense-k
interpolation of the supercell Hamiltonian and vertex gives :math:`\alpha^2F`,
:math:`\lambda`, :math:`\omega_{\log}` and the transport function in the
pair-resolved Wigner-Seitz scheme of :mod:`PAOFLOW.elphon.do_pao_eph_dense_q`.
:math:`\lambda` of the cell is intensive and equals the primitive value on the
commensurate q-grid.

Notes
-----
The dense electrons use :math:`H(\mathbf k)=\sum_{\mathbf R} e^{-2\pi i\,\mathbf k
\cdot\mathbf R} H(\mathbf R)` (``Hks = fftn(HRs)``) while the vertex kernel
evaluates :math:`d(\mathbf k)=\sum_{\mathbf R} e^{+2\pi i\,\mathbf k\cdot\mathbf R}
g(\mathbf R)`, so the cache holds

.. math::

    g_{ij,\kappa\alpha}(\mathbf R_e) = \partial_{\kappa\alpha} V_{ij}(-\mathbf R_e)
    \quad (\text{Ry/bohr}),

the index flip on the FFT grid, with the Cartesian index :math:`3\kappa+\alpha`
and a single phonon cell :math:`\mathbf R_p = 0`.
"""

from __future__ import annotations

import json
import os
import re
import warnings

import numpy as np

from .elph_bloch import AMU_RY, RY_TO_EV
from .io import MANIFEST
from .random_displacements import RY_BOHR2_TO_EV_ANG2

#: Keyword arguments of :func:`fd_cell_derivatives` accepted in ``fd_options``.
FD_DERIVATIVE_OPTIONS = (
    'basispath',
    'configuration',
    'pthr',
    'shift_type',
    'fixed_basis',
    'project_good_subspace',
    'locality_cutoff',
    'ridge',
    'verify_saves',
)

_FORCE_BLOCK = re.compile(
    r'Forces acting on atoms \(cartesian axes, Ry/au\):\s*\n\n'
    r'((?:\s*atom\s+\d+ type\s+\d+\s+force =\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\n)+)'
)
_FORCE_ROW = re.compile(r'force =\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)')


def flip_cells(a: np.ndarray, axes=(-3, -2, -1)) -> np.ndarray:
    """``a(-R)`` on a periodic FFT grid: index ``n -> (-n) mod N`` on every axis."""
    out = a
    for ax in axes:
        out = np.roll(np.flip(out, axis=ax), 1, axis=ax)
    return out


def _cartesian_entries(directional, natoms: int) -> dict:
    """``{(atom, axis): (dV, scale)}`` from unit-Cartesian directional entries."""
    entries = {}
    for d in directional:
        vec = np.asarray(d['displacement'], dtype=float)
        ax = int(np.argmax(np.abs(vec)))
        if abs(vec[ax]) < 1.0e-12 or not np.allclose(vec, np.eye(3)[ax] * vec[ax], atol=1.0e-10):
            raise ValueError(
                'the finite-difference source needs Cartesian derivatives of every atom '
                '(displacement_mode cartesian or random), got the direction %s' % vec.tolist()
            )
        entries[(int(d['sc_atom']), ax)] = (d['dV'], float(vec[ax]))
    missing = [(k, a) for k in range(natoms) for a in range(3) if (k, a) not in entries]
    if missing:
        raise ValueError(
            'derivatives missing for (atom, axis) %s...: the finite-difference source needs '
            'every atom of the cell displaced along x, y and z.' % missing[:6]
        )
    return entries


def fill_g_ReRp_from_fd(g_ReRp: np.ndarray, directional, natoms: int) -> None:
    """Write the cell derivatives into a ``g(R_e, R_p)`` buffer (Ry/bohr, ``R_p = 0``).

    Parameters
    ----------
    g_ReRp : ndarray, shape (nawf, nawf, 3 natoms, n1, n2, n3, 1, 1, 1)
        The (shared) cache buffer, overwritten.
    directional : list[dict]
        ``compute_dV`` entries ``{sc_atom, displacement, dV}`` with unit Cartesian
        displacements of every atom (``dV`` in eV/bohr, ``(nawf, nawf, n1, n2, n3, 1)``).
    natoms : int
        Atoms of the cell (``3 natoms`` Cartesian modes).
    """
    entries = _cartesian_entries(directional, natoms)
    nawf, _, nmodes = g_ReRp.shape[:3]
    grid = tuple(g_ReRp.shape[3:6])
    if nmodes != 3 * natoms or tuple(g_ReRp.shape[6:]) != (1, 1, 1):
        raise ValueError('g_ReRp must have 3 natoms modes and a single phonon cell.')
    for (k, a), (dV, scale) in entries.items():
        dV = np.asarray(dV)
        if dV.ndim != 6 or dV.shape[0] != nawf or tuple(dV.shape[2:5]) != grid:
            raise ValueError(
                'dV of atom %d has shape %s; expected (%d, %d, %s, nspin)'
                % (k, dV.shape, nawf, nawf, ', '.join(map(str, grid)))
            )
        if dV.shape[5] != 1:
            raise ValueError('spin-polarised finite-difference vertices are not supported.')
        g_ReRp[:, :, 3 * k + a, ..., 0, 0, 0] = flip_cells(dV[..., 0]) / (scale * RY_TO_EV)


def g_ReRp_from_fd(directional, natoms: int) -> np.ndarray:
    """``g(R_e, R_p)`` cache array of a cell from its finite-difference derivatives.

    Returns
    -------
    ndarray, shape (nawf, nawf, 3 natoms, n1, n2, n3, 1, 1, 1), complex
    """
    dV0 = np.asarray(directional[0]['dV'])
    nawf, grid = dV0.shape[0], tuple(dV0.shape[2:5])
    g = np.zeros((nawf, nawf, 3 * natoms) + grid + (1, 1, 1), dtype=complex)
    fill_g_ReRp_from_fd(g, directional, natoms)
    return g


def forces_from_output(path: str) -> np.ndarray | None:
    """Forces (Ry/bohr, ``(natoms, 3)``) of the last force block of a ``pw.x`` output."""
    if not os.path.isfile(path):
        return None
    with open(path, errors='replace') as fh:
        blocks = _FORCE_BLOCK.findall(fh.read())
    if not blocks:
        return None
    return np.array(_FORCE_ROW.findall(blocks[-1]), dtype=float)


def _output_path(edir: str, entry: dict) -> str:
    return os.path.join(edir, os.path.splitext(entry['input'])[0] + '.out')


def fd_force_constants(edir: str, manifest: dict | None = None, ridge: float = 1.0e-4):
    """Force constants of the cell from the forces of the displaced runs.

    Parameters
    ----------
    edir : str
        Electron-phonon directory (manifest, inputs and ``pw.x`` outputs).
    manifest : dict, optional
        The parsed manifest (read from ``edir`` by default).
    ridge : float
        Ridge of the force fit in the random-displacement mode.

    Returns
    -------
    ndarray, shape (3 natoms, 3 natoms), or None
        Symmetrised ``Phi[3 kappa + alpha, 3 kappa' + beta]`` in Ry/bohr^2 (no acoustic
        sum rule); ``None`` with a warning when the runs do not determine every row
        (an atom without ``+/-`` pairs along x, y, z, or outputs without forces).
    """
    if manifest is None:
        with open(os.path.join(edir, MANIFEST)) as fh:
            manifest = json.load(fh)
    disps = manifest['displacements']
    if manifest.get('displacement_mode') == 'random':
        from .random_displacements import NormalEquations, fit_force_constants, pair_configurations

        pairs = pair_configurations(disps)
        nat = pairs[0][2].shape[0]
        neq = NormalEquations(3 * nat, (3 * nat,), dtype=float)
        for ip, im, u in pairs:
            fp = forces_from_output(_output_path(edir, disps[ip]))
            fm = forces_from_output(_output_path(edir, disps[im]))
            if fp is None or fm is None or fp.shape != (nat, 3) or fm.shape != (nat, 3):
                continue
            neq.add(u.ravel(), (-0.5 * (fp - fm)).ravel())
        if neq.nconfig < 3 * nat:
            warnings.warn(
                'force constants: %d of %d random pairs have forces (need >= %d).'
                % (neq.nconfig, len(pairs), 3 * nat)
            )
            return None
        fc, _ = fit_force_constants(neq, [np.arange(nat)] * nat, ridge=ridge)
        return fc.transpose(0, 2, 1, 3).reshape(3 * nat, 3 * nat) / RY_BOHR2_TO_EV_ANG2

    from .dvscf_fd import _opposite_index

    rows = {}
    nat = None
    for i, d in enumerate(disps):
        if d.get('sc_atom') is None:
            continue
        vec = np.asarray(d['displacement'], dtype=float)
        ax = int(np.argmax(np.abs(vec)))
        if not np.allclose(vec, np.eye(3)[ax] * vec[ax], atol=1.0e-10):
            continue  # symmetry-adapted direction: not a Cartesian row
        j = _opposite_index(disps, i)
        if j is None or j < i:
            continue
        fi = forces_from_output(_output_path(edir, d))
        fj = forces_from_output(_output_path(edir, disps[j]))
        if fi is None or fj is None or fi.shape != fj.shape:
            continue
        nat = fi.shape[0]
        rows[(int(d['sc_atom']), ax)] = -((fi - fj) / (2.0 * vec[ax])).ravel()
    if nat is None or len(rows) < 3 * nat:
        warnings.warn(
            'force constants: %d of %s Cartesian +/- rows found in %s (every atom needs +/- '
            'pairs along x, y, z with forces in the pw.x outputs).'
            % (len(rows), 3 * nat if nat else '?', edir)
        )
        return None
    phi = np.zeros((3 * nat, 3 * nat))
    for (k, ax), row in rows.items():
        phi[3 * k + ax] = row
    return 0.5 * (phi + phi.T)


def fd_phonon_at_q(force_constants_ry, masses_amu, at, atom_positions=None):
    """``phonon_at_q`` callable of a cell from its Gamma force constants.

    Parameters
    ----------
    force_constants_ry : array_like, shape (3 natoms, 3 natoms)
        ``Phi`` in Ry/bohr^2 (the acoustic sum rule is imposed by the interpolator).
    masses_amu : array_like, shape (natoms,)
        Atomic masses (amu).
    at : array_like, shape (3, 3)
        Direct lattice vectors of the cell (rows, alat units).
    atom_positions : array_like, shape (natoms, 3), optional
        Crystal coordinates of the atoms (atom-pair Wigner-Seitz images).

    Returns
    -------
    callable
        ``phonon_at_q(q_cryst) -> (freq_thz, z)``; only ``q = 0`` is meaningful.
    """
    from .do_pao_eph_dense_q import _phonon_interp_from_force_constants

    phi = np.asarray(force_constants_ry, dtype=complex)
    masses_ry = np.asarray(masses_amu, dtype=float) * AMU_RY
    if phi.shape != (3 * masses_ry.size, 3 * masses_ry.size):
        raise ValueError('force constants %s do not match %d atoms.' % (phi.shape, masses_ry.size))
    return _phonon_interp_from_force_constants(
        phi[None, None, None], masses_ry, (1, 1, 1), np.asarray(at, dtype=float), atom_positions
    )


class _CellController:
    """Minimal data controller of the displaced-cell analysis (``opath`` only)."""

    def __init__(self, opath: str):
        self._arry = {}
        self._attr = {'opath': opath}
        self.rank = 0

    def data_dicts(self):
        return self._arry, self._attr


def fd_cell_derivatives(coupling_dir: str, **options) -> dict:
    """Cartesian derivatives of a cell from its displaced runs (``compute_dV``).

    Parameters
    ----------
    coupling_dir : str
        The electron-phonon directory of the displaced runs (manifest and saves).
    **options
        :data:`FD_DERIVATIVE_OPTIONS` of :func:`~PAOFLOW.elphon.dvscf_fd.compute_dV`
        (``fixed_basis`` defaults to ``True``).

    Returns
    -------
    dict
        The ``compute_dV`` result (``directional`` entries of every atom and direction).
    """
    from .dvscf_fd import compute_dV

    unknown = set(options) - set(FD_DERIVATIVE_OPTIONS)
    if unknown:
        raise ValueError('unknown finite-difference options %s' % sorted(unknown))
    options.setdefault('fixed_basis', True)
    edir = os.path.abspath(coupling_dir)
    controller = _CellController(os.path.dirname(edir))
    return compute_dV(controller, elphon_dir=os.path.basename(edir), **options)
