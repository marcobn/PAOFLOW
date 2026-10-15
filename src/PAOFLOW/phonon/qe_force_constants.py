"""Interatomic force constants and phonon interpolation from QE dynamical matrices.

The PAOFLOW counterpart of Quantum ESPRESSO's ``q2r.x`` + ``matdyn.x``, working
directly on the ``<prefix>.dyn0`` ... ``<prefix>.dynN`` files written by ``ph.x``
(``ldisp = .true.``):

* :func:`read_dynamical_matrices` -- the dynamical matrices of every q-point of
  the coarse grid (all star members listed in the files) and the crystal;
* :func:`real_space_force_constants` -- the Fourier transform to the real-space
  force constants of the q-grid supercell (``q2r.x``);
* :func:`write_qe_force_constants` -- the QE ``flfrc`` file, readable by
  ``matdyn.x`` and :func:`PAOFLOW.elphon.qe_matdyn.read_qe_ifc`;
* :func:`phonon_interpolator` -- Wigner-Seitz interpolation of the dynamical
  matrix to any q with atom-pair images and the simple acoustic sum rule
  (``matdyn.x``, ``asr = 'simple'``);
* :func:`phonon_dispersion` and :func:`phonon_density_of_states` -- the band
  structure along the high-symmetry path of the Bravais lattice and the DOS;
* :func:`q2r_matdyn` -- the whole chain, writing ``<fname>_band.dat``,
  ``<fname>_band.labels`` and ``<fname>_dos.dat`` in the format of the phonopy
  route, plotted by :meth:`PAOFLOW.GPAO.GPAO.plot_phonons`.

The same interpolator serves the dense-q electron-phonon routes of
:mod:`PAOFLOW.elphon.do_pao_eph_dense_q`.  Polar materials need the
non-analytic (LO-TO) term, which is not included: a warning is issued when the
dynamical matrices carry Born effective charges.

Units: force constants in Ry/bohr^2, masses in QE Rydberg units (amu * 911.444),
lattice vectors in ``alat``, q in crystal coordinates unless stated, frequencies
in THz (or cm^-1 in the writers).
"""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

AMU_RY = 911.4442421
RY_TO_THZ = 3289.842
THZ_TO_CM1 = 33.35641
# Sign of the cell vector in the atom-pair separation of the force constants
# (see :func:`~PAOFLOW.elphon.elph_bloch._ws_lattice_pairs`).
PHONON_WS_SIGN = 1


# Headers that end a dynamical-matrix block in a *.dyn file.
_BLOCK_HEADERS = ('Dynamical', 'Diagonalizing', 'Dielectric', 'Effective Charges')


def _qe_float(token: str) -> float:
    return float(token.replace('D', 'E').replace('d', 'e'))


def read_dynamical_matrix_file(path: str) -> dict:
    """Parse one QE ``*.dyn`` file: crystal header and the matrices of its q-star.

    Parameters
    ----------
    path : str
        A ``<prefix>.dynN`` file of ``ph.x``.

    Returns
    -------
    dict
        ``ntyp``, ``nat``, ``ibrav``, ``celldm`` ``(6,)``, ``at`` ``(3, 3)``
        (rows, ``alat``; ``None`` unless ``ibrav = 0``), ``species_names``,
        ``species_masses_ry`` ``(ntyp,)``, ``atom_types`` ``(nat,)`` (0-based),
        ``masses_ry`` ``(nat,)``, ``tau_cart`` ``(nat, 3)`` (``alat``),
        ``q_cart`` (list of ``(3,)``, units of 2 pi / alat), ``force_constants``
        (list of ``(3 nat, 3 nat)`` complex, Ry/bohr^2, index ``3 na + alpha``)
        ``epsilon`` ``(3, 3)`` and ``born_charges`` ``(nat, 3, 3)`` (rows: field
        direction, as the ``Effective Charges E-U`` block) or ``None`` when the
        file has no dielectric block.
    """
    with open(path) as handle:
        lines = handle.read().splitlines()
    header = lines[2].split()
    ntyp, nat, ibrav = int(header[0]), int(header[1]), int(header[2])
    celldm = np.array([_qe_float(token) for token in header[3:9]])
    line = 3
    lattice = None
    if ibrav == 0:
        line += 1  # "Basis vectors"
        lattice = np.array([[_qe_float(t) for t in lines[line + i].split()[:3]] for i in range(3)])
        line += 3
    species_names, species_masses = [], np.empty(ntyp)
    for _ in range(ntyp):
        parts = lines[line].split("'")
        index = int(parts[0].split()[0]) - 1
        species_names.append(parts[1].strip())
        species_masses[index] = _qe_float(parts[2].split()[0])
        line += 1
    atom_types = np.empty(nat, dtype=int)
    tau_cart = np.empty((nat, 3))
    for _ in range(nat):
        tokens = lines[line].split()
        atom = int(tokens[0]) - 1
        atom_types[atom] = int(tokens[1]) - 1
        tau_cart[atom] = [_qe_float(t) for t in tokens[2:5]]
        line += 1

    nmode = 3 * nat
    q_pattern = re.compile(r'q\s*=\s*\(\s*([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)')
    q_points, matrices = [], []
    epsilon, born_charges = None, None
    k = line
    while k < len(lines):
        if 'Dielectric Tensor' in lines[k]:
            rows = [r for r in lines[k + 1 : k + 6] if r.strip()][:3]
            epsilon = np.array([[_qe_float(t) for t in r.split()[:3]] for r in rows])
        if 'Effective Charges E-U' in lines[k]:
            born_charges = np.zeros((nat, 3, 3))
            j = k + 1
            for atom in range(nat):
                while 'atom' not in lines[j]:
                    j += 1
                born_charges[atom] = [
                    [_qe_float(t) for t in lines[j + 1 + a].split()[:3]] for a in range(3)
                ]
                j += 4
        if 'Dynamical' in lines[k] and 'cartesian' in lines[k]:
            j = k + 1
            while j < len(lines) and 'q =' not in lines[j]:
                j += 1
            match = q_pattern.search(lines[j])
            q_points.append(np.array([_qe_float(match.group(t)) for t in (1, 2, 3)]))
            matrix = np.zeros((nmode, nmode), dtype=complex)
            j += 1
            while j < len(lines):
                tokens = lines[j].split()
                if len(tokens) == 2 and all(t.isdigit() for t in tokens):
                    na, nb = int(tokens[0]) - 1, int(tokens[1]) - 1
                    for a in range(3):
                        row = [_qe_float(x) for x in lines[j + 1 + a].split()]
                        for b in range(3):
                            matrix[3 * na + a, 3 * nb + b] = row[2 * b] + 1j * row[2 * b + 1]
                    j += 4
                elif any(key in lines[j] for key in _BLOCK_HEADERS):
                    break
                else:
                    j += 1
            matrices.append(matrix)
            k = j
            continue
        k += 1
    return {
        'ntyp': ntyp,
        'nat': nat,
        'ibrav': ibrav,
        'celldm': celldm,
        'at': lattice,
        'species_names': species_names,
        'species_masses_ry': species_masses,
        'atom_types': atom_types,
        'masses_ry': species_masses[atom_types],
        'tau_cart': tau_cart,
        'q_cart': q_points,
        'force_constants': matrices,
        'epsilon': epsilon,
        'born_charges': born_charges,
    }


def read_dynamical_matrices(prefix: str, directory: str = '.') -> dict:
    """All dynamical matrices of a ``ph.x`` q-grid run, placed on the full grid.

    Parameters
    ----------
    prefix : str
        ``fildyn`` of ``ph.x`` (the files are ``<prefix>.dyn0`` ... ``<prefix>.dynN``).
    directory : str, optional
        Directory of the files.

    Returns
    -------
    dict
        The header of :func:`read_dynamical_matrix_file` plus ``qgrid``,
        ``alat`` (bohr), ``at`` and ``bg`` (rows; ``alat`` and 2 pi / alat
        units), ``tau_cryst`` ``(nat, 3)``, ``dyn_paths`` and
        ``force_constants_q`` ``(nq1, nq2, nq3, 3 nat, 3 nat)``, ``epsilon``,
        ``born_charges`` and ``polar`` (:func:`is_polar`).

    Raises
    ------
    ValueError
        If the q-stars of the files do not cover the full grid.

    Notes
    -----
    ``<prefix>.dyn0`` lists the grid and the irreducible q-points; each
    ``<prefix>.dynN`` holds the whole star of its irreducible q, so the files
    cover the full grid, as ``q2r.x`` requires.
    """
    from ..inputs.lattice_format import lattice_format_QE

    base = os.path.join(directory, prefix)
    with open(base + '.dyn0') as handle:
        tokens = handle.read().split()
    qgrid = tuple(int(n) for n in tokens[:3])
    nirr = int(tokens[3])
    dyn_paths = ['%s.dyn%d' % (base, i + 1) for i in range(nirr)]
    first = read_dynamical_matrix_file(dyn_paths[0])
    alat = float(first['celldm'][0])
    if first['ibrav'] == 0:
        at = np.asarray(first['at'], dtype=float)
    else:
        at = np.asarray(lattice_format_QE(first['ibrav'], first['celldm']), dtype=float) / alat
    bg = np.linalg.inv(at).T

    nmode = 3 * first['nat']
    force_constants_q = np.zeros(qgrid + (nmode, nmode), dtype=complex)
    filled = np.zeros(qgrid, dtype=bool)
    epsilon, born_charges = None, None
    grid = np.asarray(qgrid)
    for path in dyn_paths:
        parsed = first if path == dyn_paths[0] else read_dynamical_matrix_file(path)
        if parsed['born_charges'] is not None:
            epsilon, born_charges = parsed['epsilon'], parsed['born_charges']
        for q_cart, matrix in zip(parsed['q_cart'], parsed['force_constants']):
            q_cryst = at @ np.asarray(q_cart, dtype=float)
            label = tuple(np.round(q_cryst * grid).astype(int) % grid)
            if not filled[label]:
                force_constants_q[label] = matrix
                filled[label] = True
    if not filled.all():
        raise ValueError(
            'the dyn files cover only %d of the %d q-points of the %s grid.'
            % (filled.sum(), filled.size, 'x'.join(map(str, qgrid)))
        )
    result = dict(first)
    result.update({
        'qgrid': qgrid,
        'alat': alat,
        'at': at,
        'bg': bg,
        'tau_cryst': first['tau_cart'] @ np.linalg.inv(at),
        'dyn_paths': dyn_paths,
        'force_constants_q': force_constants_q,
        'epsilon': epsilon,
        'born_charges': born_charges,
        'polar': is_polar(born_charges),
    })  # fmt: skip
    result.pop('q_cart')
    result.pop('force_constants')
    return result


def is_polar(born_charges: NDArray[np.float64] | None, tolerance: float = 0.05) -> bool:
    """Whether Born effective charges are physically non-zero.

    Parameters
    ----------
    born_charges : ndarray, shape ``(nat, 3, 3)``, or None
        Born effective charges (units of e).
    tolerance : float, optional
        Largest element regarded as numerical noise, after the charge-neutrality
        sum rule.

    Returns
    -------
    bool
        ``True`` when, after imposing :math:`\\sum_a Z^*_a = 0` (``zasr =
        'simple'``), an element exceeds ``tolerance``.
    """
    if born_charges is None:
        return False
    charges = np.asarray(born_charges, dtype=float)
    neutral = charges - charges.mean(axis=0, keepdims=True)
    return bool(np.abs(neutral).max() > tolerance)


def real_space_force_constants(
    force_constants_q: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Real-space force constants of the q-grid supercell (``q2r.x``).

    Parameters
    ----------
    force_constants_q : ndarray, shape ``(nq1, nq2, nq3, 3 nat, 3 nat)``
        Dynamical matrices ``C(q)`` (not mass-weighted) on the full q-grid.

    Returns
    -------
    ndarray, shape ``(nq1, nq2, nq3, 3 nat, 3 nat)``, complex
        :math:`C(\\mathbf n)`, indexed by the supercell cell ``n``.

    Notes
    -----
    .. math::

        C(\\mathbf n) = \\frac{1}{N_q}\\sum_{\\mathbf q} C(\\mathbf q)\\,
            e^{-2\\pi i\\,\\mathbf q\\cdot\\mathbf n},
        \\qquad
        C(\\mathbf q) = \\sum_{\\mathbf n} C(\\mathbf n)\\, e^{2\\pi i\\,\\mathbf q\\cdot\\mathbf n},

    with q in crystal coordinates; the cell ``n`` is QE's ``-R``
    (:func:`write_qe_force_constants`).
    """
    nq = int(np.prod(force_constants_q.shape[:3]))
    return np.fft.fftn(force_constants_q, axes=(0, 1, 2)) / nq


def apply_acoustic_sum_rule(
    force_constants_r: NDArray[np.complex128], nat: int
) -> NDArray[np.complex128]:
    """Impose the simple acoustic sum rule on real-space force constants (``asr = 'simple'``).

    Parameters
    ----------
    force_constants_r : ndarray, shape ``(nq1, nq2, nq3, 3 nat, 3 nat)``
        Output of :func:`real_space_force_constants`.
    nat : int
        Atoms per cell.

    Returns
    -------
    ndarray
        A corrected copy.

    Notes
    -----
    :math:`C_{\\alpha\\beta}(\\mathbf 0; a, a) \\mathrel{-}= \\sum_{\\mathbf n, b}
    C_{\\alpha\\beta}(\\mathbf n; a, b)`: a rigid translation costs no energy, so
    the acoustic frequencies vanish at Gamma.
    """
    corrected = np.array(force_constants_r, copy=True)
    total = force_constants_r.sum(axis=(0, 1, 2))  # C(q = 0)
    for a in range(3):
        for b in range(3):
            for na in range(nat):
                row_sum = sum(total[3 * na + a, 3 * nb + b] for nb in range(nat))
                corrected[0, 0, 0, 3 * na + a, 3 * na + b] -= row_sum
    return corrected


def write_qe_force_constants(
    path: str, dynamical_matrices: dict, force_constants_r: NDArray[np.complex128]
) -> str:
    """Write the real-space force constants in QE's ``flfrc`` format (``q2r.x`` output).

    Parameters
    ----------
    path : str
        Output file (e.g. ``Si2.fc``).
    dynamical_matrices : dict
        Output of :func:`read_dynamical_matrices` (crystal header).
    force_constants_r : ndarray, shape ``(nq1, nq2, nq3, 3 nat, 3 nat)``
        Output of :func:`real_space_force_constants`, before the acoustic sum
        rule (as ``q2r.x``; ``matdyn.x`` applies it).

    Returns
    -------
    str
        ``path``.

    Notes
    -----
    QE stores :math:`\\Phi(\\mathbf R)` with
    :math:`C(\\mathbf q) = \\sum_{\\mathbf R}\\Phi(\\mathbf R) e^{-i\\mathbf q\\cdot\\mathbf R}`,
    i.e. :math:`\\Phi(\\mathbf m) = C(-\\mathbf m)` of
    :func:`real_space_force_constants`.  Born effective charges are not
    written (the ``F`` flag).
    """
    d = dynamical_matrices
    nat = int(d['nat'])
    nr = d['qgrid']
    with open(path, 'w') as handle:
        handle.write('%3d%5d%3d' % (d['ntyp'], nat, d['ibrav']))
        handle.write(''.join('%11.7f' % c for c in d['celldm']) + '\n')
        if d['ibrav'] == 0:
            for row in d['at']:
                handle.write('  ' + ''.join('%15.9f' % x for x in row) + '\n')
        for i, (name, mass) in enumerate(zip(d['species_names'], d['species_masses_ry'])):
            handle.write("%5d  '%-3s'%20.10f\n" % (i + 1, name, mass))
        for atom in range(nat):
            handle.write('%5d%5d' % (atom + 1, d['atom_types'][atom] + 1))
            handle.write(''.join('%18.10f' % x for x in d['tau_cart'][atom]) + '\n')
        handle.write(' F\n')
        handle.write('%4d%4d%4d\n' % tuple(nr))
        for a in range(3):
            for b in range(3):
                for na in range(nat):
                    for nb in range(nat):
                        handle.write('%4d%4d%4d%4d\n' % (a + 1, b + 1, na + 1, nb + 1))
                        for m3 in range(nr[2]):
                            for m2 in range(nr[1]):
                                for m1 in range(nr[0]):
                                    value = force_constants_r[
                                        -m1 % nr[0], -m2 % nr[1], -m3 % nr[2],
                                        3 * na + a, 3 * nb + b,
                                    ].real  # fmt: skip
                                    handle.write(
                                        '%4d%4d%4d  %18.11E\n' % (m1 + 1, m2 + 1, m3 + 1, value)
                                    )
    return path


def phonon_interpolator(
    force_constants_r: NDArray[np.complex128],
    masses_ry: NDArray[np.float64],
    qgrid: Sequence[int],
    at: NDArray[np.float64],
    atom_positions: NDArray[np.float64] | None = None,
    acoustic_sum_rule: bool = True,
) -> Callable[[NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.complex128]]]:
    """Wigner-Seitz interpolator of the dynamical matrix (``matdyn.x``).

    Parameters
    ----------
    force_constants_r : ndarray, shape ``(nq1, nq2, nq3, 3 nat, 3 nat)``
        Output of :func:`real_space_force_constants`.
    masses_ry : ndarray, shape ``(nat,)``
        Atomic masses in QE Rydberg units.
    qgrid : sequence of int
        The coarse q-grid.
    at : ndarray, shape ``(3, 3)``
        Direct lattice vectors (rows, ``alat``).
    atom_positions : ndarray, shape ``(nat, 3)``, optional
        Atomic positions (crystal coordinates): atom-pair Wigner-Seitz images, as
        ``matdyn.x``, which keep the dispersion symmetric in cells with several
        atoms.  ``None`` uses single-site images.
    acoustic_sum_rule : bool, optional
        Apply :func:`apply_acoustic_sum_rule` first (``asr = 'simple'``).

    Returns
    -------
    callable
        ``phonon_at_q(q_cryst)``: for ``q_cryst`` of shape ``(3,)`` it returns
        ``(freq_thz (nmode,), z (nmode, nmode))``, the frequencies (negative for
        unstable modes) and the mass-weighted eigenvectors as rows; for shape
        ``(nq, 3)`` the arrays gain a leading q axis.
    """
    from ..elphon.elph_bloch import _ws_lattice, _ws_lattice_pairs

    qgrid = tuple(int(n) for n in qgrid)
    masses_ry = np.asarray(masses_ry, dtype=float)
    nat = masses_ry.size
    cells_r = (
        apply_acoustic_sum_rule(force_constants_r, nat) if acoustic_sum_rule else force_constants_r
    )
    inv_sqrt_m = 1.0 / np.sqrt(np.repeat(masses_ry, 3))
    mass_weight = np.outer(inv_sqrt_m, inv_sqrt_m)  # D = C / sqrt(Ma Mb)
    if atom_positions is None:
        Nint_p, W_p, Midx_p = _ws_lattice(qgrid, at)
        cells = cells_r[Midx_p[:, 0], Midx_p[:, 1], Midx_p[:, 2]]  # (nws, nmode, nmode)
    else:
        mode_positions = np.repeat(np.asarray(atom_positions, dtype=float), 3, axis=0)
        Nint_p, W_pairs, Midx_p = _ws_lattice_pairs(qgrid, at, mode_positions, sign=PHONON_WS_SIGN)
        cells = cells_r[Midx_p[:, 0], Midx_p[:, 1], Midx_p[:, 2]] * W_pairs
        W_p = np.ones(Nint_p.shape[0])
    nmode = cells.shape[1]

    def phonon_at_q(q_cryst):
        q_cryst = np.asarray(q_cryst, dtype=float)
        if q_cryst.ndim == 1:
            phase = W_p * np.exp(2j * np.pi * (q_cryst @ Nint_p.T))
            C = np.tensordot(phase, cells, axes=([0], [0]))  # (nmode, nmode)
            D = 0.5 * (C + C.conj().T) * mass_weight  # mass-weighted -> eig = omega^2 (Ry^2)
            w2, ev = np.linalg.eigh(D)
            freq_thz = np.sign(w2) * np.sqrt(np.abs(w2)) * RY_TO_THZ
            return freq_thz, ev.T  # (nmode,), (nmode, ncart)
        phase = W_p[None, :] * np.exp(2j * np.pi * (q_cryst @ Nint_p.T))  # (nq, nws)
        C = (phase @ cells.reshape(cells.shape[0], -1)).reshape(-1, nmode, nmode)
        D = 0.5 * (C + np.conj(np.transpose(C, (0, 2, 1)))) * mass_weight
        w2, ev = np.linalg.eigh(D)
        freq_thz = np.sign(w2) * np.sqrt(np.abs(w2)) * RY_TO_THZ
        return freq_thz, np.transpose(ev, (0, 2, 1))

    return phonon_at_q


def high_symmetry_path(
    ibrav: int, alat: float, at: NDArray[np.float64]
) -> tuple[list[list[NDArray[np.float64]]], list[str]]:
    """High-symmetry q-path of a Bravais lattice (PAOFLOW's band path).

    Parameters
    ----------
    ibrav : int
        QE Bravais-lattice index (``ibrav = 0`` has no default path).
    alat : float
        Lattice parameter (bohr).
    at : ndarray, shape ``(3, 3)``
        Direct lattice vectors (rows, ``alat``).

    Returns
    -------
    segments : list of list of ndarray
        Continuous segments, each a list of crystal q-points.
    labels : list of str
        Display labels, one per point.
    """
    from ..phonon.do_phonopy import _qe_path_to_phonopy
    from ..spectrum.kpnts_interpolation_mesh import _getHighSymPoints

    if int(ibrav) == 0:
        raise ValueError('ibrav = 0 has no default path: pass the path explicitly.')
    special_points, band_path = _getHighSymPoints(int(ibrav), alat, np.asarray(at, dtype=float))
    segments, labels = _qe_path_to_phonopy(special_points, band_path)
    return [[np.asarray(p, dtype=float) for p in segment] for segment in segments], labels


def phonon_dispersion(
    phonon_at_q: Callable,
    bg: NDArray[np.float64],
    segments: Sequence[Sequence[NDArray[np.float64]]],
    labels: Sequence[str] | None = None,
    npoints: int = 51,
) -> dict:
    """Phonon frequencies along a q-path.

    Parameters
    ----------
    phonon_at_q : callable
        Output of :func:`phonon_interpolator`.
    bg : ndarray, shape ``(3, 3)``
        Reciprocal lattice vectors (rows, 2 pi / alat), for the path length.
    segments : sequence of sequence of ndarray
        Continuous path segments of crystal q-points (``|`` breaks between
        segments), e.g. from :func:`high_symmetry_path`.
    labels : sequence of str, optional
        One label per point of ``segments``.
    npoints : int, optional
        Points per straight line, end points included.

    Returns
    -------
    dict
        ``distance`` ``(nq,)`` (2 pi / alat), ``frequency_thz`` ``(nq, nmode)``,
        ``q_cryst`` ``(nq, 3)``, ``tick_distance`` and ``tick_label`` (one per
        point; two labels share a distance at a break).
    """
    q_list, distances, ticks = [], [], []
    distance = 0.0
    for segment in segments:
        segment = [np.asarray(p, dtype=float) for p in segment]
        ticks.append(distance)
        for start, stop in zip(segment[:-1], segment[1:]):
            fractions = np.linspace(0.0, 1.0, npoints)
            line = start[None, :] + fractions[:, None] * (stop - start)[None, :]
            length = np.linalg.norm((stop - start) @ bg)
            q_list.append(line)
            distances.append(distance + fractions * length)
            distance += length
            ticks.append(distance)
    q_cryst = np.concatenate(q_list)
    frequency_thz, _ = phonon_at_q(q_cryst)
    return {
        'distance': np.concatenate(distances),
        'frequency_thz': frequency_thz,
        'q_cryst': q_cryst,
        'tick_distance': np.array(ticks),
        'tick_label': list(labels) if labels is not None else [''] * len(ticks),
    }


def phonon_density_of_states(
    phonon_at_q: Callable,
    mesh: Sequence[int] = (24, 24, 24),
    sigma_thz: float = 0.1,
    frequency_step_thz: float = 0.02,
    chunk: int = 4096,
) -> dict:
    """Phonon density of states on a Gamma-centred q-mesh (Gaussian broadening).

    Parameters
    ----------
    phonon_at_q : callable
        Output of :func:`phonon_interpolator`.
    mesh : sequence of int, optional
        q-mesh.
    sigma_thz : float, optional
        Gaussian width (THz).
    frequency_step_thz : float, optional
        Spacing of the frequency grid (THz).
    chunk : int, optional
        q-points per batch.

    Returns
    -------
    dict
        ``frequency_thz`` ``(nfreq,)`` and ``dos`` ``(nfreq,)`` (states/THz per
        cell, integrating to ``3 nat``).

    Notes
    -----
    :math:`g(\\nu) = \\frac{1}{N_q}\\sum_{\\mathbf q j}\\frac{1}{\\sigma\\sqrt{2\\pi}}
    e^{-(\\nu-\\nu_{\\mathbf q j})^2/2\\sigma^2}`.
    """
    axes = [np.arange(n) / n for n in mesh]
    q_mesh = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    frequencies = np.concatenate(
        [phonon_at_q(q_mesh[s : s + chunk])[0] for s in range(0, q_mesh.shape[0], chunk)]
    ).ravel()
    low = min(0.0, frequencies.min()) - 5.0 * sigma_thz
    high = frequencies.max() + 5.0 * sigma_thz
    grid = np.arange(low, high + frequency_step_thz, frequency_step_thz)
    dos = np.zeros_like(grid)
    norm = 1.0 / (sigma_thz * np.sqrt(2.0 * np.pi) * q_mesh.shape[0])
    for s in range(0, frequencies.size, chunk):
        diff = grid[:, None] - frequencies[None, s : s + chunk]
        dos += norm * np.exp(-0.5 * (diff / sigma_thz) ** 2).sum(axis=1)
    return {'frequency_thz': grid, 'dos': dos}


def write_phonon_band_files(
    outdir: str, fname: str, dispersion: dict, units: str = 'THz'
) -> list[str]:
    """Write ``<fname>_band.dat`` and ``<fname>_band.labels`` (format of the phonopy route).

    Parameters
    ----------
    outdir : str
        Output directory.
    fname : str
        File stem.
    dispersion : dict
        Output of :func:`phonon_dispersion`.
    units : {'THz', 'cm-1'}, optional
        Frequency unit.

    Returns
    -------
    list of str
        The written files, read by :meth:`PAOFLOW.GPAO.GPAO.plot_phonons`.
    """
    scale = _unit_scale(units)
    os.makedirs(outdir, exist_ok=True)
    band = os.path.join(outdir, fname + '_band.dat')
    rows = np.column_stack([dispersion['distance'], dispersion['frequency_thz'] * scale])
    np.savetxt(band, rows, fmt='% 16.8e', header='distance  frequencies(%s)' % units)
    labels = os.path.join(outdir, fname + '_band.labels')
    with open(labels, 'w') as handle:
        handle.write('# tick_distance  label\n')
        for d, label in zip(dispersion['tick_distance'], dispersion['tick_label']):
            handle.write('% 16.8e  %s\n' % (d, label))
    return [band, labels]


def write_phonon_dos_file(outdir: str, fname: str, dos: dict, units: str = 'THz') -> str:
    """Write ``<fname>_dos.dat`` (frequency and total DOS, format of the phonopy route).

    Parameters
    ----------
    outdir : str
        Output directory.
    fname : str
        File stem.
    dos : dict
        Output of :func:`phonon_density_of_states`.
    units : {'THz', 'cm-1'}, optional
        Frequency unit (the DOS is per unit of it).

    Returns
    -------
    str
        The written file.
    """
    scale = _unit_scale(units)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, fname + '_dos.dat')
    rows = np.column_stack([dos['frequency_thz'] * scale, dos['dos'] / scale])
    np.savetxt(path, rows, fmt='% 16.8e', header='frequency(%s)  total_dos' % units)
    return path


def _unit_scale(units: str) -> float:
    if str(units).lower() in ('cm-1', 'cm^-1', 'cm'):
        return THZ_TO_CM1
    if str(units).lower() == 'thz':
        return 1.0
    raise ValueError("units must be 'THz' or 'cm-1', got %r" % units)


def q2r_matdyn(
    prefix: str,
    directory: str = '.',
    outdir: str = '.',
    fname: str = 'phonon',
    force_constants_file: str | None = None,
    segments: Sequence[Sequence[NDArray[np.float64]]] | None = None,
    labels: Sequence[str] | None = None,
    npoints: int = 51,
    mesh: Sequence[int] = (24, 24, 24),
    sigma_thz: float = 0.1,
    units: str = 'THz',
    acoustic_sum_rule: bool = True,
) -> dict:
    """``q2r.x`` + ``matdyn.x``: dispersion and DOS from the ``ph.x`` dynamical matrices.

    Parameters
    ----------
    prefix : str
        ``fildyn`` of ``ph.x``.
    directory : str, optional
        Directory of the ``<prefix>.dyn*`` files.
    outdir : str, optional
        Output directory.
    fname : str, optional
        Stem of the output files.
    force_constants_file : str, optional
        Also write the real-space force constants in QE ``flfrc`` format
        (e.g. ``'Si2.fc'``, relative to ``outdir``).
    segments, labels : optional
        q-path (crystal coordinates) and labels; default: the high-symmetry path
        of the Bravais lattice (:func:`high_symmetry_path`).
    npoints : int, optional
        Points per straight line of the path.
    mesh : sequence of int, optional
        q-mesh of the DOS; ``None`` skips the DOS.
    sigma_thz : float, optional
        Gaussian width of the DOS (THz).
    units : {'THz', 'cm-1'}, optional
        Frequency unit of the written files.
    acoustic_sum_rule : bool, optional
        Impose the simple acoustic sum rule.

    Returns
    -------
    dict
        ``dynamical_matrices``, ``dispersion``, ``dos`` (or ``None``) and
        ``files`` (written paths).
    """
    dyn = read_dynamical_matrices(prefix, directory)
    if dyn['polar']:
        warnings.warn(
            'The dynamical matrices carry non-zero Born effective charges: the '
            'non-analytic (LO-TO) term is not included, so the LO branches near Gamma '
            'are not reliable.',
            stacklevel=2,
        )
    force_constants_r = real_space_force_constants(dyn['force_constants_q'])
    os.makedirs(outdir, exist_ok=True)
    files = []
    if force_constants_file:
        files.append(
            write_qe_force_constants(
                os.path.join(outdir, force_constants_file), dyn, force_constants_r
            )
        )
    phonon_at_q = phonon_interpolator(
        force_constants_r, dyn['masses_ry'], dyn['qgrid'], dyn['at'],
        atom_positions=dyn['tau_cryst'], acoustic_sum_rule=acoustic_sum_rule,
    )  # fmt: skip
    if segments is None:
        segments, labels = high_symmetry_path(dyn['ibrav'], dyn['alat'], dyn['at'])
    dispersion = phonon_dispersion(phonon_at_q, dyn['bg'], segments, labels, npoints)
    files += write_phonon_band_files(outdir, fname, dispersion, units)
    dos = None
    if mesh is not None:
        dos = phonon_density_of_states(phonon_at_q, mesh, sigma_thz)
        files.append(write_phonon_dos_file(outdir, fname, dos, units))
    return {'dynamical_matrices': dyn, 'dispersion': dispersion, 'dos': dos, 'files': files}
