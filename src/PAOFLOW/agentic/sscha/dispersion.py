"""Phonon dispersions and fine-mesh stability checks from QE dynamical matrices (harmonic or SSCHA).

The matrices are a ``ph.x``-style set ``<prefix>.dyn0 ... dynN`` (the harmonic run, or the ``force_constants_<kind>``
export of ``PAOFLOW.sscha_force_constants``); the interpolation is PAOFLOW's ``q2r.x`` / ``matdyn.x`` equivalent
(:mod:`PAOFLOW.phonon.qe_force_constants`, atom-pair Wigner-Seitz images, simple acoustic sum rule). The analytic
long-range term of polar crystals is not added: LO-TO splittings are absent, which does not affect the stability
verdict of the soft modes.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

#: THz -> cm^-1.
THZ_TO_CM1 = 33.35641


def phonon_model(dyn_prefix: str) -> tuple[Callable, dict]:
    """Interpolator of a set of dynamical matrices.

    Parameters
    ----------
    dyn_prefix : str
        Path prefix ``<dir>/<stem>.dyn`` (``<stem>.dyn0`` lists the grid).

    Returns
    -------
    interpolate : callable
        ``q_cryst (3,) or (nq, 3) -> frequencies (cm^-1, imaginary negative)``, sorted per q.
    lattice : dict
        ``at``, ``bg`` (rows; alat and 2 pi / alat units), ``qgrid``, ``nat``.

    Raises
    ------
    ValueError
        If the prefix does not end in ``.dyn``.
    """
    from ...phonon.qe_force_constants import (
        phonon_interpolator,
        read_dynamical_matrices,
        real_space_force_constants,
    )

    directory, name = os.path.split(os.path.abspath(dyn_prefix))
    if not name.endswith('.dyn'):
        raise ValueError('the dynamical-matrix prefix must end in .dyn: %s' % dyn_prefix)
    data = read_dynamical_matrices(name[:-4], directory)
    phonon_at_q = phonon_interpolator(
        real_space_force_constants(data['force_constants_q']), data['masses_ry'], data['qgrid'], data['at'],
        atom_positions=data['tau_cryst'],
    )  # fmt: skip

    def interpolate(q_cryst: NDArray[np.float64]) -> NDArray[np.float64]:
        frequencies = np.asarray(phonon_at_q(np.asarray(q_cryst, dtype=float))[0]) * THZ_TO_CM1
        return np.sort(frequencies, axis=-1)

    lattice = dict(at=np.asarray(data['at']), bg=np.asarray(data['bg']), qgrid=tuple(data['qgrid']),
                   nat=len(data['masses_ry']))  # fmt: skip
    return interpolate, lattice


def dispersion(
    dyn_prefix: str,
    path: Sequence[Sequence[float]],
    labels: Sequence[str],
    npoints: int | Sequence[int] = 40,
) -> dict:
    """Frequencies along a path of straight segments in the Brillouin zone.

    Parameters
    ----------
    dyn_prefix : str
        Dynamical matrices (see :func:`phonon_model`).
    path : sequence of 3-vectors
        Vertices in crystal coordinates of the reciprocal lattice (e.g. fcc Gamma, X, W, K, Gamma, L).
    labels : sequence of str
        One label per vertex.
    npoints : int or sequence of int, optional
        Points per segment (one value for all, or one per segment).

    Returns
    -------
    dict
        ``distance`` (cumulative, 2 pi / alat), ``frequencies_cm1`` ``(npts, 3 nat)``, ``q_cryst``, ``ticks``
        (distance of each vertex) and ``labels``.

    Raises
    ------
    ValueError
        If ``labels`` and ``path`` differ in length or the path has fewer than two vertices.
    """
    vertices = np.asarray(path, dtype=float)
    if len(labels) != len(vertices) or len(vertices) < 2:
        raise ValueError('one label per vertex and at least two vertices are needed')
    counts = (
        [int(npoints)] * (len(vertices) - 1) if np.isscalar(npoints) else [int(n) for n in npoints]
    )
    interpolate, lattice = phonon_model(dyn_prefix)
    qs, distance, ticks, start = [], [], [0.0], 0.0
    for (a, b), n in zip(zip(vertices[:-1], vertices[1:]), counts):
        length = float(np.linalg.norm((b - a) @ lattice['bg']))
        steps = np.linspace(0.0, 1.0, n + 1)
        if qs:
            steps = steps[1:]
        qs.extend(a + s * (b - a) for s in steps)
        distance.extend(start + s * length for s in steps)
        start += length
        ticks.append(start)
    q_cryst = np.asarray(qs)
    return dict(distance=np.asarray(distance), frequencies_cm1=interpolate(q_cryst), q_cryst=q_cryst,
                ticks=np.asarray(ticks), labels=list(labels))  # fmt: skip


def grid_minimum(
    dyn_prefix: str, mesh: Sequence[int] = (12, 12, 12)
) -> tuple[NDArray[np.float64], float]:
    """Softest mode on a Gamma-centred q-mesh (the three acoustic modes at Gamma excluded).

    Parameters
    ----------
    dyn_prefix : str
        Dynamical matrices (see :func:`phonon_model`).
    mesh : sequence of int, optional
        q-mesh; finer than the supercell grid, so instabilities between its q-points show.

    Returns
    -------
    q_cryst : ndarray
        Crystal coordinates of the softest mode.
    frequency : float
        Its frequency (cm^-1, imaginary negative).
    """
    interpolate, _ = phonon_model(dyn_prefix)
    axes = [np.arange(int(n)) / int(n) for n in mesh]
    q = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    frequencies = interpolate(q)
    frequencies[0, :3] = np.inf  # Gamma acoustic
    index = np.unravel_index(np.argmin(frequencies), frequencies.shape)
    q_min = q[index[0]] - np.round(q[index[0]])
    return q_min, float(frequencies[index])
