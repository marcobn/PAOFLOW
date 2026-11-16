"""Charge shapes of the three-dimensional electrostatics: one spherical Gaussian per atom.

The bias-induced change of the electron number on an atom is spread over a normalised Gaussian
:math:`g(r) = (2\\pi\\sigma^2)^{-3/2} e^{-r^2/2\\sigma^2}`. Its width fixes the on-site Hartree energy: two such
Gaussians on the same site interact with :math:`1/(\\sqrt\\pi\\,\\sigma)` hartree. The parameter-free width matches the
Hartree self-energy of the valence pseudo-atomic density of the species,

.. math::

    \\sigma = \\frac{1}{\\sqrt\\pi\\,J},\\qquad J = \\iint \\frac{\\rho(\\mathbf r)\\rho(\\mathbf r')}{|\\mathbf r - \\mathbf r'|},
    \\qquad \\rho = \\sum_l w_l\\,\\frac{|u_l(r)|^2}{4\\pi r^2},

with :math:`u_l = rR_l` the pseudo-wavefunctions of the UPF file (``PP_PSWFC``) and :math:`w_l` their occupations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid

from PAOFLOW.inputs.read_upf import UPF


def hartree_self_energy(
    radii: NDArray[np.float64], radial_probability: NDArray[np.float64]
) -> float:
    """Hartree self-energy of a spherical charge distribution of one electron.

    Parameters
    ----------
    radii : ndarray, shape ``(n,)``
        Increasing radial mesh (bohr).
    radial_probability : ndarray, shape ``(n,)``
        :math:`u(r) = 4\\pi r^2\\rho(r)`; it is normalised here to :math:`\\int u\\,dr = 1`.

    Returns
    -------
    float
        :math:`J = \\iint \\rho\\rho'/|\\mathbf r - \\mathbf r'| = 2\\int dr\\, u(r)\\,Q(r)/r` (hartree), with
        :math:`Q(r) = \\int_0^r u`.

    Notes
    -----
    The hydrogen 1s density, :math:`u = 4r^2e^{-2r}`, gives :math:`J = 5/8`.
    """
    r = np.asarray(radii, dtype=float)
    u = np.asarray(radial_probability, dtype=float)
    u = u / np.trapezoid(u, r)
    enclosed = cumulative_trapezoid(u, r, initial=0.0)
    positive = r > 0.0
    return float(2.0 * np.trapezoid(u[positive] * enclosed[positive] / r[positive], r[positive]))


def gaussian_width_of_self_energy(self_energy: float) -> float:
    """Width of the Gaussian with the given Hartree self-energy.

    Parameters
    ----------
    self_energy : float
        :math:`J` (hartree).

    Returns
    -------
    float
        :math:`\\sigma = 1/(\\sqrt\\pi\\,J)` (bohr).
    """
    return 1.0 / (np.sqrt(np.pi) * self_energy)


def valence_charge_width(upf: UPF) -> float:
    """Gaussian width of the valence pseudo-atomic density of a UPF file.

    Parameters
    ----------
    upf : UPF
        Pseudopotential with ``PP_PSWFC`` (``upf.pswfc``: label, occupation, :math:`u = rR`) on ``upf.r``.

    Returns
    -------
    float
        :math:`\\sigma` (bohr), from the occupation-weighted density (equal weights when no shell is occupied).

    Raises
    ------
    ValueError
        Without pseudo-wavefunctions.
    """
    if not upf.pswfc:
        raise ValueError('the pseudopotential has no pseudo-wavefunctions (PP_PSWFC)')
    radii = np.asarray(upf.r, dtype=float)
    occupations = np.array([max(float(shell['occ']), 0.0) for shell in upf.pswfc])
    if occupations.sum() <= 0.0:
        occupations = np.ones_like(occupations)
    density = np.zeros_like(radii)
    for weight, shell in zip(occupations, upf.pswfc):
        probability = np.asarray(shell['wfc'], dtype=float)[: radii.size] ** 2
        density += weight * probability / np.trapezoid(probability, radii)
    return gaussian_width_of_self_energy(hartree_self_energy(radii, density))


def _pseudopotential_path(attributes: Mapping, pseudo_file: str) -> Path:
    fpath = attributes.get('fpath')
    if fpath and os.path.exists(os.path.join(fpath, pseudo_file)):
        return Path(fpath) / pseudo_file
    from PAOFLOW.transport.partition.orbital_inventory import _find_pseudopotential

    return _find_pseudopotential(dict(attributes), pseudo_file)


def atomic_charge_widths(
    elements: Sequence[str],
    charge_widths: float | Mapping[str, float] | None,
    species: Sequence[tuple[str, str]] | None = None,
    attributes: Mapping | None = None,
) -> NDArray[np.float64]:
    """Gaussian width of every atom.

    Parameters
    ----------
    elements : sequence of str
        Species label of every atom.
    charge_widths : float, mapping or None
        One width for all atoms (bohr), widths per species (the missing ones from the pseudopotentials), or None
        (all from the pseudopotentials).
    species : sequence of (label, pseudo_file), optional
        ``data_arrays['species']``.
    attributes : mapping, optional
        ``data_attributes``, to locate the UPF files (``fpath``, ``savedir``, ``workpath``).

    Returns
    -------
    ndarray, shape ``(n_atoms,)``
        :math:`\\sigma` (bohr).

    Raises
    ------
    ValueError
        For a non-positive width, or a species with neither a given width nor a pseudopotential.
    """
    if charge_widths is not None and not isinstance(charge_widths, Mapping):
        width = float(charge_widths)
        if width <= 0.0:
            raise ValueError(f'charge widths must be positive, got {width}')
        return np.full(len(elements), width)
    given = {str(key): float(value) for key, value in (charge_widths or {}).items()}
    if any(value <= 0.0 for value in given.values()):
        raise ValueError(f'charge widths must be positive, got {given}')
    pseudo_files = {str(label).strip(): str(name) for label, name in (species or [])}
    widths: dict[str, float] = {}
    for element in dict.fromkeys(str(e).strip() for e in elements):
        if element in given:
            widths[element] = given[element]
        elif element in pseudo_files and attributes is not None:
            upf = UPF(str(_pseudopotential_path(attributes, pseudo_files[element])))
            widths[element] = valence_charge_width(upf)
        else:
            raise ValueError(
                f'no charge width for {element!r}: give charge_widths (bohr) for the species without a '
                'pseudopotential'
            )
    return np.array([widths[str(e).strip()] for e in elements])
