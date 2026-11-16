"""High-level basis generator: pseudopotential -> radial shells, in memory or as ``BASIS_PS/<elem>/<shell>.dat`` files.

:func:`generate_basis_shells` is the in-memory core (one radial function per requested shell,
or per ``j`` for spin-orbit pseudopotentials), used by the ``projections()`` generator route of
:mod:`PAOFLOW.projection.radial_shell_basis`; :func:`generate_basis_for_pseudo` writes the same
functions to disk for the table route (``paoflow-genbasis-ps``). Both accept a UPF file (QE) or
a POTCAR dataset (VASP, :mod:`PAOFLOW.inputs.read_potcar`).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from glob import glob

import numpy as np
from numpy.typing import NDArray

from ..inputs.basis_presets import (
    extended_augmentation,
    standard_augmentation,
)
from .radial import is_frozen_core_shell, pseudize_shell

_L_INDEX = {'S': 0, 'P': 1, 'D': 2, 'F': 3}


@dataclass
class GeneratedShell:
    """One radial function of the generator.

    Attributes
    ----------
    label : str
        Shell label (``'3D'``).
    l : int
        Angular momentum.
    j : float or None
        Total angular momentum of a j-resolved function (fully relativistic pseudopotentials),
        ``None`` for a scalar one (also the degeneracy-weighted j-average written as the scalar
        file of a spin-orbit pseudopotential).
    r : ndarray
        Uniform mesh (bohr).
    wfc : ndarray
        ``u(r) = r R(r)``, normalised as :func:`PAOFLOW.basis_gen.radial.pseudize_shell`.
    energy : float
        Solver eigenvalue (Hartree), ``nan`` for a stored pseudo-orbital.
    filename : str
        Name of the table file of this function (``'3D.dat'``, ``'3D_j3.dat'``).
    """

    label: str
    l: int
    j: float | None
    r: NDArray[np.float64]
    wfc: NDArray[np.float64]
    energy: float
    filename: str

    @property
    def from_pseudo_orbital(self) -> bool:
        """Whether the function is a stored pseudo-orbital rather than a solver state."""
        return bool(np.isnan(self.energy))


def _minimal_shells_from_upf(upf):
    """Return the list of unique '<n><L>' labels carried by the pseudo-orbitals."""
    seen = []
    for c in upf.pswfc:
        lab = c['label'].upper()
        if lab not in seen:
            seen.append(lab)
    return seen


def _default_shells(upf, preset='extended'):
    """Resolve the full shell list (minimal + augmentation) for the preset."""
    minimal = _minimal_shells_from_upf(upf)
    # UPF pads single-letter symbols (e.g. ' O'); the augmentation rules key
    # the periodic-table block on the bare symbol, so strip before lookup.
    elem = upf.element.strip()
    if preset == 'minimal':
        return list(minimal)
    if preset == 'standard':
        return list(minimal) + standard_augmentation(elem, minimal)
    if preset == 'extended':
        return list(minimal) + extended_augmentation(elem, minimal)
    raise ValueError(f"unknown preset '{preset}'")


def _write_two_col(path, r, wfc):
    """Write a 2-column (r, wfc) ASCII file compatible with BASIS/."""
    with open(path, 'w') as f:
        for ri, wi in zip(r, wfc):
            f.write(f'{ri:24.15e} {wi:24.15e}\n')


def _j_average(R_minus, R_plus, l):
    """Degeneracy-weighted j-average of two scalar radial functions."""
    if l == 0:
        return R_plus
    Jm = 2 * (l - 0.5) + 1
    Jp = 2 * (l + 0.5) + 1
    return (Jm * R_minus + Jp * R_plus) / (Jm + Jp)


def _load_dataset(pseudo):
    """A parsed pseudopotential from a path (UPF or POTCAR) or the object itself."""
    if isinstance(pseudo, (str, os.PathLike)):
        from ..inputs.pseudopotentials import read_pseudopotential_file

        return read_pseudopotential_file(os.fspath(pseudo))
    return pseudo


def generate_basis_shells(
    pseudo,
    shells: list[str],
    r_box: float | None = None,
    n_points: int = 2000,
    prefer_pswfc: bool = True,
) -> tuple[list[GeneratedShell], list[tuple[str, str]]]:
    """Radial functions of the requested shells of one pseudopotential, in memory.

    Parameters
    ----------
    pseudo : UPF, PotcarDataset or str
        The pseudopotential (a parsed object, or the path of a UPF file / POTCAR).
    shells : list of str
        Shell labels (``['3S', '3P', '3D']``).
    r_box : float, optional
        Confining box radius (bohr). Defaults to ``min(r[-1], 10)`` of the radial mesh.
    n_points : int
        Solver mesh size (interior points = n_points - 1).
    prefer_pswfc : bool
        Return the shells the pseudopotential carries as pseudo-orbitals as stored
        (:func:`~PAOFLOW.basis_gen.radial.pseudize_shell`); ``False`` solves every shell.

    Returns
    -------
    generated : list of GeneratedShell
        In the order of ``shells``; a shell of a fully relativistic pseudopotential with
        ``l > 0`` gives three entries (``j = l - 1/2``, ``j = l + 1/2`` and their
        degeneracy-weighted average with ``j = None``).
    skipped : list of tuple
        ``(label, reason)`` of the shells that could not be generated (below the lowest
        pseudo-atom level, no physical state, or a frozen-core shell), each also reported
        with a :class:`RuntimeWarning`.

    Notes
    -----
    Every function of a pseudopotential is returned on the same uniform mesh, which the
    position-dipole tables require. The channel solutions are cached across the shells of
    one call (one generalized eigenproblem per ``(l, j)`` channel).
    """
    upf = _load_dataset(pseudo)
    elem = upf.element.strip()
    if getattr(upf, 'has_augmentation', False) and getattr(upf, 'version', 2) != 2:
        raise NotImplementedError(
            f'UPF v{upf.version} ultrasoft/PAW augmentation parsing is not implemented; '
            f'got {getattr(upf, "source", pseudo)!r}.'
        )
    so = bool(getattr(upf, 'has_spinorbit', False))
    cache: dict = {}
    generated: list[GeneratedShell] = []
    skipped: list[tuple[str, str]] = []
    for label in shells:
        if len(label) < 2 or label[1].upper() not in _L_INDEX:
            raise ValueError(f'shell label {label!r} not understood')
        n = int(label[0])
        l = _L_INDEX[label[1].upper()]
        try:
            if so and l > 0:
                # Solve both j = l-1/2 and j = l+1/2; the j-resolved functions and a
                # degeneracy-weighted j-averaged scalar function for fallback use.
                r, u_minus, e_minus = pseudize_shell(
                    upf, n, l, j=l - 0.5, r_box=r_box, n_points=n_points,
                    prefer_pswfc=prefer_pswfc, cache=cache,
                )  # fmt: skip
                _, u_plus, e_plus = pseudize_shell(
                    upf, n, l, j=l + 0.5, r_box=r_box, n_points=n_points,
                    prefer_pswfc=prefer_pswfc, cache=cache,
                )  # fmt: skip
                entries = [
                    (l - 0.5, u_minus, e_minus, f'{label}_j{int(2 * (l - 0.5))}.dat'),
                    (l + 0.5, u_plus, e_plus, f'{label}_j{int(2 * (l + 0.5))}.dat'),
                    (
                        None,
                        _j_average(u_minus, u_plus, l),
                        0.5 * (e_minus + e_plus),
                        f'{label}.dat',
                    ),
                ]
            else:
                r, u_, e_ = pseudize_shell(
                    upf, n, l, r_box=r_box, n_points=n_points, prefer_pswfc=prefer_pswfc,
                    cache=cache,
                )  # fmt: skip
                entries = [(None, u_, e_, f'{label}.dat')]
        except ValueError as err:
            warnings.warn(
                f'Skipping shell {label!r} for element {elem!r}: {err}', RuntimeWarning,
                stacklevel=2,
            )  # fmt: skip
            skipped.append((label, str(err)))
            continue

        # Skip shells the pseudopotential froze into the core: an explicit
        # shell list may request a sub-valence shell (e.g. As 3D) that has no
        # PSWFC counterpart and cannot be bound, in which case the solver
        # returns a spurious diffuse box mode that corrupts the basis.
        rep_eps = entries[0][2]
        if not np.isnan(rep_eps) and is_frozen_core_shell(upf, n, l, rep_eps):
            reason = (
                f'it has no matching pseudo-atomic wavefunction and the radial solver '
                f'returns an unbound state (eps = {rep_eps:+.4f} Ha), which indicates a '
                f'frozen-core shell the pseudopotential cannot represent'
            )
            warnings.warn(
                f'Skipping shell {label!r} for element {elem!r}: {reason}.',
                RuntimeWarning,
                stacklevel=2,
            )
            skipped.append((label, reason))
            continue
        for j, wfc, eps, fname in entries:
            generated.append(GeneratedShell(label, l, j, r, wfc, float(eps), fname))
    return generated, skipped


def generate_basis_for_pseudo(
    upf_path,
    out_dir,
    shells=None,
    preset='extended',
    r_box=None,
    n_points=2000,
    overwrite=True,
    verbose=False,
    prefer_pswfc=True,
):
    """Pseudize the requested shells of a pseudopotential and write them to disk.

    Parameters
    ----------
    upf_path : str, UPF or PotcarDataset
        Path to the UPF file or POTCAR (or a parsed dataset).
    out_dir : str
        Output root (per-element subdirectory ``out_dir/<elem>/`` is
        created automatically).
    shells : list of str, optional
        Explicit shell labels (e.g. ``['3S', '3P', '3D']``).  Defaults
        to the ``preset`` augmentation rules.
    preset : {'minimal', 'standard', 'extended'}
        Used when ``shells`` is ``None``.
    r_box : float, optional
        Confining box radius (Bohr).  Defaults to ``min(upf.r[-1], 10)``.
    n_points : int
        Solver mesh size (interior points = n_points - 1).
    overwrite : bool
        If False, existing files are skipped.
    verbose : bool
        Print one line per generated shell.
    prefer_pswfc : bool
        Stored pseudo-orbitals for the shells that have one (default); ``False`` solves
        every shell.

    Returns
    -------
    written : list of str
        Paths of the files written.
    """
    upf = _load_dataset(upf_path)
    elem = upf.element.strip()
    if shells is None:
        shells = _default_shells(upf, preset=preset)

    elem_dir = os.path.join(out_dir, elem)
    os.makedirs(elem_dir, exist_ok=True)

    generated, _ = generate_basis_shells(
        upf, shells, r_box=r_box, n_points=n_points, prefer_pswfc=prefer_pswfc
    )
    written = []
    for shell in generated:
        path = os.path.join(elem_dir, shell.filename)
        if not overwrite and os.path.exists(path):
            if verbose:
                print(f'  skip (exists) {path}')
            continue
        _write_two_col(path, shell.r, shell.wfc)
        written.append(path)
        if verbose:
            print(f'  wrote {path}   eps = {shell.energy:+.4f} Ha')

    return written


def _pseudopotential_files(pseudo_dir):
    """UPF files and POTCARs (``POTCAR``, ``*/POTCAR``) of a directory, sorted."""
    patterns = ('*.UPF', '*.upf', 'POTCAR', os.path.join('*', 'POTCAR'))
    return sorted({p for pat in patterns for p in glob(os.path.join(pseudo_dir, pat))})


def generate_basis_for_directory(
    pseudo_dir,
    out_dir,
    preset='extended',
    r_box=None,
    n_points=2000,
    overwrite=True,
    verbose=False,
    prefer_pswfc=True,
):
    """Run :func:`generate_basis_for_pseudo` over every pseudopotential in ``pseudo_dir``.

    UPF files and POTCARs (a ``POTCAR`` of the directory, possibly concatenated, and the
    ``<name>/POTCAR`` of a PAW library) are processed; every dataset of a concatenated POTCAR
    gives one element directory.

    Returns a dict ``{element_symbol: [written_paths]}``.
    """
    pseudos = _pseudopotential_files(pseudo_dir)
    if not pseudos:
        raise FileNotFoundError(f'no *.UPF / *.upf / POTCAR files found in {pseudo_dir!r}')
    out = {}
    for p in pseudos:
        if verbose:
            print(f'== {p} ==')
        if os.path.basename(p) == 'POTCAR':
            from ..inputs.read_potcar import read_potcar

            datasets = read_potcar(p)
        else:
            datasets = [_load_dataset(p)]
        for dataset in datasets:
            elem = dataset.element.strip()
            out[elem] = generate_basis_for_pseudo(
                dataset,
                out_dir,
                preset=preset,
                r_box=r_box,
                n_points=n_points,
                overwrite=overwrite,
                verbose=verbose,
                prefer_pswfc=prefer_pswfc,
            )
    return out
