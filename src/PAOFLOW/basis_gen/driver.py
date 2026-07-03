"""High-level basis generator: UPF -> BASIS_PS/<elem>/<shell>.dat files."""

from __future__ import annotations

import os
import warnings
from glob import glob

from ..inputs.basis_presets import (
    extended_augmentation,
    standard_augmentation,
)
from ..inputs.read_upf import UPF
from .radial import is_frozen_core_shell, pseudize_shell

_L_INDEX = {'S': 0, 'P': 1, 'D': 2, 'F': 3}


def _minimal_shells_from_upf(upf):
    """Return the list of unique '<n><L>' labels carried by the UPF PSWFC."""
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


def generate_basis_for_pseudo(
    upf_path,
    out_dir,
    shells=None,
    preset='extended',
    r_box=None,
    n_points=2000,
    overwrite=True,
    verbose=False,
):
    """Pseudize the requested shells of a UPF and write them to disk.

    Parameters
    ----------
    upf_path : str
        Path to the UPF file.
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

    Returns
    -------
    written : list of str
        Paths of the files written.
    """
    upf = UPF(upf_path)
    elem = upf.element.strip()
    if getattr(upf, 'has_augmentation', False) and upf.version != 2:
        raise NotImplementedError(
            f'UPF v{upf.version} ultrasoft/PAW augmentation parsing is not '
            f'implemented; got {upf_path!r}.'
        )
    if shells is None:
        shells = _default_shells(upf, preset=preset)

    elem_dir = os.path.join(out_dir, elem)
    os.makedirs(elem_dir, exist_ok=True)

    written = []
    so = bool(getattr(upf, 'has_spinorbit', False))

    for label in shells:
        if len(label) < 2 or label[1].upper() not in _L_INDEX:
            raise ValueError(f'shell label {label!r} not understood')
        n = int(label[0])
        l = _L_INDEX[label[1].upper()]

        if so and l > 0:
            # Solve both j = l-1/2 and j = l+1/2, write j-resolved files and
            # a degeneracy-weighted j-averaged scalar file for fallback use.
            r, u_minus, e_minus = pseudize_shell(
                upf, n, l, j=l - 0.5, r_box=r_box, n_points=n_points
            )
            _, u_plus, e_plus = pseudize_shell(upf, n, l, j=l + 0.5, r_box=r_box, n_points=n_points)
            files = [
                (f'{label}_j{int(2 * (l - 0.5))}.dat', u_minus, e_minus),
                (f'{label}_j{int(2 * (l + 0.5))}.dat', u_plus, e_plus),
                (f'{label}.dat', _j_average(u_minus, u_plus, l), 0.5 * (e_minus + e_plus)),
            ]
        else:
            r, u_, e_ = pseudize_shell(upf, n, l, r_box=r_box, n_points=n_points)
            files = [(f'{label}.dat', u_, e_)]

        # Skip shells the pseudopotential froze into the core: an explicit
        # shell list may request a sub-valence shell (e.g. As 3D) that has no
        # PSWFC counterpart and cannot be bound, in which case the solver
        # returns a spurious diffuse box mode that corrupts the basis.
        rep_eps = files[0][2]
        if is_frozen_core_shell(upf, n, l, rep_eps):
            warnings.warn(
                f'Skipping shell {label!r} for element {elem!r}: it has no '
                f'matching pseudo-atomic wavefunction and the radial solver '
                f'returns an unbound state (eps = {rep_eps:+.4f} Ha), which '
                f'indicates a frozen-core shell the pseudopotential cannot '
                f'represent.',
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        for fname, wfc, eps in files:
            path = os.path.join(elem_dir, fname)
            if not overwrite and os.path.exists(path):
                if verbose:
                    print(f'  skip (exists) {path}')
                continue
            _write_two_col(path, r, wfc)
            written.append(path)
            if verbose:
                print(f'  wrote {path}   eps = {eps:+.4f} Ha')

    return written


def generate_basis_for_directory(
    pseudo_dir,
    out_dir,
    preset='extended',
    r_box=None,
    n_points=2000,
    overwrite=True,
    verbose=False,
):
    """Run :func:`generate_basis_for_pseudo` over every UPF in ``pseudo_dir``.

    Returns a dict ``{element_symbol: [written_paths]}``.
    """
    out = {}
    patterns = ('*.UPF', '*.upf')
    pseudos = sorted({p for pat in patterns for p in glob(os.path.join(pseudo_dir, pat))})
    if not pseudos:
        raise FileNotFoundError(f'no *.UPF / *.upf files found in {pseudo_dir!r}')
    for p in pseudos:
        if verbose:
            print(f'== {p} ==')
        upf = UPF(p)
        elem = upf.element.strip()
        out[elem] = generate_basis_for_pseudo(
            p,
            out_dir,
            preset=preset,
            r_box=r_box,
            n_points=n_points,
            overwrite=overwrite,
            verbose=verbose,
        )
    return out
