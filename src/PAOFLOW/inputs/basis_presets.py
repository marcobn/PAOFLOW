"""Automated internal-basis (``configuration``) presets for PAOFLOW.

This module turns string presets such as ``"minimal"`` and ``"extended"``
into the per-species ``{element: [shell_labels]}`` dictionary consumed by
:func:`PAOFLOW.projection.do_atwfc_proj.build_aewfc_basis`.

* ``"minimal"`` — the set of occupied pseudo-atomic wavefunctions found in
  each species' UPF pseudopotential (``PP_PSWFC/PP_CHI`` entries with
  positive occupation).
* ``"standard"`` — the minimal set with each angular channel doubled by
  adding the next principal-quantum-number shell (e.g. ``3S, 3P`` →
  ``3S, 3P, 4S, 4P``).  Roughly doubles the orbital count of
  ``"minimal"``.
* ``"extended"`` — the minimal set augmented by a small, rule-based set
  of polarization shells, drawn from the all-electron wavefunctions
  shipped in ``BASIS/<element>/*.dat``.

The user-facing dispatch lives in :func:`PAOFLOW.PAOFLOW.projections`;
this module is intentionally side-effect-free and easy to unit test.
"""

from __future__ import annotations

import glob
import os
import re
import warnings

from .read_upf import UPF

# ---------------------------------------------------------------------------
# Periodic-table classification
# ---------------------------------------------------------------------------

# Element → periodic-table block.  Covers Z=1..96, which matches the
# elements that ship with the BASIS/ directory.
_ELEMENT_BLOCK = {
    # s-block
    'H': 's',
    'He': 's',
    'Li': 's',
    'Be': 's',
    'Na': 's',
    'Mg': 's',
    'K': 's',
    'Ca': 's',
    'Rb': 's',
    'Sr': 's',
    'Cs': 's',
    'Ba': 's',
    'Fr': 's',
    'Ra': 's',
    # p-block
    'B': 'p',
    'C': 'p',
    'N': 'p',
    'O': 'p',
    'F': 'p',
    'Ne': 'p',
    'Al': 'p',
    'Si': 'p',
    'P': 'p',
    'S': 'p',
    'Cl': 'p',
    'Ar': 'p',
    'Ga': 'p',
    'Ge': 'p',
    'As': 'p',
    'Se': 'p',
    'Br': 'p',
    'Kr': 'p',
    'In': 'p',
    'Sn': 'p',
    'Sb': 'p',
    'Te': 'p',
    'I': 'p',
    'Xe': 'p',
    'Tl': 'p',
    'Pb': 'p',
    'Bi': 'p',
    'Po': 'p',
    'At': 'p',
    'Rn': 'p',
    # d-block (3d)
    'Sc': 'd',
    'Ti': 'd',
    'V': 'd',
    'Cr': 'd',
    'Mn': 'd',
    'Fe': 'd',
    'Co': 'd',
    'Ni': 'd',
    'Cu': 'd',
    'Zn': 'd',
    # d-block (4d)
    'Y': 'd',
    'Zr': 'd',
    'Nb': 'd',
    'Mo': 'd',
    'Tc': 'd',
    'Ru': 'd',
    'Rh': 'd',
    'Pd': 'd',
    'Ag': 'd',
    'Cd': 'd',
    # d-block (5d)
    'Lu': 'd',
    'Hf': 'd',
    'Ta': 'd',
    'W': 'd',
    'Re': 'd',
    'Os': 'd',
    'Ir': 'd',
    'Pt': 'd',
    'Au': 'd',
    'Hg': 'd',
    # f-block (lanthanides)
    'La': 'f',
    'Ce': 'f',
    'Pr': 'f',
    'Nd': 'f',
    'Pm': 'f',
    'Sm': 'f',
    'Eu': 'f',
    'Gd': 'f',
    'Tb': 'f',
    'Dy': 'f',
    'Ho': 'f',
    'Er': 'f',
    'Tm': 'f',
    'Yb': 'f',
    # f-block (actinides)
    'Ac': 'f',
    'Th': 'f',
    'Pa': 'f',
    'U': 'f',
    'Np': 'f',
    'Pu': 'f',
}

# Recognised string presets.
SUPPORTED_PRESETS = ('minimal', 'standard', 'extended')


def element_block(elem: str) -> str:
    """Return the periodic-table block (``'s'``, ``'p'``, ``'d'`` or ``'f'``).

    Parameters
    ----------
    elem : str
        Chemical symbol, case-sensitive (e.g. ``'Fe'``).

    Raises
    ------
    KeyError
        If ``elem`` is not in the internal element table.
    """
    return _ELEMENT_BLOCK[elem]


# ---------------------------------------------------------------------------
# BASIS/ catalog
# ---------------------------------------------------------------------------

_SHELL_RE = re.compile(r'^([1-9])([SPDFspdf])$')


def _normalize_shell(label: str) -> str:
    """Uppercase the angular-momentum letter (e.g. ``'4p'`` → ``'4P'``)."""
    label = label.strip()
    if len(label) != 2:
        return label
    return label[0] + label[1].upper()


def available_ae_shells(basispath: str, elem: str) -> list[str]:
    """List the AE shell labels available for ``elem`` in ``basispath``.

    Globs ``<basispath>/<elem>/*.dat`` and returns the orbital labels
    (e.g. ``['1S', '2S', '2P', '3S', '3P', '3D', ...]``) sorted by
    principal quantum number then by angular momentum.

    Returns an empty list if the directory is missing or contains no
    recognisable ``nL.dat`` files.
    """
    pattern = os.path.join(basispath, elem, '*.dat')
    shells = []
    for path in glob.glob(pattern):
        name = os.path.splitext(os.path.basename(path))[0]
        name = _normalize_shell(name)
        if _SHELL_RE.match(name):
            shells.append(name)
    # Sort by (n, l-index)
    order = 'SPDF'
    shells.sort(key=lambda s: (int(s[0]), order.find(s[1])))
    return shells


# ---------------------------------------------------------------------------
# Minimal set from UPF
# ---------------------------------------------------------------------------


def _resolve_pseudo_path(data_controller, elem: str) -> str | None:
    """Locate the UPF file shipped with the QE calculation for ``elem``.

    Mirrors the lookup performed inside
    :func:`read_pswfc_from_upf <PAOFLOW.projection.do_atwfc_proj.read_pswfc_from_upf>`.
    Returns ``None`` if no matching species record exists (e.g. the
    DataController has not been populated yet, as in some unit tests).
    """
    arry, attr = data_controller.data_dicts()
    species = arry.get('species')
    fpath = attr.get('fpath')
    if not species or fpath is None:
        return None
    for at, pseudo in species:
        if re.split(r'\d+', at)[0] == elem:
            return os.path.join(fpath, pseudo)
    return None


def minimal_shells_from_upf(data_controller, elem: str) -> list[str]:
    """Return the PP_PSWFC shell labels shipped with ``elem``'s UPF.

    Reads the species' UPF file and returns the labels of every
    ``PP_CHI`` entry, preserving the order in which they appear in
    the pseudopotential.  Duplicates (rare, but possible for
    spin-orbit pseudos where two j-channels share a label) are
    collapsed.

    All PSWFC entries are included regardless of occupation: an
    unoccupied PSWFC (e.g. Pt 6P, ``occ=0``) is still a physical
    pseudo-atomic orbital generated together with the pseudopotential,
    and is typically the dominant bonding/conduction channel.  Excluding
    such shells leaves the corresponding QE bands unprojectable, which
    silently lowers the AE-basis ``standard``/``extended`` ``Pn>0.95``
    band count below that of the ``minimal`` preset (which reads every
    PSWFC unconditionally via :func:`build_pswfc_basis_all`).

    Raises
    ------
    RuntimeError
        If no UPF can be located for ``elem``.
    """
    pseudo_path = _resolve_pseudo_path(data_controller, elem)
    if pseudo_path is None:
        raise RuntimeError(
            "Cannot resolve UPF for element '%s'; populate "
            "data_arrays['species'] and data_attributes['fpath'] first." % elem
        )
    upf = UPF(pseudo_path)
    seen = []
    for chi in upf.pswfc:
        label = _normalize_shell(chi['label'])
        if label not in seen:
            seen.append(label)
    if not seen:
        raise RuntimeError(
            "UPF for '%s' has no PP_PSWFC entries; cannot derive a "
            "'minimal' basis automatically." % elem
        )
    return seen


# ---------------------------------------------------------------------------
# Augmentation rules
# ---------------------------------------------------------------------------


def standard_augmentation(elem: str, minimal_shells: list[str]) -> list[str]:
    """Augmentation rule for the ``"standard"`` preset.

    Builds on top of the minimal valence set with two complementary
    additions:

    1. **Polarization (next missing L)** — append the next angular
       momentum channel beyond the highest one present in the minimal
       set, at the largest principal quantum number ``nmax`` of the
       minimal set.  For Si (minimal ``3S, 3P``) this adds ``3D``; for
       a transition metal with minimal ``4S, 3D`` this adds ``4F``
       (only when ``elem`` is in the f-block) or ``4F`` is skipped
       otherwise.  The added shell is also clamped to the physical
       minimum ``n`` for that ``L`` (``nP ≥ 2``, ``nD ≥ 3``,
       ``nF ≥ 4``).
    2. **Radial doubling (next n, same L)** — for each ``nL`` in the
       minimal set append ``(n+1)L``.  This roughly doubles the radial
       freedom of every occupied angular channel.

    The result is intermediate between ``"minimal"`` (only the
    occupied valence) and ``"extended"`` (generous multi-row
    polarization).  For Si the standard set becomes
    ``[3S, 3P, 3D, 4S, 4P]``.

    Duplicates and shells already in ``minimal_shells`` are skipped.
    Unknown elements or an empty minimal set yield an empty list.
    The F-channel polarization is only added for f-block elements;
    on s/p/d-block elements the augmentation stops at the D channel.
    """
    block = _ELEMENT_BLOCK.get(elem)
    if block is None or not minimal_shells:
        return []

    minimal_set = set(minimal_shells)
    extra: list[str] = []

    def _add(label: str) -> None:
        if label not in minimal_set and label not in extra:
            extra.append(label)

    order = 'SPDF'
    min_n_for_L = {'S': 1, 'P': 2, 'D': 3, 'F': 4}

    # (1) Next missing L channel at nmax.
    L_indices = [order.find(sh[1]) for sh in minimal_shells]
    nmax = max(int(sh[0]) for sh in minimal_shells)
    next_L_idx = max(L_indices) + 1
    if next_L_idx < len(order):
        next_L = order[next_L_idx]
        # Don't add F unless we're actually on an f-block element.
        if next_L != 'F' or block == 'f':
            n_pol = max(nmax, min_n_for_L[next_L])
            _add(f'{n_pol}{next_L}')

    # (2) Radial doubling: (n+1)L for each nL in minimal.
    for sh in minimal_shells:
        n = int(sh[0])
        L = sh[1]
        _add(f'{n + 1}{L}')

    return extra


def extended_augmentation(elem: str, minimal_shells: list[str]) -> list[str]:
    """Rule-based AE polarization shells to add to ``minimal_shells``.

    The ``"extended"`` preset is built on top of the all-electron basis
    in ``basispath`` (``build_aewfc_basis``), so augmentation must be
    generous: AE radial functions carry core-region nodes and the
    Loewdin orthogonalization needs enough variational freedom to span
    both the valence and a few conduction-like channels.  This rule
    targets roughly the hand-tuned configurations that ship with the
    PAOFLOW examples (≈ 11 shells per atom for GaAs).

    Augmentation per block (``nmax`` = largest principal quantum number
    present in ``minimal_shells``):

    * **s/p/d/f blocks** — append ``S`` and ``P`` shells for
      ``n = nmax, nmax+1, nmax+2`` (clamped to ``n ≥ 1`` for ``S`` and
      ``n ≥ 2`` for ``P``).
    * **D channel** — append ``nD`` for
      ``n = max(3, nmax) … nmax+1`` (skips sub-valence D rows, which are
      either genuine semicore shells already present in ``minimal_shells``
      or frozen-core shells the pseudopotential cannot represent).
    * **F channel** — only for f-block elements; append ``nF`` for
      ``n = max(4, nmax−1) … nmax``.

    Shells already present in ``minimal_shells`` are skipped.  Unknown
    elements or an empty minimal set yield an empty list.
    """
    block = _ELEMENT_BLOCK.get(elem)
    if block is None or not minimal_shells:
        return []

    nmax = max(int(s[0]) for s in minimal_shells)
    minimal_set = set(minimal_shells)
    extra: list[str] = []

    def _add(label: str) -> None:
        if label not in minimal_set and label not in extra:
            extra.append(label)

    # Three rows of S and P starting at nmax.
    for n in (nmax, nmax + 1, nmax + 2):
        if n >= 1:
            _add(f'{n}S')
        if n >= 2:
            _add(f'{n}P')

    # D shells: from max(3, nmax) up through nmax+1.  We deliberately do
    # *not* start at nmax-1: a D row below the valence nmax is only physical
    # when it is a genuine semicore shell, in which case it already appears
    # in ``minimal_shells`` (e.g. Ga 3D) and is kept by the baseline.  Adding
    # a sub-valence D row that the pseudopotential froze into the core (e.g.
    # As 3D, whose 3d electrons are not in the As pseudo valence) yields an
    # unbound diffuse box mode from the radial solver and corrupts the basis.
    n_d_start = max(3, nmax)
    for n in range(n_d_start, nmax + 2):
        _add(f'{n}D')

    # F shells: f-block only.
    if block == 'f':
        n_f_start = max(4, nmax - 1)
        for n in range(n_f_start, nmax + 1):
            _add(f'{n}F')

    return extra


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------


def _unique_elements(data_controller) -> list[str]:
    """Distinct element symbols present in ``arry['atoms']`` (order-preserving)."""
    arry, _ = data_controller.data_dicts()
    elems: list[str] = []
    for atom in arry['atoms']:
        elem = re.split(r'\d+', atom)[0]
        if elem not in elems:
            elems.append(elem)
    return elems


def resolve_configuration(data_controller, preset: str) -> dict[str, list[str]]:
    """Turn a string preset into a ``{element: [augment_shells]}`` dict.

    In the current "mixed" scheme the pseudo-atomic wavefunctions from
    the UPF always serve as the baseline projection set; presets only
    control which **extra** all-electron shells are loaded from
    ``basispath`` on top of that baseline.

    * ``"minimal"`` — returns the occupied PSWFC shells read from the
      UPF for each species.  The caller should build a pure pseudo
      basis with :func:`build_pswfc_basis_all`; the returned dict is
      kept for diagnostics.
    * ``"extended"`` — returns the minimal valence set **plus** a
      generous rule-based augmentation drawn from ``basispath``
      (see :func:`extended_augmentation`).  The caller should feed
      this dict to :func:`build_aewfc_basis` (AE-only path).  Shells
      that are not present on disk are dropped with a
      :class:`RuntimeWarning`.

    To choose exactly which orbitals enter the basis, pass a per-element
    ``dict`` instead of a preset string (handled by
    :func:`resolve_configuration_dict`).  Each value may independently be
    a preset name or an explicit list of shells, e.g.
    ``{'Ga': 'standard', 'As': ['4S', '4P', '3D']}``.

    Notes
    -----
    A *mixed* scheme combining pseudo PSWFC with AE polarization is
    implemented in :func:`build_mixed_basis` but currently not wired
    through this preset: raw AE orbitals carry core-region oscillations
    that the smooth QE pseudo Bloch states cannot project onto,
    producing ill-conditioned overlaps.  Re-enabling the mixed scheme
    requires AE pseudization first.

    Parameters
    ----------
    data_controller : DataController
        Populated controller; needs ``data_arrays['atoms']``,
        ``data_arrays['species']`` and ``data_attributes['fpath']``.
        ``data_attributes['basispath']`` is additionally required for
        ``"extended"``.
    preset : str
        One of :data:`SUPPORTED_PRESETS` (case-insensitive).

    Returns
    -------
    dict
        Mapping ``{element: [shells]}`` suitable for assignment to
        ``arry['configuration']``.

    Raises
    ------
    ValueError
        If ``preset`` is not recognised, or if ``"extended"`` is
        requested without ``data_attributes['basispath']`` being set.
    """
    if not isinstance(preset, str):
        raise TypeError('configuration preset must be a string; got %r' % type(preset).__name__)
    key = preset.lower()
    if key not in SUPPORTED_PRESETS:
        raise ValueError(
            "Unknown configuration preset '%s'. Supported presets: %s "
            '(or pass an explicit {element: [shells]} dict).'
            % (preset, ', '.join(SUPPORTED_PRESETS))
        )

    _, attr = data_controller.data_dicts()
    basispath = attr.get('basispath')
    if key in ('standard', 'extended') and not basispath:
        raise ValueError(
            "configuration='%s' requires 'basispath' to be set so that "
            'augmenting AE wavefunctions can be located.' % key
        )

    config: dict[str, list[str]] = {}
    for elem in _unique_elements(data_controller):
        config[elem] = _resolve_element_shells(data_controller, elem, key, basispath)

    return config


def _resolve_element_shells(data_controller, elem: str, key: str, basispath) -> list[str]:
    """Resolve a single element's shell list for a string ``key`` preset.

    Shared by :func:`resolve_configuration` (whole-system string preset)
    and the per-element dispatch used when ``configuration`` is a dict
    that mixes preset names with explicit shell lists.
    """
    minimal = minimal_shells_from_upf(data_controller, elem)
    if key == 'minimal':
        return list(minimal)

    # 'standard' / 'extended': start from the all-electron
    # shells available on disk and select the requested subset.
    available = set(available_ae_shells(basispath, elem))
    if not available:
        warnings.warn(
            "No AE basis files found in '%s' for element '%s'; "
            'falling back to the minimal valence set.' % (os.path.join(basispath, elem), elem),
            RuntimeWarning,
            stacklevel=2,
        )
        return list(minimal)

    # 'standard' / 'extended': start from the minimal valence and
    # append a rule-based augmentation, filtered by what is
    # actually available on disk.
    shells = list(minimal)
    for sh in minimal:
        if sh not in available:
            warnings.warn(
                "Minimal shell '%s' for element '%s' is absent from "
                "'%s'; build_aewfc_basis will fail." % (sh, elem, os.path.join(basispath, elem)),
                RuntimeWarning,
                stacklevel=2,
            )
    augment = (
        standard_augmentation(elem, minimal)
        if key == 'standard'
        else extended_augmentation(elem, minimal)
    )
    for extra in augment:
        if extra in available and extra not in shells:
            shells.append(extra)
        elif extra not in available:
            warnings.warn(
                "Skipping augmenting shell '%s' for element '%s': not "
                "found in '%s'." % (extra, elem, os.path.join(basispath, elem)),
                RuntimeWarning,
                stacklevel=2,
            )
    return shells


def _select_shells(data_controller, elem: str, spec, basispath) -> list[str]:
    """Resolve one element's shells from an explicit list or preset name.

    ``spec`` may be:

    * a list/tuple of shell labels (e.g. ``['3S', '3P', '4S']``) — used
      verbatim after normalising the angular-momentum letter, so the
      caller picks exactly which orbitals enter the basis;
    * a string preset name (``'minimal'``/``'standard'``/``'extended'``)
      — resolved for this element via :func:`_resolve_element_shells`.

    When ``spec`` is an explicit list, any shell that is not present on
    disk under ``basispath`` is reported with a :class:`RuntimeWarning`
    but still kept, so the user stays in full control of the selection.
    """
    if isinstance(spec, str):
        key = spec.lower()
        if key not in SUPPORTED_PRESETS:
            raise ValueError(
                "Unknown configuration preset '%s' for element '%s'. "
                'Supported presets: %s (or pass an explicit list of shells).'
                % (spec, elem, ', '.join(SUPPORTED_PRESETS))
            )
        if key in ('standard', 'extended') and not basispath:
            raise ValueError(
                "configuration preset '%s' for element '%s' requires "
                "'basispath' to be set." % (key, elem)
            )
        return _resolve_element_shells(data_controller, elem, key, basispath)

    if isinstance(spec, (list, tuple)):
        shells = [_normalize_shell(str(s)) for s in spec]
        if basispath:
            available = set(available_ae_shells(basispath, elem))
            if available:
                missing = [sh for sh in shells if sh not in available]
                if missing:
                    warnings.warn(
                        "Selected shells %s for element '%s' are not present "
                        "in '%s'; build_aewfc_basis will fail for them."
                        % (missing, elem, os.path.join(basispath, elem)),
                        RuntimeWarning,
                        stacklevel=2,
                    )
        return shells

    raise TypeError(
        "configuration value for element '%s' must be a list of shell "
        'labels or a preset name string; got %r' % (elem, type(spec).__name__)
    )


def resolve_configuration_dict(data_controller, spec: dict) -> dict[str, list[str]]:
    """Resolve a per-element ``configuration`` dict.

    Each value may independently be a preset name string or an explicit
    list of shell labels, so the user can ask for a curated preset on
    some species while hand-picking orbitals on others::

        configuration = {
            'Ga': 'standard',                   # rule-based augmentation
            'As': ['4S', '4P', '3D'],           # only these orbitals
        }

    Explicit lists are taken verbatim (only the angular-momentum letter
    is upper-cased).

    Parameters
    ----------
    data_controller : DataController
        Populated controller (see :func:`resolve_configuration`).
    spec : dict
        Mapping ``{element: (preset_name | [shells])}``.

    Returns
    -------
    dict
        Mapping ``{element: [shells]}``.
    """
    _, attr = data_controller.data_dicts()
    basispath = attr.get('basispath')
    config: dict[str, list[str]] = {}
    for elem, value in spec.items():
        config[elem] = _select_shells(data_controller, elem, value, basispath)
    return config
