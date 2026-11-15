"""The electronic side of the dense electron-phonon pipeline: PAO Hamiltonians from a QE save.

The dense-q properties (:func:`~PAOFLOW.elphon.phonon_assisted_absorption.phonon_assisted_absorption_dense_q`,
:func:`~PAOFLOW.elphon.carrier_scattering.carrier_scattering_dense_q`) take the
electron-phonon vertex from the EPW save (projections ``A`` and ``.epb`` files)
and the electronic Hamiltonian ``HRs`` as separate inputs.  ``HRs`` may therefore
come from **another** save of the same crystal in the same PAO basis, e.g. an
eACBN0 (DFT+U+V, ortho-atomic) nscf whose PAO Hamiltonian already has the
right band gap: the band energies and eigenvectors are then those of that
Hamiltonian, while the coupling in the orbital basis stays that of the DFPT/EPW
run.  This module builds such Hamiltonians, checks that two saves share the PAO
basis, and records the Hubbard provenance of a save.

Functions
---------
pao_hamiltonian_from_save
    Projections, PAO Hamiltonian, orbital centres and lattice data of a save.
check_same_pao_basis
    Raise unless two PAOFLOW runs span the same orbital basis on the same structure.
check_orbital_dimensions
    Shape guard between the coupling projections and the Hamiltonian.
hubbard_info_from_save, format_hubbard_info
    The Hubbard (DFT+U, DFT+U+V) parameters recorded in ``data-file-schema.xml``.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .elph_bloch import pao_orbital_positions, read_nscf

HARTREE_TO_EV = 27.211386245988


@dataclass
class PAOElectrons:
    """PAO electronic structure of one QE save.

    Attributes
    ----------
    savedir : str
        The QE save directory.
    projections : ndarray or None
        ``A_{ni}(k) = <phi_i|psi_nk>``, shape ``(nbnd, nawf, nk)`` (``None`` when
        not kept).
    HRs : ndarray
        PAO Hamiltonian (eV), shape ``(nawf, nawf, n1, n2, n3, nspin)``.
    nscf : dict
        :func:`~PAOFLOW.elphon.elph_bloch.read_nscf` of the save, plus
        ``'orbital_positions'`` (crystal coordinates of the orbital centres).
    data_controller : DataController
        The PAOFLOW data controller of the run (non-local velocity, basis checks).
    expand_wedge : bool
        The symmetry expansion used by ``pao_hamiltonian``.
    hubbard : dict or None
        :func:`hubbard_info_from_save` of the save (``None`` for a plain DFT run).
    """

    savedir: str
    projections: NDArray[np.complex128] | None
    HRs: NDArray[np.complex128]
    nscf: dict
    data_controller: object
    expand_wedge: bool
    hubbard: dict | None


def pao_hamiltonian_from_save(
    savedir: str,
    basispath: str | None = None,
    configuration: str | Mapping | None = 'standard',
    pthr: float = 0.95,
    *,
    expand_wedge: bool | None = None,
    projector: str = 'internal',
    keep_projections: bool = True,
    workpath: str | None = None,
    outputdir: str = 'output',
    save_overlaps: bool = False,
    verbose: bool = False,
    unfold_kpoints: bool = False,
) -> PAOElectrons:
    """Build the PAO projections and Hamiltonian of a QE save.

    Parameters
    ----------
    savedir : str
        QE save directory (``data-file-schema.xml``, wavefunctions, UPF files).
    basispath : str, optional
        Directory of the radial basis tables (``paoflow-genbasis-ps``); only for
        ``projector='internal'``.
    configuration : str or mapping, optional
        PAO basis preset (``'minimal'``, ``'standard'``, ``'extended'``) or a
        per-element shell dictionary (``PAOFLOW.projections``).
    pthr : float, optional
        Projectability threshold.
    expand_wedge : bool, optional
        Symmetry expansion of ``pao_hamiltonian``: ``False`` for an explicit full
        k list (the EPW nscf), ``True`` for an irreducible Monkhorst-Pack set.
        ``None`` (default) infers it: the k-points are expanded unless they
        already fill the ``nk1 x nk2 x nk3`` grid.
    projector : {'internal', 'projwfc'}, optional
        Internal atomic projections (``PAOFLOW.projections``) or the
        ``projwfc.x`` ones of the save (``read_atomic_proj_QE``).
    keep_projections : bool, optional
        Keep a copy of the full projection matrix (needed only for the save that
        provides the coupling).
    workpath, outputdir, save_overlaps, verbose : optional
        Passed to :class:`PAOFLOW.PAOFLOW`.
    unfold_kpoints : bool, optional
        The save holds the irreducible wedge of a Gamma-centred ``K_POINTS automatic`` nscf:
        unfold k-points, eigenvalues and wavefunctions to the full grid by symmetry
        (:class:`PAOFLOW.PAOFLOW` ``unfold_kpoints``), so that the projections cover every k
        and the Hamiltonian is built with ``expand_wedge=False``.  Required for the save that
        provides the coupling (``source='dfpt'``) when its nscf was not run on the full grid.

    Returns
    -------
    PAOElectrons
        Projections, ``HRs``, lattice data with orbital centres, the data
        controller and the Hubbard provenance.

    Raises
    ------
    ValueError
        For an unknown ``projector``.

    Notes
    -----
    The call sequence is the one of the generated electron-phonon drivers
    (``projections`` -> ``projectability`` -> ``full_projections`` ->
    ``pao_hamiltonian`` -> ``read_nscf`` -> ``pao_orbital_positions``), so a
    driver that switches to this function reproduces its previous numbers.
    ``HRs`` is on the energy scale of this save (its Fermi level or highest
    occupied level at zero).  ``savedir`` is resolved against the current
    directory (not ``workpath``).
    """
    from ..PAOFLOW import PAOFLOW

    if projector not in ('internal', 'projwfc'):
        raise ValueError("projector must be 'internal' or 'projwfc', got %r" % projector)
    savedir = os.path.abspath(savedir)  # PAOFLOW resolves relative saves against workpath
    pf = PAOFLOW(
        workpath=workpath if workpath is not None else os.getcwd(),
        outputdir=outputdir,
        savedir=savedir,
        save_overlaps=save_overlaps,
        verbose=verbose,
        unfold_kpoints=unfold_kpoints,
    )
    if projector == 'internal':
        pf.projections(basispath=basispath, configuration=configuration)
    else:
        pf.read_atomic_proj_QE()
    pf.projectability(pthr=pthr)
    projections = None
    if keep_projections:
        projections = pf.data_controller.full_projections()[:, :, :, 0].copy()
    arrays, attributes = pf.data_controller.data_dicts()
    if expand_wedge is None:
        full_grid = int(attributes['nk1']) * int(attributes['nk2']) * int(attributes['nk3'])
        expand_wedge = int(attributes['nkpnts']) != full_grid
    pf.pao_hamiltonian(expand_wedge=expand_wedge)
    nscf = read_nscf(savedir, unfold=unfold_kpoints)
    nscf['orbital_positions'] = pao_orbital_positions(pf.data_controller, nscf['at'])
    return PAOElectrons(
        savedir=savedir,
        projections=projections,
        HRs=pf.data_controller.data_arrays['HRs'],
        nscf=nscf,
        data_controller=pf.data_controller,
        expand_wedge=bool(expand_wedge),
        hubbard=hubbard_info_from_save(savedir),
    )


def _dicts(run) -> tuple[Mapping, Mapping]:
    """``(arrays, attributes)`` of a DataController or of such a tuple."""
    if isinstance(run, tuple) and len(run) == 2:
        return run
    return run.data_dicts()


def _orbital_signature(orbital: Mapping) -> tuple:
    """Hashable identity of one PAO orbital (atom, shell labels, l, m)."""
    keys = ('atom', 'label', 'n', 'shell', 'l', 'm', 'j', 'mj')
    return tuple((key, orbital[key]) for key in keys if key in orbital)


def check_same_pao_basis(
    reference,
    other,
    *,
    names: tuple[str, str] = ('coupling save', 'electrons save'),
    tau_tol_bohr: float = 1.0e-5,
) -> dict:
    """Check that two PAOFLOW runs span the same PAO basis on the same crystal.

    Parameters
    ----------
    reference, other : DataController or (arrays, attributes)
        The two runs after ``projections``/``projectability`` (and optionally
        ``pao_hamiltonian``).
    names : tuple of str, optional
        How the two runs are called in the error message.
    tau_tol_bohr : float, optional
        Tolerance on the atomic and orbital positions (bohr).

    Returns
    -------
    dict
        Per-run summary of the quantities that may legitimately differ:
        ``nbnds``, ``nkpnts``, ``kgrid``, ``bnd`` (bands with good
        projectability), ``shift``, ``Efermi`` and ``pthr`` (each a pair).

    Raises
    ------
    ValueError
        Listing every mismatch of the orbital count, spin, species and
        pseudopotentials, atoms, positions, lattice vectors, orbital identities
        and basis configuration.

    Notes
    -----
    The Hamiltonian of ``other`` is used with the coupling of ``reference``
    only if both are expressed in the same orthonormal orbital basis: the same
    pseudopotentials, radial tables and shells, on the same atoms in the same
    order.  The projectability outcome (``bnd``, ``shift``) depends on the
    Kohn-Sham states and differs between functionals; it is reported, not
    enforced.
    """
    (arr_a, att_a), (arr_b, att_b) = _dicts(reference), _dicts(other)
    problems: list[str] = []

    def mismatch(what: str, a, b) -> None:
        problems.append('%s: %s %r != %s %r' % (what, names[0], a, names[1], b))

    for key in ('nawf', 'nspin', 'dftSO', 'natoms', 'nelec'):
        if key in att_a and key in att_b and att_a[key] != att_b[key]:
            mismatch(key, att_a[key], att_b[key])
    if 'alat' in att_a and 'alat' in att_b:
        if not np.isclose(float(att_a['alat']), float(att_b['alat']), rtol=1.0e-8, atol=0.0):
            mismatch('alat', att_a['alat'], att_b['alat'])
    species_a = [(str(s), os.path.basename(str(p))) for s, p in arr_a.get('species', [])]
    species_b = [(str(s), os.path.basename(str(p))) for s, p in arr_b.get('species', [])]
    if species_a != species_b:
        mismatch('species (label, pseudopotential)', species_a, species_b)
    atoms_a, atoms_b = list(arr_a.get('atoms', [])), list(arr_b.get('atoms', []))
    if atoms_a != atoms_b:
        mismatch('atoms', atoms_a, atoms_b)
    for key, tol in (('tau', tau_tol_bohr), ('a_vectors', 1.0e-8)):
        if key in arr_a and key in arr_b:
            a, b = np.asarray(arr_a[key], dtype=float), np.asarray(arr_b[key], dtype=float)
            if a.shape != b.shape or not np.allclose(a, b, atol=tol, rtol=0.0):
                mismatch(key, a.tolist(), b.tolist())
    basis_a, basis_b = arr_a.get('basis'), arr_b.get('basis')
    if basis_a is not None and basis_b is not None:
        sig_a = [_orbital_signature(o) for o in basis_a]
        sig_b = [_orbital_signature(o) for o in basis_b]
        if sig_a != sig_b:
            first = next((i for i, (x, y) in enumerate(zip(sig_a, sig_b)) if x != y), None)
            where = (
                'orbital %d' % first
                if first is not None
                else 'count %d vs %d' % (len(sig_a), len(sig_b))
            )
            mismatch('orbital basis (%s)' % where, sig_a[first] if first is not None else len(sig_a),
                     sig_b[first] if first is not None else len(sig_b))  # fmt: skip
        else:
            tau_a = np.array([o['tau'] for o in basis_a], dtype=float)
            tau_b = np.array([o['tau'] for o in basis_b], dtype=float)
            if not np.allclose(tau_a, tau_b, atol=tau_tol_bohr, rtol=0.0):
                mismatch('orbital centres (bohr)', tau_a.tolist(), tau_b.tolist())
    cfg_a, cfg_b = arr_a.get('configuration'), arr_b.get('configuration')
    if cfg_a is not None and cfg_b is not None and dict(cfg_a) != dict(cfg_b):
        mismatch('basis configuration', dict(cfg_a), dict(cfg_b))
    if problems:
        raise ValueError(
            'The PAO bases of the %s and the %s differ; the electron-phonon coupling of the '
            'first cannot be used with the Hamiltonian of the second (same pseudopotentials, '
            'structure, atom order, BASIS tables and configuration are required):\n  '
            % names
            + '\n  '.join(problems)
        )

    def pair(key, cast=lambda x: x):
        return (
            cast(att_a[key]) if key in att_a else None,
            cast(att_b[key]) if key in att_b else None,
        )

    grid = lambda att: tuple(int(att[k]) for k in ('nk1', 'nk2', 'nk3') if k in att)  # noqa: E731
    return {
        'nbnds': pair('nbnds', int),
        'nkpnts': pair('nkpnts', int),
        'kgrid': (grid(att_a), grid(att_b)),
        'bnd': pair('bnd'),
        'shift': pair('shift'),
        'Efermi': pair('Efermi', float),
        'pthr': pair('pthr', float),
    }


def check_orbital_dimensions(
    A: NDArray[np.complex128],
    HRs: NDArray[np.complex128],
    orbital_positions: NDArray[np.float64] | None = None,
) -> None:
    """Shape guard between the coupling projections and the electronic Hamiltonian.

    Parameters
    ----------
    A : ndarray, shape ``(nbnd, nawf, nk)``
        PAO projections of the coupling save.
    HRs : ndarray, shape ``(nawf, nawf, n1, n2, n3, nspin)``
        PAO Hamiltonian (possibly of another save).
    orbital_positions : ndarray, shape ``(nawf, 3)``, optional
        Orbital centres.

    Raises
    ------
    ValueError
        If the orbital counts disagree.
    """
    nawf_A, nawf_H = int(A.shape[1]), int(HRs.shape[0])
    if nawf_A != nawf_H or HRs.shape[1] != nawf_H:
        raise ValueError(
            'the projections span %d orbitals but the Hamiltonian %d: the coupling and the '
            'Hamiltonian must be expressed in the same PAO basis (same pseudopotentials, '
            'BASIS tables and configuration; see check_same_pao_basis).' % (nawf_A, nawf_H)
        )
    if orbital_positions is not None and np.asarray(orbital_positions).shape[0] != nawf_H:
        raise ValueError(
            'orbital_positions has %d rows for a %d-orbital Hamiltonian.'
            % (np.asarray(orbital_positions).shape[0], nawf_H)
        )


_HUBBARD_KINDS = {0: 'DFT+U', 1: 'DFT+U+J', 2: 'DFT+U+V'}
_SKIP_TAGS = ('Hubbard_ns', 'Hubbard_Occ', 'Hubbard_ns_nc', 'starting_ns', 'starting_ns_nc')


def hubbard_info_from_save(savedir: str) -> dict | None:
    """Hubbard parameters recorded in ``data-file-schema.xml`` of a QE save.

    Parameters
    ----------
    savedir : str
        QE save directory.

    Returns
    -------
    dict or None
        ``None`` for a plain DFT run.  Otherwise ``kind`` (``'DFT+U'``,
        ``'DFT+U+J'``, ``'DFT+U+V'`` from ``lda_plus_u_kind``), ``projectors``
        (``U_projection_type`` or ``Hubbard_projectors``), and ``terms``: a list
        of ``(tag, attributes, value_eV)`` for every scalar ``Hubbard_*`` child
        (``Hubbard_U``, ``Hubbard_V``, ``Hubbard_J0``, ...), in file order.
        Occupation matrices are skipped.

    Notes
    -----
    The element layout of ``<dftU>`` changed across QE versions; every scalar
    child whose tag starts with ``Hubbard_`` is kept with its attributes
    (``specie``, ``label``, ``spec1``, ``index`` ...) so that nothing is lost.
    Values are in Hartree in the XML and converted to eV.
    """
    root = ET.parse(os.path.join(savedir, 'data-file-schema.xml')).getroot()
    dftu = root.find('output/dft/dftU')
    if dftu is None:
        dftu = root.find('input/dft/dftU')
    if dftu is None:
        return None
    kind_text = dftu.findtext('lda_plus_u_kind')
    kind = (
        _HUBBARD_KINDS.get(int(kind_text), 'DFT+U (kind %s)' % kind_text) if kind_text else 'DFT+U'
    )
    projectors = dftu.findtext('U_projection_type') or dftu.findtext('Hubbard_projectors') or ''
    terms = []
    for child in dftu:
        tag = child.tag.split('}')[-1]
        if not tag.startswith('Hubbard_') or tag in _SKIP_TAGS or child.text is None:
            continue
        try:
            value = float(child.text.strip().split()[0])
        except (ValueError, IndexError):
            continue
        if len(child.text.split()) != 1:
            continue  # a matrix, not a scalar
        terms.append((tag, dict(child.attrib), value * HARTREE_TO_EV))
    return {'kind': kind, 'projectors': projectors.strip(), 'terms': terms}


def format_hubbard_info(info: dict | None, threshold_ev: float = 1.0e-6) -> str:
    """One-line description of :func:`hubbard_info_from_save` (``'PBE'`` for ``None``).

    Parameters
    ----------
    info : dict or None
        Output of :func:`hubbard_info_from_save`.
    threshold_ev : float, optional
        Terms below this magnitude are omitted.

    Returns
    -------
    str
        E.g. ``'DFT+U+V (ortho-atomic): Hubbard_U Si 3p 3.00 eV; Hubbard_V ... 1.66 eV'``.
    """
    if info is None:
        return 'PBE'
    parts = []
    for tag, attrib, value in info['terms']:
        if abs(value) < threshold_ev:
            continue
        label = ' '.join(str(attrib[k]) for k in attrib if k not in ('units',))
        parts.append(
            '%s %s %.3f eV' % (tag, label, value) if label else '%s %.3f eV' % (tag, value)
        )
    head = info['kind'] + (' (%s)' % info['projectors'] if info['projectors'] else '')
    return head + (': ' + '; '.join(parts) if parts else '')


__all__: Sequence[str] = (
    'PAOElectrons',
    'check_orbital_dimensions',
    'check_same_pao_basis',
    'format_hubbard_info',
    'hubbard_info_from_save',
    'pao_hamiltonian_from_save',
)
