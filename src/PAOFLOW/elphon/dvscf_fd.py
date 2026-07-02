"""Finite-difference ``dV = dH/du`` in the supercell PAO basis (P1).

For every reference-cell displacement the PAO real-space Hamiltonian ``HRs`` of
the ``+delta`` and ``-delta`` supercells is rebuilt (from the displaced DFT
``.save`` via the standard projection / ``pao_hamiltonian`` pipeline) and
central-differenced,

    dV_{kappa,alpha}(R) = (HRs[+delta] - HRs[-delta]) / (2 delta),

giving the derivative of the Hamiltonian with respect to displacing atom
``kappa`` along Cartesian direction ``alpha``, in the supercell PAO basis.  The
fold to the primitive cell and the assembly of ``g_mn^v(k, q)`` follow in P2.
"""

import json
import os

import numpy as np

from ..phonon.io import resolve_phonon_dir
from .io import MANIFEST


def finite_difference_dV(hrs_plus, hrs_minus, distance):
    """Central finite difference of the supercell PAO Hamiltonian.

    Parameters
    ----------
    hrs_plus, hrs_minus : ndarray
        ``HRs`` of the ``+delta`` / ``-delta`` supercells, identical shape
        ``(nawf, nawf, nk1, nk2, nk3, nspin)`` (eV).
    distance : float
        Displacement amplitude ``delta`` in Bohr.

    Returns
    -------
    ndarray
        ``dV = (HRs[+] - HRs[-]) / (2 delta)`` (eV/Bohr), same shape.
    """
    hp = np.asarray(hrs_plus)
    hm = np.asarray(hrs_minus)
    if hp.shape != hm.shape:
        raise ValueError('H(+) and H(-) shapes differ: %s vs %s' % (hp.shape, hm.shape))
    if distance == 0:
        raise ValueError('displacement distance must be non-zero.')
    return (hp - hm) / (2.0 * float(distance))


def build_supercell_HRs(
    savedir,
    workpath='.',
    configuration='standard',
    basispath=None,
    pthr=0.95,
    shift_type=1,
    return_shift=False,
):
    """Rebuild the PAO real-space Hamiltonian ``HRs`` for one supercell ``.save``.

    Runs the standard PAOFLOW pipeline (projections -> projectability ->
    ``pao_hamiltonian``) on the displaced supercell and returns its ``HRs``
    array ``(nawf, nawf, nk1, nk2, nk3, nspin)``.  If ``return_shift`` is
    ``True`` also returns the completion energy ``eta`` (``attr['shift']``),
    needed to build the good-projectability subspace projector.
    """
    from ..PAOFLOW import PAOFLOW

    pf = PAOFLOW(workpath=workpath, savedir=savedir, outputdir='.', verbose=False)
    if basispath is not None:
        pf.projections(basispath=basispath, configuration=configuration)
    else:
        pf.projections(configuration=configuration)
    pf.projectability(pthr=pthr)
    pf.pao_hamiltonian(shift_type=shift_type)

    arry, attr = pf.data_controller.data_dicts()
    hrs = np.asarray(arry['HRs'])
    if return_shift:
        return hrs, float(attr['shift'])
    return hrs


def good_subspace_projectors(HRs_ref, eta, tol=0.05):
    """Build the good-projectability subspace projectors ``P_good(K)``.

    The PAO Hamiltonian completes the unrepresented subspace with
    ``+eta(1 - P)``, so the complement eigenstates of ``H^PAO(K)`` sit *exactly*
    at ``eta`` while the physical (projectable) bands lie strictly below it.
    Diagonalising the reference ``H^PAO(K)`` (from ``HRs_ref``) and keeping the
    eigenvectors with eigenvalue ``< eta - tol`` therefore isolates the good
    subspace, and ``P_good = sum_good |v><v|``.

    Parameters
    ----------
    HRs_ref : ndarray, ``(nawf, nawf, n1, n2, n3, nspin)``
        Reference (undisplaced) supercell PAO Hamiltonian in real space.
    eta : float
        Completion energy ``attr['shift']``.
    tol : float
        Margin below ``eta`` separating good bands from the complement.

    Returns
    -------
    ndarray, ``(nawf, nawf, n1, n2, n3, nspin)`` complex
        The projector ``P_good(K)`` on the supercell k-grid (folded-``fftfreq``).
    """
    HRs_ref = np.asarray(HRs_ref)
    nawf, _, n1, n2, n3, nspin = HRs_ref.shape
    nk = n1 * n2 * n3
    Hk = np.fft.ifftn(HRs_ref, axes=(2, 3, 4)) * nk  # (nawf, nawf, n1, n2, n3, nspin)
    thr = eta - max(tol, 1.0e-3 * abs(eta))
    Pg = np.zeros_like(Hk)
    for s in range(nspin):
        M = np.moveaxis(Hk[..., s], (0, 1), (-2, -1)).reshape(nk, nawf, nawf)
        M = 0.5 * (M + np.conj(np.transpose(M, (0, 2, 1))))
        w, V = np.linalg.eigh(M)
        P = np.zeros((nk, nawf, nawf), dtype=complex)
        for K in range(nk):
            Vs = V[K][:, w[K] < thr]
            P[K] = Vs @ Vs.conj().T
        Pg[..., s] = np.moveaxis(P.reshape(n1, n2, n3, nawf, nawf), (-2, -1), (0, 1))
    return Pg


def project_dV_good(dV_R, Pg):
    """Project a real-space supercell ``dV`` onto the good subspace ``P_good``.

    Transforms ``dV(R) -> dV(K)``, applies ``P_good(K) dV(K) P_good(K)`` at every
    supercell k-point, and transforms back.  This removes the coupling to (and
    through) the ``eta``-completion complement -- the unphysical off-diagonal
    channel -- while preserving the physical good-good deformation potential.
    """
    dV_R = np.asarray(dV_R)
    nawf, _, n1, n2, n3, nspin = dV_R.shape
    nk = n1 * n2 * n3
    dV_K = np.fft.ifftn(dV_R, axes=(2, 3, 4)) * nk
    out = np.zeros_like(dV_K)
    for s in range(nspin):
        M = np.moveaxis(dV_K[..., s], (0, 1), (-2, -1)).reshape(nk, nawf, nawf)
        P = np.moveaxis(Pg[..., s], (0, 1), (-2, -1)).reshape(nk, nawf, nawf)
        Mg = P @ M @ P
        out[..., s] = np.moveaxis(Mg.reshape(n1, n2, n3, nawf, nawf), (-2, -1), (0, 1))
    return np.fft.fftn(out, axes=(2, 3, 4)) / nk


def _save_dir_for(prefix):
    """Save path of a displaced run, relative to the elphon directory."""
    return os.path.join('tmp_%s' % prefix, '%s.save' % prefix)


def _opposite_index(disps, i, tol=1.0e-8):
    """Index of the ``-displacement`` partner of ``disps[i]`` (or ``None``)."""
    vi = np.asarray(disps[i]['displacement'], dtype=float)
    ai = disps[i]['sc_atom']
    for j, dj in enumerate(disps):
        if j == i or dj['sc_atom'] != ai:
            continue
        if np.allclose(np.asarray(dj['displacement'], dtype=float), -vi, atol=tol):
            return j
    return None


def _read_save_positions(edir, prefix):
    """Cartesian atomic positions (Bohr) from a supercell's QE ``data-file-schema.xml``.

    Returns ``None`` when the file is missing or unparsable (verification is then
    skipped rather than raising a spurious error).
    """
    import xml.etree.ElementTree as ET

    xml = os.path.join(edir, _save_dir_for(prefix), 'data-file-schema.xml')
    if not os.path.isfile(xml):
        return None
    try:
        root = ET.parse(xml).getroot()
    except ET.ParseError:
        return None
    node = root.find('.//output/atomic_structure/atomic_positions')
    if node is None:
        node = root.find('.//atomic_positions')
    if node is None:
        return None
    pos = [[float(x) for x in at.text.split()] for at in node.findall('atom')]
    return np.asarray(pos, dtype=float) if pos else None


def _verify_save_displacements(edir, disps, reference_prefix, tol=1.0e-3):
    """Guard against stale supercell saves that no longer match the manifest.

    Reads the atomic positions actually stored in each displaced ``.save`` and
    checks that the displaced atom moved by the manifest displacement vector
    (relative to the reference structure, in Bohr).  Raises ``ValueError`` on a
    mismatch -- e.g. a ``.save`` left over from an earlier displacement mode whose
    ``pw.x`` was never re-run -- which would otherwise silently corrupt ``dV``.

    Verification is skipped for any displacement whose positions cannot be read.
    """
    ref_pos = _read_save_positions(edir, reference_prefix) if reference_prefix else None

    def _check(got, want, prefix):
        if not np.allclose(got, want, atol=tol):
            raise ValueError(
                'Stale/mismatched supercell save for %r: the atomic displacement '
                'stored in %s is (%.4g, %.4g, %.4g) Bohr but the manifest expects '
                '(%.4g, %.4g, %.4g) Bohr. Re-run pw.x on the current input for this '
                'displacement (or regenerate the inputs).'
                % (prefix, _save_dir_for(prefix), got[0], got[1], got[2], want[0], want[1], want[2])
            )

    if ref_pos is not None:
        for d in disps:
            pos = _read_save_positions(edir, d['prefix'])
            if pos is None or pos.shape != ref_pos.shape:
                continue
            k = int(d['sc_atom'])
            _check(pos[k] - ref_pos[k], np.asarray(d['displacement'], dtype=float), d['prefix'])
        return

    # No reference structure: verify that explicit +/- pairs are true negatives.
    for i, d in enumerate(disps):
        j = _opposite_index(disps, i)
        if j is None or j < i:
            continue
        pi = _read_save_positions(edir, d['prefix'])
        pj = _read_save_positions(edir, disps[j]['prefix'])
        if pi is None or pj is None or pi.shape != pj.shape:
            continue
        k = int(d['sc_atom'])
        _check(pi[k] - pj[k], 2.0 * np.asarray(d['displacement'], dtype=float), d['prefix'])


def compute_dV(
    data_controller,
    elphon_dir='elphon',
    configuration=None,
    basispath=None,
    pthr=0.95,
    shift_type=1,
    verify_saves=True,
    project_good_subspace=True,
):
    """Rebuild ``HRs`` per displaced supercell and finite-difference to ``dV``.

    Reads the ``displacements.json`` manifest written by the generate phase.
    For each symmetry-reduced displacement it computes the *directional*
    derivative of the PAO Hamiltonian per unit displacement along the (Cartesian)
    displacement direction: a central difference when the explicit ``-`` partner
    is present, otherwise a forward difference against the reference supercell.

    The reduced directional derivatives are the response dataset; the full
    ``dH/du_{kappa,alpha}`` tensor for every atom and Cartesian direction is
    reconstructed from them by crystal symmetry in the next stage (analogous to
    phonopy's force-constant symmetrization).

    Returns
    -------
    dict
        ``{'distance', 'reference_prefix', 'directional': [ {sc_atom,
        displacement, dV}, ... ]}`` where ``dV`` is the supercell-basis
        derivative (eV/Bohr).  Also stored as ``arry['elphon_dV']``.
    """
    arry, attr = data_controller.data_dicts()
    edir = os.path.abspath(resolve_phonon_dir(data_controller, elphon_dir))

    with open(os.path.join(edir, MANIFEST)) as fh:
        manifest = json.load(fh)
    if configuration is None:
        configuration = manifest.get('configuration', 'standard')
    distance = float(manifest['displacement_distance'])
    disps = manifest['displacements']
    reference_prefix = manifest.get('reference_prefix')

    if verify_saves:
        _verify_save_displacements(edir, disps, reference_prefix)

    def _hrs(prefix, return_shift=False):
        return build_supercell_HRs(
            _save_dir_for(prefix),
            workpath=edir,
            configuration=configuration,
            basispath=basispath,
            pthr=pthr,
            shift_type=shift_type,
            return_shift=return_shift,
        )

    # Good-projectability subspace projector, built once from the reference
    # supercell.  Projecting dV onto it removes the unphysical coupling through
    # the eta-completion complement (the off-diagonal inter-band artifact), while
    # preserving the physical (eigenvalue-pinned) deformation potential.
    Pgood = None
    if project_good_subspace:
        if not reference_prefix:
            raise ValueError('project_good_subspace needs the reference supercell prefix.')
        h_ref_g, eta_ref = _hrs(reference_prefix, return_shift=True)
        Pgood = good_subspace_projectors(h_ref_g, eta_ref)

    # The reference (u = 0) Hamiltonian is only needed for forward differences.
    needs_reference = any(_opposite_index(disps, i) is None for i in range(len(disps)))
    if needs_reference:
        if not reference_prefix:
            raise ValueError('A forward difference needs the reference supercell prefix.')
        h_ref = _hrs(reference_prefix)

    directional = []
    used = set()
    for i, d in enumerate(disps):
        if i in used:
            continue
        vec = np.asarray(d['displacement'], dtype=float)
        norm = float(np.linalg.norm(vec))
        j = _opposite_index(disps, i)
        if j is not None:
            # Central difference along the displacement direction.
            dV = finite_difference_dV(_hrs(d['prefix']), _hrs(disps[j]['prefix']), norm)
            used.add(i)
            used.add(j)
        else:
            # Forward difference against the reference (u = 0) supercell.
            dV = (_hrs(d['prefix']) - h_ref) / norm
            used.add(i)
        if Pgood is not None:
            dV = project_dV_good(dV, Pgood)
        directional.append({'sc_atom': int(d['sc_atom']), 'displacement': vec.tolist(), 'dV': dV})

    result = {
        'distance': distance,
        'reference_prefix': reference_prefix,
        'directional': directional,
    }
    arry['elphon_dV'] = result
    return result
