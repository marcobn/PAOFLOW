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
):
    """Rebuild the PAO real-space Hamiltonian ``HRs`` for one supercell ``.save``.

    Runs the standard PAOFLOW pipeline (projections -> projectability ->
    ``pao_hamiltonian``) on the displaced supercell and returns its ``HRs``
    array ``(nawf, nawf, nk1, nk2, nk3, nspin)``.
    """
    from ..PAOFLOW import PAOFLOW

    pf = PAOFLOW(workpath=workpath, savedir=savedir, outputdir='.', verbose=False)
    if basispath is not None:
        pf.projections(basispath=basispath, configuration=configuration)
    else:
        pf.projections(configuration=configuration)
    pf.projectability(pthr=pthr)
    pf.pao_hamiltonian(shift_type=shift_type)

    arry, _ = pf.data_controller.data_dicts()
    return np.asarray(arry['HRs'])


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


def compute_dV(
    data_controller,
    elphon_dir='elphon',
    configuration=None,
    basispath=None,
    pthr=0.95,
    shift_type=1,
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

    def _hrs(prefix):
        return build_supercell_HRs(
            _save_dir_for(prefix),
            workpath=edir,
            configuration=configuration,
            basispath=basispath,
            pthr=pthr,
            shift_type=shift_type,
        )

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
        directional.append({'sc_atom': int(d['sc_atom']), 'displacement': vec.tolist(), 'dV': dV})

    result = {
        'distance': distance,
        'reference_prefix': reference_prefix,
        'directional': directional,
    }
    arry['elphon_dV'] = result
    return result
