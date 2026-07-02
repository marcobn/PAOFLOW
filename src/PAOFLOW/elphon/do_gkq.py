"""Assemble the real-space electron-phonon tensor g(kappa,alpha; R_e, R_p) (P2).

Ties together the finite-difference derivatives, the Cartesian solve and the
supercell -> primitive fold:

1. :func:`PAOFLOW.elphon.dvscf_fd.compute_dV` gives the directional derivatives
   (one per symmetry-reduced / Cartesian displacement).
2. For every displaced atom the directional responses are combined into the
   three Cartesian derivatives
   (:func:`PAOFLOW.elphon.symmetry.cartesian_derivatives_from_directional`).
3. Each Cartesian derivative (supercell basis) is folded to the primitive
   electron-phonon tensor (:func:`PAOFLOW.elphon.fold.fold_dV_to_primitive`).

The result is ``g_R`` of shape
``(natom_prim * 3, nawf_prim, nawf_prim, N1e, N2e, N3e, s1, s2, s3, nspin)`` --
the Cartesian derivative ``dH/du_{kappa,alpha}`` of the primitive hopping over
the electron grid ``R_e`` and the commensurate phonon-cell grid ``R_p``.
"""

import numpy as np

from ..phonon.do_phonopy import init_phonopy
from .dvscf_fd import compute_dV
from .fold import fold_dV_to_primitive, supercell_atom_translations, supercell_naw
from .symmetry import cartesian_derivatives_from_directional


def enforce_acoustic_sum_rule(g_R, cart_index):
    """Impose the translational acoustic sum rule on the real-space e-ph tensor.

    A rigid displacement of the whole crystal (every atom ``kappa`` in every
    cell ``R_p`` moved by the same vector along ``alpha``) leaves each primitive
    hopping ``H_{mn}(R_e)`` unchanged, hence

    .. math::

        \\sum_{\\kappa, R_p} g_{\\kappa\\alpha,\\,mn}(R_e, R_p) = 0

    for every ``alpha``, ``R_e``, ``m``, ``n`` and spin.  Finite differences on a
    metal built with the ``eta``-shift completion satisfy this only up to
    numerical noise, so -- exactly as EPW/phonopy enforce the ASR on the
    dynamical matrix and e-ph coupling -- the residual is removed by subtracting
    its mean equally from every ``(kappa, R_p)`` slice sharing a Cartesian
    direction ``alpha``.

    Parameters
    ----------
    g_R : ndarray
        Shape ``(natom*3, nawf, nawf, N1e, N2e, N3e, s1, s2, s3, nspin)``.
    cart_index : list[tuple[int, int]]
        ``(prim_atom, alpha)`` label of each leading row of ``g_R``.

    Returns
    -------
    g_R : ndarray
        The corrected tensor (modified in place and returned).
    residual : float
        Frobenius norm of the rigid-shift residual *before* enforcement,
        ``||sum_{kappa,R_p} g||``, summed over the three Cartesian directions --
        a small value (relative to ``||g||``) certifies a trustworthy tensor.
    """
    g_R = np.asarray(g_R)
    alphas = np.array([a for (_, a) in cart_index], dtype=int)
    n_rp = int(np.prod(g_R.shape[6:9]))

    residual_sq = 0.0
    for alpha in range(3):
        rows = np.where(alphas == alpha)[0]
        if rows.size == 0:
            continue
        block = g_R[rows]  # (n_kappa, nawf, nawf, Ne.., Rp.., nspin)
        # Sum over the displaced atoms (axis 0) and their cells R_p (axes 6,7,8).
        total = block.sum(axis=(0, 6, 7, 8))  # (nawf, nawf, Ne.., nspin)
        residual_sq += float(np.vdot(total, total).real)
        correction = total / (rows.size * n_rp)
        # Broadcast back over kappa and R_p and subtract.
        g_R[rows] -= correction[None, :, :, :, :, :, None, None, None, :]

    return g_R, float(np.sqrt(residual_sq))


def assemble_eph_tensor(
    data_controller,
    elphon_dir='elphon',
    configuration=None,
    basispath=None,
    pthr=0.95,
    shift_type=1,
    enforce_asr=True,
):
    """Build the primitive Cartesian electron-phonon real-space tensor.

    Requires directional derivatives that span the three Cartesian directions
    for every displaced atom (``displacement_mode='cartesian'``, or
    ``'symmetry'`` once the Wigner-D expansion is applied).

    Parameters
    ----------
    enforce_asr : bool, optional
        If ``True`` (default), impose the translational acoustic sum rule on the
        assembled tensor via :func:`enforce_acoustic_sum_rule`.  The pre-
        enforcement residual is reported in the returned ``'asr_residual'``.

    Returns
    -------
    dict
        ``{'g_R', 'cart_index', 's2p', 'translations', 'naw', 'asr_residual'}``
        where ``g_R`` has shape ``(natom_prim*3, nawf_prim, nawf_prim, N1e, N2e,
        N3e, s1, s2, s3, nspin)`` and ``cart_index`` lists the ``(prim_atom,
        alpha)`` label of each leading row.  Also stored on the controller as
        ``arry['elphon_g_R']``.
    """
    arry, attr = data_controller.data_dicts()

    result = arry.get('elphon_dV')
    if result is None:
        result = compute_dV(
            data_controller,
            elphon_dir=elphon_dir,
            configuration=configuration,
            basispath=basispath,
            pthr=pthr,
            shift_type=shift_type,
        )
    directional = result['directional']

    phonon = init_phonopy(data_controller)
    s2p, translations = supercell_atom_translations(phonon)

    if configuration is None:
        configuration = attr.get('elphon_configuration', 'standard')
    from ..phonon.io import _pp_filenames

    pp_dir = attr.get('fpath', '.')
    naw = supercell_naw(phonon, configuration, _pp_filenames(data_controller), pp_dir)

    supercell_matrix = phonon.supercell_matrix

    # Group directional derivatives by the primitive atom they displace.
    by_prim = {}
    for d in directional:
        p = int(s2p[int(d['sc_atom'])])
        by_prim.setdefault(p, []).append(d)

    nprim = int(s2p.max()) + 1
    g_blocks = []
    cart_index = []
    for p in range(nprim):
        entries = by_prim.get(p, [])
        directions = np.array([e['displacement'] for e in entries], dtype=float)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        units = directions / norms
        responses = np.array([e['dV'] for e in entries])
        # Cartesian derivatives (supercell basis) for this atom: (3, nawf_sc, ...).
        d_cart = cartesian_derivatives_from_directional(units, responses)
        for alpha in range(3):
            g_alpha = fold_dV_to_primitive(d_cart[alpha], s2p, translations, naw, supercell_matrix)
            g_blocks.append(g_alpha)
            cart_index.append((p, alpha))

    g_R = np.stack(g_blocks, axis=0)  # (natom_prim*3, nawf_prim, nawf_prim, Ne..., Rp..., nspin)

    asr_residual = None
    if enforce_asr:
        g_R, asr_residual = enforce_acoustic_sum_rule(g_R, cart_index)

    out = {
        'g_R': g_R,
        'cart_index': cart_index,
        's2p': s2p,
        'translations': translations,
        'naw': naw,
        'asr_residual': asr_residual,
    }
    arry['elphon_g_R'] = out
    return out
