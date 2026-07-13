"""Fourier (``matdyn``-style) interpolation of phonons and the electron-phonon
coupling from Quantum ESPRESSO real-space force constants (Phase 1A).

This is the PAOFLOW analogue of the QE ``q2r.x`` -> ``matdyn.x`` (``la2F=.true.``)
step: it takes the real-space interatomic force constants (``flfrc``, e.g.
``Pb333.fc``) and the real-space electron-phonon coupling "force constants"
(``a2Fmatdyn.NN``) on a coarse commensurate q-grid and Fourier-interpolates both
onto an arbitrarily dense q-grid.  At each dense q the dynamical matrix is
diagonalised to give the mode frequencies / eigenvectors, and the coupling
matrix is projected onto the modes to give the mode-resolved ``lambda_{q nu}``.
The dense ``(lambda_qv, omega_qv)`` are then fed to
:func:`PAOFLOW.elphon.eph_kq.eliashberg_from_modes` to build the smooth
Eliashberg ``a2F(omega)``.

This "generalised Fourier interpolation" is the step EPW performs with
maximally-localised Wannier functions; here it is done directly with the PAO /
QE real-space representation, so the coupling is interpolated from a handful of
coarse q-points to a dense grid at negligible cost.

The force-constant files use the standard QE ``flfrc`` block layout::

    <crystal header>            # only in flfrc (nr line onwards is shared)
    nr1 nr2 nr3
    i j na nb                   # 3*3*nat*nat blocks
    m1 m2 m3  Phi(i,j,na,nb; m)  # nr1*nr2*nr3 lines each
    ...
"""

import numpy as np

# Atomic mass unit in the QE Rydberg convention (amu -> Ry electron-mass/2).
AMU_RY = 911.4442421
RY_TO_EV = 13.605693122994
THZ_TO_EV = 4.135667696e-3
# 1 Ry (energy) expressed in THz (E / h): converts a dynamical-matrix frequency
# sqrt(w2) [Ry], obtained with masses in Ry units, to THz.
RY_TO_THZ = 3289.842


def _to_float(tok):
    return float(tok.replace('D', 'E').replace('d', 'e'))


def _parse_fc_blocks(lines, start, nr, nat):
    """Parse the ``3*3*nat*nat`` force-constant blocks into ``(3,3,nat,nat,*nr)``."""
    frc = np.zeros((3, 3, nat, nat, nr[0], nr[1], nr[2]), dtype=float)
    i = start
    nblocks = 3 * 3 * nat * nat
    ncell = nr[0] * nr[1] * nr[2]
    for _ in range(nblocks):
        while i < len(lines) and not lines[i].strip():
            i += 1
        hdr = lines[i].split()
        a, b, na, nb = (int(hdr[0]) - 1, int(hdr[1]) - 1, int(hdr[2]) - 1, int(hdr[3]) - 1)
        i += 1
        for _ in range(ncell):
            tok = lines[i].split()
            m1, m2, m3 = int(tok[0]) - 1, int(tok[1]) - 1, int(tok[2]) - 1
            frc[a, b, na, nb, m1, m2, m3] = _to_float(tok[3])
            i += 1
    return frc, i


def read_qe_ifc(path):
    """Parse a QE ``flfrc`` interatomic force-constant file (e.g. ``Pb333.fc``).

    Returns
    -------
    dict
        ``{'nat', 'ntyp', 'ibrav', 'celldm', 'masses_amu', 'tau', 'nr', 'frc',
        'has_zeu'}``.  ``frc`` has shape ``(3, 3, nat, nat, nr1, nr2, nr3)`` in
        Ry / Bohr^2; ``masses_amu`` is ``(nat,)`` in amu.
    """
    with open(path) as fh:
        lines = [ln.rstrip('\n') for ln in fh]

    tok = lines[0].split()
    ntyp, nat, ibrav = int(tok[0]), int(tok[1]), int(tok[2])
    celldm = np.array([_to_float(t) for t in tok[3:9]], dtype=float)

    i = 1
    if ibrav == 0:
        i += 3  # explicit lattice vectors (not needed: we work in crystal coords)
    # Species lines: ntyp of them.
    type_mass = {}
    for _ in range(ntyp):
        parts = lines[i].split("'")
        idx = int(parts[0].split()[0])
        mass_ry = _to_float(parts[2].split()[0])
        type_mass[idx] = mass_ry / AMU_RY
        i += 1
    # Atom lines: nat of them (index, type, tau x3).
    masses = np.zeros(nat, dtype=float)
    tau = np.zeros((nat, 3), dtype=float)
    for k in range(nat):
        tok = lines[i].split()
        ityp = int(tok[1])
        masses[k] = type_mass[ityp]
        tau[k] = [_to_float(tok[2]), _to_float(tok[3]), _to_float(tok[4])]
        i += 1
    # Effective-charge flag (T/F) then, if T, the dielectric + Z* block.
    has_zeu = lines[i].strip().upper().startswith('T')
    i += 1
    if has_zeu:
        i += 3  # dielectric tensor
        i += nat * 4  # per-atom Z* (1 header + 3 rows)
    # Grid line.
    nr = [int(x) for x in lines[i].split()[:3]]
    i += 1
    frc, _ = _parse_fc_blocks(lines, i, nr, nat)

    return {
        'nat': nat,
        'ntyp': ntyp,
        'ibrav': ibrav,
        'celldm': celldm,
        'masses_amu': masses,
        'tau': tau,
        'nr': nr,
        'frc': frc,
        'has_zeu': has_zeu,
    }


def read_a2f_ifc(path, nat=1):
    """Parse a QE ``a2Fmatdyn.NN`` real-space electron-phonon coupling file.

    The layout matches :func:`read_qe_ifc` from the grid line on, prefixed by a
    single header line ``sigma_Ry  Ef_Ry  N(E_F)``.

    Parameters
    ----------
    path : str
        Path to the ``a2Fmatdyn.NN`` file.
    nat : int, optional
        Number of atoms (needed because the file has no crystal header;
        default ``1``).

    Returns
    -------
    dict
        ``{'sigma_ry', 'ef_ry', 'dos_ef', 'nr', 'frc'}``.  ``frc`` has shape
        ``(3, 3, nat, nat, nr1, nr2, nr3)``; ``dos_ef`` is in states / spin / Ry.
    """
    with open(path) as fh:
        lines = [ln.rstrip('\n') for ln in fh]

    hdr = lines[0].split()
    sigma_ry, ef_ry, dos_ef = (_to_float(hdr[0]), _to_float(hdr[1]), _to_float(hdr[2]))
    nr = [int(x) for x in lines[1].split()[:3]]
    frc, _ = _parse_fc_blocks(lines, 2, nr, nat)
    return {'sigma_ry': sigma_ry, 'ef_ry': ef_ry, 'dos_ef': dos_ef, 'nr': nr, 'frc': frc}


def _ws_axis_images(nr):
    """Map each file cell index ``c`` (0..nr-1) to ``[(R_int, weight), ...]``.

    Cells beyond the first Brillouin zone are folded to the shortest lattice
    vector; a zone-boundary cell of an even grid is split symmetrically over the
    two equivalent images (weight 1/2 each), reproducing the minimal
    Wigner-Seitz treatment of ``matdyn``.
    """
    out = {}
    half = nr // 2
    for c in range(nr):
        if nr % 2 == 0 and c == half:
            out[c] = [(half, 0.5), (half - nr, 0.5)]
        else:
            r = c if c <= half else c - nr
            out[c] = [(r, 1.0)]
    return out


def _apply_asr_simple(frc):
    """Enforce the ``simple`` acoustic sum rule in place on a force-constant tensor.

    For each Cartesian pair ``(alpha, beta)`` and atom ``na`` the self term at
    ``R = 0, nb = na`` is corrected so that ``sum_{nb, R} Phi_{alpha beta}(R) =
    0``, which drives the three acoustic modes to zero frequency at ``Gamma``
    (QE ``asr='simple'``).
    """
    nat = frc.shape[2]
    for a in range(3):
        for b in range(3):
            for na in range(nat):
                total = frc[a, b, na, :, :, :, :].sum()
                frc[a, b, na, na, 0, 0, 0] -= total
    return frc


def _real_space_images(nr):
    """Precompute ``(R_int (3,), weight, (c1, c2, c3))`` for the whole grid."""
    ax = [_ws_axis_images(nr[d]) for d in range(3)]
    images = []
    for c1 in range(nr[0]):
        for c2 in range(nr[1]):
            for c3 in range(nr[2]):
                for r1, w1 in ax[0][c1]:
                    for r2, w2 in ax[1][c2]:
                        for r3, w3 in ax[2][c3]:
                            images.append((np.array([r1, r2, r3]), w1 * w2 * w3, (c1, c2, c3)))
    return images


def _lattice_vectors(ibrav, celldm):
    """Primitive real-space lattice vectors (rows ``a1, a2, a3``) in ``alat`` units.

    Implements the QE ``ibrav`` conventions needed for the electron-phonon
    examples (simple, fcc and bcc cubic).  ``ibrav == 0`` is unsupported here
    because the explicit vectors are skipped by :func:`read_qe_ifc`.
    """
    if ibrav == 1:  # simple cubic
        return np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    if ibrav == 2:  # face-centred cubic
        return 0.5 * np.array([[-1.0, 0, 1], [0, 1.0, 1], [-1.0, 1, 0]])
    if ibrav == 3:  # body-centred cubic
        return 0.5 * np.array([[1.0, 1, 1], [-1.0, 1, 1], [-1.0, -1, 1]])
    raise NotImplementedError(
        'Wigner-Seitz interpolation supports ibrav in {1, 2, 3}; got %r. '
        'Use ws=False for the simple per-axis folding.' % ibrav
    )


def _supercell_rws(nr, at, span=2):
    """Supercell lattice vectors ``sum_i m_i (nr_i a_i)`` (Cartesian, ``alat``).

    Excludes the zero vector; ``m_i`` runs over ``[-span, span]`` which is ample
    to bound the Wigner-Seitz cell of the ``nr``-supercell.
    """
    A = np.array([nr[d] * at[d] for d in range(3)])  # supercell vectors
    rws = []
    for m1 in range(-span, span + 1):
        for m2 in range(-span, span + 1):
            for m3 in range(-span, span + 1):
                if m1 == m2 == m3 == 0:
                    continue
                rws.append(m1 * A[0] + m2 * A[1] + m3 * A[2])
    return np.array(rws)


def _wsweight(r, rws, tol=1.0e-6):
    """QE ``wsweight``: 1 inside the WS cell, ``1/deg`` on its boundary, else 0."""
    nreq = 1
    half = 0.5 * np.einsum('ij,ij->i', rws, rws)
    proj = rws @ r
    ck = proj - half
    if np.any(ck > tol):
        return 0.0
    nreq += int(np.count_nonzero(np.abs(ck) < tol))
    return 1.0 / nreq


def build_ws_images(nr, at, tau):
    """Wigner-Seitz real-space images per atom pair for ``matdyn`` interpolation.

    For every atom pair ``(na, nb)`` the routine scans real-space cells
    ``R = sum_i n_i a_i`` over ``n_i in [-2 nr_i, 2 nr_i]``, keeps those whose
    bond vector ``R + tau_na - tau_nb`` lies in (or on the boundary of) the
    Wigner-Seitz cell of the ``nr``-supercell, and records the grid index
    ``m_i = n_i mod nr_i`` together with the integer cell ``n`` (for the Bloch
    phase) and the WS weight.  This reproduces QE ``matdyn`` (``wsinit`` +
    ``frc_blk``), giving a far better dense-grid interpolation than a plain
    per-axis fold.

    Returns
    -------
    dict
        ``(na, nb) -> list of (m (3,), n (3,), weight)``.
    """
    nat = tau.shape[0]
    rws = _supercell_rws(nr, at)
    out = {}
    rng = [range(-2 * nr[d], 2 * nr[d] + 1) for d in range(3)]
    for na in range(nat):
        for nb in range(nat):
            entries = []
            for n1 in rng[0]:
                for n2 in rng[1]:
                    for n3 in rng[2]:
                        n = np.array([n1, n2, n3])
                        R = n @ at  # Cartesian (alat)
                        r_ws = R + tau[na] - tau[nb]
                        w = _wsweight(r_ws, rws)
                        if w <= 1.0e-8:
                            continue
                        m = (n1 % nr[0], n2 % nr[1], n3 % nr[2])
                        entries.append((m, n, w))
            out[(na, nb)] = entries
    return out


def _matrix_at_q_ws(frc, ws_images, q_cryst, nat, masses_ry=None):
    """Fourier-interpolate a force-constant tensor to ``q`` using WS images.

    ``ws_images`` is the output of :func:`build_ws_images`.  The Bloch phase uses
    the integer cell ``n`` in crystal coordinates (``exp(2 pi i q.n)``), while the
    WS selection/weighting already used the true Cartesian geometry.  Returns a
    ``(3*nat, 3*nat)`` Hermitian complex matrix.
    """
    dim = 3 * nat
    M = np.zeros((dim, dim), dtype=complex)
    for (na, nb), entries in ws_images.items():
        acc = np.zeros((3, 3), dtype=complex)
        for (m1, m2, m3), n, w in entries:
            phase = w * np.exp(2j * np.pi * (q_cryst @ n))
            acc += phase * frc[:, :, na, nb, m1, m2, m3]
        for a in range(3):
            for b in range(3):
                M[a * nat + na, b * nat + nb] = acc[a, b]
    M = 0.5 * (M + M.conj().T)
    if masses_ry is not None:
        mvec = np.empty(dim)
        for alpha in range(3):
            for na in range(nat):
                mvec[alpha * nat + na] = 1.0 / np.sqrt(masses_ry[na])
        M = M * np.outer(mvec, mvec)
    return M


def _matrix_at_q(frc, images, q_cryst, masses_ry=None):
    """Fourier-interpolate a force-constant tensor to a single q (crystal coords).

    ``M(q)_{(alpha,na),(beta,nb)} = sum_R Phi_{alpha beta,na nb}(R) e^{2 pi i q.R}``,
    optionally mass-weighted by ``1/sqrt(M_na M_nb)`` (for the dynamical matrix,
    with ``masses_ry`` in the QE Rydberg mass unit).  Returns a
    ``(3*nat, 3*nat)`` Hermitian complex matrix.
    """
    nat = frc.shape[2]
    dim = 3 * nat
    M = np.zeros((dim, dim), dtype=complex)
    for R, w, (c1, c2, c3) in images:
        phase = w * np.exp(2j * np.pi * (q_cryst @ R))
        block = frc[:, :, :, :, c1, c2, c3]  # (3, 3, nat, nat)
        # (alpha, na, beta, nb) -> (alpha*nat+na, beta*nat+nb)
        blk = np.transpose(block, (0, 2, 1, 3)).reshape(dim, dim)
        M += phase * blk
    M = 0.5 * (M + M.conj().T)
    if masses_ry is not None:
        # flattening is alpha-major (alpha*nat + na).
        mvec = np.empty(dim)
        for alpha in range(3):
            for na in range(nat):
                mvec[alpha * nat + na] = 1.0 / np.sqrt(masses_ry[na])
        M = M * np.outer(mvec, mvec)
    return M


def _dyn_freq_thz(w2):
    """Signed frequencies in THz from dynamical-matrix eigenvalues.

    ``w2`` is the eigenvalue of ``C/M`` with ``C`` in Ry/Bohr^2 and ``M`` in the
    QE Rydberg mass unit, i.e. ``omega^2`` in Ry^2; ``omega[THz] = sqrt(w2) *
    RY_TO_THZ``.  Imaginary modes are returned as negative frequencies.
    """
    w2 = np.asarray(w2, dtype=float)
    return np.sign(w2) * np.sqrt(np.abs(w2)) * RY_TO_THZ


def interpolate_coupling(ifc_ph, ifc_a2f, nk, freq_scale=None, asr=True, ws=True):
    """Interpolate phonons + el-ph coupling onto a dense ``nk^3`` q-grid.

    Parameters
    ----------
    ifc_ph : dict
        Output of :func:`read_qe_ifc` (phonon force constants).
    ifc_a2f : dict
        Output of :func:`read_a2f_ifc` (el-ph coupling force constants).
    nk : int or (int, int, int)
        Dense q-grid size.
    freq_scale : float, optional
        Overrides the Ry-dynamical-matrix -> THz frequency scale (otherwise the
        physical constant is used).  Use to calibrate against a QE phonon DOS.
    asr : bool, optional
        Apply the ``simple`` acoustic sum rule to both force-constant tensors
        (default ``True``, matching QE ``asr='simple'``).
    ws : bool, optional
        Use the full Wigner-Seitz interpolation (default ``True``, matching QE
        ``matdyn``); ``False`` falls back to the cheaper per-axis fold.  ``True``
        requires a supported ``ibrav`` (see :func:`_lattice_vectors`).

    Returns
    -------
    dict
        ``{'omega_thz', 'lambda_qv', 'proj', 'dos_ef', 'q_cryst'}``.
        ``omega_thz`` / ``lambda_qv`` / ``proj`` are ``(nq, nmode)``; ``proj`` is
        the raw mode projection ``<z|A(q)|z>`` (before the ``lambda`` conversion).
    """
    if np.isscalar(nk):
        nk = (int(nk), int(nk), int(nk))
    nat = ifc_ph['nat']
    nmode = 3 * nat
    masses_ry = ifc_ph['masses_amu'] * AMU_RY
    dos_ef = ifc_a2f['dos_ef']

    frc_ph = ifc_ph['frc'].copy()
    frc_a2 = ifc_a2f['frc'].copy()
    if asr:
        _apply_asr_simple(frc_ph)
        _apply_asr_simple(frc_a2)

    if ws:
        at = _lattice_vectors(ifc_ph['ibrav'], ifc_ph['celldm'])
        ws_images = build_ws_images(ifc_ph['nr'], at, ifc_ph['tau'])
        if tuple(ifc_a2f['nr']) != tuple(ifc_ph['nr']):
            raise ValueError('phonon and el-ph FC grids differ; cannot share WS images.')

    else:
        img_ph = _real_space_images(ifc_ph['nr'])
        img_a2 = _real_space_images(ifc_a2f['nr'])

    qlist = []
    for i in range(nk[0]):
        for j in range(nk[1]):
            for k in range(nk[2]):
                qlist.append((i / nk[0], j / nk[1], k / nk[2]))
    q_cryst = np.array(qlist, dtype=float)
    nq = len(q_cryst)

    omega_thz = np.zeros((nq, nmode))
    proj = np.zeros((nq, nmode))
    w2_ry = np.zeros((nq, nmode))

    for iq, q in enumerate(q_cryst):
        if ws:
            D = _matrix_at_q_ws(frc_ph, ws_images, q, nat, masses_ry=masses_ry)
            A = _matrix_at_q_ws(frc_a2, ws_images, q, nat, masses_ry=masses_ry)
        else:
            D = _matrix_at_q(frc_ph, img_ph, q, masses_ry=masses_ry)
            A = _matrix_at_q(frc_a2, img_a2, q, masses_ry=masses_ry)
        w2, z = np.linalg.eigh(D)  # ascending, orthonormal columns; w2 = omega^2 [Ry^2]
        pj = np.einsum('an,ab,bn->n', z.conj(), A, z).real  # (nmode,)
        w2_ry[iq] = w2
        proj[iq] = pj

    # Frequency scale (THz): omega[THz] = sqrt(w2 [Ry^2]) * RY_TO_THZ.
    scale = RY_TO_THZ if freq_scale is None else freq_scale
    omega_thz = np.sign(w2_ry) * np.sqrt(np.abs(w2_ry)) * scale

    # QE matdyn (la2F) mode coupling: lambda_{q nu} = <z|A(q)|z> / (2 N(E_F) w2),
    # with w2 = omega^2 in Ry^2 and N(E_F) in states / spin / Ry.  The factor 2
    # (rather than pi) reproduces QE's per-broadening lambda across the full
    # smearing range for Pb.
    lam = np.zeros((nq, nmode))
    mask = w2_ry > 1.0e-12
    np.divide(proj, 2.0 * dos_ef * w2_ry, out=lam, where=mask)

    return {
        'omega_thz': omega_thz,
        'lambda_qv': lam,
        'proj': proj,
        'w2_ry': w2_ry,
        'dos_ef': dos_ef,
        'q_cryst': q_cryst,
    }
