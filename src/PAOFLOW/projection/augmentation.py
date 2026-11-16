"""Augmented overlap of ultrasoft and PAW pseudopotentials for the internal projections.

The pseudo-wavefunctions of an ultrasoft or PAW calculation are orthonormal in the metric

.. math::

    S = 1 + \\sum_{a}\\sum_{ij} q^{a}_{ij}\\,|\\beta^{a}_i\\rangle\\langle\\beta^{a}_j|,

not in the plain plane-wave one, so the projection of a Bloch state on an atomic orbital is
:math:`\\langle\\phi|S|\\psi\\rangle`, the Gram matrix of the atomic orbitals is
:math:`\\langle\\phi|S|\\phi\\rangle` (what ``projwfc.x`` uses and writes with
``lwrite_overlaps``) and the bands must not be re-orthonormalised. Re-orthonormalising the
truncated set of ``nbnd`` pseudo-wavefunctions with the plain overlap, the former workaround of
:func:`PAOFLOW.projection.do_atwfc_proj.read_QE_wfc`, mixes the bands where the ``nbnd`` cut
crosses a multiplet and breaks the point symmetry of the PAO Hamiltonian at those k-points
(GaAs, October 2026: 0.65 eV at 16 of 4096 grid points, a 5-25 meV spread of the interpolated
eigenvalues between equivalent k and a 2-5 % violation of the tensor symmetry of the
second-order optical responses).

The projectors :math:`\\beta^{a}_i(\\mathbf k+\\mathbf G)` and the integrated augmentation charges
:math:`q^{a}_{ij}` (``PP_Q`` of the UPF) come from
:class:`PAOFLOW.elphon.pseudopotential_terms.KleinmanBylanderProjectors`.

With spin-orbit coupling (fully relativistic ultrasoft / PAW pseudopotentials, ``npol = 2``) the
charges are rotated into the spinor basis, QE's ``qq_so``,

.. math::

    S = 1 + \\sum_{a}\\sum_{kl}\\sum_{s_1 s_2} q^{a,s_1 s_2}_{kl}\\,
    |\\beta^{a}_k s_1\\rangle\\langle\\beta^{a}_l s_2|,
    \\qquad
    q^{s_1 s_2}_{kl} = \\sum_{ij}\\sum_{s} q_{ij}\\, f^{s_1 s}_{ki}\\, f^{s s_2}_{jl},

with :math:`f` the rotation from the real-harmonic projectors to the :math:`|l\\,j\\,m_j\\rangle`
spinors (QE ``fcoef``), and the inner products run over both spinor components.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..elphon.pseudopotential_terms import KleinmanBylanderProjectors


def k_plus_g_vectors(gkspace: dict) -> NDArray[np.float64]:
    """``k + G`` of a wavefunction block (bohr^-1), as built by ``calc_atwfc_k``.

    Parameters
    ----------
    gkspace : dict
        G-space descriptor of :func:`PAOFLOW.projection.do_atwfc_proj.read_QE_wfc`
        (``'xk'``, ``'mill'``, ``'bg'`` in bohr^-1).

    Returns
    -------
    ndarray ``(npw, 3)``
    """
    k_plus_g = np.dot(gkspace['mill'].T, gkspace['bg'].T)
    return k_plus_g + np.asarray(gkspace['xk'], dtype=float)[None, :]


def is_augmented(upf) -> bool:
    """Whether a pseudopotential (UPF or POTCAR dataset) carries augmentation charges (ultrasoft or PAW)."""
    return getattr(upf, 'ptype', 'NC') != 'NC' or bool(getattr(upf, 'has_augmentation', False))


class AugmentedOverlap:
    """The operator ``S - 1`` of a cell: projectors at ``k + G`` and the charges ``q_ij``.

    Parameters
    ----------
    kb : KleinmanBylanderProjectors
        Built with ``require_norm_conserving=False``; its :attr:`Q` (scalar relativistic) or
        :attr:`Q_so` (spin-orbit coupling, ``spin_orbit=True``) holds the block matrix of the
        integrated augmentation charges.
    tau_alat : ndarray ``(nat, 3)``
        Atomic positions (Cartesian, alat units).
    tpiba : float
        ``2 pi / alat`` (bohr^-1).
    """

    def __init__(self, kb: KleinmanBylanderProjectors, tau_alat: NDArray[np.float64], tpiba: float):
        self.spin_orbit = bool(getattr(kb, 'spin_orbit', False))
        if self.spin_orbit:
            if getattr(kb, 'Q_so', None) is None:
                raise ValueError('the projectors carry no spinor augmentation charges')
            self.Q = None
            self.Q_so = np.asarray(kb.Q_so, dtype=np.complex128)
        else:
            if kb.Q is None:
                raise ValueError('the projectors carry no augmentation charges')
            self.Q = np.asarray(kb.Q, dtype=float)
            self.Q_so = None
        self.kb = kb
        self.tau_alat = np.asarray(tau_alat, dtype=float)
        self.tpiba = float(tpiba)

    @property
    def nproj(self) -> int:
        """Number of projectors (all atoms, all ``m``)."""
        return self.kb.nproj

    def beta(self, k_plus_g: NDArray[np.float64]) -> NDArray[np.complex128]:
        """``beta_p(k+G)`` on one sphere.

        Parameters
        ----------
        k_plus_g : ndarray ``(npw, 3)``
            ``k + G`` (Cartesian, bohr^-1).

        Returns
        -------
        ndarray ``(nproj, npw)``, complex
        """
        return self.kb.at(np.asarray(k_plus_g, dtype=float) / self.tpiba, self.tau_alat, self.tpiba)

    def becp(
        self, beta: NDArray[np.complex128], functions: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """``<beta_p|f_i>`` of a set of functions on one sphere.

        Parameters
        ----------
        beta : ndarray ``(nproj, npw)``
            :meth:`beta` of the sphere.
        functions : ndarray ``(nf, npw)`` or ``(nf, 2 npw)``
            Plane-wave coefficients (rows = functions); with spin-orbit coupling the two
            spinor components are concatenated (upper then lower, as QE stores them).

        Returns
        -------
        ndarray ``(nproj, nf)``, or ``(2, nproj, nf)`` with spin-orbit coupling (one block per
        spinor component), complex
        """
        npw = beta.shape[1]
        if not self.spin_orbit:
            return beta.conj() @ functions.T
        if functions.shape[1] != 2 * npw:
            raise ValueError(
                'spin-orbit coupling: expected spinor functions with 2 x %d coefficients, got %d'
                % (npw, functions.shape[1])
            )
        return np.stack([beta.conj() @ functions[:, :npw].T, beta.conj() @ functions[:, npw:].T])

    def overlap(
        self,
        a: NDArray[np.complex128],
        b: NDArray[np.complex128],
        beta: NDArray[np.complex128],
        becp_a: NDArray[np.complex128] | None = None,
        becp_b: NDArray[np.complex128] | None = None,
    ) -> NDArray[np.complex128]:
        """``<a_i|S|b_j>`` for two sets of functions on the same sphere.

        Parameters
        ----------
        a, b : ndarray ``(na, npw)``, ``(nb, npw)``
            Plane-wave coefficients (rows = functions); ``(na, 2 npw)`` spinors with spin-orbit
            coupling.
        beta : ndarray ``(nproj, npw)``
            :meth:`beta` on the same sphere.
        becp_a, becp_b : ndarray, optional
            :meth:`becp` of ``a`` and ``b`` when already available.

        Returns
        -------
        ndarray ``(na, nb)``, complex
        """
        if becp_a is None:
            becp_a = self.becp(beta, a)
        if becp_b is None:
            becp_b = self.becp(beta, b)
        plain = a.conj() @ b.T
        if not self.spin_orbit:
            return plain + becp_a.conj().T @ self.Q @ becp_b
        return plain + sum(
            becp_a[s1].conj().T @ self.Q_so[s1, s2] @ becp_b[s2]
            for s1 in range(2)
            for s2 in range(2)
        )


def plain_overlap_warning(data_controller, requested: bool = False) -> None:
    """Report that the projections use the plain plane-wave overlap.

    Parameters
    ----------
    data_controller : DataController
        The run (``species``, ``dft``).
    requested : bool, optional
        The user asked for the plain overlap (``projections(plain_overlap=True)``).

    Returns
    -------
    None
        Prints a ``WARNING`` when any species is ultrasoft / PAW (every VASP run), for which the
        plain overlap is an approximation; for norm-conserving pseudopotentials, for which it
        is the exact metric, a note is printed in verbose mode only.
    """
    from ..inputs.pseudopotentials import load_pseudopotentials

    arry, attr = data_controller.data_dicts()
    if attr.get('dft', 'QE') == 'VASP':
        augmented = True
    else:
        try:
            augmented = any(
                is_augmented(d) for d in load_pseudopotentials(data_controller).values()
            )
        except Exception:
            augmented = True
    if not augmented:
        if attr.get('verbose'):
            print('Norm-conserving pseudopotentials: projections with the plain overlap (exact).')
        return
    reason = (
        'as requested (plain_overlap=True)'
        if requested
        else 'because the augmentation charges and projectors are not available'
    )
    print(
        'WARNING: the projections use the plain plane-wave overlap with re-orthonormalised '
        'bands %s. With ultrasoft / PAW pseudopotentials%s this metric is approximate: '
        'the point symmetry of the PAO Hamiltonian is broken at a few k-points, the '
        'real-space Hamiltonian decays slowly and the projectabilities are overestimated '
        '(see docs/internals/internal-basis.md).'
        % (reason, ' (every VASP run)' if attr.get('dft') == 'VASP' else '')
    )


def build_augmentation(data_controller) -> AugmentedOverlap | None:
    """The augmented overlap of the run, or ``None`` when the plain overlap is the right one.

    Parameters
    ----------
    data_controller : DataController
        Required arrays: ``species``, ``atoms``, ``tau``; attributes ``dft``, ``fpath``,
        ``ecutwfc``, ``omega``, ``alat``, ``dftSO``.

    Returns
    -------
    AugmentedOverlap or None
        ``None`` when every species is norm-conserving, when an augmented species comes from a
        UPF v1 file whose projectors and charges are not parsed, or when the POTCAR datasets of
        a VASP run are not found (a warning names the problem and the plain-overlap path is
        kept). With spin-orbit coupling the charges of the fully relativistic pseudopotentials
        are rotated into the spinor basis (``kb.Q_so``); a scalar-relativistic species in such
        a run, and every VASP dataset (scalar relativistic, spin-orbit coupling added by VASP
        at run time), couples the spins diagonally, as in QE.

    Notes
    -----
    VASP datasets are read through :mod:`PAOFLOW.inputs.pseudopotentials` (``fpath/POTCAR``,
    ``attr['potcar_dir']`` or ``$PAOFLOW_POTCAR_DIR``); their projectors at ``k+G`` come from
    the reciprocal-space tables of the file, the ones VASP applies with ``LREAL = .FALSE.``.
    """
    from ..inputs.pseudopotentials import load_pseudopotentials

    arry, attr = data_controller.data_dicts()
    try:
        upf_by_species = load_pseudopotentials(data_controller)
    except FileNotFoundError as err:
        if attr.get('dft', 'QE') != 'VASP':
            raise
        # VASP run without POTCAR datasets: the PAW metric is unknown, keep the plain-overlap
        # path (band re-orthonormalisation, the behaviour before November 2026)
        if getattr(data_controller, 'rank', 0) == 0:
            print(
                'WARNING: %s. The projections use the plain overlap with re-orthonormalised '
                'bands (approximate for PAW: symmetry breaking at a few k-points, inflated '
                'projectabilities); pass potcar_dir= to PAOFLOW() for the augmented metric.' % err
            )
        return None
    if not any(is_augmented(upf) for upf in upf_by_species.values()):
        return None
    unparsed = [
        name
        for name, upf in upf_by_species.items()
        if is_augmented(upf) and (getattr(upf, 'qqq', None) is None or not getattr(upf, 'beta', []))
    ]
    if unparsed:
        # UPF v1 files: the reader does not parse PP_NONLOCAL / PP_QIJ, so the projectors and
        # charges of S are unknown. Keep the plain-overlap path (band re-orthonormalisation)
        # rather than failing; the references of the transport examples were made with it.
        if getattr(data_controller, 'rank', 0) == 0:
            print(
                'WARNING: ultrasoft / PAW species %s: the augmentation charges and projectors '
                'are not available from the pseudopotential file (UPF v1), the projections use '
                'the plain overlap with re-orthonormalised bands (approximate). Convert the '
                'file to UPF v2 (QE upfconv.x) for the augmented metric.' % ', '.join(unparsed)
            )
        return None
    qmax = np.sqrt(float(attr['ecutwfc'])) + 1.0
    kb = KleinmanBylanderProjectors(
        upf_by_species,
        list(arry['atoms']),
        float(attr['omega']),
        qmax,
        spin_orbit=bool(attr.get('dftSO', False)),
        require_norm_conserving=False,
    )
    alat = float(attr['alat'])
    return AugmentedOverlap(kb, np.asarray(arry['tau'], dtype=float) / alat, 2.0 * np.pi / alat)
