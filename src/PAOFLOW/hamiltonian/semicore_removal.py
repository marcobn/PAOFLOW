"""Removal of semicore states from the PAO Hamiltonian.

Pseudopotentials with semicore shells (e.g. Te 4d, Zr 4s4p) put those shells in the
PAO basis. They form flat bands tens of eV below the Fermi level that play no role
in low-energy properties but enlarge every matrix (supercells, ribbons, transport
principal layers) by up to a factor of two. The functions here drop those orbitals
and project ``H(k)`` onto the remaining valence orbitals, so that every eigenvalue
above the cut-off energy is kept exactly.

The public entry point is :meth:`PAOFLOW.PAOFLOW.remove_semicore`, which calls, in
order, :func:`find_semicore_orbitals`, :func:`project_valence_hamiltonian` (MPI rank
0 only) and :func:`update_orbital_metadata` (all ranks).
"""

from __future__ import annotations

from os.path import join
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import fftpack as FFT

if TYPE_CHECKING:
    from ..DataController import DataController


# Arrays that exist only after later workflow steps (interpolation, eigenvectors on
# the dense grid, gradients, smearing). They are built for the full orbital basis and
# cannot be reduced consistently, so remove_semicore must run before those steps.
LATER_STEP_ARRAYS = ('Hksp', 'E_k', 'v_k', 'dHksp', 'pksp', 'd2Hksp', 'deltakp', 'deltakp2')

# Arrays of the phenomenological (ad hoc) spin-orbit model, built from per-atom
# orbital layouts that do not follow the pseudopotential shells.
ADHOC_SOC_ARRAYS = ('orb_pseudo', 'orb_atom', 'lambda_p', 'lambda_d')


def check_semicore_prerequisites(data_controller: 'DataController') -> None:
    """Verify that the semicore states can be removed at this point of the workflow.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``HRs`` (shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``),
        ``basis`` (one entry per orbital, as built by ``read_atomic_proj_QE`` or
        ``projections``), ``species``, ``shells``, ``jchia``.

    Returns
    -------
    None
        Nothing is modified.

    Raises
    ------
    RuntimeError
        If the Hamiltonian is missing, the basis is non-orthogonal (``Sks`` or
        ``SRs`` present), the phenomenological spin-orbit model is active, the cell
        was already doubled or the Hamiltonian already interpolated or
        differentiated.
    """
    arry, attr = data_controller.data_dicts()

    # the real-space PAO Hamiltonian must exist (pao_hamiltonian has been called)
    if 'HRs' not in arry:
        raise RuntimeError('remove_semicore: call pao_hamiltonian() first')

    # the valence projection assumes an orthonormal PAO basis
    if 'Sks' in arry or 'SRs' in arry:
        raise RuntimeError('remove_semicore: not implemented for a non-orthogonal basis '
                           '(overlaps Sks/SRs present, e.g. acbn0 or save_overlaps)')

    # the ad hoc spin-orbit arrays use their own per-atom orbital layout
    if attr.get('adhoc_SO', False) or any(key in arry for key in ADHOC_SOC_ARRAYS):
        raise RuntimeError('remove_semicore: not implemented together with adhoc_spin_orbit; '
                           'call remove_semicore() before adhoc_spin_orbit()')

    # arrays of later steps were built on the full basis
    later_steps = [key for key in LATER_STEP_ARRAYS if key in arry]
    if later_steps:
        raise RuntimeError('remove_semicore: call it right after pao_hamiltonian(); found arrays '
                           f'of later steps: {", ".join(later_steps)}')

    # 'basis' is not replicated by doubling_Hamiltonian, so a mismatch means the cell
    # was already enlarged (or the basis metadata is missing)
    if 'basis' not in arry or len(arry['basis']) != attr['nawf']:
        raise RuntimeError('remove_semicore: the orbital basis does not match nawf; call '
                           'remove_semicore() after pao_hamiltonian() and before doubling_Hamiltonian()')


def find_semicore_orbitals(
    data_controller: 'DataController',
    emin: float,
    weight_threshold: float,
) -> NDArray[np.bool_]:
    """Identify the PAO orbitals that carry the states below ``emin``.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``HRs`` (shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``),
        ``basis``.
        Required attributes: ``nawf``, ``nspin``.
    emin : float
        Energy (eV, same reference as the PAO bands) below which the states are
        semicore.
    weight_threshold : float
        Minimum k-averaged weight of an orbital in the states below ``emin`` for it
        to be classified as semicore.

    Returns
    -------
    NDArray[np.bool_], shape ``(nawf,)``
        ``True`` for the semicore orbitals.

    Raises
    ------
    RuntimeError
        If the selection splits a shell, i.e. a pseudo-atomic wavefunction label is
        semicore on some orbitals of a species and valence on others; ``emin`` must
        then be moved.

    Notes
    -----
    The weight of orbital :math:`i` in the states below ``emin`` is averaged over
    the :math:`N_k` points of the grid and the :math:`N_s` spin channels,

    .. math::

        w_i = \\frac{1}{N_k N_s} \\sum_{\\mathbf{k}, \\sigma}
            \\sum_{n:\\, \\varepsilon_{n\\mathbf{k}\\sigma} < E_{\\mathrm{min}}}
            \\left| \\langle i | \\psi_{n\\mathbf{k}\\sigma} \\rangle \\right|^2 .

    A semicore orbital has :math:`w_i \\simeq 1` and a valence orbital
    :math:`w_i \\simeq 0`, so the classification does not depend on the exact
    value of ``weight_threshold``. Only whole shells are accepted, removed on every
    atom of a species, so that the per-species shell lists stay meaningful.
    """
    arry, attr = data_controller.data_dicts()
    nawf, nspin = attr['nawf'], attr['nspin']

    # H(k) on the regular grid: HRs = ifftn(Hks) in PAOFLOW, so Hks = fftn(HRs)
    hamiltonian_kspace = FFT.fftn(arry['HRs'], axes=[2, 3, 4]).reshape(nawf, nawf, -1, nspin)
    nkpoints = hamiltonian_kspace.shape[2]

    # k- and spin-averaged weight of every orbital in the eigenstates below emin
    semicore_weight = np.zeros(nawf)
    for ik in range(nkpoints):
        for ispin in range(nspin):
            energies, eigenvectors = np.linalg.eigh(hamiltonian_kspace[:, :, ik, ispin])
            semicore_weight += np.sum(np.abs(eigenvectors[:, energies < emin]) ** 2, axis=1)
    semicore_weight /= nkpoints * nspin
    is_semicore = semicore_weight > weight_threshold

    # every (species, shell label) must be either fully semicore or fully valence
    removed_shells = {(orbital['atom'], orbital['label'])
                      for orbital, semicore in zip(arry['basis'], is_semicore) if semicore}
    for orbital, semicore in zip(arry['basis'], is_semicore):
        if ((orbital['atom'], orbital['label']) in removed_shells) != semicore:
            raise RuntimeError(f'remove_semicore: shell {orbital["atom"]} {orbital["label"]} is only '
                               f'partially below emin = {emin} eV; change emin')
    return is_semicore


def project_valence_hamiltonian(
    data_controller: 'DataController',
    is_semicore: NDArray[np.bool_],
    emin: float,
) -> float:
    """Project the PAO Hamiltonian onto the valence orbitals, keeping every state above ``emin``.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``HRs`` (shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``).
        Optional array: ``Hks`` (same shape; present on MPI rank 0 after
        ``pao_hamiltonian``), replaced when present.
        Required attributes: ``nawf``, ``nspin``.
    is_semicore : NDArray[np.bool_], shape ``(nawf,)``
        Semicore orbitals, from :func:`find_semicore_orbitals`.
    emin : float
        Energy (eV) below which the states are removed.

    Returns
    -------
    float
        Smallest singular value of the valence block of the kept eigenvectors over
        the whole grid (1 means the valence orbitals do not mix with the semicore
        states at all). ``HRs`` (and ``Hks`` if present) are replaced in
        ``data_controller.data_arrays`` by the ``(nval, nval, nk1, nk2, nk3, nspin)``
        valence Hamiltonian. To be called on MPI rank 0 only; ``HRs`` must then be
        broadcast.

    Raises
    ------
    RuntimeError
        If the number of eigenvalues below ``emin`` differs from the number of
        semicore orbitals at some k-point (a semicore band crosses ``emin``).

    Notes
    -----
    At each k-point let :math:`\\varepsilon_{\\mathbf{k}}` be the eigenvalues above
    ``emin`` and :math:`A_{\\mathbf{k}}` the valence-orbital rows of their
    eigenvectors (a square :math:`N_v \\times N_v` matrix, since there are as many
    states above ``emin`` as valence orbitals). Its closest unitary matrix, from the
    polar decomposition :math:`A_{\\mathbf{k}} = U \\Sigma W^{\\dagger}`, is

    .. math::

        Q_{\\mathbf{k}} = U W^{\\dagger}
            = A_{\\mathbf{k}} \\left(A_{\\mathbf{k}}^{\\dagger} A_{\\mathbf{k}}\\right)^{-1/2},

    and the valence Hamiltonian is

    .. math::

        H^{\\mathrm{v}}(\\mathbf{k}) = Q_{\\mathbf{k}} \\, \\mathrm{diag}(\\varepsilon_{\\mathbf{k}})
            \\, Q_{\\mathbf{k}}^{\\dagger} .

    Its eigenvalues are exactly :math:`\\varepsilon_{\\mathbf{k}}`, and its
    eigenvectors are the valence parts of the original ones, re-normalised
    (Löwdin-symmetric, so the gauge stays smooth in k and ``HRs`` stays short
    ranged). The singular values :math:`\\Sigma` measure how much of each valence
    state lived on the semicore orbitals.
    """
    arry, attr = data_controller.data_dicts()
    nawf, nspin = attr['nawf'], attr['nspin']
    is_valence = ~is_semicore
    nsemicore, nvalence = int(is_semicore.sum()), int(is_valence.sum())

    # H(k) on the regular grid, flattened over k
    hamiltonian_kspace = FFT.fftn(arry['HRs'], axes=[2, 3, 4])
    kgrid_shape = hamiltonian_kspace.shape[2:5]
    hamiltonian_kspace = hamiltonian_kspace.reshape(nawf, nawf, -1, nspin)
    nkpoints = hamiltonian_kspace.shape[2]

    valence_hamiltonian = np.empty((nvalence, nvalence, nkpoints, nspin), dtype=complex)
    smallest_singular_value = 1.0
    for ik in range(nkpoints):
        for ispin in range(nspin):
            energies, eigenvectors = np.linalg.eigh(hamiltonian_kspace[:, :, ik, ispin])

            # the nsemicore lowest states must be exactly the ones below emin
            if np.sum(energies < emin) != nsemicore:
                raise RuntimeError(f'remove_semicore: {np.sum(energies < emin)} states below {emin} eV at '
                                   f'k-point {ik}, expected {nsemicore}')

            # valence-orbital rows of the states above emin, and their closest unitary
            left_vectors, singular_values, right_vectors_h = np.linalg.svd(eigenvectors[is_valence, nsemicore:])
            smallest_singular_value = min(smallest_singular_value, singular_values.min())
            valence_unitary = left_vectors @ right_vectors_h

            # H_v(k) = Q diag(eps) Q^dagger reproduces the eigenvalues above emin
            valence_hamiltonian[:, :, ik, ispin] = (valence_unitary * energies[nsemicore:]) @ valence_unitary.conj().T
    hamiltonian_kspace = None

    # back to real space with PAOFLOW's convention, on the same grid
    valence_hamiltonian = valence_hamiltonian.reshape(nvalence, nvalence, *kgrid_shape, nspin)
    arry['HRs'] = FFT.ifftn(valence_hamiltonian, axes=[2, 3, 4])
    if 'Hks' in arry:
        arry['Hks'] = valence_hamiltonian
    return smallest_singular_value


def update_orbital_metadata(
    data_controller: 'DataController',
    is_semicore: NDArray[np.bool_],
) -> None:
    """Remove the semicore orbitals from every per-orbital array and count.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Arrays updated when present: ``basis``, ``Dnm`` (shape
        ``(nawf, nawf, 3)``), ``Sj`` and ``Lj`` (spin and orbital angular-momentum
        operators, shape ``(3, nawf, nawf)``), ``naw`` (orbitals per atom),
        ``shells``, ``jchia`` and ``lchia`` (per-species shell lists), ``sh_l`` and
        ``sh_j`` (per-shell lists of the spin operator), ``my_eigsmat`` (DFT
        eigenvalues, shape ``(nbnds, nkpnts, nspin)``).
        Attributes updated: ``nawf``, ``nelec``, and ``nbnds`` and ``bnd`` when
        present.
    is_semicore : NDArray[np.bool_], shape ``(nawf,)``
        Semicore orbitals, from :func:`find_semicore_orbitals`.

    Returns
    -------
    None
        All modifications are made in place. To be called on every MPI rank with
        the same ``is_semicore``.

    Notes
    -----
    The removed orbitals form whole shells, so the operators ``Dnm``, ``Sj`` and
    ``Lj`` (block diagonal in atom and shell) are reduced exactly by keeping the
    valence rows and columns. The semicore states are the ``nsemicore`` lowest
    states of every k-point, so they are also the lowest DFT bands: they are
    dropped from ``my_eigsmat``, ``nbnds`` and ``bnd``, and their electrons from
    ``nelec`` (one per spinor state with spin-orbit coupling, two per orbital
    state otherwise).
    """
    from ..inputs.read_upf import UPF

    arry, attr = data_controller.data_dicts()
    is_valence = ~is_semicore
    nsemicore = int(is_semicore.sum())

    # shells removed, as (species, pseudo-wavefunction label)
    removed_shells = {(orbital['atom'], orbital['label'])
                      for orbital, semicore in zip(arry['basis'], is_semicore) if semicore}

    # orbitals per atom: atom of each orbital from its position, then count the valence ones
    if 'naw' in arry:
        atom_positions = np.asarray(arry['tau'], dtype=float)
        orbital_atom = np.array([np.argmin(np.linalg.norm(atom_positions - np.asarray(orbital['tau']), axis=1))
                                 for orbital in arry['basis']])
        arry['naw'] = np.array([int(np.sum(is_valence[orbital_atom == atom])) for atom in range(len(atom_positions))])

    # per-orbital lists and matrices: keep the valence rows/columns
    arry['basis'] = [orbital for orbital, keep in zip(arry['basis'], is_valence) if keep]
    if 'Dnm' in arry:
        arry['Dnm'] = arry['Dnm'][is_valence][:, is_valence]
    for operator in ('Sj', 'Lj'):
        if operator in arry:
            arry[operator] = arry[operator][:, is_valence][:, :, is_valence]

    # per-species shell lists, one entry per pseudo-atomic wavefunction of the UPF file
    lchia_is_separate = 'lchia' in arry and arry['lchia'] is not arry['shells']
    for species, pseudo_file in arry['species']:
        labels = [wavefunction['label'] for wavefunction in UPF(join(attr['fpath'], pseudo_file)).pswfc]
        keep = [(species, label) not in removed_shells for label in labels]
        arry['shells'][species] = [shell for shell, k in zip(arry['shells'][species], keep) if k]
        if species in arry.get('jchia', {}) and len(arry['jchia'][species]) == len(keep):
            arry['jchia'][species] = [j for j, k in zip(arry['jchia'][species], keep) if k]
        if lchia_is_separate and species in arry['lchia']:
            arry['lchia'][species] = [lc for lc, k in zip(arry['lchia'][species], keep) if k]

    # per-shell lists of the spin operator: rebuilt from the reduced shells
    if 'sh_l' in arry or 'sh_j' in arry:
        arry['sh_l'], arry['sh_j'] = spin_operator_shell_lists(arry['atoms'], arry['shells'])

    # DFT eigenvalues: the semicore states are the lowest bands
    if 'my_eigsmat' in arry:
        arry['my_eigsmat'] = arry['my_eigsmat'][nsemicore:]

    # counts: orbitals, bands and electrons
    electrons_per_state = 1 if attr.get('dftSO', False) else 2
    attr['nawf'] = int(is_valence.sum())
    attr['nelec'] -= electrons_per_state * nsemicore
    for key in ('nbnds', 'bnd'):
        if attr.get(key) is not None:
            attr[key] -= nsemicore


def spin_operator_shell_lists(
    atoms: list[str],
    shells: dict[str, list[int]],
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Per-shell angular momenta used by the spin operator, from the per-species shells.

    Parameters
    ----------
    atoms : list of str
        Species of every atom, in the order of the basis.
    shells : dict of str to list of int
        Angular momentum of every pseudo-atomic wavefunction of each species.

    Returns
    -------
    tuple of NDArray
        ``(sh_l, sh_j)``: orbital and total angular momentum of every shell, as
        built by :meth:`PAOFLOW.PAOFLOW.spin_operator` when they are not given.
    """
    shell_l, shell_j = [], []
    for atom in atoms:
        # s shells appear once in the relativistic UPF, p/d/f twice (j = l -+ 1/2)
        doubled = []
        for l in shells[atom]:
            doubled += [l, l] if l == 0 else [l]
        shell_l += doubled[::2]
        for l in doubled[::2]:
            shell_j += [0.5, 0.5] if l == 0 else [l - 0.5, l + 0.5]
    return np.array(shell_l), np.array(shell_j)[::2]
