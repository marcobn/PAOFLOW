r"""Static uniform spin susceptibility with the LSDA spin kernel on the real-space grid of the crystal.

A uniform Zeeman field splits the two spin channels of a spin-degenerate ground state by :math:`\pm h`
(:math:`h = g_s\mu_B B/2`). The induced magnetization density :math:`\delta m = \delta n_\uparrow -
\delta n_\downarrow` changes the LSDA potential by :math:`\pm f_s(n_0)\,\delta m(\mathbf r)`
(:func:`PAOFLOW.basis_gen.xc.lsda_spin_kernel`, :math:`n_0` valence plus core density), so

.. math::

    \delta m = \chi_0\big[h + f_s(n_0)\,\delta m\big],

with :math:`\chi_0` the static, :math:`\mathbf q = 0` independent-particle response of the PAO bands (spin
factor 2 included):

.. math::

    \delta P_{nm}(\mathbf k) = \frac{f_m - f_n}{\epsilon_m - \epsilon_n}\,\delta V_{nm}(\mathbf k),
    \qquad \delta P_{nn} = f'(\epsilon_n)\,\delta V_{nn}

(the Fermi-surface term on the diagonal and inside degenerate pairs; smeared occupations as in QE). The
ground state is spin symmetric, so the chemical potential does not move at first order. The equation is
solved by GMRES on :math:`\delta m(\mathbf r)`; the susceptibility per cell is
:math:`\chi = -\int\delta m\,d\mathbf r / h` (:math:`\mu_B` per eV), the bare one the same without the kernel
(the Pauli term :math:`2N(E_F)`, :math:`N` per spin). This is the full-grid counterpart of the on-site
Stoner mean field (:mod:`PAOFLOW.hamiltonian.stoner`): the kernel acts on the actual shape of
:math:`\delta m`, including its interstitial and its :math:`s`-:math:`p` parts. The dynamical, finite-q
response (paramagnons) needs a :math:`\mathbf k,\mathbf k+\mathbf q` channel and is not done here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .coupled_response import HARTREE_EV


def core_density_on_grid(data_controller, nr: tuple[int, int, int], at_bohr: NDArray) -> NDArray:
    """Pseudo-core (NLCC) density of the crystal on the cell grid ``nr`` (flat, bohr^-3).

    The radial ``rho_atc`` of every species is Fourier transformed and summed over the atoms at
    ``tau`` (bohr); zero for pseudopotentials without a core correction.
    """
    from ..inputs.pseudopotentials import load_pseudopotentials

    arry = data_controller.data_arrays
    at = np.asarray(at_bohr, dtype=float)
    omega = abs(float(np.linalg.det(at)))
    bg = 2.0 * np.pi * np.linalg.inv(at).T  # rows: reciprocal vectors (bohr^-1)
    freq = [np.fft.fftfreq(n, 1.0 / n) for n in nr]
    mill = np.stack(np.meshgrid(*freq, indexing='ij'), axis=-1).reshape(-1, 3)
    g = mill @ bg
    gnorm = np.linalg.norm(g, axis=1)
    rho_g = np.zeros(mill.shape[0], dtype=complex)
    upfs = load_pseudopotentials(data_controller)
    tau = np.asarray(arry['tau'], dtype=float)
    for name, upf in upfs.items():
        core = getattr(upf, 'rho_atc', None)
        if core is None:
            continue
        r = np.asarray(upf.r, dtype=float)
        core = np.asarray(core, dtype=float)[: r.size]
        r = r[: core.size]
        shells, inverse = np.unique(np.round(gnorm, 10), return_inverse=True)
        form = np.array(
            [4.0 * np.pi * np.trapezoid(r**2 * core * np.sinc(q * r / np.pi), r) for q in shells]
        )
        structure = sum(
            np.exp(-1j * g @ tau[a]) for a, atom in enumerate(arry['atoms']) if atom == name
        )
        rho_g += form[inverse] * structure / omega
    box = rho_g.reshape(nr)
    return np.real(np.fft.ifftn(box) * float(np.prod(nr))).reshape(-1)


def valence_density_from_qe(savedir: str, nr: tuple[int, int, int]) -> NDArray:
    """QE's valence density (``charge-density.dat``, spin-summed) on the cell grid ``nr`` (flat, bohr^-3)."""
    from .coupled_crystal import read_charge_density

    mill, rho, _ = read_charge_density(savedir)
    total = rho[0]
    half = np.array(nr)[:, None] // 2
    keep = np.all(np.abs(mill) < half, axis=0)
    box = np.zeros(nr, dtype=complex)
    box[mill[0, keep] % nr[0], mill[1, keep] % nr[1], mill[2, keep] % nr[2]] = total[keep]
    return np.real(np.fft.ifftn(box) * float(np.prod(nr))).reshape(-1)


def static_spin_susceptibility(
    data_controller,
    nk=(12, 12, 12),
    smearing: str = 'marzari-vanderbilt',
    degauss: float = 0.2721,
    valence_density: str = 'pao',
    core: bool = True,
    grid_factor: float = 2.0,
    density_threshold: float = 1.0e-8,
    tol: float = 1.0e-8,
    maxiter: int = 100,
    cache_gb: float = 8.0,
    comm=None,
) -> dict:
    """Uniform static spin susceptibility with the full-grid LSDA spin kernel (see the module notes).

    Parameters
    ----------
    data_controller : DataController
        After ``pao_hamiltonian`` with the internal projection basis (``HRs``, ``basis``); a
        spin-degenerate run (``nspin = 1`` without spin-orbit coupling).
    nk : sequence of int
        k-grid of the response.
    smearing, degauss
        Occupations (:func:`PAOFLOW.hamiltonian.stoner.occupation`) and their width (eV).
    valence_density : {'pao', 'qe'}
        Ground-state valence density of the kernel: from the PAO bands, or QE's charge density
        (``charge-density.dat`` in the save directory).
    core : bool
        Add the pseudo-core density of the pseudopotentials (as QE's LSDA does).
    grid_factor : float
        Density grid relative to the wavefunction sphere (2: alias-free products).
    density_threshold : float
        Density (bohr^-3) below which the kernel is set to zero.
    tol, maxiter
        GMRES.
    cache_gb : float
        Memory cap (per rank) for the periodic parts on the grid.
    comm : MPI communicator, optional

    Returns
    -------
    dict
        ``'chi'`` and ``'chi_bare'`` (:math:`\\mu_B` per eV per cell), ``'enhancement'``,
        ``'stoner_effective'`` (:math:`I = 2(1/\\chi_0 - 1/\\chi)`, eV per :math:`\\mu_B`, the on-site
        Stoner parameter giving the same enhancement), ``'mu'`` (eV), ``'density_of_states'`` (per
        spin, eV^-1), ``'magnetization_density'`` (flat, per Hartree of h), ``'nr'``,
        ``'iterations'``, ``'valence_electrons_on_grid'`` (the integral of the valence density, a
        check of the grid).
    """
    from ..basis_gen.xc import lsda_spin_kernel
    from ..hamiltonian.stoner import chemical_potential, occupation, occupation_derivative
    from .coupled_crystal import build_crystal_channel

    arry, attr = data_controller.data_dicts()
    if np.asarray(arry['HRs']).shape[-1] != 1 or attr.get('dftSO', False):
        raise NotImplementedError(
            'the grid spin response needs a spin-degenerate run (nspin = 1, no spin-orbit): use '
            'self_consistent_spin (on-site Stoner) for spinors and collinear runs'
        )
    spin_factor = 2.0
    # any finite temperature passes the metallic check; the occupations are replaced below
    ch = build_crystal_channel(
        data_controller, nk, grid_factor=grid_factor, comm=comm, fermi_energy=0.0, temperature=300.0
    )
    energies = ch.energies * HARTREE_EV  # (nk_local, n) eV
    mu = chemical_potential(energies, float(attr['nelec']) / spin_factor, degauss, smearing, comm)
    ch.occupations = spin_factor * occupation(energies, mu, degauss, smearing)
    ch.occupation_derivative = (
        spin_factor * occupation_derivative(energies, mu, degauss, smearing) * HARTREE_EV
    )  # per Hartree
    ch.n0 = None
    if cache_gb > 0:
        ch.cache_tables(cache_gb)
    dv = ch.omega / int(np.prod(ch.nr))
    if valence_density == 'qe':
        n0 = valence_density_from_qe(attr['fpath'], ch.nr)
    elif valence_density == 'pao':
        n0 = ch.ground_state_density()
    else:
        raise ValueError(f'valence_density must be pao or qe, not {valence_density!r}')
    valence_electrons = float(np.sum(n0)) * dv
    if core:
        n0 = n0 + core_density_on_grid(data_controller, ch.nr, ch.at_bohr)
    kernel = np.zeros(n0.size)
    keep = n0 > density_threshold
    kernel[keep] = lsda_spin_kernel(n0[keep])  # Hartree bohr^3
    out = solve_static_spin_response(ch, kernel, tol=tol, maxiter=maxiter)
    out.update({'mu': mu, 'nr': ch.nr, 'valence_electrons_on_grid': valence_electrons})
    return out


def solve_static_spin_response(
    ch, kernel: NDArray, tol: float = 1.0e-8, maxiter: int = 100, degeneracy_tol: float = 1.0e-7
) -> dict:
    """Solve :math:`\\delta m = \\chi_0[h + f_s\\,\\delta m]` on the grid of a channel, for a unit h.

    Parameters
    ----------
    ch : CrystalChannel
        Energies (Hartree), eigenvectors and periodic parts of the local k-points; ``occupations``
        ``(nk, n)`` and ``occupation_derivative`` ``(nk, n)`` (per Hartree), spin factor 2 included.
    kernel : ndarray ``(N_r,)``
        Spin kernel on the grid (Hartree bohr^3); zero gives the bare response.
    tol, maxiter
        GMRES.
    degeneracy_tol : float
        Pairs closer than this (Hartree) take the Fermi-surface term :math:`(f'_n + f'_m)/2`.

    Returns
    -------
    dict
        ``'chi'``, ``'chi_bare'`` (:math:`\\mu_B` per eV per cell), ``'enhancement'``,
        ``'stoner_effective'``, ``'density_of_states'``, ``'magnetization_density'`` (flat, per
        Hartree of h), ``'iterations'``.
    """
    from scipy.sparse.linalg import LinearOperator, gmres

    f = np.asarray(ch.occupations, dtype=float)
    fp = np.asarray(ch.occupation_derivative, dtype=float)
    e = ch.energies
    gap = e[:, None, :] - e[:, :, None]  # e_m - e_n
    degenerate = np.abs(gap) < degeneracy_tol
    coeff = np.where(
        degenerate,
        0.5 * (fp[:, :, None] + fp[:, None, :]),
        (f[:, None, :] - f[:, :, None]) / np.where(degenerate, 1.0, gap),
    )
    n_r = int(np.prod(ch.nr))
    dv = ch.omega / n_r
    kernel = np.asarray(kernel, dtype=float).reshape(-1)

    def density(dP: NDArray) -> NDArray:
        dn = np.zeros(n_r, dtype=complex)
        for ik in range(ch.nk_local):
            U = ch.eigenvectors[ik]
            dP_orb = U @ dP[ik] @ U.conj().T
            u = ch.periodic_table(ik)
            dn += ch.weights[ik] * np.einsum('rn,rn->r', u, u.conj() @ dP_orb.T)
        return np.real(ch.allreduce(dn))

    # unit h (Hartree): uniform on both channels with opposite signs, the identity in the PAO basis
    bare = density(coeff * np.eye(ch.n)[None])
    count = [0]

    def matvec(x: NDArray) -> NDArray:
        count[0] += 1
        potential = kernel * x
        response = np.stack(
            [coeff[ik] * ch.band_potential(potential, ik) for ik in range(ch.nk_local)]
        )
        return x - density(response)

    if np.any(kernel != 0.0):
        operator = LinearOperator((n_r, n_r), matvec=matvec, dtype=float)
        dm, info = gmres(
            operator, bare, rtol=tol, atol=0.0, restart=min(maxiter, 50), maxiter=maxiter
        )
        if info != 0:
            raise RuntimeError(f'GMRES did not converge (info {info})')
    else:
        dm = bare
    chi_bare = -float(np.sum(bare)) * dv / HARTREE_EV
    chi = -float(np.sum(dm)) * dv / HARTREE_EV
    return {
        'chi': chi,
        'chi_bare': chi_bare,
        'enhancement': chi / chi_bare,
        'stoner_effective': 2.0 * (1.0 / chi_bare - 1.0 / chi),
        'density_of_states': 0.5 * chi_bare,
        'magnetization_density': dm,
        'iterations': count[0],
    }


__all__ = [
    'core_density_on_grid',
    'solve_static_spin_response',
    'static_spin_susceptibility',
    'valence_density_from_qe',
]
