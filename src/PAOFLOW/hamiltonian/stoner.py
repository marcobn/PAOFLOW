r"""Self-consistent spin response of the PAO Hamiltonian: the LSDA exchange field in the Stoner mean field.

A magnetization :math:`m(\mathbf r)` changes the LSDA exchange-correlation potential by
:math:`v_\sigma = \bar v + \sigma f_s(n)\,m` (:func:`PAOFLOW.basis_gen.xc.lsda_spin_kernel`). With the
magnetization of each PAO shell :math:`g` spread over its radial density, :math:`m(\mathbf r) = \sum_g m_g
R_g^2(r)/4\pi`, the orbitals of shell :math:`g` see the exchange splitting :math:`\Delta_g =
\sum_{g'} I_{gg'}\,m_{g'}` with the Stoner integrals (Gunnarsson 1976; Janak 1977)

.. math::

    I_{gg'} = -2\int_0^\infty r^2\,dr\,\frac{R_g^2(r)\,R_{g'}^2(r)}{4\pi}\,f_s(n(r)),

:math:`n` the atomic valence density plus the core charge of the pseudopotential. The mean-field Hamiltonian
is :math:`H(\mathbf k) + V_B - \sum_g \mathbf h_g\cdot\mathbf S_g`, :math:`\mathbf h_g = \sum_{g'} I_{gg'}
(\mathbf m_{g'} - \mathbf m^0_{g'})`, :math:`\mathbf S_g` the spin operator restricted to the shell, solved
self-consistently with the moments :math:`\mathbf m_g = 2\langle\mathbf S_g\rangle` of the occupied states.
:math:`\mathbf m^0` are the moments of :math:`H` without field: a magnetic DFT Hamiltonian already holds its
exchange splitting. In the linear regime the result is the RPA-Stoner enhancement
:math:`\chi = \chi_0/(1 - I\chi_0)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

if TYPE_CHECKING:
    from ..DataController import DataController

HARTREE_EV = 27.211386245988


# --------------------------------------------------------------------------- shells and Stoner integrals
def species_densities(data_controller: DataController) -> dict[str, tuple[NDArray, NDArray]]:
    """Atomic valence plus core density of every species, from its pseudopotential.

    Returns
    -------
    dict
        ``{species: (r, n)}``, radial grid (bohr) and density (bohr^-3).
    """
    from ..inputs.pseudopotentials import load_pseudopotentials

    out = {}
    for name, upf in load_pseudopotentials(data_controller).items():
        r = np.asarray(upf.r, dtype=float)
        valence = np.zeros_like(r)
        if getattr(upf, 'atrho', None) is not None:
            safe = np.where(r > 1e-12, r, 1.0)
            valence = np.where(
                r > 1e-12, np.asarray(upf.atrho, dtype=float) / (4.0 * np.pi * safe**2), 0.0
            )
        core = np.zeros_like(r)
        if getattr(upf, 'rho_atc', None) is not None:
            core = np.asarray(upf.rho_atc, dtype=float)[: r.size]
        out[name] = (r, valence + core)
    return out


def crystal_densities(data_controller: DataController) -> dict[str, tuple[NDArray, NDArray]]:
    r"""Crystal valence density, spherically averaged about an atom of every species, plus its core.

    :math:`\bar n(r) = \sum_{\mathbf G}\rho(\mathbf G)\,e^{i\mathbf G\cdot\boldsymbol\tau}\,j_0(Gr)` from
    QE's ``charge-density.dat`` (first atom of the species), on the radial grid of its
    pseudopotential. Unlike the free-atom density, it does not vanish between the atoms, where the
    spin kernel :math:`f_s\propto n^{-2/3}` of a diffuse shell would otherwise diverge.

    Returns
    -------
    dict
        ``{species: (r, n)}``, radial grid (bohr) and density (bohr^-3).
    """
    from ..inputs.pseudopotentials import load_pseudopotentials
    from ..response.coupled_crystal import read_charge_density

    arrays, attributes = data_controller.data_dicts()
    mill, rho, _ = read_charge_density(attributes['fpath'])
    bg = np.asarray(arrays['b_vectors'], dtype=float) * 2.0 * np.pi / float(attributes['alat'])
    g = mill.T @ bg
    gnorm = np.linalg.norm(g, axis=1)
    tau = np.asarray(arrays['tau'], dtype=float)
    out = {}
    for name, upf in load_pseudopotentials(data_controller).items():
        r = np.asarray(upf.r, dtype=float)
        site = tau[list(arrays['atoms']).index(name)]
        weights = rho[0] * np.exp(1j * g @ site)
        valence = np.array([np.real(np.sum(weights * np.sinc(gnorm * x / np.pi))) for x in r])
        core = np.zeros_like(r)
        if getattr(upf, 'rho_atc', None) is not None:
            core = np.asarray(upf.rho_atc, dtype=float)[: r.size]
        out[name] = (r, np.maximum(valence, 0.0) + core)
    return out


def shell_groups(arrays: dict, attributes: dict) -> list[dict]:
    """The PAO shells of every atom, with their orbitals in the current basis.

    Parameters
    ----------
    arrays, attributes : dict
        DataController dictionaries; ``basis`` (the projection basis) and ``HRs``.

    Returns
    -------
    list of dict
        One per (atom, shell): ``'atom'`` (position key), ``'species'``, ``'label'``, ``'l'``,
        ``'orbitals'`` (indices in ``HRs``: both spinor components for block-ordered spinors made
        by ``adhoc_spin_orbit``/``spinor_basis``, the orbitals of one channel for ``nspin = 2``),
        ``'r'`` and ``'radial'`` (:math:`R^2(r)`, normalized :math:`\\int R^2 r^2 dr = 1`).

    Raises
    ------
    ValueError
        Without a projection basis (models: pass the shells and the Stoner matrix explicitly).
    """
    basis = arrays.get('basis')
    nawf = arrays['HRs'].shape[0]
    if basis is None or len(basis) not in (nawf, nawf // 2):
        raise ValueError('no projection basis for the orbitals: pass shells and stoner explicitly')
    doubled = len(basis) == nawf // 2 and nawf % 2 == 0 and len(basis) != nawf
    groups: dict = {}
    for index, entry in enumerate(basis):
        key = (tuple(np.round(np.asarray(entry['tau'], dtype=float), 6)), entry['label'])
        group = groups.setdefault(
            key,
            {'atom': key[0], 'species': entry['atom'], 'label': entry['label'], 'l': int(entry['l']),
             'orbitals': [], 'radials': []},
        )  # fmt: skip
        group['orbitals'].append(index)
        if doubled:
            group['orbitals'].append(index + len(basis))
        r = np.asarray(entry['r'], dtype=float)
        wfc = np.asarray(entry['wfc'], dtype=float)[: r.size]
        r = r[: wfc.size]
        # chi = r R (UPF) or R: normalize R^2 r^2 to one
        as_chi = np.trapezoid(wfc**2, r)
        as_radial = np.trapezoid(wfc**2 * r**2, r)
        safe = np.where(r > 1e-12, r, 1.0)
        radial = (wfc**2 / safe**2) if abs(as_chi - 1.0) < abs(as_radial - 1.0) else wfc**2
        radial = np.where(r > 1e-12, radial, 0.0)
        group['radials'].append((r, radial / np.trapezoid(radial * r**2, r)))
    out = []
    for group in groups.values():
        r = group['radials'][0][0]
        radial = np.mean([np.interp(r, rr, rad) for rr, rad in group['radials']], axis=0)
        out.append({**{k: v for k, v in group.items() if k != 'radials'}, 'orbitals': np.array(sorted(group['orbitals'])),
                    'r': r, 'radial': radial})  # fmt: skip
    return out


def stoner_matrix(
    groups: list[dict], densities: dict[str, tuple[NDArray, NDArray]]
) -> NDArray[np.float64]:
    """Stoner integrals :math:`I_{gg'}` (eV) between the shells of each atom; zero between atoms.

    Parameters
    ----------
    groups : list of dict
        From :func:`shell_groups`.
    densities : dict
        From :func:`species_densities`.

    Returns
    -------
    ndarray, shape ``(ngroups, ngroups)``
        Splitting per unit moment, :math:`\\Delta_g = \\sum_{g'} I_{gg'} m_{g'}` (eV per :math:`\\mu_B`).
    """
    from ..basis_gen.xc import lsda_spin_kernel

    size = len(groups)
    matrix = np.zeros((size, size))
    for a in range(size):
        for b in range(size):
            if groups[a]['atom'] != groups[b]['atom']:
                continue
            r = groups[a]['r']
            rr, density = densities[groups[a]['species']]
            kernel = lsda_spin_kernel(np.interp(r, rr, density, right=0.0))
            other = np.interp(r, groups[b]['r'], groups[b]['radial'], right=0.0)
            j = np.trapezoid(r**2 * groups[a]['radial'] * other * kernel / (4.0 * np.pi), r)
            matrix[a, b] = -2.0 * j * HARTREE_EV
    return 0.5 * (matrix + matrix.T)


# --------------------------------------------------------------------------- occupations
def occupation(
    energies: NDArray, mu: float, width: float, kind: str = 'marzari-vanderbilt'
) -> NDArray:
    """Smeared occupations (QE conventions, ``x = (mu - e) / width``).

    ``kind``: ``'marzari-vanderbilt'`` (cold), ``'gaussian'`` or ``'fermi-dirac'``.
    """
    x = (mu - np.asarray(energies)) / width
    if kind == 'gaussian':
        return 0.5 * (1.0 + erf(x))
    if kind == 'fermi-dirac':
        return 0.5 * (1.0 + np.tanh(0.5 * np.clip(x, -400.0, 400.0)))
    if kind == 'marzari-vanderbilt':
        xp = x - 1.0 / np.sqrt(2.0)
        return 0.5 * erf(xp) + np.exp(-np.minimum(xp**2, 200.0)) / np.sqrt(2.0 * np.pi) + 0.5
    raise ValueError(f'unknown smearing {kind!r}')


def occupation_derivative(
    energies: NDArray, mu: float, width: float, kind: str = 'marzari-vanderbilt'
) -> NDArray:
    r""":math:`df/d\epsilon` (eV^-1) of :func:`occupation`: minus the smeared delta over the width.

    The Marzari-Vanderbilt delta is :math:`e^{-x_p^2}(2 - \sqrt2\,x)/\sqrt\pi`, :math:`x_p = x - 1/\sqrt2`
    (QE's ``w0gauss``); it is negative on a small window above the Fermi level.
    """
    x = (mu - np.asarray(energies)) / width
    if kind == 'gaussian':
        delta = np.exp(-np.minimum(x**2, 200.0)) / np.sqrt(np.pi)
    elif kind == 'fermi-dirac':
        delta = 0.25 / np.cosh(0.5 * np.clip(x, -400.0, 400.0)) ** 2
    elif kind == 'marzari-vanderbilt':
        xp = x - 1.0 / np.sqrt(2.0)
        delta = np.exp(-np.minimum(xp**2, 200.0)) * (2.0 - np.sqrt(2.0) * x) / np.sqrt(np.pi)
    else:
        raise ValueError(f'unknown smearing {kind!r}')
    return -delta / width


def chemical_potential(
    energies: NDArray, electrons: float, width: float, kind: str, comm=None
) -> float:
    """Chemical potential holding ``electrons`` per cell (uniform k-weights, one electron per band)."""
    from scipy.optimize import brentq

    local_nk = energies.shape[0]
    total_nk = local_nk if comm is None else comm.allreduce(local_nk)

    def excess(mu: float) -> float:
        count = float(np.sum(occupation(energies, mu, width, kind)))
        if comm is not None:
            count = comm.allreduce(count)
        return count / total_nk - electrons

    lo = float(np.min(energies)) - 10.0 * width
    hi = float(np.max(energies)) + 10.0 * width
    if comm is not None:
        from mpi4py import MPI

        lo, hi = comm.allreduce(lo, op=MPI.MIN), comm.allreduce(hi, op=MPI.MAX)
    return brentq(excess, lo, hi, xtol=1e-12)


# --------------------------------------------------------------------------- the self-consistent loop
def bloch_hamiltonians(
    hrs: NDArray,
    kgrid: tuple[int, int, int],
    comm=None,
    a_vectors: NDArray | None = None,
    alat: float | None = None,
    positions_cryst: NDArray | None = None,
) -> NDArray[np.complex128]:
    """H(k) ``(nk_local, n, n, nspin)`` on a uniform grid from ``HRs`` (PAOFLOW's Fourier convention).

    Parameters
    ----------
    hrs : ndarray ``(n, n, n1, n2, n3, nspin)``
    kgrid : tuple of int
        Grid ``k = (i/n1, j/n2, l/n3)``; split over the ranks of ``comm``.
    a_vectors, alat, positions_cryst : optional
        With the lattice, the pair-resolved Wigner-Seitz sums of
        :func:`PAOFLOW.hamiltonian.bloch_derivatives.bloch_terms_from_HRs` (the interpolation of the
        grid response and of the optics); without, the plain sum over the signed cells of ``HRs``.
        Both are exact on the grid of ``HRs``.
    """
    kpoints = np.stack(np.meshgrid(*[np.arange(n) / n for n in kgrid], indexing='ij'), -1).reshape(
        -1, 3
    )
    if comm is not None:
        kpoints = np.array_split(kpoints, comm.Get_size())[comm.Get_rank()]
    if a_vectors is not None:
        from .bloch_derivatives import bloch_terms_from_HRs, hamiltonian_block

        out = np.empty((kpoints.shape[0], hrs.shape[0], hrs.shape[1], hrs.shape[-1]), dtype=complex)
        for spin in range(hrs.shape[-1]):
            terms = bloch_terms_from_HRs(hrs, a_vectors, float(alat), positions_cryst, ispin=spin)
            for start in range(0, kpoints.shape[0], 256):
                block = kpoints[start : start + 256]
                out[start : start + block.shape[0], :, :, spin] = hamiltonian_block(terms, block)
        return out
    grid = hrs.shape[2:5]
    signed = np.array(list(np.ndindex(*grid)))
    signed = np.where(signed <= np.array(grid) // 2, signed, signed - np.array(grid))
    flat = hrs.reshape(hrs.shape[0], hrs.shape[1], -1, hrs.shape[-1])
    phases = np.exp(-2j * np.pi * kpoints @ signed.T)  # (nk, nR)
    return np.einsum('ijrs,kr->kijs', flat, phases)


def _spin_projectors(
    groups: list[dict], spin_operator: NDArray | None, nawf: int, nspin: int
) -> NDArray:
    """``(ngroups, 3, n, n)`` shell spin operators (spinors) or ``(ngroups, n)`` shell masks (collinear)."""
    if nspin == 2:
        masks = np.zeros((len(groups), nawf))
        for g, group in enumerate(groups):
            masks[g, group['orbitals']] = 1.0
        return masks
    out = np.zeros((len(groups), 3, nawf, nawf), dtype=complex)
    for g, group in enumerate(groups):
        idx = group['orbitals']
        for a in range(3):
            out[g, a][np.ix_(idx, idx)] = spin_operator[a][np.ix_(idx, idx)]
    return out


def solve_spin_response(
    hk: NDArray[np.complex128],
    onsite: NDArray[np.complex128],
    groups: list[dict],
    stoner: NDArray[np.float64],
    spin_operator: NDArray | None,
    electrons: float,
    width: float,
    kind: str = 'marzari-vanderbilt',
    mixing: float = 0.5,
    tol: float = 1.0e-7,
    max_iter: int = 300,
    seed: NDArray | None = None,
    reference: NDArray | None = None,
    check_stability: bool = True,
    warmup: int = 30,
    comm=None,
) -> dict:
    """Self-consistent Stoner mean field on a k-grid.

    Parameters
    ----------
    hk : ndarray ``(nk_local, n, n, nspin)``
        Bloch Hamiltonians (eV): spinor (``nspin = 1``) or collinear (``nspin = 2``).
    onsite : ndarray ``(n, n, nspin)``
        k-independent external term (eV), e.g. the Zeeman coupling.
    groups, stoner
        Shells (:func:`shell_groups`) and their Stoner matrix (eV per :math:`\\mu_B`).
    spin_operator : ndarray ``(3, n, n)`` or None
        :math:`\\mathbf S` of the spinor basis (unused for ``nspin = 2``).
    electrons : float
        Electrons per cell (one per band).
    width, kind
        Smearing (eV) and its type (:func:`occupation`).
    mixing : float
        Anderson mixing parameter.
    tol : float
        Convergence of the shell moments (:math:`\\mu_B`).
    seed : ndarray ``(ngroups, 3)``, optional
        Initial moments added to the reference (to probe a Stoner instability).
    reference : ndarray ``(ngroups, 3)``, optional
        :math:`\\mathbf m^0`, the moments without field (default: computed from ``hk``).
    check_stability : bool, optional
        After convergence, perturb the moments and iterate without mixing: a growing perturbation
        (a Stoner instability, e.g. the non-magnetic state of a ferromagnet) restarts the loop from
        it, so the solution returned is the stable one.
    warmup : int, optional
        Linear-mixing iterations before Anderson mixing; linear mixing is repelled by an unstable
        fixed point. ``warmup = 0`` and ``check_stability = False`` stay on the solution connected
        to the reference: the linear response, whose susceptibility is negative across a Stoner
        instability.

    Returns
    -------
    dict
        ``'moments'`` ``(ngroups, 3)``, ``'reference'``, ``'exchange'`` (the converged
        :math:`-\\sum_g\\mathbf h_g\\cdot\\mathbf S_g`, ``(n, n, nspin)``), ``'mu'``, ``'iterations'``,
        ``'converged'``, ``'stable'``, ``'history'`` (moment change per iteration).
    """
    nawf, nspin = hk.shape[1], hk.shape[-1]
    ops = _spin_projectors(groups, spin_operator, nawf, nspin)
    ng = len(groups)

    def exchange_term(moments: NDArray) -> NDArray:
        field = stoner @ (moments - ref)  # (ng, 3) eV
        term = np.zeros((nawf, nawf, nspin), dtype=complex)
        if nspin == 2:
            for g in range(ng):
                term[:, :, 0] -= 0.5 * field[g, 2] * np.diag(ops[g])
                term[:, :, 1] += 0.5 * field[g, 2] * np.diag(ops[g])
        else:
            term[:, :, 0] = -np.einsum('ga,gaij->ij', field, ops)
        return term

    total_nk = hk.shape[0] if comm is None else comm.allreduce(hk.shape[0])

    def moments_of(term: NDArray, external: NDArray) -> tuple[NDArray, float]:
        energies, vectors = [], []
        for s in range(nspin):
            e, v = np.linalg.eigh(hk[..., s] + (external[:, :, s] + term[:, :, s])[None])
            energies.append(e)
            vectors.append(v)
        energies_all = np.concatenate(energies, axis=1) if nspin == 2 else energies[0]
        mu = chemical_potential(energies_all, electrons, width, kind, comm)
        out = np.zeros((ng, 3))
        for s in range(nspin):
            f = occupation(energies[s], mu, width, kind)  # (nk, nb)
            # the density matrix summed over the local k: moments are linear in it
            rho = np.einsum(
                'kib,kjb->ij', vectors[s] * f[:, None, :], vectors[s].conj(), optimize=True
            )
            if comm is not None:
                rho = comm.allreduce(rho)
            if nspin == 2:
                out[:, 2] += (1.0 if s == 0 else -1.0) * (ops @ np.real(np.diag(rho)))
            else:
                out += 2.0 * np.real(np.einsum('gaij,ji->ga', ops, rho))
        return out / total_nk, mu

    zero = np.zeros((nawf, nawf, nspin), dtype=complex)
    ref = moments_of(zero, zero)[0] if reference is None else np.asarray(reference, dtype=float)

    def iterate(
        m_start: NDArray, budget: int, linear_only: bool = False
    ) -> tuple[NDArray, bool, int, float]:
        """Linear-mixing warm-up (repelled by unstable fixed points), then Anderson mixing."""
        m_in, mu = m_start.copy(), 0.0
        inputs, residuals = [], []
        for iteration in range(1, budget + 1):
            m_out, mu = moments_of(exchange_term(m_in), onsite)
            residual = m_out - m_in
            history.append(float(np.abs(residual).max()))
            if history[-1] < tol:
                return m_out, True, iteration, mu
            if linear_only or iteration <= warmup:
                m_in = m_in + mixing * residual
                continue
            inputs.append(m_in.ravel().copy())
            residuals.append(residual.ravel().copy())
            if len(inputs) > 6:
                inputs.pop(0)
                residuals.pop(0)
            if len(inputs) > 1:
                dr = np.diff(np.array(residuals), axis=0)
                dx = np.diff(np.array(inputs), axis=0)
                coeff, *_ = np.linalg.lstsq(dr.T, residuals[-1], rcond=None)
                m_in = (inputs[-1] - coeff @ dx + mixing * (residuals[-1] - coeff @ dr)).reshape(
                    m_in.shape
                )
            else:
                m_in = m_in + mixing * residual
        return m_in, False, budget, mu

    def unstable(m_fixed: NDArray) -> NDArray | None:
        """Plain iterations from a perturbed solution: the perturbed moments if the perturbation grows."""
        direction = m_fixed - ref
        if np.abs(direction).max() < 1e-6:
            direction = np.zeros_like(m_fixed)
            direction[:, 2] = 1.0
        delta = 1e-3 * direction / np.abs(direction).max()
        m_probe = m_fixed + delta
        for _ in range(probe_steps):
            m_probe = moments_of(exchange_term(m_probe), onsite)[0]
        grown = np.abs(m_probe - m_fixed).max() > 1.02 * np.abs(delta).max()  # any net growth
        return m_probe if grown else None

    history: list[float] = []
    probe_steps = 12
    m_in = ref + (0.0 if seed is None else np.asarray(seed, dtype=float))
    total_iterations, stable = 0, True
    linear_only = False
    for _restart in range(4):
        m_in, converged, used, mu = iterate(m_in, max_iter * (4 if linear_only else 1), linear_only)
        total_iterations += used
        if not converged or not check_stability:
            break
        restart = unstable(m_in)
        if restart is None:
            stable = True
            break
        stable, m_in, linear_only = False, restart, True  # linear mixing is repelled by m = 0
    iteration = total_iterations
    return {
        'moments': m_in,
        'reference': ref,
        'exchange': exchange_term(m_in),
        'mu': mu,
        'iterations': iteration,
        'converged': converged,
        'stable': stable,
        'history': history,
    }
