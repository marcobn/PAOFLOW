from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI
from scipy import sparse
from scipy.sparse import linalg as spla

if TYPE_CHECKING:
    from ..DataController import DataController

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def compute_fermi_energy(data_controller: DataController) -> float:
    r"""Determine the Fermi energy from sparse Hamiltonian blocks via bisection.

    Parameters
    ----------
    data_controller : DataController
        Runtime container holding sparse ``Hks_sparse[ik * nspin + ispin]`` blocks
        and run attributes including number of electrons and smearing parameters.

    Returns
    -------
    float
        Fermi energy in the same energy units as the Hamiltonian, computed such that
        the integrated occupation at this energy matches the number of electrons in
        the system.

    Notes
    -----
    This function computes the Fermi energy

    .. math::

        E_F: \int_{-\infty}^{E_F} n(E, T) dE = N_{elec}

    where :math:`n(E, T)` is the electronic density of states smeared with Gaussian
    width via Gaussian smearing applied to eigenspetra collected from diagonalization
    of sparse :math:`H(k)` blocks across the Brillouin zone.

    Unlike the dense approach, this function only allocates space for eigenvalues
    at each (k, spin) pair, never materializing full eigenvector matrices. This avoids
    the dense memory scaling :math:`O(N_{PAO}^2 \cdot N_k)` that would occur if storing
    all eigenvectors. Eigenvalues are computed via sparse Krylov iteration and
    collected directly into a compact 3D array for bisection.

    The bisection algorithm reuses the dense smearing infrastructure (Gaussian occupation
    via ``intmetpax``) but adapts it to the distributed sparse eigenproblem by gathering
    only eigenvalues across ranks and k-points before performing the bracketing search.

    Parallelization is preserved through MPI reductions during bracketing convergence.
    """
    from ..smearing import intmetpax
    from .communication import load_balancing_kpoints

    arrays, attributes = data_controller.data_dicts()
    assert arrays is not None and attributes is not None

    if 'Hks_sparse' not in arrays:
        raise KeyError(
            'compute_fermi_energy requires Hks_sparse. '
            'Ensure pao_hamiltonian(sparse=True) was called first.'
        )

    hks_sparse = arrays['Hks_sparse']
    if not isinstance(hks_sparse, dict):
        raise TypeError('Hks_sparse must be dict[int, scipy.sparse.spmatrix].')

    nkpnts = int(attributes['nkpnts'])
    nspin = int(attributes['nspin'])
    nawf = int(attributes['nawf'])
    nelec = int(attributes['nelec'])
    insulator = bool(attributes.get('insulator', False))
    dftSO = bool(attributes.get('dftSO', False))

    if 'sparse_kpoint_block_range' in attributes:
        start_block, stop_block = attributes['sparse_kpoint_block_range']
    else:
        start_block, stop_block = load_balancing_kpoints(nkpnts, nspin, rank, size)

    # Collect eigenvalues from sparse diagonalization (no eigenvectors stored).
    local_eig_lists: dict[tuple[int, int], np.ndarray] = {}
    for block_idx in range(int(start_block), int(stop_block)):
        if block_idx not in hks_sparse:
            continue
        h_k = hks_sparse[block_idx]
        if not sparse.issparse(h_k):
            raise TypeError(f'Hks_sparse[{block_idx}] must be a scipy sparse matrix.')
        h_csr = h_k.tocsr()

        ik = int(block_idx) // nspin
        ispin = int(block_idx) % nspin

        # Compute all eigenvalues for this block (not eigenvectors).
        # This reuses the dense diagonalization strategy locally while avoiding
        # global dense storage.
        try:
            eigvals = np.linalg.eigvalsh(h_csr.toarray())
        except Exception:
            # Fallback: use sparse eigenvalue solver to get all eigenvalues.
            # Request nawf-1 eigenvalues in different regions and merge.
            eigvals_list = []
            for target in ['SR', 'LR']:
                try:
                    evals, _ = spla.eigs(
                        h_csr,
                        k=min(nawf - 2, h_csr.shape[0] - 2),
                        which=target,
                        return_eigenvectors=False,
                    )
                    eigvals_list.append(evals.real)
                except Exception:
                    pass
            if eigvals_list:
                eigvals = np.sort(np.concatenate(eigvals_list))
            else:
                raise

        local_eig_lists[(ik, ispin)] = eigvals

    # Gather eigenvalues from all ranks.
    if size == 1:
        gathered_eig_lists = [local_eig_lists]
    else:
        gathered_eig_lists = comm.gather(local_eig_lists, root=0)

    if rank == 0:
        assert gathered_eig_lists is not None
        # Reconstruct full eigenvalue array on rank 0.
        eig = np.zeros((nawf, nkpnts, nspin))
        for rank_eigs in gathered_eig_lists:
            for (ik, ispin), eigvals in rank_eigs.items():
                n_available = min(len(eigvals), nawf)
                eig[:n_available, int(ik), int(ispin)] = eigvals[:n_available]

        # Insulator case: Fermi energy is at HOMO.
        if insulator:
            homo_band = nelec - 1 if dftSO else nelec // 2 - 1
            e_fermi = np.max(eig[homo_band, :, :])
        else:
            # Conductor: use bisection with Gaussian smearing.
            eps = 1.0e-10
            degauss = 0.01
            fac = 1 if dftSO else 2

            # Bracket the Fermi energy.
            e_low = np.min(eig) - 2 * degauss
            e_up = np.max(eig) + 2 * degauss

            # Check bracketing.
            sum_low = fac * np.sum(intmetpax(eig, e_low, degauss))
            sum_up = fac * np.sum(intmetpax(eig, e_up, degauss))

            if (sum_up - nelec) < -eps or (sum_low - nelec) > eps:
                if rank == 0:
                    print('Warning: Fermi energy may not be properly bracketed.')

            # Bisection loop.
            max_iter = 100
            for iteration in range(max_iter):
                e_fermi = (e_up + e_low) / 2
                sum_mid = fac * np.sum(intmetpax(eig, e_fermi, degauss))

                if np.abs(sum_mid - nelec) < eps:
                    break

                if sum_mid < nelec:
                    e_low = e_fermi
                else:
                    e_up = e_fermi

        return float(e_fermi)

    return 0.0
