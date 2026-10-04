#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016-2018 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
#
# Reference:
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

import numpy as np


# Compute Z2 invariant and topological properties on a selected path in the BZ
def do_berry_curvature(data_controller):
    from mpi4py import MPI
    from ..utils.constants import ANGSTROM_AU
    from ..utils.get_R_grid_fft import get_R_grid_fft
    from ..utils.communication import scatter_full, gather_full
    from ..spectrum.kpnts_interpolation_mesh import kpnts_interpolation_mesh

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    arrays, attributes = data_controller.data_dicts()

    npool = attributes['npool']

    if 'kq' not in arrays:
        kpnts_interpolation_mesh(data_controller)

    HRs = arrays['HRs']
    bnd = attributes['nawf']
    nkpi = arrays['kq'].shape[1]
    nawf, _, nk1, nk2, nk3, nspin = HRs.shape

    if 'Rfft' not in arrays:
        get_R_grid_fft(data_controller, nk1, nk2, nk3)

    ipol = attributes['ipol']
    jpol = attributes['jpol']
    spol = attributes['spol']

    curvature = attributes['curvature']

    alat = attributes['alat'] / ANGSTROM_AU

    # Compute momenta and kinetic energy
    kq_aux = scatter_full(arrays['kq'].T, npool)
    kq_aux = kq_aux.T

    # Compute R*H(R)
    Rfft = np.reshape(arrays['Rfft'], (nk1 * nk2 * nk3, 3), order='C')
    HRs = np.reshape(HRs, (nawf, nawf, nk1 * nk2 * nk3, nspin), order='C')
    HRs = np.moveaxis(HRs, 2, 0)

    HRs_aux = scatter_full(HRs, npool)
    Rfft_aux = scatter_full(Rfft, npool)

    HRs = np.reshape(np.moveaxis(HRs, 0, 2), (nawf, nawf, nk1, nk2, nk3, nspin), order='C')

    Oj = arrays['Oj']
    jks = np.zeros((kq_aux.shape[1], 3, bnd, bnd, nspin), dtype=complex)

    pks = np.zeros((kq_aux.shape[1], 3, bnd, bnd, nspin), dtype=complex)
    for l in range(3):
        dHRs = np.zeros((HRs_aux.shape[0], nawf, nawf, nspin), dtype=complex)
        for ispin in range(nspin):
            for n in range(nawf):
                for m in range(nawf):
                    dHRs[:, n, m, ispin] = (
                        1.0j * alat * ANGSTROM_AU * Rfft_aux[:, l] * HRs_aux[:, n, m, ispin]
                    )

        dHRs = gather_full(dHRs, npool)
        if rank != 0:
            dHRs = np.zeros((nk1 * nk2 * nk3, nawf, nawf, nspin), dtype=complex)
        comm.Bcast(dHRs)
        dHRs = np.moveaxis(dHRs, 0, 2)

        # Compute dH(k)/dk on the path
        dHks_aux = band_loop_H(dHRs, Rfft, kq_aux, nawf, nspin)

        dHRs = None

        # Compute momenta
        for ik in range(dHks_aux.shape[0]):
            for ispin in range(nspin):
                pks[ik, l, :, :, ispin], jks[ik, l, :, :, ispin] = path_matrix_elements_k(
                    arrays['v_k'][ik, :, :, ispin], dHks_aux[ik, :, :, ispin], Oj[spol], bnd
                )

    Omj_zk = np.zeros((pks.shape[0], 1), dtype=float)
    Omj_znk = np.zeros((pks.shape[0], bnd), dtype=float)
    for ik in range(pks.shape[0]):
        Omj_znk[ik], Omj_zk[ik] = path_berry_k(
            arrays['E_k'][ik, :, 0], pks[ik, :, :, :, 0], jks[ik, :, :, :, 0], attributes, bnd
        )

    velk = np.zeros((pks.shape[0], 3, bnd, nspin), dtype=float)
    for n in range(bnd):
        velk[:, :, n, :] = np.real(pks[:, :, n, n, :])
    pks = jks = None
    write_berry_path(data_controller, velk, Omj_zk, Omj_znk, curvature, spol, ipol, jpol, nkpi)


def _project(v_k, operator):
    """``(v^dagger O v)`` for a dense or ``scipy.sparse`` operator; the dense
    association order is kept for ndarrays."""
    from scipy.sparse import issparse

    if issparse(operator):
        return np.conj(v_k.T) @ np.asarray(operator @ v_k)
    return np.conj(v_k.T).dot(operator).dot(v_k)


def _anticommutator(O, dH):
    """:math:`\\tfrac12\\{O, \\partial H\\}`, with ``np.dot`` for ndarrays as before."""
    from scipy.sparse import issparse

    if issparse(O) or issparse(dH):
        return 0.5 * (O @ dH + dH @ O)
    return 0.5 * (np.dot(O, dH) + np.dot(dH, O))


def path_matrix_elements_k(v_k, dHk, O, bnd: int):
    """Momentum and current matrix elements of one direction at one path point.

    Parameters
    ----------
    v_k : np.ndarray, shape ``(nawf, nawf)``
        Every eigenvector at this point (columns).
    dHk : np.ndarray or scipy.sparse matrix, shape ``(nawf, nawf)``
        ``dH/dk_l`` (lattice part only, as built here from ``Rfft``).
    O : np.ndarray or scipy.sparse matrix, shape ``(nawf, nawf)``, or None
        Spin or orbital component of the curvature operator; ``None`` skips
        the current (``jks`` is then ``None``).
    bnd : int
        Bands kept.

    Returns
    -------
    (pks, jks) : tuple of np.ndarray, shape ``(bnd, bnd)``
        :math:`V^\\dagger \\partial_l H V` and
        :math:`V^\\dagger \\tfrac12\\{O, \\partial_l H\\} V`.
    """
    pks = _project(v_k, dHk)[:bnd, :bnd]
    if O is None:
        return pks, None
    jks = _project(v_k, _anticommutator(O, dHk))[:bnd, :bnd]
    return pks, jks


def path_berry_k(E, pks, jks, attributes, bnd: int):
    """Spin/orbital Berry curvature of one path point, per band and summed.

    Parameters
    ----------
    E : np.ndarray, shape ``(nbands,)``
        Eigenvalues (the first ``bnd`` are used).
    pks, jks : np.ndarray, shape ``(3, bnd, bnd)``
        :func:`path_matrix_elements_k` of the three directions.
    attributes : dict
        Supplies ``ipol``, ``jpol`` and the overrides ``bc_delta``,
        ``bc_mu`` and ``bc_smearing``.
    bnd : int

    Returns
    -------
    (Omj_znk, Omj_zk) : tuple
        Band-resolved curvature ``(bnd,)`` and its occupation-weighted sum.
    """
    ipol = attributes['ipol']
    jpol = attributes['jpol']
    # --- broadening / occupation controls (overridable via attributes) ---
    #   bc_delta    : Lorentzian broadening in the energy denominator (eV).
    #   bc_mu       : chemical potential (E_k are referenced to E_F, so 0.0 = E_F).
    #                 NB: this used to be hard-coded to -0.02 eV, an arbitrary
    #                 offset from the Fermi level; the default is now the Fermi
    #                 level (0.0).
    #   bc_smearing : occupation smearing kT (eV). 0 => sharp T=0 step (original
    #                 behaviour). A finite value uses a Fermi-Dirac occupation so
    #                 the band-summed curvature is CONTINUOUS along the path even
    #                 for metals -- the sharp step makes the sum jump wherever a
    #                 band crosses the Fermi level (e.g. along Gamma-M).
    deltab = attributes.get('bc_delta', 0.05)
    mu = attributes.get('bc_mu', 0.0)
    kbt = attributes.get('bc_smearing', 0.0)

    Omj_znk = np.zeros(bnd, dtype=float)
    for n in range(bnd):
        for m in range(bnd):
            if m != n:
                Omj_znk[n] += (
                    -2.0
                    * np.imag(jks[ipol, n, m] * pks[jpol, m, n])
                    / ((E[m] - E[n]) ** 2 + deltab**2)
                )
    en = E[:bnd]
    if kbt > 0.0:
        # Fermi-Dirac occupation -> continuous band-summed curvature
        occ = 1.0 / (np.exp(np.clip((en - mu) / kbt, -60.0, 60.0)) + 1.0)
    else:
        occ = 0.5 * (1.0 - np.sign(en - mu))  # T=0.0K sharp step (original)
    return Omj_znk, np.sum(Omj_znk[:] * occ)


def write_berry_path(data_controller, velk, Omj_zk, Omj_znk, curvature, spol, ipol, jpol, nkpi):
    """Gather and write the path velocities and Berry curvatures (collective).

    Parameters
    ----------
    velk : np.ndarray, shape ``(nk_local, 3, bnd, nspin)``
        Band velocities, the real diagonal of ``pks``.
    Omj_zk : np.ndarray, shape ``(nk_local, 1)``
    Omj_znk : np.ndarray, shape ``(nk_local, bnd)``
    curvature : str
        ``'Spin'`` or ``'Orbital'``, for the file names.
    """
    from mpi4py import MPI

    from ..utils.communication import gather_full
    from ..utils.constants import LL

    rank = MPI.COMM_WORLD.Get_rank()
    arrays, attributes = data_controller.data_dicts()
    npool = attributes['npool']
    bnd = velk.shape[2]

    indices = (curvature, LL[spol], LL[ipol], LL[jpol])
    lrng = list(range(nkpi)) if rank == 0 else None

    velk = gather_full(velk, npool)
    for l in range(3):
        fvk = 'velocity_' + str(l)
        data_controller.write_bands(fvk, (velk[:, l, :bnd, :] if rank == 0 else None))
    velk = None

    Omj_zk = gather_full(Omj_zk, npool)
    Omj_znk = gather_full(Omj_znk, npool)
    fOmj_zk = 'Omegaj_%s_%s_%s%s.dat' % (indices)

    data_controller.write_file_row_col(fOmj_zk, lrng, (Omj_zk[:, 0] if rank == 0 else None))

    # Band-resolved Berry curvature Omega_n(k): smooth per band (no occupation
    # step), written in the bands_0.dat layout (k-index + one column per band).
    # Use this for band-coloured "Berry-curvature texture" plots and as the
    # occupation-independent alternative to the summed curve above.
    if attributes.get('bc_write_bands', True):
        fOmj_znk = 'Omegaj_%s_%s_%s%s_bands' % (indices)
        data_controller.write_bands(fOmj_znk, (Omj_znk[:, :, None] if rank == 0 else None))
    Omj_zk = Omj_znk = fOmj_zk = None


def band_loop_H(HRaux, R, kq, nawf, nspin):
    kdot = np.zeros((kq.shape[1], R.shape[0]), dtype=complex, order='C')
    kdot = np.tensordot(R, 2.0j * np.pi * kq, axes=([1], [0]))
    np.exp(kdot, kdot)

    Haux = np.zeros((nawf, nawf, kq.shape[1], nspin), dtype=complex, order='C')

    for ispin in range(nspin):
        Haux[:, :, :, ispin] = np.tensordot(HRaux[:, :, :, ispin], kdot, axes=([2], [0]))

    kdot = None
    Haux = np.transpose(Haux, (2, 0, 1, 3))
    return Haux
