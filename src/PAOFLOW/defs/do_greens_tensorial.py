# -*- coding: utf-8 -*-
"""
Tensorized Green's-function routines for PAOFLOW.

Retarded Green's function:

    G^R(k,E) = [(E - Ef + i eta) I - H(k)]^{-1}

or, with an overlap matrix S(k),

    G^R(k,E) = [(E - Ef + i eta) S(k) - H(k)]^{-1}.

The implementation uses blocked batched NumPy linear solves over
energy, k-point, and spin indices.  It avoids Python loops over k and spin.
"""

from __future__ import annotations

import os
import numpy as np


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _get_mpi():
    """Return MPI objects if mpi4py is available; otherwise serial defaults."""

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        return MPI, comm, comm.Get_rank(), comm.Get_size()
    except Exception:
        return None, None, 0, 1


def _first_key(dictionary, keys):
    """Return the first key present in dictionary, or None."""

    for key in keys:
        if key in dictionary:
            return key
    return None


def _as_hk_nawf_nawf_nk_nspin(Hk):
    """Convert H(k) to shape (nawf, nawf, nkpts, nspin).

    Accepted layouts are:

        (nawf, nawf, nkpts, nspin)
        (nkpts, nawf, nawf, nspin)
    """

    Hk = np.asarray(Hk)

    if Hk.ndim != 4:
        raise ValueError(
            "Hamiltonian must be 4D: (nawf,nawf,nkpts,nspin) "
            "or (nkpts,nawf,nawf,nspin)."
        )

    if Hk.shape[0] == Hk.shape[1]:
        return Hk

    if Hk.shape[1] == Hk.shape[2]:
        return np.transpose(Hk, (1, 2, 0, 3))

    raise ValueError(
        "Could not identify Hamiltonian layout. Expected "
        "(nawf,nawf,nkpts,nspin) or (nkpts,nawf,nawf,nspin), "
        f"got {Hk.shape}."
    )


def _as_sk_nawf_nawf_nk_nspin(Sk, target_shape):
    """Convert S(k) to shape (nawf, nawf, nkpts, nspin)."""

    Sk = _as_hk_nawf_nawf_nk_nspin(Sk)

    if Sk.shape != target_shape:
        raise ValueError(
            f"Overlap shape {Sk.shape} is incompatible with Hamiltonian "
            f"shape {target_shape}."
        )

    return Sk


def _get_fermi(attributes, arrays=None, fermi=None):
    """Get Fermi energy from explicit input, attributes, or arrays."""

    if fermi is not None:
        return float(fermi)

    candidate_keys = ("Efermi", "fermi", "fermi_energy", "Ef", "efermi")

    for key in candidate_keys:
        if key in attributes:
            return float(np.asarray(attributes[key]).reshape(-1)[0])

    if arrays is not None:
        for key in candidate_keys:
            if key in arrays:
                return float(np.asarray(arrays[key]).reshape(-1)[0])

    return 0.0


# def _get_k_weights(arrays, nkpts, k_weights_key=None):
#     """Return normalized k-point weights.

#     If no suitable array is found, uniform weights are used.
#     """

#     if k_weights_key is not None:
#         if k_weights_key not in arrays:
#             raise KeyError(f"k_weights_key='{k_weights_key}' not found in arrays.")
#         weights = np.asarray(arrays[k_weights_key], dtype=float).reshape(-1)
#     else:
#         key = _first_key(
#             arrays,
#             ("kq_wt", "kqwt", "k_weights", "kweights", "wk", "weights"),
#         )
#         if key is None:
#             weights = np.ones(nkpts, dtype=float)
#         else:
#             weights = np.asarray(arrays[key], dtype=float).reshape(-1)

#     # In MPI-distributed runs, a global weight vector may not match the local
#     # number of k-points.  In that case use uniform local weights.  The final
#     # MPI Reduce will still collect local contributions.
#     if weights.size != nkpts:
#         weights = np.ones(nkpts, dtype=float)

#     total = np.sum(weights)
#     if abs(total) < 1.0e-14:
#         raise ValueError("The sum of k-point weights is zero.")

#     return weights / total


def _get_k_weights(arrays, attributes, nkpts, k_weights_key=None, comm=None):
    """Read and globally normalize k-point weights.

    This function is MPI-safe.  Each rank may only hold its local subset
    of k-points, but the normalization is performed using the global
    weight sum over all MPI ranks.

    If no weights are found, uniform local weights are used and normalized
    by the global number of k-points.
    """

    if k_weights_key is not None:
        if k_weights_key not in arrays:
            raise KeyError(f"k_weights_key='{k_weights_key}' not found in arrays.")
        weights = np.asarray(arrays[k_weights_key], dtype=float).reshape(-1)

    else:
        key = _first_key(
            arrays,
            (
                "kq_wt",
                "kqwt",
                "k_weights",
                "kweights",
                "wk",
                "weights",
            ),
        )

        if key is not None:
            weights = np.asarray(arrays[key], dtype=float).reshape(-1)
        else:
            weights = None

    # If no usable weights are found, use one weight per local k-point.
    if weights is None or weights.size != nkpts:
        weights = np.ones(nkpts, dtype=float)

    local_sum = float(np.sum(weights))

    if comm is not None:
        global_sum = comm.allreduce(local_sum)
    else:
        global_sum = local_sum

    if abs(global_sum) < 1.0e-14:
        raise ValueError("The global sum of k-point weights is zero.")

    return weights / global_sum


# -----------------------------------------------------------------------------
# Tensorized Green's functions
# -----------------------------------------------------------------------------


def green_local(
    Hk,
    energies,
    eta=1.0e-3,
    fermi=0.0,
    Sk=None,
    k_weights=None,
    energy_block=32,
):
    """Compute the local/BZ-averaged retarded Green's function.

    Parameters
    ----------
    Hk : ndarray
        Hamiltonian with shape (nawf,nawf,nkpts,nspin) or
        (nkpts,nawf,nawf,nspin).
    energies : array_like
        Energy grid in eV.
    eta : float
        Positive imaginary broadening in eV.
    fermi : float
        Fermi energy in eV.
    Sk : ndarray or None
        Optional overlap matrix with same accepted layout as Hk.
    k_weights : ndarray or None
        k-point weights with shape (nkpts,). Uniform if None.
    energy_block : int
        Number of energies solved simultaneously.

    Returns
    -------
    G_loc : ndarray
        Shape (ne, nawf, nawf, nspin).
    """

    Hk = _as_hk_nawf_nawf_nk_nspin(Hk)
    energies = np.asarray(energies, dtype=float)

    if energies.ndim != 1:
        raise ValueError("energies must be a 1D array.")
    if eta <= 0.0:
        raise ValueError("eta must be positive for a retarded Green function.")
    if energy_block < 1:
        raise ValueError("energy_block must be at least 1.")

    nawf, nawf2, nkpts, nspin = Hk.shape
    if nawf != nawf2:
        raise ValueError("Hamiltonian must be square in orbital space.")

    if Sk is not None:
        Sk = _as_sk_nawf_nawf_nk_nspin(Sk, Hk.shape)

    if k_weights is None:
        k_weights = np.ones(nkpts, dtype=float) / float(nkpts)
    else:
        k_weights = np.asarray(k_weights, dtype=float).reshape(-1)
        if k_weights.size != nkpts:
            raise ValueError(f"k_weights must have length {nkpts}.")
        # total = np.sum(k_weights)
        # if abs(total) < 1.0e-14:
        #     raise ValueError("The sum of k-point weights is zero.")
        # k_weights = k_weights / total
        if k_weights is None:
            k_weights = np.ones(nkpts, dtype=float) / float(nkpts)
        else:
            k_weights = np.asarray(k_weights, dtype=float).reshape(-1)

            if k_weights.size != nkpts:
                raise ValueError(
                    f"k_weights must have shape ({nkpts},), got {k_weights.shape}."
                )

    ne = energies.size
    dtype = np.result_type(Hk.dtype, np.complex128)
    eye = np.eye(nawf, dtype=dtype)

    # Batch layout: Hbatch[k, s, a, b]
    Hbatch = np.transpose(Hk, (2, 3, 0, 1)).astype(dtype, copy=False)

    if Sk is None:
        Sbatch = None
    else:
        Sbatch = np.transpose(Sk, (2, 3, 0, 1)).astype(dtype, copy=False)

    G_loc = np.zeros((ne, nawf, nawf, nspin), dtype=dtype)

    for i0 in range(0, ne, energy_block):
        i1 = min(i0 + energy_block, ne)
        z = energies[i0:i1] - fermi + 1.0j * eta

        # mat[e, k, s, a, b]
        if Sbatch is None:
            mat = (
                z[:, None, None, None, None] * eye[None, None, None, :, :]
                - Hbatch[None, :, :, :, :]
            )
        else:
            mat = (
                z[:, None, None, None, None] * Sbatch[None, :, :, :, :]
                - Hbatch[None, :, :, :, :]
            )

        rhs = np.broadcast_to(eye, mat.shape)
        Gbatch = np.linalg.solve(mat, rhs)  # Gbatch[e, k, s, a, b]

        G_loc[i0:i1] = np.einsum(
            "k,eksab->eabs",
            k_weights,
            Gbatch,
            optimize=True,
        )

    return G_loc


def green_k(
    Hk,
    energies,
    eta=1.0e-3,
    fermi=0.0,
    Sk=None,
    energy_block=32,
):
    """Compute the k-resolved retarded Green's function.

    Returns
    -------
    G_k : ndarray
        Shape (ne, nawf, nawf, nkpts, nspin).
    """

    Hk = _as_hk_nawf_nawf_nk_nspin(Hk)
    energies = np.asarray(energies, dtype=float)

    if energies.ndim != 1:
        raise ValueError("energies must be a 1D array.")
    if eta <= 0.0:
        raise ValueError("eta must be positive for a retarded Green function.")
    if energy_block < 1:
        raise ValueError("energy_block must be at least 1.")

    nawf, nawf2, nkpts, nspin = Hk.shape
    if nawf != nawf2:
        raise ValueError("Hamiltonian must be square in orbital space.")

    if Sk is not None:
        Sk = _as_sk_nawf_nawf_nk_nspin(Sk, Hk.shape)

    ne = energies.size
    dtype = np.result_type(Hk.dtype, np.complex128)
    eye = np.eye(nawf, dtype=dtype)

    Hbatch = np.transpose(Hk, (2, 3, 0, 1)).astype(dtype, copy=False)

    if Sk is None:
        Sbatch = None
    else:
        Sbatch = np.transpose(Sk, (2, 3, 0, 1)).astype(dtype, copy=False)

    G_k = np.empty((ne, nawf, nawf, nkpts, nspin), dtype=dtype)

    for i0 in range(0, ne, energy_block):
        i1 = min(i0 + energy_block, ne)
        z = energies[i0:i1] - fermi + 1.0j * eta

        if Sbatch is None:
            mat = (
                z[:, None, None, None, None] * eye[None, None, None, :, :]
                - Hbatch[None, :, :, :, :]
            )
        else:
            mat = (
                z[:, None, None, None, None] * Sbatch[None, :, :, :, :]
                - Hbatch[None, :, :, :, :]
            )

        rhs = np.broadcast_to(eye, mat.shape)
        Gbatch = np.linalg.solve(mat, rhs)  # Gbatch[e, k, s, a, b]
        G_k[i0:i1] = np.transpose(Gbatch, (0, 3, 4, 1, 2))

    return G_k


# -----------------------------------------------------------------------------
# DOS utilities
# -----------------------------------------------------------------------------


def dos_from_green(G_loc):
    """Compute total DOS from local Green's function.

    Parameters
    ----------
    G_loc : ndarray
        Shape (ne, nawf, nawf, nspin).

    Returns
    -------
    dos : ndarray
        Shape (ne, nspin).
    """

    G_loc = np.asarray(G_loc)
    if G_loc.ndim != 4:
        raise ValueError("G_loc must have shape (ne,nawf,nawf,nspin).")

    trace_g = np.trace(G_loc, axis1=1, axis2=2)
    return -np.imag(trace_g) / np.pi


def pdos_from_green(G_loc):
    """Compute orbital-projected DOS from local Green's function.

    Parameters
    ----------
    G_loc : ndarray
        Shape (ne, nawf, nawf, nspin).

    Returns
    -------
    pdos : ndarray
        Shape (ne, nawf, nspin).
    """

    G_loc = np.asarray(G_loc)
    if G_loc.ndim != 4:
        raise ValueError("G_loc must have shape (ne,nawf,nawf,nspin).")

    diag_g = np.diagonal(G_loc, axis1=1, axis2=2)  # (ne,nspin,nawf)
    pdos = -np.imag(diag_g) / np.pi
    return np.transpose(pdos, (0, 2, 1))


def _write_dos_files(energies, dos, output_dir=".", prefix="green_dos"):
    """Write green_dos_<spin>.dat files."""

    energies = np.asarray(energies, dtype=float)
    dos = np.asarray(dos, dtype=float)

    if dos.ndim != 2:
        raise ValueError("dos must have shape (ne,nspin).")
    if energies.size != dos.shape[0]:
        raise ValueError("energies and DOS have incompatible lengths.")

    os.makedirs(output_dir, exist_ok=True)

    for ispin in range(dos.shape[1]):
        filename = os.path.join(output_dir, f"{prefix}_{ispin}.dat")
        np.savetxt(
            filename,
            np.column_stack((energies, dos[:, ispin])),
            # header="Energy(eV) DOS",
        )


# -----------------------------------------------------------------------------
# PAOFLOW driver
# -----------------------------------------------------------------------------


def do_greens(
    data_controller,
    energies,
    eta=1.0e-3,
    fermi=0.0,
    hamiltonian_key=None,
    overlap_key=None,
    k_weights_key=None,
    return_local=True,
    return_kresolved=False,
    compute_dos=True,
    write_files=True,
    energy_block=32,
):
    """Compute Green's functions from PAOFLOW data_controller.

    Parameters
    ----------
    data_controller : object
        PAOFLOW data controller. Must provide data_dicts().
    energies : array_like
        Energy grid in eV.
    eta : float
        Positive imaginary broadening in eV.
    fermi : float or None
        Fermi level in eV. If None, read from attributes when possible.
    hamiltonian_key : str or None
        Key for H(k) in arrays. If None, common PAOFLOW names are tried.
    overlap_key : str or None
        Optional key for S(k) in arrays.
    k_weights_key : str or None
        Optional key for k-point weights in arrays.
    return_local : bool
        Store arrays['G_loc'] on rank 0.
    return_kresolved : bool
        Store arrays['G_k']. Can be very large.
    compute_dos : bool
        Compute arrays['dos_green'] and arrays['pdos_green'] from G_loc.
    write_files : bool
        Write green_dos_<spin>.dat on rank 0.
    energy_block : int
        Number of energies solved simultaneously.

    Returns
    -------
    results : dict
        Computed quantities. In MPI, reduced local quantities are returned
        only on rank 0.
    """

    MPI, comm, rank, size = _get_mpi()
    arrays, attributes = data_controller.data_dicts()

    energies = np.asarray(energies, dtype=float)
    if energies.ndim != 1:
        raise ValueError("energies must be a 1D array.")
    if eta <= 0.0:
        raise ValueError("eta must be positive for a retarded Green function.")
    if energy_block < 1:
        raise ValueError("energy_block must be at least 1.")

    if hamiltonian_key is None:
        hamiltonian_key = _first_key(
            arrays,
            ("Hksp", "Hks", "Hk", "hamiltonian"),
        )

    if hamiltonian_key is None:
        raise KeyError(
            "Could not find a Hamiltonian array. Pass hamiltonian_key explicitly, "
            "for example hamiltonian_key='Hksp'."
        )
    if hamiltonian_key not in arrays:
        raise KeyError(f"hamiltonian_key='{hamiltonian_key}' not found in arrays.")

    Hk = _as_hk_nawf_nawf_nk_nspin(arrays[hamiltonian_key])
    _, _, nkpts, _ = Hk.shape

    if overlap_key is not None:
        if overlap_key not in arrays:
            raise KeyError(f"overlap_key='{overlap_key}' not found in arrays.")
        Sk = _as_sk_nawf_nawf_nk_nspin(arrays[overlap_key], Hk.shape)
    else:
        Sk = None

    ef = _get_fermi(attributes, arrays=arrays, fermi=fermi)
    # k_weights = _get_k_weights(arrays, nkpts, k_weights_key=k_weights_key)

    k_weights = _get_k_weights(
        arrays,
        attributes,
        nkpts,
        k_weights_key=k_weights_key,
        comm=comm,
    )



    arrays["green_energies"] = energies
    arrays["green_eta"] = eta
    arrays["green_fermi"] = ef
    arrays["green_energy_block"] = int(energy_block)

    results = {}

    if return_local or compute_dos:
        G_loc_local = green_local(
            Hk,
            energies,
            eta=eta,
            fermi=ef,
            Sk=Sk,
            k_weights=k_weights,
            energy_block=energy_block,
        )

        if size > 1:
            G_loc = np.empty_like(G_loc_local) if rank == 0 else None
            comm.Reduce(G_loc_local, G_loc, op=MPI.SUM, root=0)
        else:
            G_loc = G_loc_local

        if rank == 0:
            if return_local:
                arrays["G_loc"] = G_loc
                results["G_loc"] = G_loc

            if compute_dos:
                dos = dos_from_green(G_loc)
                pdos = pdos_from_green(G_loc)

                arrays["dos_green"] = dos
                arrays["pdos_green"] = pdos

                results["dos_green"] = dos
                results["pdos_green"] = pdos

                if write_files:
                    output_dir = attributes.get("opath", ".")
                    _write_dos_files(
                        energies,
                        dos,
                        output_dir=output_dir,
                        prefix="green_dos",
                    )

    if return_kresolved:
        G_k = green_k(
            Hk,
            energies,
            eta=eta,
            fermi=ef,
            Sk=Sk,
            energy_block=energy_block,
        )

        arrays["G_k"] = G_k
        results["G_k"] = G_k

    if rank == 0:
        results["green_energies"] = energies
        results["green_eta"] = eta
        results["green_fermi"] = ef
        results["green_energy_block"] = int(energy_block)

    return results
