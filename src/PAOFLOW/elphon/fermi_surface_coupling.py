r"""State-resolved electron-phonon coupling on the Fermi surface.

Input of the anisotropic Migdal-Eliashberg equations
(:mod:`PAOFLOW.elphon.anisotropic_eliashberg`), accumulated during the dense-q
loop of :func:`~PAOFLOW.elphon.do_pao_eph_dense_q.eliashberg_dense_q`.

The Fermi-surface states are the bands ``n`` at the dense k-points with
``|e_nk - E_F| < fsthick`` (EPW ``fsthick``).  Only irreducible states are kept:
``Delta_nk`` and ``Z_nk`` are invariant under the crystal point group (with time
reversal), so ``k`` runs over one representative per star of the dense k-grid.
With ``Lambda_{nk,mk'}(omega)`` the coupling of the pair at phonon frequency
``omega``, the folded pair coupling of two irreducible states ``a = (n, i)`` and
``b = (m, i')`` is the average over the star of ``a``, weighted by the
Fermi-surface delta, of the coupling to every member of the star of ``b``:

.. math::

    \Lambda_{ab}(\omega) = \frac{1}{D_a}\sum_{q\in\mathrm{IBZ}} w_q
        \sum_{k\in\star(i),\,k+q\in\star(i')}
        \delta(\epsilon_{nk})\,\Lambda_{nk, m\,k+q}(\omega),
    \qquad D_a = \sum_{k\in\star(i)}\delta(\epsilon_{nk}),

with ``w_q`` the star multiplicity of ``q``.  Summing irreducible q with their
multiplicities instead of all q uses ``Lambda_{Sk, Sk+Sq} = Lambda_{k, k+q}``
and lets the existing loop over irreducible q and all k fill the folded coupling
directly.  With exact symmetry ``Lambda_ab = sum_{k' in star(i')}
Lambda_{n k_i, m k'}``; the delta weighting keeps ``N_F`` and the isotropic
``lambda`` exact when the interpolated bands are only approximately symmetric.
The pair coupling is

.. math::

    \Lambda_{nk,mk+q}(\omega) = \frac{1}{N_q}\sum_\nu
        \delta(\epsilon_{m k+q})\,\frac{2|g_{mn\nu}(k,q)|^2}{\omega_{q\nu}}\,
        \delta(\omega - \omega_{q\nu}),

so that ``lambda_nk = sum_b int Lambda_ab`` is EPW's ``lambda_nk`` and
``sum_a W_a lambda_a / N_F`` the isotropic ``lambda``, with the state weights
``W_a = D_a / N_k`` summing to ``N_F``.  The Coulomb weights

.. math::

    M_{ab} = \frac{1}{D_a}\sum_{q\in\mathrm{IBZ}} \frac{w_q}{N_q}
        \sum_{k\in\star(i),\,k+q\in\star(i')}
        \delta(\epsilon_{nk})\,\delta(\epsilon_{m\,k+q})

restrict the ``mu*`` term to the same pairs as the phonon kernel (EPW sums both
over ``k+q``).  When the q-grid is coarser than the k-grid, ``k+q`` reaches only
a sublattice of the k-grid; a Coulomb term spread uniformly over the whole Fermi
surface would then couple sublattices that the phonons do not, and favour
spurious gaps of opposite sign on them.  The frequency dependence is stored on
the uniform grid ``omega_j = j * d_omega``: every mode is split between its two
neighbouring grid points with linear weights, which conserves ``lambda``
exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .elph_bloch import RY_TO_EV, RY_TO_THZ


class FermiSurfacePairCoupling:
    """Accumulator of the folded pair coupling ``Lambda_ab(omega)``.

    Parameters
    ----------
    electrons : dict
        Dense electron cache of
        :func:`~PAOFLOW.elphon.elph_bloch.precompute_dense_electrons`.
    star_index : ndarray, shape ``(Nk^3,)``
        k-star of every dense k-point
        (:func:`~PAOFLOW.elphon.do_pao_eph_dense_q.irreducible_mesh`).
    representatives : ndarray, shape ``(nk_irr,)``
        Flat index of the representative of each k-star.
    star_sizes : ndarray, shape ``(nk_irr,)``
        Multiplicity of each k-star.
    fsthick_ev : float
        Half-width of the Fermi window (eV).
    omega_max_ev : float
        Highest phonon frequency (eV); the frequency grid extends to
        ``1.05 * omega_max_ev``.
    n_freq : int, optional
        Number of phonon-frequency grid points (default 100).
    isig : int, optional
        Smearing index of ``electrons`` (default 0).

    Notes
    -----
    Memory is ``8 N^2 n_freq`` bytes, with ``N`` the number of irreducible
    Fermi-surface states.
    """

    def __init__(
        self,
        electrons: dict[str, Any],
        star_index: NDArray[np.int_],
        representatives: NDArray[np.int_],
        star_sizes: NDArray[np.float64],
        fsthick_ev: float,
        omega_max_ev: float,
        n_freq: int = 100,
        isig: int = 0,
    ) -> None:
        self.nk_side = int(electrons['Nk'])
        energies_ry = electrons['E']
        fermi_ry = float(electrons['ef_sig'][isig])
        self.delta_ry = electrons['dk'][isig]  # (nkd, nawf), 1/Ry
        self.star_index = np.asarray(star_index, dtype=int)
        self.star_sizes = np.asarray(star_sizes, dtype=float)
        self.fsthick_ev = float(fsthick_ev)
        self.sigma_ev = float(electrons['sigmas'][isig]) * RY_TO_EV

        # Irreducible states: the window is decided at the star representative so
        # that every member of a star carries the same state labels.
        representatives = np.asarray(representatives, dtype=int)
        energy_rep_ev = (energies_ry[representatives] - fermi_ry) * RY_TO_EV
        irr_index, band_index = np.nonzero(np.abs(energy_rep_ev) < self.fsthick_ev)
        self.nstates = irr_index.size
        state_of_irreducible = -np.ones(energy_rep_ev.shape, dtype=int)
        state_of_irreducible[irr_index, band_index] = np.arange(self.nstates)
        self.state_of = state_of_irreducible[self.star_index]  # (nkd, nawf)
        self.k_in_window = np.any(self.state_of >= 0, axis=1)

        nkd = self.nk_side**3
        self.state_band = band_index
        self.state_irreducible_k = irr_index
        self.state_k_cryst = electrons['K'][representatives[irr_index]]
        self.state_energy_ev = energy_rep_ev[irr_index, band_index]
        self.state_multiplicity = self.star_sizes[irr_index]
        # D_a: Fermi-surface delta summed over the star of each state (1/Ry).
        member = self.state_of >= 0
        self.state_delta_sum = np.zeros(self.nstates)
        np.add.at(self.state_delta_sum, self.state_of[member], self.delta_ry[member])
        self.state_weight = self.state_delta_sum / RY_TO_EV / nkd
        self.fermi_ev = fermi_ry * RY_TO_EV

        self.n_freq = int(n_freq)
        self.freq_step_ev = 1.05 * float(omega_max_ev) / self.n_freq
        self.freq_ev = self.freq_step_ev * np.arange(1, self.n_freq + 1)
        self.coupling = np.zeros((self.nstates, self.nstates, self.n_freq))
        self.coulomb = np.zeros((self.nstates, self.nstates))

    def accumulate(
        self,
        k_indices: NDArray[np.int_],
        k_shift: NDArray[np.int_],
        g2_tilde: NDArray[np.float64],
        freqs_thz: NDArray[np.float64],
        mode_weights: NDArray[np.float64],
        q_weight: float,
    ) -> None:
        """Add the pairs ``(nk, m k+q)`` of one block of k-points.

        Parameters
        ----------
        k_indices : ndarray, shape ``(nb,)``
            Flat dense-grid indices of the k-points.
        k_shift : ndarray, shape ``(3,)``
            ``q * Nk`` (integers): ``k + q`` is the grid point shifted by it.
        g2_tilde : ndarray, shape ``(nb, nmode, nawf, nawf)``
            ``|<m k+q| dV_nu |n k>|^2`` with mass-weighted eigenvectors and no
            ``1/(2 omega)`` factor (Ry^3), indexed ``[k, nu, m, n]``, as built in
            :func:`~PAOFLOW.elphon.elph_bloch.lambda_q_dense_ws_fast`.
        freqs_thz : ndarray, shape ``(nmode,)``
            Phonon frequencies at ``q`` (THz).
        mode_weights : ndarray, shape ``(nmode,)``
            Weight of each mode: the q-star multiplicity divided by the number
            of q-points, or 0 for modes excluded from the coupling.
        q_weight : float
            q-star multiplicity divided by the number of q-points, the weight
            of the Coulomb pairs (no mode exclusions).

        Notes
        -----
        A mode adds ``delta(e_{m k+q}) |g~|^2 / omega^2 = 2 delta |g|^2 / omega``
        (the pair ``lambda``) times ``mode_weights * delta(e_nk) / D_a``.
        """
        nk = self.nk_side
        rows = self.state_of[k_indices]  # (nb, nawf): states n at k
        grid = np.stack(np.unravel_index(k_indices, (nk, nk, nk)), axis=-1)
        kq_indices = np.ravel_multi_index(tuple(((grid + k_shift) % nk).T), (nk, nk, nk))
        cols = self.state_of[kq_indices]  # (nb, nawf): states m at k+q
        block, band_m, band_n = np.nonzero((cols[:, :, None] >= 0) & (rows[:, None, :] >= 0))
        if block.size == 0:
            return
        row_states, col_states = rows[block, band_n], cols[block, band_m]
        pair_factor = (
            self.delta_ry[kq_indices[block], band_m]
            * self.delta_ry[k_indices[block], band_n]
            / self.state_delta_sum[row_states]
        )
        np.add.at(
            self.coulomb.reshape(-1),
            row_states * self.nstates + col_states,
            pair_factor * (q_weight / RY_TO_EV),
        )

        omega_ry = np.abs(np.asarray(freqs_thz, dtype=float)) / RY_TO_THZ
        active = (np.asarray(mode_weights) != 0.0) & (omega_ry > 1.0e-8)
        if not np.any(active):
            return
        modes = np.nonzero(active)[0]
        mode_factor = np.asarray(mode_weights, dtype=float)[modes] / omega_ry[modes] ** 2
        values = (
            g2_tilde[block[:, None], modes[None, :], band_m[:, None], band_n[:, None]]
            * pair_factor[:, None]
            * mode_factor[None, :]
        )  # (npairs, nactive)

        lower, upper_fraction = self._grid_split(omega_ry[modes] * RY_TO_EV)
        pair_offset = (row_states * self.nstates + col_states) * self.n_freq
        flat = self.coupling.reshape(-1)
        np.add.at(flat, (pair_offset[:, None] + lower[None, :]).ravel(),
                  (values * (1.0 - upper_fraction)[None, :]).ravel())  # fmt: skip
        np.add.at(flat, (pair_offset[:, None] + lower[None, :] + 1).ravel(),
                  (values * upper_fraction[None, :]).ravel())  # fmt: skip

    def _grid_split(
        self, omega_ev: NDArray[np.float64]
    ) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
        """Lower grid index and linear weight of the upper neighbour of each frequency.

        Frequencies below the first grid point go entirely to it, those above
        the last to the last one.
        """
        position = omega_ev / self.freq_step_ev - 1.0  # 0-based grid coordinate
        lower = np.clip(np.floor(position).astype(int), 0, self.n_freq - 2)
        upper_fraction = np.clip(position - lower, 0.0, 1.0)
        return lower, upper_fraction

    def reduce(self, comm: Any) -> None:
        """Sum the coupling accumulated on every rank of ``comm`` (in place)."""
        if comm is not None and comm.Get_size() > 1:
            from mpi4py import MPI

            comm.Allreduce(MPI.IN_PLACE, self.coupling, op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, self.coulomb, op=MPI.SUM)

    def result(self, bg: ArrayLike, at: ArrayLike, nq_side: int) -> dict[str, Any]:
        """The Fermi-surface coupling as a dictionary of arrays.

        Parameters
        ----------
        bg, at : array_like, shape ``(3, 3)``
            Reciprocal and direct lattice vectors (rows).
        nq_side : int
            Dense q-grid size.

        Returns
        -------
        dict
            ``coupling`` ``(N, N, n_freq)`` (dimensionless pair ``lambda`` per
            frequency point), ``freq_ev`` ``(n_freq,)``, ``coulomb`` ``(N, N)``
            (``M_ab``, 1/eV per spin, rows summing to about ``N_F``); per state ``band``,
            ``k_cryst`` ``(N, 3)``, ``energy_ev`` (``e - E_F``), ``weight``
            (``W_a``, 1/eV per spin, summing to ``N_F``), ``multiplicity`` and
            ``irreducible_k``; and ``fermi_ev``, ``fsthick_ev``, ``sigma_ev``,
            ``nk_dense``, ``nq_dense``, ``bg``, ``at``.  See
            :func:`write_fs_coupling`.
        """
        return {
            'coupling': self.coupling,
            'coulomb': self.coulomb,
            'freq_ev': self.freq_ev,
            'band': self.state_band,
            'k_cryst': self.state_k_cryst,
            'energy_ev': self.state_energy_ev,
            'weight': self.state_weight,
            'multiplicity': self.state_multiplicity,
            'irreducible_k': self.state_irreducible_k,
            'fermi_ev': self.fermi_ev,
            'fsthick_ev': self.fsthick_ev,
            'sigma_ev': self.sigma_ev,
            'nk_dense': self.nk_side,
            'nq_dense': int(nq_side),
            'bg': np.asarray(bg, dtype=float),
            'at': np.asarray(at, dtype=float),
        }


def write_fs_coupling(path: str, coupling: dict[str, Any]) -> None:
    """Save the Fermi-surface coupling of :meth:`FermiSurfacePairCoupling.result`.

    Parameters
    ----------
    path : str
        Output ``.npz`` file.
    coupling : dict
        The coupling dictionary.
    """
    np.savez(path, **coupling)


def read_fs_coupling(path: str) -> dict[str, Any]:
    """Load a Fermi-surface coupling written by :func:`write_fs_coupling`.

    Parameters
    ----------
    path : str
        The ``.npz`` file.

    Returns
    -------
    dict
        The arrays, with the 0-d entries converted to Python scalars.
    """
    with np.load(path) as data:
        return {key: (data[key].item() if data[key].ndim == 0 else data[key]) for key in data}
