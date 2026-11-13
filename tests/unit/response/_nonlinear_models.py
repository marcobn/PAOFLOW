"""Hand-built tight-binding models in PAOFLOW's ``HRs`` layout for the non-linear optics tests.

Every model gives ``HRs`` (``H(k) = sum_n exp(-2 pi i k.n) HRs[..., n]``), the lattice vectors
(rows, alat units), ``alat`` (bohr) and the orbital centres (crystal coordinates), plus an
*exact* atomic-gauge Bloch Hamiltonian ``H^A(k)`` with its analytic first and second
k-derivatives, built directly from the hopping list ``(i, j, R, h)`` (orbital ``i`` in the
home cell to orbital ``j`` in cell ``R``):

    H^A_ij(k) = sum_R exp(i k_cart . d) h_ij(R),   d = alat (R + tau_j - tau_i) @ at.
"""

from __future__ import annotations

import numpy as np

from PAOFLOW.models import models as tb_models

ALAT = 1.0

# ``HRs[i, j, -R mod ng] += h``: ``H(k) = sum_n exp(-2 pi i k.n) HRs[n]`` is the lattice-gauge sum
hopping_list_to_HRs = tb_models.hoppings_to_HRs


class ExactModel:
    """A hopping list with analytic atomic-gauge Bloch operators."""

    def __init__(self, hops, nawf, at, alat, tau_cryst, ng, nelec):
        self.hops = hops
        self.nawf = nawf
        self.at = np.asarray(at, dtype=float)
        self.alat = float(alat)
        self.tau = np.asarray(tau_cryst, dtype=float)
        self.ng = tuple(int(n) for n in ng)
        self.nelec = nelec
        self.HRs = hopping_list_to_HRs(hops, nawf, self.ng)
        self.bg = np.linalg.inv(self.at).T  # rows: reciprocal vectors in 2 pi / alat
        self.volume = self.alat**3 * abs(np.linalg.det(self.at))

    @classmethod
    def from_model(cls, model):
        """ExactModel of a :class:`PAOFLOW.models.models.HoppingModel`."""
        return cls(
            list(model.hoppings), model.number_of_orbitals, model.lattice_vectors, model.alat,
            model.orbital_positions, model.real_space_grid, nelec=model.nelec,
        )  # fmt: skip

    def k_cart(self, k_cryst):
        return np.asarray(k_cryst, dtype=float) @ self.bg * 2.0 * np.pi / self.alat

    def atomic_gauge(self, k_cryst):
        """``(H^A, dH^A (3,n,n), d2H^A (3,3,n,n))`` at one k (eV, eV bohr, eV bohr^2)."""
        n = self.nawf
        kc = self.k_cart(k_cryst)
        H = np.zeros((n, n), dtype=complex)
        dH = np.zeros((3, n, n), dtype=complex)
        d2H = np.zeros((3, 3, n, n), dtype=complex)
        for i, j, R, h in self.hops:
            d = self.alat * ((np.asarray(R, dtype=float) + self.tau[j] - self.tau[i]) @ self.at)
            phase = np.exp(1j * (kc @ d)) * h
            H[i, j] += phase
            dH[:, i, j] += 1j * d * phase
            d2H[:, :, i, j] += -np.outer(d, d) * phase
        return H, dH, d2H

    def phases(self, k_cryst):
        """``D = diag(exp(i k.tau_j))`` relating the lattice and atomic gauges: ``H^A = D^+ H D``."""
        kc = self.k_cart(k_cryst)
        tau_cart = self.alat * (self.tau @ self.at)
        return np.diag(np.exp(1j * (tau_cart @ kc)))


def random_hopping_model(seed=0, nawf=4, time_reversal=True, cutoff=1.0, gap=True):
    """Simple-cubic cell with two sites, generic hoppings (real if ``time_reversal``)."""
    rng = np.random.default_rng(seed)
    tau = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])[:nawf]
    sym = rng.standard_normal((nawf, nawf))
    sym = 0.5 * (sym + sym.T)
    anti = rng.standard_normal((nawf, nawf))
    anti = 0.5 * (anti + anti.T)  # symmetric: h_ji(-R) = conj(h_ij(R)) with the odd d-factor
    odd = rng.standard_normal((nawf, nawf))
    odd = 0.5 * (odd - odd.T)  # antisymmetric real part odd in d: Hermitian, breaks inversion
    onsite = np.array([-3.0, -1.5, 1.5, 3.0])[:nawf] if gap else np.zeros(nawf)
    hops = []
    for cell in np.ndindex(5, 5, 5):
        R = np.array(cell) - 2
        for i in range(nawf):
            for j in range(nawf):
                d = R + tau[j] - tau[i]
                r2 = float(d @ d)
                if r2 > cutoff**2 + 1e-9:
                    continue
                envelope = np.exp(-1.5 * r2)
                h = sym[i, j] * envelope * (1.0 + 0.2 * d[0] * d[1])
                h = h + odd[i, j] * envelope * (d[1] + 0.3 * d[2] - 0.5 * d[0])
                if not time_reversal:
                    h = h + 1j * anti[i, j] * envelope * (d[0] + 0.5 * d[2])
                if not np.any(R) and i == j:
                    h = h + onsite[i]
                hops.append((i, j, R, h))
    at = np.eye(3)
    return ExactModel(hops, nawf, at, ALAT, tau, (4, 4, 4), nelec=nawf)


def honeycomb_model(delta=0.5, t=-1.0, t2=0.0, phi=0.0, nelec=2, alat=4.65, strain=0.0):
    """Gapped graphene or the Haldane model: :func:`PAOFLOW.models.models.honeycomb_model`."""
    return ExactModel.from_model(
        tb_models.honeycomb_model(delta, t, t2, phi, nelec=nelec, alat=alat, strain=strain)
    )


def rice_mele_chain(t=1.0, dt=0.3, delta=0.4, nelec=2, alat=2.0):
    """Rice-Mele chain along x: :func:`PAOFLOW.models.models.rice_mele_model`."""
    return ExactModel.from_model(tb_models.rice_mele_model(t, dt, delta, nelec=nelec, alat=alat))


def weyl_model(t=1.0, b=0.3):
    """Two-band lattice Weyl semimetal: :func:`PAOFLOW.models.models.weyl_model`."""
    return ExactModel.from_model(tb_models.weyl_model(t, b, alat=ALAT))


def single_band_model(hoppings, at, alat, nelec, ng):
    """One band plus a decoupled empty band: :func:`PAOFLOW.models.models.single_band_model`."""
    model = tb_models.single_band_model(hoppings, at, alat, nelec, ng)
    return ExactModel.from_model(model), list(model.hoppings)
