"""A tiny plane-wave crystal (one atom, Gaussian local potential, one s projector) for exact tests.

The Hamiltonian ``H = T + V_loc + V_NL`` is built from the same tables and projectors that
:mod:`PAOFLOW.elphon.pseudopotential_terms` uses for the perturbations, so identities such as
the translational sum rule hold to machine precision.
"""

from types import SimpleNamespace

import numpy as np

from PAOFLOW.elphon.plane_waves import PlaneWaveBlock, g_sphere
from PAOFLOW.elphon.pseudopotential_terms import KleinmanBylanderProjectors, LocalPotentialTable


def toy_upf(zp=0.0, depth=1.2, width=1.0, beta_width=0.8, d0=0.7, mesh=601):
    """A norm-conserving-like UPF object: Gaussian ``vloc`` (Hartree) and one ``l = 0`` projector."""
    r = 1.0e-3 * np.exp(0.012 * np.arange(mesh))
    rab = 0.012 * r
    vloc = (
        -depth * np.exp(-((r / width) ** 2)) / 2.0 - zp * np.where(r > 1.0e-6, 1.0 / r, 0.0) / 2.0
    )
    beta = r * np.exp(-((r / beta_width) ** 2))
    return SimpleNamespace(
        r=r,
        rab=rab,
        vloc=vloc,
        val=zp,
        beta=[
            {
                'l': 0,
                'j': None,
                'wfc': beta,
                'cutoff_index': mesh,
                'cutoff_radius': None,
                'label': 's',
            }
        ],
        dion=np.array([[d0 / 2.0]]),
        ptype='NC',
        has_augmentation=False,
        has_spinorbit=False,
    )


class ToyCrystal:
    """Simple-cubic cell with one atom, exact plane-wave eigenstates on QE-like spheres."""

    def __init__(
        self,
        alat=6.0,
        ecutwfc_ry=6.0,
        ecutrho_ry=30.0,
        tau_alat=(0.1, 0.2, 0.3),
        fft_grid=(12, 12, 12),
    ):
        self.alat = float(alat)
        self.tpiba = 2.0 * np.pi / self.alat
        self.omega = self.alat**3
        self.at = np.eye(3)
        self.bg = np.eye(3)
        self.tau_alat = np.array([tau_alat], dtype=float)
        self.species = ['X']
        self.upf = {'X': toy_upf()}
        self.ecutwfc = float(ecutwfc_ry)
        self.fft_grid = tuple(fft_grid)
        gcut = ecutrho_ry / self.tpiba**2
        self.g_miller, self.g_cart = g_sphere(self.fft_grid, self.bg, gcut)
        self.vloc = {'X': LocalPotentialTable(self.upf['X'], self.omega, np.sqrt(ecutrho_ry) + 2.0)}
        self.kb = KleinmanBylanderProjectors(
            self.upf, self.species, self.omega, np.sqrt(self.ecutwfc) + 2.0
        )

    def sphere(self, k_cart):
        """Miller indices of ``|k+G|^2 <= ecutwfc`` (any order)."""
        mill = self.g_miller.T
        kpg = (mill @ self.bg + k_cart) * self.tpiba
        keep = np.einsum('gi,gi->g', kpg, kpg) <= self.ecutwfc
        return np.ascontiguousarray(mill[keep].T)

    def hamiltonian(self, k_cart, miller):
        kpg = miller.T @ self.bg + np.asarray(k_cart, float)  # (npw, 3), 2pi/alat
        kin = np.einsum('gi,gi->g', kpg, kpg) * self.tpiba**2
        dg = miller.T[:, None, :] - miller.T[None, :, :]  # G - G'
        dg_cart = dg @ self.bg
        vq = self.vloc['X'](np.sqrt(np.einsum('abi,abi->ab', dg_cart, dg_cart)) * self.tpiba)
        vloc = vq * np.exp(-2j * np.pi * (dg_cart @ self.tau_alat[0]))
        beta = self.kb.at(kpg, self.tau_alat, self.tpiba)  # (nproj, npw)
        vnl = beta.T @ self.kb.D @ beta.conj()
        return np.diag(kin) + vloc + vnl, kpg

    def block(self, k_cart, nbnd=6):
        """Exact eigenstates at ``k`` as a :class:`PlaneWaveBlock` (and energies in Ry)."""
        miller = self.sphere(k_cart)
        h, kpg = self.hamiltonian(k_cart, miller)
        e, v = np.linalg.eigh(h)
        coef = np.ascontiguousarray(v[:, :nbnd].T)
        return PlaneWaveBlock(np.asarray(k_cart, float), miller, coef, 1), e[:nbnd], kpg
