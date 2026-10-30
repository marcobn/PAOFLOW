"""Orbit-resolved electron-phonon mass enhancement of quantum-oscillation orbits.

The de Haas-van Alphen frequencies are those of the ground-state Fermi surface (the
electron-phonon self-energy has no real part at the Fermi level), but the cyclotron mass
that enters the Lifshitz-Kosevich temperature damping is the quasiparticle mass

.. math::

    m^*_{\\mathrm{LK}} = m_{\\mathrm{band}}\\,(1 + \\lambda_{\\mathrm{orbit}}),
    \\qquad
    \\lambda_{\\mathrm{orbit}} = \\frac{\\oint \\lambda_{n\\mathbf{k}}\\, w(\\mathbf{k})\\, dk}
                                      {\\oint w(\\mathbf{k})\\, dk},

with :math:`\\lambda_{n\\mathbf{k}}` the state-resolved coupling of the Fermi-surface
states (``fs_coupling.npz`` of the dense-q Eliashberg run) and the integral taken along the
extremal orbit found by pyskeaf.  The exact cyclotron weight is
:math:`w = 1/|\\mathbf{v}_\\perp|` (the time spent on each segment); this module uses the
arclength weight :math:`w = 1`, which is exact for orbits of uniform velocity and otherwise
differs by the velocity anisotropy along the orbit (reported as the spread of
:math:`\\lambda_{n\\mathbf{k}}` on the orbit).

Typical use, after ``main.py`` (Fermi-surface coupling) and ``PAOFLOW.pyskeaf``::

    from PAOFLOW.agentic.qo_analysis.mass_enhancement import enhanced_masses, write_mass_table
    table = enhanced_masses('output/dhva', 'output/c12_k60_q60/fs_coupling.npz', alat_bohr=9.5326,
                            sym_rots=nscf['s_cryst'])
    write_mass_table(table, 'output/dhva/masses.md')
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .skeaf_output import find_frequency_files, read_frequency_table

BOHR_TO_ANG = 0.529177209
_HEADER_RE = re.compile(
    r'Azimuthal\(deg\)\s*=\s*([-\d.Ee+]+)\s*,\s*Polar\(deg\)\s*=\s*([-\d.Ee+]+)\s*,\s*'
    r'Number of orbits\s*=\s*(\d+)'
)
_SLICE_RE = re.compile(
    r'Slice\s*=\s*(\d+)\s*,\s*Freq\(kT, average of all copies\)\s*=\s*([-\d.Ee+]+)'
)
_ORBIT_RE = re.compile(
    r'Orbit #\s*=\s*(\d+)\s*,\s*Freq\(kT, this largest copy used for outline\)\s*=\s*([-\d.Ee+]+)'
)
_POINTS_RE = re.compile(r'Points\s*=\s*(\d+)')


def read_orbit_outlines(path: str | Path) -> list[dict[str, Any]]:
    """Parse a pyskeaf ``orbitoutlines_invAng`` file.

    Parameters
    ----------
    path : str or Path
        ``qo_EF_<token>_orbitoutlines_invAng[_<band>].out`` written by
        :func:`PAOFLOW.pyskeaf.results.write_orbit_outlines`.

    Returns
    -------
    list of dict
        One entry per orbit: ``azimuthal_deg``, ``polar_deg``, ``slice``,
        ``frequency_kT`` (average of the copies), ``rep_frequency_kT`` (the copy
        drawn) and ``points`` ``(npts, 3)``, the orbit in Cartesian Å⁻¹ of the
        BXSF frame (2π included).
    """
    orbits: list[dict[str, Any]] = []
    lines = Path(path).read_text().splitlines()
    i = 0
    theta = phi = None
    while i < len(lines):
        line = lines[i]
        header = _HEADER_RE.search(line)
        if header:
            theta, phi = float(header.group(1)), float(header.group(2))
            i += 1
            continue
        block = _SLICE_RE.search(line)
        if block:
            orbit = _ORBIT_RE.search(lines[i + 1])
            npts = int(_POINTS_RE.search(lines[i + 2]).group(1))
            rows = lines[i + 4 : i + 4 + npts]  # skip the 'kx ky kz' line
            points = np.array([[float(x) for x in row.split()] for row in rows], dtype=float)
            orbits.append(
                {
                    'azimuthal_deg': theta,
                    'polar_deg': phi,
                    'slice': int(block.group(1)),
                    'frequency_kT': float(block.group(2)),
                    'rep_frequency_kT': float(orbit.group(2)) if orbit else float('nan'),
                    'points': points,
                }
            )
            i += 4 + npts
            continue
        i += 1
    return orbits


def lattice_point_group(at: NDArray[np.float64], tol: float = 1e-6) -> NDArray[np.int_]:
    """Integer rotations (crystal coordinates of k) that leave the reciprocal lattice invariant.

    Parameters
    ----------
    at : ndarray, shape ``(3, 3)``
        Direct lattice vectors (rows), any units.
    tol : float
        Tolerance on the metric.

    Returns
    -------
    ndarray, shape ``(nsym, 3, 3)``, int
        Matrices ``T`` with ``T G* T^T = G*`` for the reciprocal metric
        :math:`G^* = b\\,b^T`, acting on crystal k-coordinates ``k' = T k``.  This is
        the point group of the Bravais lattice (48 operations for cubic lattices),
        the correct k-space group for a one-atom cell and a superset of the
        crystal's group otherwise.

    Notes
    -----
    The candidates are all integer matrices with entries in ``-2 … 2``; the
    rotations of a primitive cell of any Bravais lattice have entries within that
    range.
    """
    at = np.asarray(at, dtype=float)
    bg = np.linalg.inv(at).T
    gstar = bg @ bg.T
    gstar = gstar / np.abs(gstar).max()
    values = np.arange(-2, 3)
    grids = np.meshgrid(*([values] * 9), indexing='ij')
    cand = np.stack([g.ravel() for g in grids], axis=1).reshape(-1, 3, 3)
    dets = np.rint(np.linalg.det(cand)).astype(int)
    cand = cand[np.abs(dets) == 1]
    metric = np.einsum('nij,jk,nlk->nil', cand, gstar, cand)
    keep = np.all(np.abs(metric - gstar[None]) < tol, axis=(1, 2))
    return cand[keep].astype(int)


def kspace_rotations_from_real(sym_rots: NDArray[np.int_]) -> NDArray[np.int_]:
    """k-space crystal rotations ``(S^-1)^T`` of real-space crystal rotations ``S`` (QE ``s_cryst``)."""
    rots = np.asarray(sym_rots, dtype=float)
    return np.rint(np.transpose(np.linalg.inv(rots), (0, 2, 1))).astype(int)


class FermiSurfaceLambda:
    """State-resolved :math:`\\lambda_{n\\mathbf{k}}` on the full dense k-grid.

    Parameters
    ----------
    fs_coupling : dict or path
        The Fermi-surface coupling (``fs_coupling.npz`` of
        :func:`PAOFLOW.elphon.fermi_surface_coupling.write_fs_coupling`, or the
        dictionary returned by :func:`read_fs_coupling`): irreducible states with
        ``k_cryst`` (crystal coordinates on the ``nk_dense`` grid), ``band``,
        ``energy_ev`` and either ``lambda_nk`` or the pair coupling ``coupling`` /
        ``freq_ev`` / ``weight`` from which it is derived
        (:func:`~PAOFLOW.elphon.anisotropic_eliashberg.coupling_strength`).
    sym_rots : ndarray ``(nsym, 3, 3)`` or None
        Real-space crystal rotations of the structure (``read_nscf(save)['s_cryst']``).
        ``None`` uses the point group of the Bravais lattice
        (:func:`lattice_point_group`), exact for one atom per cell.

    Notes
    -----
    The irreducible states are unfolded to the whole grid with the rotations and
    stored as ``lambda[k_flat, band]`` (NaN where no state lies within the Fermi
    window), so that a k-point on an orbit is looked up by its nearest grid point.
    """

    def __init__(self, fs_coupling: dict[str, Any] | str | Path, sym_rots=None) -> None:
        if not isinstance(fs_coupling, dict):
            from PAOFLOW.elphon.fermi_surface_coupling import read_fs_coupling

            fs_coupling = read_fs_coupling(str(fs_coupling))
        self.nk = int(fs_coupling['nk_dense'])
        self.at = np.asarray(fs_coupling['at'], dtype=float)
        self.fsthick_ev = float(fs_coupling.get('fsthick_ev', np.nan))
        band = np.asarray(fs_coupling['band'], dtype=int)
        if 'lambda_nk' not in fs_coupling:
            # the file of write_fs_coupling stores the pair coupling Lambda_ab(omega);
            # the state-resolved lambda_nk is its frequency integral summed over partners
            from PAOFLOW.elphon.anisotropic_eliashberg import coupling_strength

            fs_coupling = dict(fs_coupling, lambda_nk=coupling_strength(fs_coupling)['lambda_nk'])
        lam = np.asarray(fs_coupling['lambda_nk'], dtype=float)
        energy = np.asarray(fs_coupling['energy_ev'], dtype=float)
        k_int = np.rint(np.asarray(fs_coupling['k_cryst'], dtype=float) * self.nk).astype(int)
        rots = (
            lattice_point_group(self.at)
            if sym_rots is None
            else kspace_rotations_from_real(sym_rots)
        )
        self.nbands = int(band.max()) + 1
        self.lam = np.full((self.nk**3, self.nbands), np.nan)
        self.energy = np.full((self.nk**3, self.nbands), np.nan)
        for rot in rots:
            kr = np.einsum('ij,nj->ni', rot, k_int) % self.nk
            flat = (kr[:, 0] * self.nk + kr[:, 1]) * self.nk + kr[:, 2]
            self.lam[flat, band] = lam
            self.energy[flat, band] = energy
        self.nstates_full = int(np.isfinite(self.lam).sum())

    def lookup(self, k_cryst: NDArray[np.float64], band: int | None = None) -> NDArray[np.float64]:
        """:math:`\\lambda_{n\\mathbf{k}}` at the grid points nearest to ``k_cryst`` ``(n, 3)``.

        With ``band=None`` the state closest to the Fermi level at that grid point
        is used.  NaN where the grid point carries no Fermi-window state.
        """
        k_int = np.rint(np.asarray(k_cryst, dtype=float) * self.nk).astype(int) % self.nk
        flat = (k_int[:, 0] * self.nk + k_int[:, 1]) * self.nk + k_int[:, 2]
        if band is not None:
            if band >= self.nbands:
                return np.full(flat.size, np.nan)
            return self.lam[flat, band]
        energies = np.where(np.isfinite(self.energy[flat]), np.abs(self.energy[flat]), np.inf)
        best = np.argmin(energies, axis=1)
        out = self.lam[flat, best]
        out[~np.isfinite(energies[np.arange(flat.size), best])] = np.nan
        return out


def cartesian_to_crystal(
    points_invang: NDArray[np.float64], at: NDArray[np.float64], alat_bohr: float
):
    """Crystal coordinates of k-points given in Cartesian Å⁻¹ (2π included).

    ``at`` holds the direct lattice vectors in units of ``alat`` (rows);
    :math:`k_i = \\mathbf{k}\\cdot\\mathbf{a}_i / 2\\pi`.
    """
    a_ang = np.asarray(at, dtype=float) * float(alat_bohr) * BOHR_TO_ANG
    return np.asarray(points_invang, dtype=float) @ a_ang.T / (2.0 * np.pi)


def orbit_lambda(
    points_invang: NDArray[np.float64],
    fsl: FermiSurfaceLambda,
    alat_bohr: float,
    band: int | None = None,
) -> dict[str, float]:
    """Arclength-averaged :math:`\\lambda_{n\\mathbf{k}}` along one orbit.

    Parameters
    ----------
    points_invang : ndarray ``(npts, 3)``
        Orbit outline in Cartesian Å⁻¹ (pyskeaf ``outline_ang``), a closed contour.
    fsl : FermiSurfaceLambda
        The unfolded state-resolved coupling.
    alat_bohr : float
        Lattice parameter of ``fsl.at``.
    band : int or None
        PAO band index (0-based) of the orbit's sheet; ``None`` takes the state
        nearest to the Fermi level at each grid point.

    Returns
    -------
    dict
        ``lambda_orbit`` (arclength average over the matched points), ``lambda_min``,
        ``lambda_max``, ``matched_fraction`` (arclength fraction with a Fermi-window
        state on the grid), ``n_points``.
    """
    pts = np.asarray(points_invang, dtype=float)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    weight = 0.5 * (seg + np.roll(seg, 1))  # arclength attributed to each vertex
    lam = fsl.lookup(cartesian_to_crystal(pts, fsl.at, alat_bohr), band)
    ok = np.isfinite(lam)
    total = weight.sum()
    if not ok.any() or total <= 0.0:
        return {
            'lambda_orbit': np.nan,
            'lambda_min': np.nan,
            'lambda_max': np.nan,
            'matched_fraction': 0.0,
            'n_points': int(pts.shape[0]),
        }
    return {
        'lambda_orbit': float((weight[ok] * lam[ok]).sum() / weight[ok].sum()),
        'lambda_min': float(lam[ok].min()),
        'lambda_max': float(lam[ok].max()),
        'matched_fraction': float(weight[ok].sum() / total),
        'n_points': int(pts.shape[0]),
    }


def _outline_path(frequency_file: Path) -> Path:
    return frequency_file.with_name(
        frequency_file.name.replace('freqvsangle', 'orbitoutlines_invAng')
    )


def enhanced_masses(
    skeaf_dir: str | Path,
    fs_coupling: dict[str, Any] | str | Path,
    alat_bohr: float,
    sym_rots=None,
    fermi_token: str = '0',
    bands: list[int] | None = None,
    angle_tolerance_deg: float = 1e-3,
    frequency_tolerance: float = 0.02,
) -> pd.DataFrame:
    """Band masses and electron-phonon-enhanced masses of every pyskeaf orbit.

    Parameters
    ----------
    skeaf_dir : str or Path
        Folder with the ``qo_EF_<token>_freqvsangle_<band>.out`` and
        ``..._orbitoutlines_invAng_<band>.out`` files of a pyskeaf run
        (``Fermi_surf_band_<band>.bxsf`` numbering, 1-based).
    fs_coupling : dict or path
        Fermi-surface coupling of the dense-q Eliashberg run (PAO band indices, 0-based).
    alat_bohr : float
        Lattice parameter of the coupling's ``at`` (bohr).
    sym_rots : ndarray or None
        Real-space crystal rotations (``read_nscf(save)['s_cryst']``); ``None`` for the
        lattice point group.
    fermi_token, bands
        File selection, as :func:`~PAOFLOW.agentic.qo_analysis.skeaf_output.read_orbits`.
    angle_tolerance_deg, frequency_tolerance
        Matching of an outline to its row of the frequency table (same angle,
        relative frequency difference).

    Returns
    -------
    pandas.DataFrame
        One row per orbit: ``band`` (BXSF numbering), ``azimuthal_deg``, ``polar_deg``,
        ``freq_kT``, ``carrier``, ``mstar_band`` (|m*| of pyskeaf, in m_e),
        ``lambda_orbit``, ``lambda_min``, ``lambda_max``, ``matched_fraction``,
        ``mstar_enhanced`` = ``mstar_band`` (1 + ``lambda_orbit``), ``num_copies``.
    """
    fsl = FermiSurfaceLambda(fs_coupling, sym_rots)
    files = find_frequency_files(skeaf_dir, fermi_token)
    if bands:
        files = {b: p for b, p in files.items() if b in set(bands)}
    rows = []
    for band, freq_file in sorted(files.items()):
        table = read_frequency_table(freq_file)
        outline_file = _outline_path(Path(freq_file))
        if table.empty:
            continue
        if not outline_file.is_file():
            raise FileNotFoundError(
                '%s has no orbit outlines (%s): run pyskeaf with write_auxiliary_files=True '
                '(PAOFLOW.pyskeaf(verbose=True)) so the orbit coordinates are written.'
                % (Path(freq_file).name, outline_file.name)
            )
        for orbit in read_orbit_outlines(outline_file):
            same_angle = (
                np.abs(table.azimuthal_deg - orbit['azimuthal_deg']) < angle_tolerance_deg
            ) & (np.abs(table.polar_deg - orbit['polar_deg']) < angle_tolerance_deg)
            cand = table[same_angle]
            if cand.empty:
                continue
            rel = np.abs(cand.freq_kT - orbit['frequency_kT']) / np.maximum(
                np.abs(orbit['frequency_kT']), 1e-9
            )
            j = int(np.argmin(rel.to_numpy()))
            if rel.iloc[j] > frequency_tolerance:
                continue
            row = cand.iloc[j]
            stats = orbit_lambda(orbit['points'], fsl, alat_bohr, band=int(band) - 1)
            rows.append(
                {
                    'band': int(band),
                    'azimuthal_deg': float(row.azimuthal_deg),
                    'polar_deg': float(row.polar_deg),
                    'freq_kT': float(row.freq_kT),
                    'carrier': row.carrier,
                    'mstar_band': float(abs(row.mstar_me)),
                    'lambda_orbit': stats['lambda_orbit'],
                    'lambda_min': stats['lambda_min'],
                    'lambda_max': stats['lambda_max'],
                    'matched_fraction': stats['matched_fraction'],
                    'mstar_enhanced': float(abs(row.mstar_me)) * (1.0 + stats['lambda_orbit']),
                    'num_copies': int(row.num_orbit_copies),
                }
            )
    columns = ['band', 'azimuthal_deg', 'polar_deg', 'freq_kT', 'carrier', 'mstar_band', 'lambda_orbit',
               'lambda_min', 'lambda_max', 'matched_fraction', 'mstar_enhanced', 'num_copies']  # fmt: skip
    return pd.DataFrame(rows, columns=columns)


def write_mass_table(
    table: pd.DataFrame,
    path: str | Path,
    title: str = 'Orbit masses with electron-phonon enhancement',
) -> Path:
    """Write the table of :func:`enhanced_masses` as markdown (and a CSV next to it)."""
    path = Path(path)
    csv = path.with_suffix('.csv')
    table.to_csv(csv, index=False)
    lines = [
        f'# {title}',
        '',
        (
            'm*_band from pyskeaf (|m*|, m_e); λ_orbit = arclength average of λ_nk over the orbit; '
            'm*_LK = m*_band (1 + λ_orbit). matched = fraction of the orbit with a Fermi-window state '
            'on the coupling grid.'
        ),
        '',
        '| band | azimuthal (°) | polar (°) | F (kT) | carrier | m*_band | λ_orbit (min–max) | matched | m*_LK | copies |',
        '|---|---|---|---|---|---|---|---|---|---|',
    ]
    for _, r in table.iterrows():
        lines.append(
            f'| {r.band:d} | {r.azimuthal_deg:.2f} | {r.polar_deg:.2f} | {r.freq_kT:.4f} | {r.carrier} | '
            f'{r.mstar_band:.3f} | {r.lambda_orbit:.3f} ({r.lambda_min:.2f}–{r.lambda_max:.2f}) | '
            f'{r.matched_fraction:.2f} | {r.mstar_enhanced:.3f} | {r.num_copies:d} |'
        )
    path.write_text('\n'.join(lines) + '\n')
    return path
