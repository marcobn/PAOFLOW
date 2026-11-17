r"""Reader of VASP ``POTCAR`` PAW datasets, exposed with the attributes of :class:`~PAOFLOW.inputs.read_upf.UPF`.

The internal projections of a PAW run need the projectors :math:`\tilde p_i`, the integrated
augmentation charges :math:`q_{ij}` and, for the position dipoles and the valence orbitals, the
all-electron and pseudo partial waves :math:`\phi_i`, :math:`\tilde\phi_i` of every species; the
radial basis generator (:mod:`PAOFLOW.basis_gen`) also needs the local potential, the atomic
density and the non-local strengths. A ``POTCAR`` carries all of them; :class:`PotcarDataset`
converts them to bohr / Hartree and stores them under the names the UPF consumers read
(``r``, ``rab``, ``vloc``, ``atrho``, ``rho_atc``, ``beta``, ``dion``, ``qqq``, ``aewfc``,
``pswfc_partial``, ``pswfc``), so that
:class:`~PAOFLOW.elphon.pseudopotential_terms.KleinmanBylanderProjectors`,
:func:`~PAOFLOW.basis_gen.radial.pseudize_shell` and
:func:`~PAOFLOW.hamiltonian.position_dipoles.build_projector_augmentation` apply unchanged.

Layout of a dataset (keyword-delimited blocks in eV and angstrom; a concatenated ``POTCAR``
holds several datasets, each ending with ``End of Dataset``):

- the title line (``TITEL``, e.g. ``PAW_PBE Ga_d 06Jul2010``), the valence charge, the PSCTR
  parameters up to ``END of PSCTR-controll parameters`` with the ``Atomic configuration``
  (n, l, j, E, occupation) and the ``Description`` of the reference energies (l, E, TYP, RCUT);
- ``local part`` (PSGMAX and :math:`V_{loc}(q)`), the partial core and the atomic pseudo
  charge density in reciprocal space (``NPSPTS = 1000`` values on
  :math:`q_i = i\,\mathrm{PSGMAX}/1000`), then the line ``PSMAXN  T|F``;
- one ``Non local Part`` per l channel: ``l nproj RMAX``, the strengths :math:`D_{ij}` (eV) and,
  per projector, the ``Reciprocal Space Part``
  :math:`P_i(q) = 4\pi\int \tilde p_i(r) j_l(qr) r^2 dr` on :math:`q_i = i\,\mathrm{PSMAXN}/100`
  (what VASP uses with ``LREAL = .FALSE.``) and a real-space-optimised ``Real Space Part``;
- ``PAW radial sets``: the ``augmentation charges`` :math:`q_{ij}` of all projector pairs, the
  atomic ``uccopancies``, the logarithmic ``grid``, the potentials and densities on it (radial
  densities are stored as :math:`\sqrt{4\pi}\, r^2 n(r)`) and, per projector, the
  ``pseudo wavefunction`` and ``ae wavefunction`` :math:`u = r\phi`.

Conventions verified on the ``potpaw_PBE`` library (November 2026): :math:`q_{ij}` equals
:math:`\int (u_i u_j - \tilde u_i \tilde u_j)\, dr` of the stored partial waves; with
:math:`\langle \tilde p_{lm}|\mathbf k+\mathbf G\rangle = \Omega^{-1/2}\, i^l\,
Y^*_{lm}(\mathbf k+\mathbf G)\, P_i(|\mathbf k+\mathbf G|)\, e^{i(\mathbf k+\mathbf G)\cdot\tau}`
the WAVECAR bands are orthonormal in :math:`S = 1 + \sum q_{ij}|\tilde p_i\rangle\langle\tilde p_j|`
to :math:`3\times10^{-7}`; the real-space table of the file is only an approximation of the
projector (its transform differs from :math:`P_i(q)` by 3-8 %), so the real-space projectors
are rebuilt from :math:`P_i(q)` and re-biorthogonalised to the pseudo partial waves.
"""

from __future__ import annotations

import copy
import os
import re
import warnings
from functools import cache

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import simpson, solve_ivp
from scipy.interpolate import CubicSpline
from scipy.special import erf, spherical_jn

AUTOA = 0.529177249  # bohr in angstrom (vasp/src/constant.inc)
RYTOEV = 13.605826
HARTREE_EV = 2.0 * RYTOEV
NPSPTS = 1000  # points of the reciprocal-space tables of the local part and the densities
NPSNL = 100  # points of the projector tables
L_LETTERS = 'SPDF'
_PSMAXN_LINE = re.compile(r'^\s*([-+0-9.Ee]+)\s+([TF])\s*$')
_PSCTR_END = 'END of PSCTR-controll parameters'
_DATASET_END = 'End of Dataset'
_LICENSE_NOTICE_SHOWN = False
VASP_LICENSE_NOTICE = 'VASP input: ensure a valid VASP license is present'


def vasp_license_notice() -> None:
    """Print the VASP license notice once per process (MPI rank 0 only).

    Called whenever a VASP run is read (:meth:`DataController.read_vasp_output`) or a
    ``POTCAR`` is parsed (:func:`read_potcar`).
    """
    global _LICENSE_NOTICE_SHOWN
    if _LICENSE_NOTICE_SHOWN:
        return
    _LICENSE_NOTICE_SHOWN = True
    try:
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_rank() != 0:
            return
    except ImportError:
        pass
    print(VASP_LICENSE_NOTICE)


def split_potcar_blocks(text: str) -> list[list[str]]:
    """Split the text of a (possibly concatenated) ``POTCAR`` into the lines of its datasets.

    Parameters
    ----------
    text : str
        Content of the file.

    Returns
    -------
    list of list of str
        The lines of every dataset, from its title line to ``End of Dataset`` included.
    """
    blocks, current = [], []
    for line in text.splitlines():
        if not current and not line.strip():
            continue
        current.append(line)
        if line.strip() == _DATASET_END:
            blocks.append(current)
            current = []
    if any(line.strip() for line in current):
        blocks.append(current)
    return blocks


def potcar_title(lines: list[str]) -> str:
    """The ``TITEL`` of a dataset (its first line, whitespace normalised)."""
    return ' '.join(lines[0].split())


def _floats(tokens: list[str]) -> list[float]:
    return [float(tok) for tok in tokens]


def _scan_blocks(lines: list[str]) -> list[tuple[str, list[float]]]:
    """Keyword-delimited numeric blocks of the data part of a dataset, in file order.

    A line whose first character is a letter or a parenthesis opens a block named by the line;
    the following numeric lines are its values. The ``PSMAXN  T`` line becomes the block
    ``PSMAXN`` with the largest q and the real-space flag (``1.0`` for ``T``).
    """
    blocks: list[tuple[str, list[float]]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0].isalpha() or stripped[0] == '(':
            blocks.append((stripped, []))
            continue
        match = _PSMAXN_LINE.match(line)
        if match:
            blocks.append(
                ('PSMAXN', [float(match.group(1)), 1.0 if match.group(2) == 'T' else 0.0])
            )
            continue
        if not blocks:
            raise ValueError('POTCAR data before the first keyword: %r' % stripped)
        blocks[-1][1].extend(_floats(stripped.split()))
    return blocks


def _inverse_bessel_transform(
    q: NDArray[np.float64], f_q: NDArray[np.float64], l: int, r: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""``f(r) = (1 / 2 pi^2) int f(q) j_l(qr) q^2 dq`` of a table on a uniform q grid (Simpson)."""
    bessel = spherical_jn(l, np.multiply.outer(r, q))
    return simpson(bessel * (f_q * q * q)[None, :], x=q, axis=1) / (2.0 * np.pi**2)


def _local_potential_from_reciprocal(
    q: NDArray[np.float64],
    table: NDArray[np.float64],
    zval: float,
    r: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Recover the ionic local potential from a POTCAR reciprocal table.

    Parameters
    ----------
    q : ndarray, shape (nq,)
        Reciprocal radii in inverse bohr, including zero.
    table : ndarray, shape (nq,)
        The ``local part`` table in Hartree times bohr cubed.
    zval : float
        Ionic valence charge.
    r : ndarray, shape (nr,)
        Radial mesh in bohr.

    Returns
    -------
    ndarray, shape (nr,)
        Ionic local potential in Hartree.

    Notes
    -----
    VASP stores a Coulomb-compensated table. Following its ``POTTORHO``
    convention, remove a Gaussian compensation in reciprocal space and
    restore its analytic Coulomb potential after the inverse transform:

    .. math::

        V_{\mathrm{loc}}(r) = \mathcal F^{-1}\!\left[
        V(q)-\frac{4\pi Z_v}{q^2}(1-e^{-q^2/4})\right]
        - \frac{Z_v\operatorname{erf}(r)}{r}.

    The Gaussian exponent is one in atomic units. Cubic resampling of the
    finite reciprocal table reduces quadrature ringing on the logarithmic
    radial mesh; this remains a reconstruction from a finite table.
    """
    q_fine = np.linspace(0.0, q[-1], 8 * q.size + 1)
    compensation = np.empty_like(q_fine)
    compensation[0] = np.pi * zval
    compensation[1:] = -4.0 * np.pi * zval * np.expm1(-0.25 * q_fine[1:] ** 2) / q_fine[1:] ** 2
    short_range = CubicSpline(q, table)(q_fine) - compensation
    coulomb = np.empty_like(r)
    positive = r > 0.0
    coulomb[positive] = zval * erf(r[positive]) / r[positive]
    coulomb[~positive] = 2.0 * zval / np.sqrt(np.pi)
    return _inverse_bessel_transform(q_fine, short_range, 0, r) - coulomb


class PotcarDataset:
    r"""One PAW dataset of a ``POTCAR``, in atomic units, with the attributes of a UPF.

    Parameters
    ----------
    lines : list of str
        The lines of one dataset (from its title line to ``End of Dataset``).
    source : str, optional
        File the dataset was read from (for messages).
    r_extend : float or None, optional
        The stored radial grid ends at 2.6-6 angstrom depending on the dataset (4.9 bohr for
        Si), too short for the box solver of the basis generator and for the Bessel transforms
        of the orbitals, which run to 10 bohr: the mesh is continued geometrically to this
        radius (bohr) with the local potential as :math:`-Z_v/r`, the atomic density from its
        reciprocal-space table and the pseudo-orbitals by inward integration (see
        :attr:`pswfc`). ``None`` keeps the stored grid.
    nonlocal_strength : {'screened', 'file'}, optional
        Which :math:`D_{ij}` :attr:`dion` holds, see :attr:`dion_screened` and :attr:`dion_file`.

    Attributes
    ----------
    title, name, element : str
        ``TITEL`` (``'PAW_PBE Ga_d 06Jul2010'``), the dataset name (``'Ga_d'``) and the chemical
        symbol (``'Ga'``).
    val, zval : float
        Valence charge.
    ptype, type : str
        ``'PAW'``.
    has_augmentation, nlcc, has_spinorbit : bool
        ``True``, ``True``, ``False`` (VASP datasets are scalar relativistic; spin-orbit
        coupling is added at run time, so every ``beta[i]['j']`` is ``None``).
    qexc : str
        Functional label understood by :func:`PAOFLOW.basis_gen.xc.select_functional`
        (``LEXCH = PE`` -> ``'PBE'``, ``CA`` -> ``'LDA'``, else the raw code).
    r, rab : ndarray ``(mesh,)``
        Logarithmic mesh (bohr), stored part first, and ``dr/di``.
    npoints_stored : int
        Number of stored points of the mesh.
    vloc : ndarray ``(mesh,)``
        Local pseudopotential :math:`\tilde v_{loc}` (Hartree; ``local pseudopotential core``,
        :math:`-Z_v/r` beyond the stored grid).
    atrho : ndarray ``(mesh,)``
        :math:`4\pi r^2 \tilde n_v(r)` of the atomic pseudo valence density, compensation
        charges included (integrates to ``val``).
    rho_atc : ndarray ``(mesh,)``
        Pseudised core density :math:`\tilde n_c(r)` (bohr^-3).
    beta : list of dict
        Per projector: ``l``, ``j`` (``None``), ``wfc`` (:math:`r \tilde p'_i(r)` on :attr:`r`,
        the real-space projector rebuilt from the reciprocal table, truncated at ``RMAX`` and
        re-biorthogonalised to the pseudo partial waves of its channel, bohr^-1/2),
        ``cutoff_index``, ``cutoff_radius`` (``RMAX``, bohr), ``label``, ``energy`` (the
        reference energy, Hartree).
    projector_q_tables : list of tuple
        Per projector ``(q, P)``: the reciprocal-space table :math:`P_i(q) = 4\pi\int \tilde
        p_i j_l(qr) r^2 dr` of the file (bohr^-1, bohr^3/2), the projector VASP applies to the
        plane waves; :class:`~PAOFLOW.elphon.pseudopotential_terms.KleinmanBylanderProjectors`
        uses it instead of transforming ``wfc``.
    dion, dion_file, dion_screened : ndarray ``(nbeta, nbeta)``
        Non-local strengths (Hartree, block diagonal in l). ``dion_file`` is the ``D_ij`` of
        the file (unscreened); ``dion_screened`` the strengths of the screened reference atom,
        :math:`D_{jk} = \langle\tilde\phi_j|\tilde\chi_k\rangle + \varepsilon_k q_{jk}` with
        :math:`\tilde\chi_k = (\varepsilon_k - T - \tilde v_{eff})\tilde\phi_k` (Kresse and
        Joubert), for which the reference partial waves are exact eigenstates of the radial
        generalized problem :math:`H u = \varepsilon S u`; ``dion`` is the selected one.
    dion_screened_asymmetry : float
        Largest asymmetry of the screened strengths before symmetrisation (Hartree; a measure
        of the screening model).
    qqq : ndarray ``(nbeta, nbeta)``
        Integrated augmentation charges :math:`q_{ij}`, zero between different l.
    aewfc, pswfc_partial : list of ndarray
        All-electron and pseudo partial waves :math:`u = r\phi` on :attr:`r` (bohr^-1/2), in
        the order of :attr:`beta`.
    reference_energies : ndarray ``(nbeta,)``
        Reference energies (Hartree).
    pswfc : list of dict
        Pseudo-orbitals ``label`` (``'3D'``), ``occ``, ``wfc`` (:math:`r\tilde\phi` on
        :attr:`r`), ``projector``, ``energy``: the pseudo partial wave of every projector whose
        reference energy is a level of the ``Atomic configuration`` table and whose partial
        wave extends to the end of the stored grid (the bound states of the reference atom;
        reference energies chosen elsewhere, e.g. the -0.1 Ha of the p channel of the
        transition metals, or the empty d level of As and I, give scattering partial waves cut
        off outside the sphere, which are not orbitals). Beyond the stored grid the orbital is continued by inward
        integration of :math:`u'' = [l(l+1)/r^2 + 2(\tilde v_{eff} - \varepsilon)]u` in the
        screened potential of the pseudo-atom.
    shells : list of int
        ``l`` of every pseudo-orbital.
    jchia, lchia : list
        Empty (scalar relativistic).
    atomic_configuration : list of tuple
        ``(n, l, j, E_eV, occupation)`` rows of the header.
    description : list of tuple
        ``(l, E_eV, typ, rcut_bohr)`` rows of the header (projector and local channels).
    occupancies : ndarray ``(nbeta, nbeta)``
        The atomic ``uccopancies`` (density matrix of the reference atom in the projector
        basis).
    psmaxn : float
        Largest q of the projector tables (bohr^-1).
    lreal_flag : bool
        The flag of the ``PSMAXN`` line.
    duality : ndarray ``(nbeta, nbeta)``
        :math:`\int \tilde p_i \tilde\phi_j r^2 dr` of the raw inverse-transformed projectors
        (``1`` on the diagonal to 1-5 % before the re-biorthogonalisation).
    """

    def __init__(
        self,
        lines: list[str],
        source: str = '',
        r_extend: float | None = 12.0,
        nonlocal_strength: str = 'screened',
    ):
        if nonlocal_strength not in ('screened', 'file'):
            raise ValueError("nonlocal_strength must be 'screened' or 'file'")
        self.source = source
        self.nonlocal_strength = nonlocal_strength
        self.version = 2
        self.ptype = self.type = 'PAW'
        self.has_augmentation = True
        self.has_spinorbit = False
        self.nlcc = True
        self.jchia: list[float] = []
        self.lchia: list[float] = []
        self._parse(lines)
        self._finalise(r_extend)

    # ------------------------------------------------------------------ parsing
    def _parse(self, lines: list[str]) -> None:
        self.title = potcar_title(lines)
        tokens = self.title.split()
        self.name = tokens[1] if len(tokens) > 1 else tokens[0]
        match = re.match(r'^([A-Z][a-z]?)', self.name)
        if match is None:
            raise ValueError('%s: cannot read the element of %r' % (self.source, self.title))
        self.element = match.group(1)
        self.zval = self.val = float(lines[1].split()[0])
        try:
            end = next(i for i, line in enumerate(lines) if line.strip() == _PSCTR_END)
        except StopIteration:
            raise ValueError('%s: %r has no PSCTR block' % (self.source, self.title)) from None
        self._parse_psctr(lines[2:end])
        blocks = _scan_blocks(lines[end + 1 :])
        self._parse_data(blocks)

    def _parse_psctr(self, lines: list[str]) -> None:
        params: dict[str, str] = {}
        self.valence_configuration = ''
        self.atomic_configuration: list[tuple[int, int, float, float, float]] = []
        self.description: list[tuple[int, float, int, float]] = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            valence_match = re.match(r'VRHFIN\s*=\s*[^:]+:\s*(.*)', stripped)
            if valence_match:
                self.valence_configuration = valence_match.group(1).strip()
            if stripped.startswith('Atomic configuration'):
                n_entries = int(lines[i + 1].split()[0])
                for row in lines[i + 3 : i + 3 + n_entries]:
                    n, l, j, energy, occ = row.split()[:5]
                    self.atomic_configuration.append(
                        (int(n), int(l), float(j), float(energy), float(occ))
                    )
                i += 3 + n_entries
                continue
            if stripped.startswith('Description'):
                i += 2
                while i < len(lines):
                    fields = lines[i].split()
                    if len(fields) < 4 or not re.match(r'^\d+$', fields[0]):
                        break
                    self.description.append(
                        (int(fields[0]), float(fields[1]), int(fields[2]), float(fields[3]))
                    )
                    i += 1
                continue
            if stripped.startswith('Error from kinetic energy'):
                break
            for part in lines[i].split(';'):
                match = re.match(r'\s*([A-Z]+)\s*=\s*(\S+)', part)
                if match and match.group(1) not in params:
                    params[match.group(1)] = match.group(2)
            i += 1
        self.psctr = params
        if params.get('LPAW', 'T').upper().startswith('F'):
            raise NotImplementedError(
                '%s: %r is an ultrasoft dataset (LPAW = F); only PAW datasets are supported'
                % (self.source, self.title)
            )
        exchange = params.get('LEXCH', 'PE').upper()
        self.qexc = {'PE': 'PBE', 'CA': 'LDA', 'PZ': 'LDA'}.get(exchange, exchange)
        self.mass = float(params.get('POMASS', 0.0))
        self.rcore = float(params.get('RCORE', 0.0))
        self.icore = int(params['ICORE']) if 'ICORE' in params else None

    def _parse_data(self, blocks: list[tuple[str, list[float]]]) -> None:
        names = [name for name, _ in blocks]

        def block(name: str) -> list[float]:
            return blocks[names.index(name)][1]

        local = block('local part')
        self.psgmax = local[0] * AUTOA  # 1/angstrom -> 1/bohr
        self.local_potential_q = np.array(local[1 : 1 + NPSPTS])
        self.core_density_q = np.array(block('core charge-density (partial)')[:NPSPTS])
        self.atomic_density_q = np.array(block('atomic pseudo charge-density')[:NPSPTS])
        psmaxn, flag = block('PSMAXN')[:2]
        self.psmaxn = psmaxn * AUTOA
        self.lreal_flag = bool(flag)
        # non-local part: one block per l channel
        self._channels: list[tuple[int, int, float, NDArray[np.float64]]] = []
        reciprocal, real_space = [], []
        i = 0
        while i < len(blocks):
            name, values = blocks[i]
            if name == 'Non local Part':
                l, nproj, rmax = int(values[0]), int(values[1]), values[2]
                dion = np.array(values[3 : 3 + nproj * nproj]).reshape(nproj, nproj)
                self._channels.append((l, nproj, rmax / AUTOA, dion))
                i += 1
                for _ in range(nproj):
                    assert blocks[i][0] == 'Reciprocal Space Part', blocks[i][0]
                    reciprocal.append(np.array(blocks[i][1][:NPSNL]))
                    assert blocks[i + 1][0] == 'Real Space Part', blocks[i + 1][0]
                    real_space.append(np.array(blocks[i + 1][1][:NPSNL]))
                    i += 2
                continue
            i += 1
        self._reciprocal_tables = reciprocal
        self.real_space_tables = real_space
        if 'PAW radial sets' not in names:
            raise ValueError('%s: %r has no PAW radial sets' % (self.source, self.title))
        nbeta = sum(nproj for _, nproj, _, _ in self._channels)
        self.nproj = nbeta
        aug = np.array(block('augmentation charges (non sperical)')[: nbeta * nbeta])
        self.qqq_file = aug.reshape(nbeta, nbeta)
        occ = block('uccopancies in atom')
        self.occupancies = (
            np.array(occ[: nbeta * nbeta]).reshape(nbeta, nbeta)
            if len(occ) >= nbeta * nbeta
            else np.zeros((nbeta, nbeta))
        )
        self._grid_angstrom = np.array(block('grid'))
        npts = self._grid_angstrom.size
        self._potpsc_ev = (
            np.array(block('local pseudopotential core')[:npts])
            if 'local pseudopotential core' in names
            else None
        )
        self._potae_ev = np.array(block('aepotential')[:npts])
        pseudo_potential_name = (
            'pspotential valence only' if 'pspotential valence only' in names else 'pspotential'
        )
        self._potps_ev = np.array(block(pseudo_potential_name)[:npts])
        self._core_ae = np.array(block('core charge-density')[:npts])
        self._core_ps = np.array(block('core charge-density (pseudized)')[:npts])
        self._ps_waves = [
            np.array(values[:npts]) for name, values in blocks if name == 'pseudo wavefunction'
        ]
        self._ae_waves = [
            np.array(values[:npts]) for name, values in blocks if name == 'ae wavefunction'
        ]
        if len(self._ps_waves) != nbeta or len(self._ae_waves) != nbeta:
            raise ValueError(
                '%s: %r has %d projectors but %d / %d pseudo / ae partial waves'
                % (self.source, self.title, nbeta, len(self._ps_waves), len(self._ae_waves))
            )
        if aug.size != nbeta * nbeta:
            raise ValueError(
                '%s: %r: %d augmentation charges for %d projectors'
                % (self.source, self.title, aug.size, nbeta)
            )

    # --------------------------------------------------------------- finalising
    def _finalise(self, r_extend: float | None) -> None:
        r_stored = self._grid_angstrom / AUTOA
        self.npoints_stored = r_stored.size
        step = float(np.log(r_stored[1] / r_stored[0]))
        if r_extend is not None and r_extend > r_stored[-1]:
            n_extra = int(np.ceil(np.log(r_extend / r_stored[-1]) / step))
            tail = r_stored[-1] * np.exp(step * np.arange(1, n_extra + 1))
            self.r = np.concatenate([r_stored, tail])
        else:
            self.r = r_stored.copy()
        self.rab = self.r * step
        self.npoints = self.r.size
        n_stored, n_total = self.npoints_stored, self.npoints

        def padded(values: NDArray[np.float64]) -> NDArray[np.float64]:
            out = np.zeros(n_total)
            out[:n_stored] = values
            return out

        # local potential (Hartree), Coulomb tail beyond the stored grid
        if self._potpsc_ev is None:
            q_local = np.arange(NPSPTS) * self.psgmax / NPSPTS
            local_table = self.local_potential_q / (HARTREE_EV * AUTOA**3)
            self.vloc = padded(
                _local_potential_from_reciprocal(q_local, local_table, self.val, r_stored)
            )
            self.local_potential_source = 'reciprocal table'
        else:
            self.vloc = padded(self._potpsc_ev / HARTREE_EV)
            self.local_potential_source = 'radial table'
        self.vloc[n_stored:] = -self.val / self.r[n_stored:]
        tail_mismatch = abs(self.vloc[n_stored - 1] + self.val / self.r[n_stored - 1])
        if tail_mismatch > 1.0e-2:
            warnings.warn(
                '%s: the local potential differs from -Z_v/r by %.3f Ha at the end of the '
                'stored grid' % (self.title, tail_mismatch),
                RuntimeWarning,
                stacklevel=3,
            )
        # atomic pseudo valence density (compensation charges included) from its q table
        q_density = np.arange(NPSPTS) * self.psgmax / NPSPTS
        # the table is the electron count rho(q) (rho(0) = Z_v), dimensionless: no unit factor;
        # resampled on a finer grid so that the transform does not ring at large r
        q_fine = np.linspace(0.0, q_density[-1], 8 * NPSPTS + 1)
        density = _inverse_bessel_transform(
            q_fine, CubicSpline(q_density, self.atomic_density_q)(q_fine), 0, self.r
        )
        self.atrho = 4.0 * np.pi * self.r**2 * density
        charge = float(np.sum(self.atrho * self.rab))
        if abs(charge - self.val) > 1.0e-3 * max(1.0, self.val):
            warnings.warn(
                '%s: the atomic pseudo charge density integrates to %.4f instead of %.1f'
                % (self.title, charge, self.val),
                RuntimeWarning,
                stacklevel=3,
            )
        # densities on the radial grid are stored as sqrt(4 pi) r^2 n(r)
        core = self._core_ps / (np.sqrt(4.0 * np.pi) * self._grid_angstrom**2) * AUTOA**3
        self.rho_atc = padded(core)
        self.rho_core_ae = padded(
            self._core_ae / (np.sqrt(4.0 * np.pi) * self._grid_angstrom**2) * AUTOA**3
        )
        # partial waves (r phi, bohr^-1/2)
        self.aewfc = [padded(u * np.sqrt(AUTOA)) for u in self._ae_waves]
        self.pswfc_partial = [padded(u * np.sqrt(AUTOA)) for u in self._ps_waves]
        # projector bookkeeping
        lll, rmax_of, dion_file = [], [], np.zeros((self.nproj, self.nproj))
        offset = 0
        for l, nproj, rmax, dion in self._channels:
            lll.extend([l] * nproj)
            rmax_of.extend([rmax] * nproj)
            dion_file[offset : offset + nproj, offset : offset + nproj] = dion / HARTREE_EV
            offset += nproj
        self.lll = lll
        self.lmax = max(lll) if lll else 0
        self.dion_file = dion_file
        qqq = self.qqq_file.copy()
        same_l = np.equal.outer(lll, lll)
        qqq[~same_l] = 0.0
        self.qqq = qqq
        self._match_description()
        # reciprocal-space projector tables (bohr units)
        q_table = np.arange(NPSNL) * self.psmaxn / NPSNL
        self.projector_q_tables = [
            (q_table, table / AUTOA**1.5) for table in self._reciprocal_tables
        ]
        # real-space projectors: inverse transform of the tables, truncated at RMAX, then
        # re-biorthogonalised to the pseudo partial waves within every channel
        raw = []
        for i, (l, (q_table, table)) in enumerate(zip(lll, self.projector_q_tables)):
            inside = self.r <= rmax_of[i]
            spline = CubicSpline(q_table, table)
            q_fine = np.linspace(0.0, q_table[-1], 4001)
            projector = np.zeros(n_total)
            projector[inside] = _inverse_bessel_transform(q_fine, spline(q_fine), l, self.r[inside])
            raw.append(projector * self.r)
        self.duality = np.zeros((self.nproj, self.nproj))
        for i in range(self.nproj):
            for j in range(self.nproj):
                if lll[i] == lll[j]:
                    self.duality[i, j] = float(np.sum(raw[i] * self.pswfc_partial[j] * self.rab))
        self.beta = []
        for channel_l in sorted(set(lll)):
            idx = [i for i, l in enumerate(lll) if l == channel_l]
            overlap = self.duality[np.ix_(idx, idx)]
            mixing = np.linalg.inv(overlap)
            for row, i in enumerate(idx):
                wfc = sum(mixing[row, col] * raw[k] for col, k in enumerate(idx))
                cutoff_index = int(np.count_nonzero(self.r <= rmax_of[i]))
                self.beta.append(
                    {
                        'l': channel_l,
                        'j': None,
                        'wfc': wfc,
                        'cutoff_index': cutoff_index,
                        'cutoff_radius': float(rmax_of[i]),
                        'label': '%s l=%d E=%.3f eV' % (self.name, channel_l, self._energies_ev[i]),
                        'energy': float(self.reference_energies[i]),
                        'index': i,
                    }
                )
        self.beta.sort(key=lambda b: b['index'])
        for b in self.beta:
            del b['index']
        if not self.atomic_configuration:
            from .potcar_atomic import recover_legacy_atomic_data

            recover_legacy_atomic_data(self)
        # pseudo-orbitals: the bound reference states
        self._build_pseudo_orbitals(n_stored)
        self.dion_screened = self._screened_dion()
        self.dion = self.dion_screened if self.nonlocal_strength == 'screened' else self.dion_file

    def _match_description(self) -> None:
        """Reference energies of the projectors from the ``Description`` rows."""
        channel_ls = {l for l, _, _, _ in self._channels}
        block_ls = [l for l, nproj, _, _ in self._channels for _ in range(nproj)]
        rows = [row for row in self.description if row[2] == 23 and row[0] in channel_ls]
        if [row[0] for row in rows] != block_ls:
            rows = [row for row in self.description if row[0] in channel_ls]
            if [row[0] for row in rows] != block_ls:
                raise ValueError(
                    '%s: %r: the Description rows %s do not match the non-local blocks %s'
                    % (self.source, self.title, [row[0] for row in self.description], block_ls)
                )
        self._energies_ev = np.array([row[1] for row in rows])
        self.reference_energies = self._energies_ev / HARTREE_EV
        self.rcut_channels = np.array([row[3] for row in rows])

    def _build_pseudo_orbitals(self, n_stored: int) -> None:
        legacy_orbitals = getattr(self, '_legacy_pseudo_orbitals', None)
        if legacy_orbitals is not None:
            self.pswfc = legacy_orbitals
            self.shells = [self.lll[orbital['projector']] for orbital in self.pswfc]
            return
        self.pswfc: list[dict] = []
        self.shells: list[int] = []
        effective_potential = None
        for i, l in enumerate(self.lll):
            energy_ev = self._energies_ev[i]
            levels = [
                (n, occ)
                for n, l_cfg, _, e_cfg, occ in self.atomic_configuration
                if l_cfg == l and abs(e_cfg - energy_ev) < 1.0e-3
            ]
            if not levels or self.pswfc_partial[i][n_stored - 1] == 0.0:
                # no atomic level at this energy, or a scattering partial wave cut off outside
                # the sphere (the empty d level of As / I): not an orbital
                continue
            n, occ = levels[0]
            wfc = self.pswfc_partial[i].copy()
            if self.npoints > n_stored:
                if effective_potential is None:
                    effective_potential = self._effective_potential()
                wfc = self._continue_bound_state(
                    wfc, l, float(self.reference_energies[i]), effective_potential, n_stored
                )
            self.pswfc.append(
                {
                    'label': '%d%s' % (n, L_LETTERS[l]),
                    'occ': float(occ),
                    'wfc': wfc,
                    'projector': i,
                    'energy': float(self.reference_energies[i]),
                }
            )
            self.shells.append(l)

    def _effective_potential(self) -> NDArray[np.float64]:
        """Screened potential of the pseudo-atom, :math:`\\tilde v_{loc} + V_H + V_{xc}` (Hartree)."""
        if hasattr(self, '_atomic_effective_potential'):
            return self._atomic_effective_potential
        from ..basis_gen.atomic_potential import frozen_effective_potential

        return frozen_effective_potential(self)

    def _continue_bound_state(
        self,
        u: NDArray[np.float64],
        l: int,
        energy: float,
        v_eff: NDArray[np.float64],
        n_stored: int,
    ) -> NDArray[np.float64]:
        r"""Continue a bound radial function beyond the stored grid.

        Integrates :math:`u'' = [l(l+1)/r^2 + 2(\tilde v_{eff} - \varepsilon)] u` inward from the
        end of the extended mesh (WKB start :math:`e^{-\kappa r}`) to the end of the stored grid
        and scales the solution to the stored value there. A mismatch of the logarithmic
        derivative at the junction above 5 % (the screening model of the potential) is reported.
        """
        r = self.r
        kappa_sq = CubicSpline(r, l * (l + 1) / r**2 + 2.0 * (v_eff - energy))

        def rhs(x: float, y: NDArray[np.float64]) -> list[float]:
            return [y[1], float(kappa_sq(x)) * y[0]]

        r_end, r_join = float(r[-1]), float(r[n_stored - 1])
        kappa_end = np.sqrt(max(float(kappa_sq(r_end)), 1.0e-12))
        solution = solve_ivp(
            rhs,
            (r_end, r_join),
            [1.0, -kappa_end],
            t_eval=r[n_stored - 1 :][::-1],
            rtol=1.0e-10,
            atol=1.0e-14,
            max_step=0.05,
        )
        values = solution.y[0][::-1]  # on r[n_stored-1:], increasing r
        scale = u[n_stored - 1] / values[0] if values[0] != 0.0 else 0.0
        out = u.copy()
        out[n_stored:] = scale * values[1:]
        # junction check: logarithmic derivative of the stored function vs the continuation
        stored = CubicSpline(r[:n_stored], u[:n_stored])
        log_derivative_stored = (
            float(stored(r_join, 1) / u[n_stored - 1]) if u[n_stored - 1] else 0.0
        )
        log_derivative_tail = float(solution.y[1][-1] / solution.y[0][-1])
        mismatch = abs(log_derivative_stored - log_derivative_tail) / max(
            abs(log_derivative_stored), 1.0e-6
        )
        if mismatch > 0.05:
            warnings.warn(
                '%s: the l=%d pseudo-orbital at %.3f eV is continued beyond %.2f bohr with a '
                'logarithmic derivative that differs by %.0f %% from the stored one'
                % (self.title, l, energy * HARTREE_EV, r_join, 100.0 * mismatch),
                RuntimeWarning,
                stacklevel=4,
            )
        return out

    def _screened_dion(self) -> NDArray[np.float64]:
        r"""Strengths of the screened reference atom (Kresse-Joubert identity).

                For every projector pair of one channel,
                :math:`D_{jk} = arepsilon_k (\langle	ilde\phi_j|	ilde\phi_k
        angle_R + q_{jk})
                - \langle	ilde\phi_j|T + 	ilde v_{eff}|	ilde\phi_k
        angle_R`, the integrals running over
                the projector support :math:`r \le R`. The kinetic matrix element is integrated by parts,
                :math:`rac12\int_0^R u_j' u_k' dr - rac12 u_j(R) u_k'(R)`, with first derivatives of a
                spline fitted to the non-zero part of every partial wave only (the scattering partial
                waves are cut off abruptly just outside :math:`R`). The column of a bound reference state
                (a pseudo-orbital) is taken as it is and copied into its row, so that the orbital is an
                exact eigenstate of the radial generalized problem when its
                :math:`	ilde\chi_k = (arepsilon_k - T - 	ilde v_{eff})	ilde\phi_k` lies in the span of
                the projectors; the remaining block is symmetrised and its asymmetry recorded.
        """
        legacy_strengths = getattr(self, '_atomic_screened_dion', None)
        if legacy_strengths is not None:
            return legacy_strengths
        v_eff = self._effective_potential()
        r, rab = self.r, self.rab
        bound = {p['projector'] for p in self.pswfc}
        dion = np.zeros((self.nproj, self.nproj))
        asymmetry = 0.0
        for channel_l in sorted(set(self.lll)):
            idx = [i for i, l in enumerate(self.lll) if l == channel_l]
            n = max(self.beta[i]['cutoff_index'] for i in idx)
            r_in, w_in = r[:n], rab[:n]
            local = v_eff[:n] + 0.5 * channel_l * (channel_l + 1) / r_in**2
            values, slopes = [], []
            for i in idx:
                u = self.pswfc_partial[i]
                nonzero = np.nonzero(u)[0]
                last = int(nonzero[-1]) + 1 if nonzero.size else n
                spline = CubicSpline(r[:last], u[:last])
                values.append(u[:n])
                slopes.append(np.where(r_in <= r[last - 1], spline(r_in, 1), 0.0))
            block = np.zeros((len(idx), len(idx)))
            for a, j in enumerate(idx):
                for b, k in enumerate(idx):
                    overlap = float(np.sum(values[a] * values[b] * w_in))
                    kinetic = 0.5 * float(np.sum(slopes[a] * slopes[b] * w_in)) - 0.5 * float(
                        values[a][-1] * slopes[b][-1]
                    )
                    potential = float(np.sum(values[a] * local * values[b] * w_in))
                    block[a, b] = (
                        self.reference_energies[k] * (overlap + self.qqq[j, k])
                        - kinetic
                        - potential
                    )
            asymmetry = max(asymmetry, float(np.abs(block - block.T).max()))
            symmetric = 0.5 * (block + block.T)
            for b, k in enumerate(idx):
                if k in bound:
                    symmetric[:, b] = block[:, b]
                    symmetric[b, :] = block[:, b]
            if sum(k in bound for k in idx) > 1:
                symmetric = 0.5 * (symmetric + symmetric.T)
            dion[np.ix_(idx, idx)] = symmetric
        self.dion_screened_asymmetry = asymmetry
        return dion

    # ------------------------------------------------------------------ helpers
    def augmented_norm(
        self, u: NDArray[np.float64], l: int, r: NDArray[np.float64] | None = None
    ) -> float:
        r"""``<u|S|u>`` of a radial function of the ``l`` channel.

                Parameters
                ----------
                u : ndarray
                    :math:`r\chi(r)` on :attr:`r` (or on ``r``).
                l : int
                    Angular momentum of the function (only the projectors of that channel couple to it).
                r : ndarray, optional
                    Mesh of ``u`` when it is not :attr:`r` (interpolated).

                Returns
                -------
                float
                    :math:`\int u^2 dr + \sum_{ij} q_{ij}\langle	ilde p_i|u
        angle\langle	ilde p_j|u
        angle`
                    over the projectors of the channel.
        """
        if r is not None:
            u = np.nan_to_num(CubicSpline(r, u, extrapolate=False)(self.r), nan=0.0)
        plain = float(np.sum(u * u * self.rab))
        idx = [i for i, b in enumerate(self.beta) if b['l'] == l]
        becp = np.array([float(np.sum(self.beta[i]['wfc'] * u * self.rab)) for i in idx])
        return plain + float(becp @ self.qqq[np.ix_(idx, idx)] @ becp)

    def with_nonlocal_strength(self, kind: str) -> PotcarDataset:
        """A shallow copy with :attr:`dion` set to ``'file'`` or ``'screened'`` strengths."""
        if kind not in ('screened', 'file'):
            raise ValueError("kind must be 'screened' or 'file'")
        other = copy.copy(self)
        other.nonlocal_strength = kind
        other.dion = self.dion_screened if kind == 'screened' else self.dion_file
        return other

    def summary(self) -> str:
        """One line per projector channel (l, reference energy, pseudo-orbital label)."""
        labels = {p['projector']: p['label'] for p in self.pswfc}
        lines = ['%s (%s): %d projectors' % (self.title, self.source, self.nproj)]
        for i, l in enumerate(self.lll):
            lines.append(
                '  l=%d  E_ref=%9.4f eV  q_ii=%7.4f  %s'
                % (l, self._energies_ev[i], self.qqq[i, i], labels.get(i, 'scattering state'))
            )
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return 'PotcarDataset(%r, nproj=%d, pswfc=%s)' % (
            self.title,
            self.nproj,
            [p['label'] for p in self.pswfc],
        )


@cache
def _read_potcar_cached(path: str, r_extend: float | None, nonlocal_strength: str) -> tuple:
    vasp_license_notice()
    with open(path) as handle:
        text = handle.read()
    return tuple(
        PotcarDataset(block, source=path, r_extend=r_extend, nonlocal_strength=nonlocal_strength)
        for block in split_potcar_blocks(text)
    )


def read_potcar(
    path: str, *, r_extend: float | None = 12.0, nonlocal_strength: str = 'screened'
) -> list[PotcarDataset]:
    """All PAW datasets of a (possibly concatenated) ``POTCAR``.

    Parameters
    ----------
    path : str
        ``POTCAR`` file.
    r_extend, nonlocal_strength : optional
        See :class:`PotcarDataset`.

    Returns
    -------
    list of PotcarDataset
        In file order (parsed once per path and options; the objects are shared).
    """
    return list(_read_potcar_cached(os.path.abspath(path), r_extend, nonlocal_strength))


def find_potcar_datasets(
    titles: list[str],
    fpath: str | None,
    potcar_dir: str | None = None,
    *,
    r_extend: float | None = 12.0,
    nonlocal_strength: str = 'screened',
) -> list[PotcarDataset]:
    """The PAW datasets of a VASP run, matched to the ``TITEL`` of every atom type.

    Parameters
    ----------
    titles : list of str
        ``TITEL`` of every atom type, in the order of ``vasprun.xml``
        (``'PAW_PBE Ga_d 06Jul2010'``).
    fpath : str or None
        Run folder; its ``POTCAR`` (what VASP read, possibly concatenated) is searched first.
    potcar_dir : str, optional
        A POTCAR library (``<dir>/<name>/POTCAR``, ``<name>`` the second word of the title) or
        a folder with a concatenated ``POTCAR``; the environment variable ``PAOFLOW_POTCAR_DIR``
        is searched after it.
    r_extend, nonlocal_strength : optional
        See :class:`PotcarDataset`.

    Returns
    -------
    list of PotcarDataset
        One per title.

    Raises
    ------
    FileNotFoundError
        When a title is found nowhere; the message lists the places searched.

    Notes
    -----
    A dataset with the same name but another date (``PAW_PBE Si 05Jan2001`` for
    ``PAW_PBE Si 06Sep2000``) is accepted with a warning: the ``<psi|S|psi>`` check of the
    projections tells whether it is the one the run used.
    """
    options = {'r_extend': r_extend, 'nonlocal_strength': nonlocal_strength}
    directories = [d for d in (potcar_dir, os.environ.get('PAOFLOW_POTCAR_DIR')) if d]
    out: list[PotcarDataset] = []
    for title in titles:
        wanted = ' '.join(title.split())
        name = wanted.split()[1] if len(wanted.split()) > 1 else wanted
        candidates = []
        if fpath:
            candidates.append(os.path.join(fpath, 'POTCAR'))
        for directory in directories:
            candidates.append(os.path.join(directory, name, 'POTCAR'))
            candidates.append(os.path.join(directory, 'POTCAR'))
        found = None
        same_name = None
        for candidate in candidates:
            if not os.path.isfile(candidate):
                continue
            for dataset in read_potcar(candidate, **options):
                if dataset.title == wanted:
                    found = dataset
                    break
                if same_name is None and dataset.name == name:
                    same_name = dataset
            if found is not None:
                break
        if found is None and same_name is not None:
            warnings.warn(
                'no POTCAR dataset titled %r; using %r from %s'
                % (wanted, same_name.title, same_name.source),
                RuntimeWarning,
                stacklevel=2,
            )
            found = same_name
        if found is None:
            raise FileNotFoundError(
                'no POTCAR dataset %r in %s (set potcar_dir= or PAOFLOW_POTCAR_DIR to the PAW '
                'library, or copy the POTCAR of the run into the save directory)'
                % (wanted, ', '.join(candidates) or 'no location')
            )
        out.append(found)
    return out
