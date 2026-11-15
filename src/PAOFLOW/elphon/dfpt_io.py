"""Readers for the ``ph.x`` outputs that feed the native coarse electron-phonon vertex.

The DFPT coupling source of :mod:`PAOFLOW.elphon.dfpt_vertex` needs, per irreducible q:
the induced self-consistent potential ``dvscf`` (one record per displacement pattern on
the dense FFT grid), the displacement patterns ``u`` (``patterns.N.xml``) and the
dynamical matrix (``dyn`` file).  Two directory layouts are recognised:

* the ``pp.py`` collection of EPW: ``<dir>/<prefix>.dvscf_qN``, ``<prefix>.dyn_qN`` and
  ``<prefix>.phsave/patterns.N.xml``;
* the raw ``ph.x`` run: ``<dir>/_ph0/<prefix>.dvscf1`` (first q),
  ``_ph0/<prefix>.q_N/<prefix>.dvscf1``, ``_ph0/<prefix>.phsave/patterns.N.xml`` and
  ``<dir>/<prefix>.dynN`` (``fildyn`` written in the run directory).

The crystal space group and the plane-wave setup come from the nscf ``data-file-schema.xml``.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..phonon.qe_force_constants import read_dynamical_matrix_file

HARTREE_TO_RY = 2.0


@dataclass(frozen=True)
class DFPTFiles:
    """Paths of the ``ph.x`` outputs of one q-grid run.

    Attributes
    ----------
    prefix : str
        ``prefix`` of the QE run.
    layout : {'pp', 'ph'}
        ``pp.py`` collection or raw ``ph.x`` run directory.
    dvscf : tuple of str
        ``dvscf`` file of every irreducible q (``ph.x`` order).
    patterns : tuple of str
        ``patterns.N.xml`` of every irreducible q.
    dyn : tuple of str
        Dynamical-matrix file of every irreducible q.
    dyn0 : str or None
        ``<prefix>.dyn0`` (q-grid header) when present.
    """

    prefix: str
    layout: str
    dvscf: tuple[str, ...]
    patterns: tuple[str, ...]
    dyn: tuple[str, ...]
    dyn0: str | None

    @property
    def nq_irr(self) -> int:
        """Number of irreducible q-points."""
        return len(self.dvscf)


def _infer_prefix(dfpt_dir: str) -> str:
    names = os.listdir(dfpt_dir)
    pp = sorted(n for n in names if re.fullmatch(r'.+\.dvscf_q\d+', n))
    if pp:
        return pp[0].rsplit('.dvscf_q', 1)[0]
    ph0 = os.path.join(dfpt_dir, '_ph0')
    if os.path.isdir(ph0):
        raw = sorted(n for n in os.listdir(ph0) if re.fullmatch(r'.+\.dvscf1', n))
        if raw:
            return raw[0][: -len('.dvscf1')]
    raise FileNotFoundError(
        'no <prefix>.dvscf_qN (pp.py) or _ph0/<prefix>.dvscf1 (ph.x) in %s' % dfpt_dir
    )


def locate_dfpt_files(dfpt_dir: str, prefix: str | None = None) -> DFPTFiles:
    """Find the ``dvscf``, pattern and dynamical-matrix files of a ``ph.x`` q-grid run.

    Parameters
    ----------
    dfpt_dir : str
        ``pp.py`` ``save/`` directory or the ``ph.x`` run directory (holding ``_ph0/``).
    prefix : str, optional
        QE prefix; inferred from the ``dvscf`` file names when omitted.

    Returns
    -------
    DFPTFiles
        The files of every irreducible q in ``ph.x`` order.

    Raises
    ------
    FileNotFoundError
        When neither layout is found or a q is incomplete.
    """
    if prefix is None:
        prefix = _infer_prefix(dfpt_dir)
    pp = sorted(
        (int(m.group(1)), n)
        for n in os.listdir(dfpt_dir)
        for m in [re.fullmatch(re.escape(prefix) + r'\.dvscf_q(\d+)', n)]
        if m
    )
    if pp:
        layout = 'pp'
        nq = len(pp)
        dvscf = tuple(
            os.path.join(dfpt_dir, '%s.dvscf_q%d' % (prefix, i)) for i in range(1, nq + 1)
        )
        patterns = tuple(
            os.path.join(dfpt_dir, '%s.phsave' % prefix, 'patterns.%d.xml' % i)
            for i in range(1, nq + 1)
        )
        dyn = tuple(os.path.join(dfpt_dir, '%s.dyn_q%d' % (prefix, i)) for i in range(1, nq + 1))
        dyn0 = None
    else:
        ph0 = os.path.join(dfpt_dir, '_ph0')
        if not os.path.isdir(ph0):
            raise FileNotFoundError('no dvscf files for prefix %r in %s' % (prefix, dfpt_dir))
        layout = 'ph'
        qdirs = sorted(
            int(m.group(1))
            for n in os.listdir(ph0)
            for m in [re.fullmatch(re.escape(prefix) + r'\.q_(\d+)', n)]
            if m
        )
        nq = max(qdirs) if qdirs else 1
        dvscf = [os.path.join(ph0, '%s.dvscf1' % prefix)]
        dvscf += [
            os.path.join(ph0, '%s.q_%d' % (prefix, i), '%s.dvscf1' % prefix)
            for i in range(2, nq + 1)
        ]
        dvscf = tuple(dvscf)
        patterns = tuple(
            os.path.join(ph0, '%s.phsave' % prefix, 'patterns.%d.xml' % i) for i in range(1, nq + 1)
        )
        dyn = tuple(os.path.join(dfpt_dir, '%s.dyn%d' % (prefix, i)) for i in range(1, nq + 1))
        dyn0 = os.path.join(dfpt_dir, '%s.dyn0' % prefix)
        if not os.path.exists(dyn0):
            dyn0 = None
    for group in (dvscf, patterns, dyn):
        missing = [p for p in group if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError('missing ph.x outputs: %s' % ', '.join(missing[:3]))
    return DFPTFiles(prefix, layout, dvscf, patterns, dyn, dyn0)


def read_patterns(path: str) -> dict:
    """Displacement patterns of one irreducible q (``patterns.N.xml`` of ``ph.x``).

    Parameters
    ----------
    path : str
        The ``patterns.N.xml`` file.

    Returns
    -------
    dict
        ``u`` ``(3 nat, 3 nat)`` complex, column ``nu`` = pattern ``nu`` in the global mode
        order (Cartesian index ``3 kappa + alpha`` along the rows); ``npert`` (perturbations
        per irreducible representation); ``nirr``; ``minus_q`` (``MINUS_Q_SYM``); ``nat``.

    Notes
    -----
    The ``dvscf`` record ``nu`` is the response to the pattern ``u[:, nu]``; the Cartesian
    responses follow from ``dV_c = sum_nu conj(u[c, nu]) dV_nu`` (``u`` unitary).
    """
    root = ET.parse(path).getroot()
    info = root.find('IRREPS_INFO')
    nirr = int(info.findtext('NUMBER_IRR_REP'))
    minus_q = (info.findtext('MINUS_Q_SYM') or 'false').strip().lower() == 'true'
    columns, npert = [], []
    for irr in range(1, nirr + 1):
        rep = info.find('REPRESENTION.%d' % irr)
        if rep is None:
            rep = info.find('REPRESENTATION.%d' % irr)
        n = int(rep.findtext('NUMBER_OF_PERTURBATIONS'))
        npert.append(n)
        for ip in range(1, n + 1):
            text = rep.find('PERTURBATION.%d' % ip).findtext('DISPLACEMENT_PATTERN')
            vals = np.array(text.split(), dtype=float).reshape(-1, 2)
            columns.append(vals[:, 0] + 1j * vals[:, 1])
    u = np.array(columns).T  # (3nat, nmodes)
    if u.shape[0] != u.shape[1]:
        raise ValueError('%s: %d patterns for %d Cartesian modes' % (path, u.shape[1], u.shape[0]))
    return {'u': u, 'npert': npert, 'nirr': nirr, 'minus_q': minus_q, 'nat': u.shape[0] // 3}


def read_dvscf(
    path: str,
    fft_grid: tuple[int, int, int],
    nmodes: int,
    nspin_mag: int = 1,
) -> NDArray[np.complex128]:
    """Induced self-consistent potential of ``ph.x`` (``fildvscf``), all patterns of one q.

    Parameters
    ----------
    path : str
        ``<prefix>.dvscf_qN`` (pp.py) or ``_ph0/.../<prefix>.dvscf1`` (ph.x).
    fft_grid : tuple of int
        Dense FFT grid ``(nr1, nr2, nr3)`` of the scf run.
    nmodes : int
        ``3 nat``.
    nspin_mag : int, optional
        Spin components of the potential (1 for non-magnetic runs).

    Returns
    -------
    ndarray, shape ``(nmodes, nspin_mag, nr1, nr2, nr3)``, complex
        Lattice-periodic part of the Hartree + exchange-correlation response (Ry) to the
        displacement pattern ``u[:, nu]`` with modulation ``exp(i q . R)``; the bare ionic
        term is not included.  Index ``[.., i, j, k]`` is the grid point ``i/nr1, j/nr2, k/nr3``.

    Raises
    ------
    ValueError
        When the file size is not ``nmodes`` records of ``nr1 nr2 nr3 nspin_mag`` complex numbers.
    """
    nr1, nr2, nr3 = (int(n) for n in fft_grid)
    nrec = nr1 * nr2 * nr3 * nspin_mag
    size = os.path.getsize(path)
    if size != nmodes * nrec * 16:
        raise ValueError(
            '%s has %d bytes, expected %d modes x %d grid points x %d spin x 16 bytes = %d '
            '(dense FFT grid %s; is nr1x padded or nspin_mag wrong?)'
            % (path, size, nmodes, nr1 * nr2 * nr3, nspin_mag, nmodes * nrec * 16, fft_grid)
        )
    raw = np.fromfile(path, dtype=np.complex128).reshape(nmodes, nspin_mag, nr3, nr2, nr1)
    return np.ascontiguousarray(raw.transpose(0, 1, 4, 3, 2))


def read_irreducible_q(files: DFPTFiles, bg: NDArray[np.float64]) -> NDArray[np.float64]:
    """Irreducible q-points of the ``ph.x`` run (crystal coordinates), from the dyn files.

    Parameters
    ----------
    files : DFPTFiles
        From :func:`locate_dfpt_files`.
    bg : ndarray, shape ``(3, 3)``
        Reciprocal lattice vectors (rows, ``2 pi / alat``).

    Returns
    -------
    ndarray, shape ``(nq_irr, 3)``
        The first q of every dyn file (the irreducible representative; the star follows it).
    """
    q_irr = []
    for path in files.dyn:
        q_cart = read_dynamical_matrix_file(path)['q_cart'][0]
        q_irr.append(np.linalg.solve(np.asarray(bg, dtype=float).T, q_cart))
    return np.array(q_irr)


def read_long_range_tensors(
    files: DFPTFiles,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Dielectric tensor and Born effective charges stored with the Gamma dynamical matrix.

    Parameters
    ----------
    files : DFPTFiles
        From :func:`locate_dfpt_files`.

    Returns
    -------
    epsilon : ndarray ``(3, 3)`` or None
    born_charges : ndarray ``(nat, 3, 3)`` or None
        ``None`` when ``ph.x`` ran without ``epsil``.
    """
    dyn = read_dynamical_matrix_file(files.dyn[0])
    return dyn['epsilon'], dyn['born_charges']


@dataclass(frozen=True)
class SpaceGroup:
    """Crystal space-group operations of the scf/nscf run, in QE's conventions.

    Attributes
    ----------
    rotations : ndarray, shape ``(nsym, 3, 3)``, int
        Rotation ``R`` acting on crystal coordinates of positions: the operation maps
        ``x -> R x - t`` (QE stores the fractional translation with this sign).
    translations : ndarray, shape ``(nsym, 3)``
        Fractional translations ``t`` (crystal coordinates).
    equivalent_atoms : ndarray, shape ``(nsym, nat)``, int
        ``equivalent_atoms[isym, kappa]`` is the atom onto which ``kappa`` is mapped
        (QE ``irt``, 0-based).
    time_reversal : ndarray, shape ``(nsym,)``, bool
        Operations combined with time reversal (magnetic runs only).
    names : tuple of str
        QE's symmetry labels.
    """

    rotations: NDArray[np.int_]
    translations: NDArray[np.float64]
    equivalent_atoms: NDArray[np.int_]
    time_reversal: NDArray[np.bool_]
    names: tuple[str, ...]

    @property
    def nsym(self) -> int:
        """Number of operations."""
        return int(self.rotations.shape[0])


def read_space_group(save_dir: str, check: bool = True) -> SpaceGroup:
    """Crystal symmetries from ``data-file-schema.xml`` (the ``crystal_symmetry`` entries).

    Parameters
    ----------
    save_dir : str
        QE ``outdir/prefix.save`` directory.
    check : bool, optional
        Verify every operation against the atomic positions (``R x_kappa - t`` equals the
        position of ``equivalent_atoms[isym, kappa]`` modulo lattice vectors).

    Returns
    -------
    SpaceGroup

    Raises
    ------
    ValueError
        If an operation does not map the atoms as declared (convention mismatch).
    """
    root = ET.parse(os.path.join(save_dir, 'data-file-schema.xml')).getroot()
    out = root.find('output')
    astruct = out.find('atomic_structure')
    alat = float(astruct.attrib['alat'])
    cell = astruct.find('cell')
    at = np.array([[float(x) for x in cell.findtext(v).split()] for v in ('a1', 'a2', 'a3')]) / alat
    tau = (
        np.array(
            [[float(x) for x in a.text.split()] for a in astruct.findall('atomic_positions/atom')]
        )
        / alat
    )
    x_cryst = tau @ np.linalg.inv(at)
    rots, fts, irts, trev, names = [], [], [], [], []
    for sym in out.findall('symmetries/symmetry'):
        info = sym.find('info')
        if info is None or (info.text or '').strip() != 'crystal_symmetry':
            continue
        names.append(info.attrib.get('name', ''))
        rots.append(np.array([float(v) for v in sym.findtext('rotation').split()]).reshape(3, 3))
        ft = sym.findtext('fractional_translation')
        fts.append([float(v) for v in ft.split()] if ft else [0.0, 0.0, 0.0])
        irts.append([int(v) - 1 for v in sym.findtext('equivalent_atoms').split()])
        trev.append(info.attrib.get('time_reversal', 'false').strip().lower() == 'true')
    if not rots:
        raise ValueError('%s lists no crystal symmetries' % save_dir)
    rotations = np.rint(np.array(rots)).astype(int)
    if np.abs(np.array(rots) - rotations).max() > 1e-6:
        raise ValueError('non-integer crystal rotations in %s' % save_dir)
    group = SpaceGroup(
        rotations,
        np.array(fts, dtype=float),
        np.array(irts, dtype=int),
        np.array(trev, dtype=bool),
        tuple(names),
    )
    if check:
        for isym in range(group.nsym):
            mapped = x_cryst @ group.rotations[isym].T - group.translations[isym]
            delta = mapped - x_cryst[group.equivalent_atoms[isym]]
            delta -= np.rint(delta)
            if np.abs(delta).max() > 1e-5:
                raise ValueError(
                    'symmetry %d (%s) does not map the atoms as declared (max |delta| = %.2e)'
                    % (isym, group.names[isym], np.abs(delta).max())
                )
    return group


def read_nscf_plane_wave_setup(save_dir: str) -> dict:
    """Plane-wave and spin setup of the nscf run (``data-file-schema.xml``).

    Parameters
    ----------
    save_dir : str
        QE ``outdir/prefix.save`` directory.

    Returns
    -------
    dict
        ``ecutwfc_ry``, ``ecutrho_ry``, ``fft_grid`` and ``fft_smooth`` (dense and smooth
        grids), ``nspin``, ``lsda``, ``noncolin``, ``spinorbit``, ``npol``, ``nbnd``,
        ``nelec``, ``alat`` (bohr), ``at`` and ``bg`` (rows, alat and 2 pi/alat),
        ``omega`` (bohr^3), ``tau_alat`` ``(nat, 3)``, ``atom_species`` (species name of
        every atom), ``pseudo_files`` (species -> UPF file name), ``species_order``.
    """
    root = ET.parse(os.path.join(save_dir, 'data-file-schema.xml')).getroot()
    out = root.find('output')
    basis = out.find('basis_set')
    astruct = out.find('atomic_structure')
    alat = float(astruct.attrib['alat'])
    cell = astruct.find('cell')
    at = np.array([[float(x) for x in cell.findtext(v).split()] for v in ('a1', 'a2', 'a3')]) / alat
    rl = basis.find('reciprocal_lattice')
    bg = np.array([[float(x) for x in rl.findtext(v).split()] for v in ('b1', 'b2', 'b3')])
    atoms = astruct.findall('atomic_positions/atom')
    tau_alat = np.array([[float(x) for x in a.text.split()] for a in atoms]) / alat
    species_nodes = out.findall('atomic_species/species')
    pseudo_files = {sp.attrib['name']: sp.findtext('pseudo_file').strip() for sp in species_nodes}
    fft = basis.find('fft_grid')
    fft_smooth = basis.find('fft_smooth')
    grid = tuple(int(fft.attrib['nr%d' % i]) for i in (1, 2, 3))
    smooth = (
        tuple(int(fft_smooth.attrib['nr%d' % i]) for i in (1, 2, 3))
        if fft_smooth is not None
        else grid
    )
    mag = out.find('magnetization')
    noncolin = (
        (mag.findtext('noncolin') or 'false').strip().lower() == 'true'
        if mag is not None
        else False
    )
    lsda = (mag.findtext('lsda') or 'false').strip().lower() == 'true' if mag is not None else False
    spinorbit = (
        (mag.findtext('spinorbit') or 'false').strip().lower() == 'true'
        if mag is not None
        else False
    )
    bs = out.find('band_structure')
    nbnd_text = bs.findtext('nbnd') or bs.findtext('nbnd_up')
    return {
        'ecutwfc_ry': float(basis.findtext('ecutwfc')) * HARTREE_TO_RY,
        'ecutrho_ry': float(basis.findtext('ecutrho')) * HARTREE_TO_RY,
        'fft_grid': grid,
        'fft_smooth': smooth,
        'nspin': 4 if noncolin else (2 if lsda else 1),
        'lsda': lsda,
        'noncolin': noncolin,
        'spinorbit': spinorbit,
        'npol': 2 if noncolin else 1,
        'nbnd': int(nbnd_text),
        'nelec': float(bs.findtext('nelec')),
        'alat': alat,
        'at': at,
        'bg': bg,
        'omega': abs(np.dot(at[0], np.cross(at[1], at[2]))) * alat**3,
        'tau_alat': tau_alat,
        'atom_species': [a.attrib['name'] for a in atoms],
        'pseudo_files': pseudo_files,
        'species_order': [sp.attrib['name'] for sp in species_nodes],
    }
