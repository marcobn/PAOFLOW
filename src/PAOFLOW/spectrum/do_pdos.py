import re
from typing import Any

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def do_pdos(data_controller, emin, emax, ne, delta):
    """Compute the projected density of states with fixed Gaussian smearing.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``E_k`` (shape ``(nkpnts, nawf, nspin)``),
        ``v_k`` (shape ``(nkpnts, nawf, nawf, nspin)``).
        Required attributes: ``nawf``, ``nspin``, ``nkpnts``, ``shift``.
    emin : float
        Lower energy bound of the PDOS grid (eV).
    emax : float
        Upper energy bound; clipped to ``min(shift, emax)`` (eV).
    ne : int
        Number of energy grid points.
    delta : float
        Gaussian smearing width (eV).

    Returns
    -------
    None
        Writes the following files to the output directory via
        :meth:`DataController.write_file_row_col`:

        - ``{atom_indexed}_{n_orb}_pdos_{ispin}.dat`` (for example
                    ``Si1_3px_pdos_0.dat``) for each orbital and spin channel.
        - ``pdos_sum_{ispin}.dat`` — the total projected DOS.

    Notes
    -----
    The PDOS for orbital ``m`` at energy ``E`` is

    .. math::

        P_m(E) = \\frac{1}{N_k \\sqrt{\\pi}\\,\\delta}
            \\sum_{\\mathbf{k},n} |\\langle m | n, \\mathbf{k} \\rangle|^2
            \\exp\\!\\left(-\\left(\\frac{E - \\varepsilon_{n\\mathbf{k}}}{\\delta}\\right)^2\\right)

    MPI reduction (:func:`MPI.Reduce`) is used to accumulate the per-rank
    partial sums on rank 0.
    """
    arrays, attributes = data_controller.data_dicts()

    nawf = attributes['nawf']
    nspin = attributes['nspin']
    nktot = attributes['nkpnts']

    # PDOS calculation with gaussian smearing

    emax = np.amin(np.array([attributes['shift'], emax]))
    ene = np.linspace(emin, emax, ne)

    orbital_prefixes = _build_orbital_prefixes(arrays, nawf)

    for ispin in range(nspin):
        pdosaux = np.zeros((nawf, ne), dtype=float)
        v_kaux = np.real(np.abs(arrays['v_k'][:, :, :, ispin]) ** 2)

        E_k = arrays['E_k'][:, :, ispin]

        for n in range(ne):
            taux = np.exp(-(((ene[n] - E_k) / delta) ** 2)) / np.sqrt(np.pi)
            pdosaux[:, n] = np.einsum('kb,kmb->m', taux, v_kaux)

        pdos = np.zeros((nawf, ne), dtype=float) if rank == 0 else None

        comm.Reduce(pdosaux, pdos, op=MPI.SUM)
        pdosaux = None

        if rank == 0:
            assert pdos is not None
            pdos /= float(nktot) * np.sqrt(np.pi) * delta
            pdos_sum = np.zeros(ne, dtype=float)
            for m in range(nawf):
                pdos_sum += pdos[m]
                fpdos = '%s_pdos_%d.dat' % (orbital_prefixes[m], ispin)
                data_controller.write_file_row_col(fpdos, ene, pdos[m])
        else:
            pdos_sum = None
            for m in range(nawf):
                fpdos = '%s_pdos_%d.dat' % (orbital_prefixes[m], ispin)
                data_controller.write_file_row_col(fpdos, ene, None)

        fpdos = 'pdos_sum_%d.dat' % ispin
        data_controller.write_file_row_col(fpdos, ene, pdos_sum)


def do_pdos_adaptive(data_controller, emin, emax, ne):
    """Compute the projected density of states with adaptive smearing.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``E_k`` (shape ``(nkpnts, nawf, nspin)``),
        ``v_k`` (shape ``(nkpnts, nawf, nawf, nspin)``),
        ``deltakp`` (shape ``(nkpnts, nawf, nspin)`` or similar —
        per-k adaptive smearing widths).
        Required attributes: ``nawf``, ``nspin``, ``nkpnts``, ``shift``,
        ``smearing`` (``'gauss'`` or ``'m-p'``).
    emin : float
        Lower energy bound (eV).
    emax : float
        Upper energy bound; clipped to ``min(shift, emax)`` (eV).
    ne : int
        Number of energy grid points.

    Returns
    -------
    None
        Writes the following files to the output directory:

        - ``{atom_indexed}_{n_orb}_pdosdk_{ispin}.dat`` (for example
                    ``Si1_3px_pdosdk_0.dat``) for each orbital and spin channel.
        - ``pdosdk_sum_{ispin}.dat`` — the total adaptive PDOS.

    Notes
    -----
    At each energy point the per-k smearing width ``deltakp`` is used to
    evaluate either a Gaussian or a Methfessel-Paxton broadening function
    via :func:`smearing.gaussian` or :func:`smearing.metpax`.  This
    approach follows Yates *et al.*, Phys. Rev. B **75**, 195121 (2007).
    """
    from ..utils.smearing import gaussian, metpax

    arrays = data_controller.data_arrays
    attributes = data_controller.data_attributes

    # PDoS Calculation with Gaussian Smearing
    emax = np.amin(np.array([attributes['shift'], emax]))
    ene = np.linspace(emin, emax, ne)

    nawf = attributes['nawf']

    orbital_prefixes = _build_orbital_prefixes(arrays, nawf)

    for ispin in range(attributes['nspin']):
        E_k = np.real(arrays['E_k'][:, :, ispin])

        pdosaux = np.zeros((nawf, ne), dtype=float)

        v_kaux = np.real(np.abs(arrays['v_k'][:, :, :, ispin]) ** 2)

        taux = np.zeros((arrays['deltakp'].shape[0], nawf), dtype=float)

        for n in range(ne):
            if attributes['smearing'] == 'gauss':
                taux = gaussian(ene[n], E_k, arrays['deltakp'][:, :, ispin])
            elif attributes['smearing'] == 'm-p':
                taux = metpax(ene[n], E_k, arrays['deltakp'][:, :, ispin])
            pdosaux[:, n] = np.einsum('kb,kmb->m', taux, v_kaux)

        pdos = np.zeros((nawf, ne), dtype=float) if rank == 0 else None

        comm.Reduce(pdosaux, pdos, op=MPI.SUM)
        pdosaux = None

        if rank == 0:
            assert pdos is not None
            pdos /= float(attributes['nkpnts'])
            pdos_sum = np.zeros(ne, dtype=float)
            for m in range(nawf):
                pdos_sum += pdos[m]
                fpdos = '%s_pdosdk_%d.dat' % (orbital_prefixes[m], ispin)
                data_controller.write_file_row_col(fpdos, ene, pdos[m])
        else:
            pdos_sum = None
            for m in range(nawf):
                fpdos = '%s_pdosdk_%d.dat' % (orbital_prefixes[m], ispin)
                data_controller.write_file_row_col(fpdos, ene, None)

        fpdos = 'pdosdk_sum_%d.dat' % ispin
        data_controller.write_file_row_col(fpdos, ene, pdos_sum)


def _sanitize_token(token: Any) -> str:
    """Convert an arbitrary label into a filesystem-friendly token.

    Parameters
    ----------
    token : Any
        Input label (atom symbol, orbital string, etc.).

    Returns
    -------
    str
        Sanitized label containing only ``[A-Za-z0-9_-]``. Any run of
        disallowed characters is replaced by ``'_'``. Empty inputs fall back
        to ``'orb'``.
    """
    token = str(token).strip()
    token = re.sub(r'[^A-Za-z0-9_\-]+', '_', token)
    return token or 'orb'


def _orbital_component_from_lm(lval: int, mval: int) -> str:
    """Map angular quantum numbers ``(l, m)`` to cubic-harmonic labels.

    Parameters
    ----------
    lval : int
        Orbital angular momentum quantum number.
        Typical values are ``0`` (s), ``1`` (p), ``2`` (d), ``3`` (f).
    mval : int
        PAOFLOW basis component index in the internal ordering
        ``m = 1, ..., 2*l+1``.

    Returns
    -------
    str
        Component label in the same cubic-harmonic convention used by the
        projection code (for example ``'px'``, ``'dxy'``, ``'fz3'``).

    Notes
    -----
    The ordering matches the real-cubic-harmonic blocks used in
    :func:`PAOFLOW.projection.do_atwfc_proj.calc_ylmg`.
    For ``l=1``, the mapping is ``[pz, px, py]`` for ``m=1,2,3``.
    """
    lm_map = {
        0: ['s'],
        1: ['pz', 'px', 'py'],
        2: ['dz2', 'dzx', 'dyz', 'dx2y2', 'dxy'],
        3: ['fz3', 'fxz2', 'fyz2', 'fzx2y2', 'fxyz', 'fxx3yy2', 'fy3xx2y'],
    }

    names = lm_map.get(int(lval), None)
    mind = int(mval) - 1
    if names is None or mind < 0 or mind >= len(names):
        return 'l%dm%d' % (int(lval), int(mval))
    return names[mind]


def _principal_n_from_label(label: Any) -> str | None:
    """Extract principal quantum number from a shell label.

    Parameters
    ----------
    label : Any
        Shell label such as ``'3P'``, ``'4D'`` or ``'2S'``.

    Returns
    -------
    str or None
        Leading principal quantum number as a string (for example ``'3'``),
        or ``None`` when no leading integer is present.
    """
    match = re.match(r'^\s*(\d+)', str(label))
    return match.group(1) if match else None


def _build_orbital_prefixes(arrays: dict[str, Any], nawf: int) -> list[str]:
    """Build deterministic per-orbital filename prefixes.

    Parameters
    ----------
    arrays : dict[str, Any]
        Data-controller array dictionary. Uses ``'basis'`` when available,
        otherwise ``'atomic_basis'``. Each basis record is expected to expose
        ``atom``, ``tau``, ``l``, ``m`` and ``label``.
    nawf : int
        Number of atomic wavefunctions/orbitals.

    Returns
    -------
    list[str]
        Prefix list of length ``nawf``. Each entry follows
        ``{AtomSite}_{n}{component}``, for example ``Si1_3px``.

    Notes
    -----
    Site indices (``Si1``, ``Si2``, ...) are assigned by first-seen order of
    unique atomic positions ``tau`` for each species, ensuring reproducible
    labels in multi-atom cells. If basis metadata is unavailable, a safe
    fallback ``orb_<index>`` is returned.
    """
    basis = arrays.get('basis') or arrays.get('atomic_basis')
    if basis is None or len(basis) < nawf:
        return ['orb_%d' % m for m in range(nawf)]

    species_counts = {}
    tau_to_tag = {}
    prefixes = []
    for m in range(nawf):
        rec = basis[m]

        atom = _sanitize_token(rec.get('atom', 'atom'))
        tau = rec.get('tau', None)
        tau_key = None
        if tau is not None:
            tau_key = tuple(np.asarray(tau, dtype=float).tolist())

        if tau_key is not None:
            if tau_key not in tau_to_tag:
                species_counts[atom] = species_counts.get(atom, 0) + 1
                tau_to_tag[tau_key] = '%s%d' % (atom, species_counts[atom])
            atom_tag = tau_to_tag[tau_key]
        else:
            species_counts[atom] = species_counts.get(atom, 0) + 1
            atom_tag = '%s%d' % (atom, species_counts[atom])

        lval = rec.get('l', None)
        mval = rec.get('m', None)
        label = rec.get('label', '')

        if lval is None or mval is None:
            prefixes.append('%s_orb%d' % (atom_tag, m))
            continue

        component = _orbital_component_from_lm(lval, mval)
        nval = _principal_n_from_label(label)
        orb = (nval + component) if nval is not None else component

        prefixes.append('%s_%s' % (atom_tag, _sanitize_token(orb)))

    return prefixes
