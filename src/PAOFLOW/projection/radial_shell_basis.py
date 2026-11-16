"""Projection basis from radial shells generated at run time (no table files).

The generator route of ``PAOFLOW.projections()``: the shells of ``arry['configuration']`` are
built from the pseudopotential of every species (UPF for QE, POTCAR dataset for VASP) by
:func:`PAOFLOW.basis_gen.driver.generate_basis_shells`: the shells the pseudopotential carries as
pseudo-orbitals are taken as stored, the others are eigenstates of the pseudo-atom in a box
selected by the ghost filter of :mod:`PAOFLOW.basis_gen.radial`. The radial functions are
generated on rank 0, broadcast, and turned into the per-orbital records of the projection
(the schema of :func:`PAOFLOW.projection.do_atwfc_proj.build_pswfc_basis_all`), atom by atom
in the order of the configuration, every shell of a species on one uniform mesh (which the
position-dipole tables require). With spin-orbit coupling a scalar shell is duplicated into
the two ``j`` blocks exactly as :func:`~PAOFLOW.projection.do_atwfc_proj.build_aewfc_basis` does
with scalar tables, and j-resolved shells of fully relativistic pseudopotentials fill the two
blocks with their own functions.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

from ..basis_gen.driver import GeneratedShell, _write_two_col, generate_basis_shells
from ..inputs.pseudopotentials import load_pseudopotentials
from .do_atwfc_proj import _parallel_radialfft, _resolve_ae_shell_files, assign_jm

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

_SHELL_CACHE: dict = {}


@dataclass
class RadialShell:
    """One radial function of the basis of a species.

    Attributes
    ----------
    label : str
        Shell label (``'4P'``).
    l : int
        Angular momentum.
    r, wfc : ndarray
        Mesh (bohr) and ``u(r) = r R(r)``.
    source : str
        Where the function comes from (for the verbose report).
    energy : float
        Solver eigenvalue (Hartree), ``nan`` for a stored pseudo-orbital.
    j : float or None
        Total angular momentum of a j-resolved function, ``None`` for a scalar one.
    """

    label: str
    l: int
    r: NDArray[np.float64]
    wfc: NDArray[np.float64]
    source: str
    energy: float = float('nan')
    j: float | None = None


def shells_from_generator(
    dataset,
    labels: list[str],
    *,
    prefer_pswfc: bool = True,
    r_box: float | None = None,
    n_points: int = 2000,
    spin_orbit: bool = False,
) -> tuple[list[RadialShell], list[tuple[str, str]]]:
    """The radial shells of one species from the generator.

    Parameters
    ----------
    dataset : UPF or PotcarDataset
        The pseudopotential of the species.
    labels : list of str
        Requested shell labels, in order.
    prefer_pswfc, r_box, n_points : optional
        As :func:`PAOFLOW.basis_gen.driver.generate_basis_shells`.
    spin_orbit : bool, optional
        The run has spin-orbit coupling: the j-resolved functions of a fully relativistic
        pseudopotential are kept (two shells per ``l > 0`` label); otherwise its j-averaged
        scalar function is used.

    Returns
    -------
    shells : list of RadialShell
    skipped : list of tuple
        ``(label, reason)`` of the shells the generator could not provide.
    """
    generated, skipped = generate_basis_shells(
        dataset, labels, r_box=r_box, n_points=n_points, prefer_pswfc=prefer_pswfc
    )
    relativistic = bool(getattr(dataset, 'has_spinorbit', False))
    shells: list[RadialShell] = []
    for shell in generated:
        if relativistic and shell.l > 0:
            if spin_orbit and shell.j is None:
                continue  # the j-average: only for scalar runs
            if not spin_orbit and shell.j is not None:
                continue
        source = (
            'pseudo-orbital of %s' % getattr(dataset, 'title', 'the pseudopotential')
            if shell.from_pseudo_orbital
            else 'box solver, eps = %+.4f Ha' % shell.energy
        )
        shells.append(
            RadialShell(shell.label, shell.l, shell.r, shell.wfc, source, shell.energy, shell.j)
        )
    return shells, skipped


def shells_from_pseudo_orbitals(dataset, labels: list[str]) -> list[RadialShell]:
    """The pseudo-orbitals of a pseudopotential (UPF ``PP_PSWFC`` or POTCAR bound states).

    Parameters
    ----------
    dataset : UPF or PotcarDataset
    labels : list of str
        Shell labels; every label must be a pseudo-orbital of the dataset.

    Returns
    -------
    list of RadialShell
        On the mesh of the pseudopotential, in the order of ``labels``.

    Raises
    ------
    RuntimeError
        A label without pseudo-orbital.
    """
    title = getattr(dataset, 'title', 'the pseudopotential')
    shells = []
    for label in labels:
        match = [p for p in dataset.pswfc if p['label'].upper() == label.upper()]
        if not match:
            raise RuntimeError('no pseudo-orbital %s in %s' % (label, title))
        l = 'SPDF'.index(label[1].upper())
        shells.append(
            RadialShell(
                label.upper(),
                l,
                np.asarray(dataset.r, dtype=float),
                np.asarray(match[0]['wfc'], dtype=float),
                'pseudo-orbital of %s' % title,
            )
        )
    return shells


def shells_from_tables(
    basispath: str, elem: str, labels: list[str], spin_orbit: bool = False
) -> list[RadialShell]:
    """Radial shells read from ``basispath/<elem>/<label>.dat`` tables.

    Parameters
    ----------
    basispath : str
        Table directory (the shipped ``BASIS/`` or a ``BASIS_PS`` of the generator).
    elem : str
        Element.
    labels : list of str
        Shell labels, in order.
    spin_orbit : bool, optional
        Use the j-resolved companions ``<label>_j<2j>.dat`` when both exist (two shells with
        ``j`` set), as :func:`~PAOFLOW.projection.do_atwfc_proj.build_aewfc_basis` does.

    Returns
    -------
    list of RadialShell
    """
    shells = []
    for label in labels:
        l = 'SPDF'.index(label[1].upper())
        for j, path in _resolve_ae_shell_files(os.path.join(basispath, elem), label, l, spin_orbit):
            data = np.loadtxt(path)
            shells.append(
                RadialShell(label.upper(), l, data[:, 0], data[:, 1], 'table %s' % path, j=j)
            )
    return shells


def _resample(shell: RadialShell, mesh: NDArray[np.float64]) -> RadialShell:
    """The shell on another mesh (cubic spline, zero outside its own mesh)."""
    from scipy.interpolate import CubicSpline

    if shell.r.shape == mesh.shape and np.allclose(shell.r, mesh, rtol=0.0, atol=1.0e-10):
        return shell
    spline = CubicSpline(shell.r, shell.wfc, extrapolate=False)
    wfc = np.nan_to_num(spline(mesh), nan=0.0)
    return RadialShell(shell.label, shell.l, mesh, wfc, shell.source, shell.energy, shell.j)


def build_basis_from_radial_shells(
    data_controller, shells_by_element: dict[str, list[RadialShell]]
) -> tuple[list[dict], dict]:
    """Basis records of all atoms from the radial shells of every element.

    Parameters
    ----------
    data_controller : DataController
        Required arrays: ``atoms``, ``tau``; attributes ``ecutrho``, ``omega``, ``verbose``,
        ``dftSO``.
    shells_by_element : dict
        ``{element: [RadialShell, ...]}`` in the order of the configuration. A label present
        twice with ``j = l - 1/2`` and ``j = l + 1/2`` is a j-resolved pair.

    Returns
    -------
    basis : list of dict
        Records ``atom, tau, l, m, label, r, wfc, qmesh, wfc_g`` (and ``jm`` with spin-orbit
        coupling), atom by atom.
    shells : dict
        ``{species: [l, ...]}`` (``l`` listed twice per ``l > 0`` shell with spin-orbit
        coupling, as :func:`~PAOFLOW.projection.do_atwfc_proj.build_aewfc_basis`).

    Notes
    -----
    Every shell of an element is resampled on one mesh (that of its first table shell, else
    of its first shell), so that the orbital records of a species share one mesh. Sets
    ``arry['jchia']``.
    """
    arry, attr = data_controller.data_dicts()
    verbose = attr['verbose']
    so = bool(attr.get('dftSO', False))
    qmesh = np.arange(0, np.sqrt(attr['ecutrho']) + 4, 0.01)
    volume = attr['omega']

    # group the shells of every element by label, on one mesh per element
    grouped: dict[str, list[tuple[str, int, list[RadialShell]]]] = {}
    for elem, shells in shells_by_element.items():
        if not shells:
            grouped[elem] = []
            continue
        reference = next((s for s in shells if s.source.startswith('table')), shells[0])
        mesh = np.asarray(reference.r, dtype=float)
        groups: list[tuple[str, int, list[RadialShell]]] = []
        for shell in shells:
            shell = _resample(shell, mesh)
            if groups and groups[-1][0] == shell.label and groups[-1][1] == shell.l:
                groups[-1][2].append(shell)
            else:
                groups.append((shell.label, shell.l, [shell]))
        grouped[elem] = groups

    per_atom, tasks = [], []
    for na in range(len(arry['atoms'])):
        atom = arry['atoms'][na]
        elem = re.split(r'\d+', atom)[0]
        tau = arry['tau'][na]
        records = []
        for label, l, members in grouped.get(elem, []):
            entries = []
            for shell in members:
                entries.append({'j': shell.j, 'r': shell.r, 'wfc': shell.wfc, 'task': len(tasks)})
                tasks.append((shell.r, shell.wfc, l))
                if verbose and rank == 0:
                    print('atom: %-2s  %s%s from %s' % (
                        atom, label, '' if shell.j is None else ' (j=%.1f)' % shell.j,
                        shell.source))  # fmt: skip
            records.append((label, l, entries))
        per_atom.append((atom, tau, records))
    wfc_g_list = _parallel_radialfft(tasks, qmesh, volume)

    basis, shells = [], {}
    arry['jchia'] = {}
    for atom, tau, records in per_atom:
        ash, jchia = [], []
        for label, l, entries in records:
            ash.append(l)
            if so and l > 0:
                ash.append(l)
                jchia.extend([l - 0.5, l + 0.5])
                resolved = [e for e in entries if e['j'] is not None]
                if len(resolved) == 2:
                    blocks = sorted(resolved, key=lambda e: e['j'])
                else:
                    blocks = [entries[0], entries[0]]
            elif so:
                jchia.append(0.5)
                blocks = [entries[0], entries[0]]
            else:
                blocks = [next((e for e in entries if e['j'] is None), entries[0])]
            for entry in blocks:
                wfc_g = wfc_g_list[entry['task']]
                for m in range(1, 2 * l + 2):
                    basis.append(
                        {
                            'atom': atom,
                            'tau': tau,
                            'l': l,
                            'm': m,
                            'label': label,
                            'r': entry['r'],
                            'wfc': np.array(entry['wfc'], dtype=float),
                            'qmesh': qmesh,
                            'wfc_g': wfc_g,
                        }
                    )
        if atom not in shells:
            shells[atom] = ash
            arry['jchia'][atom] = jchia
    if so:
        assign_jm(basis)
    return basis, shells


def build_generated_basis(
    data_controller,
    *,
    prefer_pswfc: bool = True,
    r_box: float | None = None,
    n_points: int = 2000,
    save_basis: bool = False,
) -> tuple[list[dict], dict]:
    """Basis of the run from the generator (the shells of ``arry['configuration']``).

    Parameters
    ----------
    data_controller : DataController
        Required arrays: ``configuration``, ``species``, ``atoms``, ``tau``; attributes
        ``dft``, ``fpath``, ``potcar_dir`` (VASP), ``ecutrho``, ``omega``, ``verbose``,
        ``dftSO``, ``opath``.
    prefer_pswfc : bool, optional
        Stored pseudo-orbitals for the shells that have one (default); ``False`` solves
        every shell.
    r_box : float, optional
        Confining box radius of the solver (bohr); default ``min(r[-1], 10)``.
    n_points : int, optional
        Solver mesh size.
    save_basis : bool, optional
        Write the generated functions to ``opath/BASIS_PS/<elem>/<shell>.dat`` (the table
        format of ``paoflow-genbasis-ps``) for inspection and reuse.

    Returns
    -------
    basis, shells
        As :func:`build_basis_from_radial_shells`.

    Notes
    -----
    The generation runs on rank 0 (one dense generalized eigenproblem per channel) and the
    functions are broadcast; results are cached per run folder, element and options.
    ``arry['configuration'][elem]`` is reduced to the shells actually generated (skipped
    shells are listed in a warning on rank 0).
    """
    arry, attr = data_controller.data_dicts()
    so = bool(attr.get('dftSO', False))
    configuration = arry['configuration']
    if rank == 0:
        datasets = load_pseudopotentials(data_controller)
        by_element = {}
        for name, data in datasets.items():
            by_element.setdefault(re.split(r'\d+', name)[0], data)
        shells_by_element: dict[str, list[RadialShell]] = {}
        skipped_by_element: dict[str, list[tuple[str, str]]] = {}
        for elem, labels in configuration.items():
            if elem not in by_element:
                raise RuntimeError('no pseudopotential for element %r in the run' % elem)
            key = (attr.get('fpath'), elem, tuple(labels), prefer_pswfc, r_box, n_points, so)
            if key not in _SHELL_CACHE:
                _SHELL_CACHE[key] = shells_from_generator(
                    by_element[elem], list(labels), prefer_pswfc=prefer_pswfc, r_box=r_box,
                    n_points=n_points, spin_orbit=so,
                )  # fmt: skip
            shells_by_element[elem], skipped_by_element[elem] = _SHELL_CACHE[key]
        payload = (shells_by_element, skipped_by_element)
    else:
        payload = None
    shells_by_element, skipped_by_element = comm.bcast(payload, root=0)

    for elem, skipped in skipped_by_element.items():
        if skipped:
            if rank == 0:
                print('WARNING: %s: shells not generated: %s' % (
                    elem, '; '.join('%s (%s)' % (label, reason) for label, reason in skipped)))  # fmt: skip
            skipped_labels = {label for label, _ in skipped}
            configuration[elem] = [lbl for lbl in configuration[elem] if lbl not in skipped_labels]

    if save_basis and rank == 0:
        root = os.path.join(attr['opath'], 'BASIS_PS')
        for elem, shells in shells_by_element.items():
            os.makedirs(os.path.join(root, elem), exist_ok=True)
            for shell in shells:
                suffix = '' if shell.j is None else '_j%d' % int(2 * shell.j)
                _write_two_col(
                    os.path.join(root, elem, '%s%s.dat' % (shell.label, suffix)), shell.r, shell.wfc
                )
        if attr.get('verbose'):
            print('Generated basis written to %s' % root)

    return build_basis_from_radial_shells(data_controller, shells_by_element)


def build_mixed_basis_with_tables(
    data_controller, pseudo_labels: dict[str, list[str]]
) -> tuple[list[dict], dict]:
    """Basis with the pseudo-orbitals of the pseudopotentials and the other shells from tables.

    The spin-orbit counterpart of :func:`~PAOFLOW.projection.do_atwfc_proj.build_basis_from_sources`
    (which keeps the scalar numerics of the QE route): pseudo-orbitals from the UPF or POTCAR
    datasets, the remaining shells of ``arry['configuration']`` from ``attr['basispath']``
    (j-resolved tables when present), records built by :func:`build_basis_from_radial_shells`.

    Parameters
    ----------
    data_controller : DataController
        As :func:`build_generated_basis`, plus ``attr['basispath']``.
    pseudo_labels : dict
        ``{element: [labels taken from the pseudopotential]}``.

    Returns
    -------
    basis, shells
        As :func:`build_basis_from_radial_shells`.
    """
    arry, attr = data_controller.data_dicts()
    so = bool(attr.get('dftSO', False))
    datasets = load_pseudopotentials(data_controller)
    by_element = {}
    for name, data in datasets.items():
        by_element.setdefault(re.split(r'\d+', name)[0], data)
    shells_by_element: dict[str, list[RadialShell]] = {}
    for elem, labels in arry['configuration'].items():
        from_pseudo = {lbl.upper() for lbl in pseudo_labels.get(elem, [])}
        shells = []
        for label in labels:
            if label.upper() in from_pseudo:
                shells.extend(shells_from_pseudo_orbitals(by_element[elem], [label]))
            else:
                shells.extend(shells_from_tables(attr['basispath'], elem, [label], so))
        shells_by_element[elem] = shells
    return build_basis_from_radial_shells(data_controller, shells_by_element)


__all__ = [
    'GeneratedShell',
    'RadialShell',
    'build_basis_from_radial_shells',
    'build_generated_basis',
    'build_mixed_basis_with_tables',
    'shells_from_generator',
    'shells_from_pseudo_orbitals',
    'shells_from_tables',
]
