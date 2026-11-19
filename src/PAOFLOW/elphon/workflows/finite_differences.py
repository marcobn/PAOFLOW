r"""Finite-difference coupling of a crystal: the displaced runs of its ideal supercell (``'fd'``).

The ``'fd'`` coupling source (:mod:`PAOFLOW.elphon.fd_vertex`) differentiates the PAO
Hamiltonian of a cell that is its own unit cell.  For a crystal that cell is the ideal
:math:`n_1\times n_2\times n_3` supercell of the primitive cell: its only phonon wavevector is
:math:`\Gamma`, whose :math:`3N` modes are the primitive phonons on the commensurate
:math:`n_1\times n_2\times n_3` q-grid, and :math:`\lambda` of the cell equals the primitive
:math:`\lambda` on that grid at equal k-sampling.

:func:`write_supercell_displacements` writes, from the scf save of the primitive cell, the
``pw.x`` inputs of the reference supercell and of its displaced copies (every atom along
:math:`\pm x, \pm y, \pm z`, or random all-atom patterns), the ``displacements.json`` manifest
of :meth:`PAOFLOW.PAOFLOW.PAOFLOW.electron_phonon` and ``run_fd.sh``.  The supercell is built
directly as the unit cell (``supercell_matrix = 1``), so no scf of the ideal supercell is
needed before the displaced runs: the reference run of the manifest is that scf.  After the
runs, :class:`~PAOFLOW.elphon.workflows.system.ElphonSystem` with ``coupling_source='fd'``,
``fd_dir`` this directory and ``save_dir`` the reference save feeds the dense-k drivers.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

from .thermal import fd_inputs, fd_runs_done, write_fd_run_script

__all__ = [
    'fd_inputs',
    'fd_runs_done',
    'reference_save',
    'supercell_prefix',
    'write_supercell_displacements',
]


def supercell_prefix(prefix: str) -> str:
    """QE prefix of the supercell runs (``<prefix>_sc``; the reference adds ``_ref``).

    Parameters
    ----------
    prefix : str
        Prefix of the primitive cell.

    Returns
    -------
    str
    """
    return '%s_sc' % prefix


def reference_save(fd_dir: str, prefix: str) -> str:
    """The save of the undisplaced supercell (the electrons of the ``'fd'`` source).

    Parameters
    ----------
    fd_dir : str
        The displaced-runs directory.
    prefix : str
        Prefix of the primitive cell.

    Returns
    -------
    str
        ``<fd_dir>/tmp_<prefix>_sc_ref/<prefix>_sc_ref.save``.
    """
    ref = supercell_prefix(prefix) + '_ref'
    return os.path.join(fd_dir, 'tmp_' + ref, ref + '.save')


def write_supercell_displacements(
    primitive_save: str,
    fd_dir: str,
    supercell: Sequence[int],
    *,
    prefix: str,
    kgrid: Sequence[int],
    distance: float = 0.06,
    mode: str = 'cartesian',
    nconfig: int | None = None,
    seed: int = 0,
    nbnd: int | None = None,
    pp_dir: str | None = None,
    configuration='standard',
) -> dict:
    r"""``pw.x`` inputs of the ideal supercell and of its displaced copies, with ``run_fd.sh``.

    Parameters
    ----------
    primitive_save : str
        scf save of the primitive cell (structure, cutoffs, smearing and pseudopotentials of
        the inputs).
    fd_dir : str
        The displaced-runs directory (created): ``supercell.in``, ``disp-NNN.in``,
        ``displacements.json`` and ``run_fd.sh``.
    supercell : sequence of int
        Diagonal supercell :math:`(n_1, n_2, n_3)` in primitive cells.
    prefix : str
        Prefix of the primitive cell; the runs use ``<prefix>_sc_ref`` and
        ``<prefix>_sc_dispNNN``.
    kgrid : sequence of int
        Monkhorst-Pack grid of the supercell scf runs.
    distance : float, optional
        Displacement amplitude (bohr).
    mode : {'cartesian', 'random'}, optional
        ``'cartesian'``: every atom of the supercell along :math:`\pm x, \pm y, \pm z`
        (:math:`6N` cells); ``'random'``: ``nconfig`` all-atom random patterns and their mirrors
        (:math:`2M` cells), fitted by least squares
        (:mod:`PAOFLOW.elphon.random_displacements`).
    nconfig : int, optional
        Random configurations (``'random'`` mode; default :math:`3N`, a full-rank fit without a
        locality cutoff).
    seed : int, optional
        Seed of the random directions.
    nbnd : int, optional
        Bands of the supercell runs (default: sized to the PAO basis of ``configuration``).
    pp_dir : str, optional
        ``pseudo_dir`` written into the inputs, as seen from ``fd_dir`` (default: the folder
        of the primitive save's pseudopotentials).
    configuration : str or mapping, optional
        PAO basis preset that sizes ``nbnd``.

    Returns
    -------
    dict
        ``fd_dir``, ``atoms`` (of the supercell), ``cells`` (displaced cells), ``mode``,
        ``inputs`` (the ``pw.x`` inputs, reference first) and ``run_script``.

    Raises
    ------
    ValueError
        For an unknown ``mode`` or a non-positive supercell.
    """
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    from ...PAOFLOW import PAOFLOW
    from ...phonon.do_phonopy import init_phonopy
    from ..displacements import generate_eph_displacements
    from ..io import write_eph_displaced_supercells

    if mode not in ('cartesian', 'random'):
        raise ValueError("mode must be 'cartesian' or 'random', got %r" % mode)
    dims = [int(n) for n in supercell]
    if len(dims) != 3 or min(dims) < 1:
        raise ValueError('supercell must be three positive integers, got %r' % (supercell,))
    fd_dir = os.path.abspath(fd_dir)
    os.makedirs(fd_dir, exist_ok=True)

    pf = PAOFLOW(savedir=primitive_save, outputdir=fd_dir, verbose=False)
    controller = pf.data_controller
    arry, attr = controller.data_dicts()
    attr['phonon_supercell_matrix'] = np.diag(dims)
    ideal = init_phonopy(controller).supercell
    # The supercell is the unit cell of the finite differences (supercell_matrix = 1): every
    # one of its atoms is a reference atom, displaced explicitly or in the random patterns.
    cell = PhonopyAtoms(
        symbols=list(ideal.symbols),
        cell=np.asarray(ideal.cell, dtype=float),
        scaled_positions=np.asarray(ideal.scaled_positions, dtype=float),
        masses=np.asarray(ideal.masses, dtype=float),
    )
    attr['phonon_supercell_matrix'] = np.eye(3, dtype=int)
    phonon = Phonopy(
        cell, supercell_matrix=np.eye(3, dtype=int), primitive_matrix='P', calculator='qe'
    )
    arry['phonopy'] = phonon
    natoms = len(cell)
    if mode == 'random' and nconfig is None:
        nconfig = 3 * natoms
    if nbnd is None and pp_dir is not None:
        # pseudo_dir is written as seen from fd_dir; the band count reads the files from here.
        from ..basis import supercell_nbnd

        pseudos = pp_dir if os.path.isabs(pp_dir) else os.path.join(fd_dir, pp_dir)
        try:
            nbnd = supercell_nbnd(controller, cell, configuration=configuration, pp_dir=pseudos)
        except (FileNotFoundError, ValueError):
            nbnd = None  # write_eph_displaced_supercells tries again and warns
    cells, meta = generate_eph_displacements(
        phonon, float(distance), is_plusminus=True, displacement_mode=mode,
        nconfig=nconfig, seed=seed,
    )  # fmt: skip
    write_eph_displaced_supercells(
        controller, cells, meta, elphon_dir=fd_dir, pp_dir=pp_dir,
        prefix=supercell_prefix(prefix), kgrid=[int(k) for k in kgrid],
        configuration=configuration, nbnd=nbnd, is_plusminus=True,
        displacement_mode=mode, seed=seed if mode == 'random' else None,
    )  # fmt: skip
    script = write_fd_run_script(fd_dir)
    return {
        'fd_dir': fd_dir,
        'atoms': natoms,
        'cells': len(cells),
        'mode': mode,
        'inputs': fd_inputs(fd_dir),
        'run_script': script,
    }
