"""One configuration per system: :class:`ElphonSystem`, its electrons and its coupling arguments.

A system folder (``examples/elphon/<System>``) holds the QE runs under ``qe/`` and one
``system.py`` that builds an :class:`ElphonSystem`; the workflow drivers of its
subfolders (``superconductivity/``, ``optics/``, ``transport/``, ``arpes/``, ...) load that
file with :func:`load_system` and call the functions of this subpackage.  The physics stays
in the procedural backends (``eliashberg_dense_q``, ``carrier_scattering_dense_q``, ...);
this layer only wires inputs, caches and outputs.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..coupling_cache import basis_tag
from ..elph_bloch import atom_masses
from ..long_range import LongRangeModel
from ..pao_electrons import (
    PAOElectrons,
    check_same_pao_basis,
    format_hubbard_info,
    pao_hamiltonian_from_save,
)
from ..wavefunction_symmetry import stores_irreducible_wedge

SYSTEM_FILE = 'system.py'


def is_root(comm=None) -> bool:
    """Whether this MPI rank is rank 0 (``True`` without MPI).

    Parameters
    ----------
    comm : MPI communicator, optional
        ``MPI.COMM_WORLD`` by default.

    Returns
    -------
    bool
        ``True`` on rank 0.
    """
    if comm is None:
        try:
            from mpi4py import MPI
        except ImportError:  # pragma: no cover - mpi4py is a core dependency
            return True
        comm = MPI.COMM_WORLD
    return comm.Get_rank() == 0


def find_system_file(start: str) -> str:
    """Path of the nearest ``system.py`` at or above ``start``.

    Parameters
    ----------
    start : str
        A driver's ``__file__`` or a directory.  Symbolic links are not resolved, so a
        driver linked into a run directory finds the ``system.py`` of that directory.

    Returns
    -------
    str
        Absolute path of the configuration file.

    Raises
    ------
    FileNotFoundError
        If no ``system.py`` exists in the directory of ``start`` or any of its parents.
    """
    path = os.path.abspath(start)
    directory = path if os.path.isdir(path) else os.path.dirname(path)
    while True:
        candidate = os.path.join(directory, SYSTEM_FILE)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(directory)
        if parent == directory:
            raise FileNotFoundError(
                'no %s found in %s or its parents' % (SYSTEM_FILE, os.path.dirname(path))
            )
        directory = parent


def load_system(start: str) -> ModuleType:
    """Import the nearest ``system.py`` above a driver and return it as a module.

    Parameters
    ----------
    start : str
        A driver's ``__file__`` or a directory (see :func:`find_system_file`).

    Returns
    -------
    module
        The configuration module: ``SYSTEM`` (an :class:`ElphonSystem`) and the
        per-workflow default dictionaries (``SUPERCONDUCTIVITY``, ``OPTICS``,
        ``TRANSPORT``, ``ARPES``, ``PHONONS``) that the drivers use as defaults.

    Notes
    -----
    The module is registered under a name derived from its path, so two systems can be
    loaded in one process.
    """
    path = find_system_file(start)
    name = 'paoflow_elphon_system_' + os.path.basename(os.path.dirname(path)).replace('-', '_')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, 'SYSTEM'):
        raise AttributeError('%s defines no SYSTEM (an ElphonSystem)' % path)
    return module


@dataclass
class Electrons:
    """PAO electronic structure of a system, as the electron-phonon workflows use it.

    Attributes
    ----------
    projections : ndarray
        ``A_{ni}(k) = <phi_i|psi_nk>`` of the coupling save (rotate the coupling to the
        PAO gauge).
    HRs : ndarray
        PAO Hamiltonian (eV) that provides the band energies and eigenvectors: of the
        coupling save, or of ``electrons_save`` (eACBN0 electrons).
    nscf : dict
        Lattice data of the coupling save (``read_nscf``) with ``'orbital_positions'``.
    data_controller : DataController
        Of the coupling save (position dipoles, photoemission basis, basis signature).
    provenance : dict
        ``coupling_save``, ``electrons_save``, ``electrons_dft``, ``electrons_hubbard``
        (merged into the output dictionaries).
    coupling : PAOElectrons
        The run on the coupling save.
    electrons : PAOElectrons or None
        The run on ``electrons_save`` (``None`` when the electrons are those of the coupling
        save).
    """

    projections: NDArray[np.complex128]
    HRs: NDArray[np.complex128]
    nscf: dict
    data_controller: object
    provenance: dict
    coupling: PAOElectrons
    electrons: PAOElectrons | None = None

    @property
    def lattice(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """``(kpts_cryst, bg, at)`` of the coarse grid, the positional lattice arguments."""
        return self.nscf['kpts_cryst'], self.nscf['bg'], self.nscf['at']

    @property
    def spin_degeneracy(self) -> float:
        """2 without spin-orbit coupling, 1 with it (from the coupling save)."""
        _, attributes = self.data_controller.data_dicts()
        return 1.0 if attributes.get('dftSO', False) else 2.0


@dataclass
class ElphonSystem:
    """Configuration of one crystal for the electron-phonon workflows.

    Parameters
    ----------
    prefix : str
        QE prefix of the scf, ph.x and nscf runs.
    root : str
        The system folder; relative paths below are resolved against it.
    masses_amu : sequence of float
        One mass per species, in ``ATOMIC_SPECIES`` order (expanded to the atoms of the
        cell by :func:`~PAOFLOW.elphon.elph_bloch.atom_masses`).
    nelec : float
        Valence electrons of the cell (locates the Fermi level or the gap).
    coupling_source : {'dfpt', 'epw'}
        ``'dfpt'``: PAOFLOW's native vertex from the ``ph.x`` ``dvscf`` files and the nscf
        wavefunctions; ``'epw'``: EPW's ``.epb`` files of an ``epbwrite`` run.
    coarse_grid : tuple of int
        nscf k-grid (EPW ``nk1..3``).
    qgrid : tuple of int, optional
        ``ph.x`` q-grid (EPW ``nq1..3``); must divide ``coarse_grid``.  Default: the
        k-grid.
    qe_dir : str
        Directory of the QE runs: ``phonon/`` (scf + ph.x), ``nscf/`` (the coarse-grid
        nscf) and, optionally, ``optics/`` (an nscf with many bands for the extended-basis
        dielectric function).
    phonon_dir, nscf_dir, optics_dir : str, optional
        Overrides of the three run directories (default ``<qe_dir>/phonon`` etc.).  The
        phonon directory holds ``_ph0/`` and ``<prefix>.dyn*``, or the ``pp.py`` ``save/``;
        the nscf directory holds ``<prefix>.save`` and, for the EPW source, the ``.epb``
        files.
    save_dir : str, optional
        The coarse-grid nscf save (default ``<nscf_dir>/<prefix>.save``).
    basis_dir : str, optional
        Radial tables of ``paoflow-genbasis-ps`` (``None``: PAOFLOW's ``BASIS``).
    configuration : str or mapping
        PAO basis preset (``'standard'``) or a per-element shell dictionary.
    pthr : float
        Projectability threshold.
    projector : {'internal', 'projwfc'}
        Internal atomic projections or the ``projwfc.x`` ones of the save.
    save_overlaps : bool
        ``projwfc`` overlaps (``projwfc`` projector only).
    long_range : bool
        Subtract the analytic dipole (Fröhlich) vertex on the coarse grid and restore it at
        every dense q when the crystal is polar (``ph.x`` with ``epsil``), and the
        quadrupole term when ``quadrupoles`` is given.  ``False``: plain interpolation of
        the full coupling (``allow_missing_long_range``).
    quadrupoles : str, optional
        EPW-format ``quadrupole.fmt`` (``coupling/quadrupoles.py``).
    cache_dir : str, optional
        Directory of the coarse-vertex cache ``g(R_e, R_p)`` shared by every workflow
        (``None`` disables it).
    spin_degeneracy : float, optional
        2 without spin-orbit coupling, 1 with it; ``None`` reads it from the save.
    """

    prefix: str
    root: str
    masses_amu: Sequence[float]
    nelec: float
    coupling_source: str = 'dfpt'
    coarse_grid: tuple[int, int, int] = (6, 6, 6)
    qgrid: tuple[int, int, int] | None = None
    qe_dir: str = 'qe'
    phonon_dir: str | None = None
    nscf_dir: str | None = None
    optics_dir: str | None = None
    save_dir: str | None = None
    basis_dir: str | None = 'BASIS_PS'
    configuration: str | Mapping = 'standard'
    pthr: float = 0.95
    projector: str = 'internal'
    save_overlaps: bool = False
    long_range: bool = True
    quadrupoles: str | None = None
    cache_dir: str | None = 'cache'
    spin_degeneracy: float | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.coupling_source not in ('dfpt', 'epw'):
            raise ValueError(
                "coupling_source must be 'dfpt' or 'epw', got %r" % self.coupling_source
            )
        self.root = os.path.abspath(self.root)
        self.coarse_grid = tuple(int(n) for n in self.coarse_grid)
        self.qgrid = self.coarse_grid if self.qgrid is None else tuple(int(n) for n in self.qgrid)
        if any(k % q for k, q in zip(self.coarse_grid, self.qgrid)):
            raise ValueError(
                'the q-grid %s must divide the k-grid %s' % (self.qgrid, self.coarse_grid)
            )
        self.masses_amu = [float(m) for m in self.masses_amu]

    # ------------------------------------------------------------------ paths
    def path(self, relative: str | None) -> str | None:
        """Absolute path of ``relative`` (resolved against ``root``; absolute paths pass)."""
        if relative is None:
            return None
        relative = os.path.expanduser(relative)
        return relative if os.path.isabs(relative) else os.path.join(self.root, relative)

    @property
    def qe_path(self) -> str:
        """The QE directory."""
        return self.path(self.qe_dir)

    @property
    def phonon_path(self) -> str:
        """The ``ph.x`` run directory (``_ph0/``, dyn files, or the ``pp.py`` ``save/``)."""
        return (
            self.path(self.phonon_dir) if self.phonon_dir else os.path.join(self.qe_path, 'phonon')
        )

    @property
    def nscf_path(self) -> str:
        """The coarse-grid nscf run directory (and the ``.epb`` files of the EPW source)."""
        return self.path(self.nscf_dir) if self.nscf_dir else os.path.join(self.qe_path, 'nscf')

    @property
    def optics_path(self) -> str:
        """The optics nscf run directory (extended-basis dielectric function)."""
        return (
            self.path(self.optics_dir) if self.optics_dir else os.path.join(self.qe_path, 'optics')
        )

    @property
    def save_path(self) -> str:
        """The coarse-grid nscf save."""
        if self.save_dir:
            return self.path(self.save_dir)
        return os.path.join(self.nscf_path, self.prefix + '.save')

    @property
    def coupling_dir(self) -> str:
        """Where the coarse coupling is read from: the phonon directory (dfpt) or the nscf one (epw)."""
        return self.phonon_path if self.coupling_source == 'dfpt' else self.nscf_path

    @property
    def basis_path(self) -> str | None:
        """The radial tables of the PAO basis (``None``: PAOFLOW's default ``BASIS``)."""
        return self.path(self.basis_dir)

    @property
    def cache_path(self) -> str | None:
        """The coarse-vertex cache directory (``None`` when disabled)."""
        return self.path(self.cache_dir)

    @property
    def quadrupoles_path(self) -> str | None:
        """The quadrupole file, or ``None``."""
        return self.path(self.quadrupoles)

    # ------------------------------------------------------------------ checks
    def check_inputs(self) -> None:
        """Stop early with a clear message when the QE outputs the coupling needs are missing.

        Raises
        ------
        FileNotFoundError
            Missing nscf save, ``ph.x`` outputs (dfpt) or ``.epb`` files (epw).
        """
        if not os.path.isdir(self.save_path):
            raise FileNotFoundError(
                '%s not found: run the pw.x nscf in %s first.' % (self.save_path, self.nscf_path)
            )
        if self.coupling_source == 'epw':
            if not any(f.startswith(self.prefix + '.epb') for f in os.listdir(self.nscf_path)):
                raise FileNotFoundError(
                    'no %s.epb* files in %s: run epw.x with epbwrite = .true. first.'
                    % (self.prefix, self.nscf_path)
                )
            return
        phonon = self.phonon_path
        if not os.path.isdir(phonon):
            raise FileNotFoundError('%s not found: run pw.x scf and ph.x there first.' % phonon)
        if not (
            os.path.isdir(os.path.join(phonon, '_ph0'))
            or os.path.isdir(os.path.join(phonon, 'save'))
            or any('dvscf' in f for f in os.listdir(phonon))
        ):
            raise FileNotFoundError(
                'no ph.x outputs (_ph0/, save/ or *.dvscf_q*) in %s: run ph.x with fildvscf first.'
                % phonon
            )

    # --------------------------------------------------------------- electrons
    def _run_options(self, workpath: str | None, outputdir: str | None) -> dict:
        return dict(
            basispath=self.basis_path,
            configuration=self.configuration,
            pthr=self.pthr,
            projector=self.projector,
            workpath=workpath if workpath is not None else self.root,
            outputdir=outputdir if outputdir is not None else os.path.join(self.root, 'output'),
            save_overlaps=self.save_overlaps,
            verbose=False,
        )

    def electrons(
        self,
        electrons_save: str | None = None,
        *,
        workpath: str | None = None,
        outputdir: str | None = None,
        comm=None,
    ) -> Electrons:
        """PAO projections and Hamiltonian; the Hamiltonian optionally from another save.

        Parameters
        ----------
        electrons_save : str, optional
            QE save of an eACBN0 (DFT+U+V, ortho-atomic) nscf of the same crystal in the
            same PAO basis: its PAO Hamiltonian gives the band energies and eigenvectors,
            while the projections, the coupling, the k list and the symmetries stay those
            of the coupling save (the coupling in the orbital basis is the DFPT one).
        workpath, outputdir : str, optional
            Of the :class:`PAOFLOW.PAOFLOW` runs (default: ``root`` and ``root/output``).
        comm : MPI communicator, optional
            Rank 0 prints the basis check of ``electrons_save``.

        Returns
        -------
        Electrons
            Projections, ``HRs``, lattice data with the orbital centres, the data controller
            of the coupling save and the provenance.

        Notes
        -----
        An nscf on the irreducible wedge (``K_POINTS automatic``) is unfolded to the full
        grid by symmetry (``unfold_kpoints``); an explicit full list is used as is.  The
        Hamiltonian is built with ``expand_wedge=False`` in both cases.
        """
        options = self._run_options(workpath, outputdir)
        save = self.save_path
        coupling = pao_hamiltonian_from_save(
            save, expand_wedge=False, unfold_kpoints=stores_irreducible_wedge(save), **options
        )

        def kind(run: PAOElectrons) -> str:
            return run.hubbard['kind'] if run.hubbard else 'PBE'

        provenance = {
            'coupling_save': save,
            'electrons_save': '',
            'electrons_dft': kind(coupling),
            'electrons_hubbard': format_hubbard_info(coupling.hubbard),
        }
        if electrons_save is None:
            return Electrons(
                coupling.projections,
                coupling.HRs,
                coupling.nscf,
                coupling.data_controller,
                provenance,
                coupling,
            )
        electrons_save = os.path.abspath(electrons_save)
        electrons = pao_hamiltonian_from_save(electrons_save, keep_projections=False, **options)
        summary = check_same_pao_basis(coupling.data_controller, electrons.data_controller)
        if is_root(comm):
            print(
                '  electrons from %s: %s' % (electrons_save, format_hubbard_info(electrons.hubbard))
            )
            print(
                '  same PAO basis as %s (bands with projectability > %g: %s vs %s)'
                % (save, self.pthr, summary['bnd'][0], summary['bnd'][1])
            )
        nscf = dict(coupling.nscf, orbital_positions=electrons.nscf['orbital_positions'])
        provenance.update(
            electrons_save=electrons_save,
            electrons_dft=kind(electrons),
            electrons_hubbard=format_hubbard_info(electrons.hubbard),
        )
        return Electrons(
            coupling.projections,
            electrons.HRs,
            nscf,
            coupling.data_controller,
            provenance,
            coupling,
            electrons,
        )

    def hamiltonian_of(
        self, savedir: str, *, workpath: str | None = None, outputdir: str | None = None
    ):
        """PAO Hamiltonian of another QE save in this system's basis (e.g. a scissor reference).

        Parameters
        ----------
        savedir : str
            The QE save.
        workpath, outputdir : str, optional
            As :meth:`electrons`.

        Returns
        -------
        PAOElectrons
            Without projections.
        """
        options = self._run_options(workpath, outputdir)
        return pao_hamiltonian_from_save(savedir, keep_projections=False, **options)

    # ---------------------------------------------------------------- coupling
    def masses(self, nscf: Mapping) -> NDArray[np.float64]:
        """One mass per atom of the cell, from the per-species ``masses_amu``."""
        return atom_masses(self.masses_amu, nscf['species'], nscf['atom_names'])

    def long_range_model(self) -> LongRangeModel | None:
        """Dipole/quadrupole model of the coupling, or ``None``.

        Returns
        -------
        LongRangeModel or None
            ``None`` when ``long_range`` is off, when ``ph.x`` was run without ``epsil`` (no
            dielectric tensor) or when the crystal is non-polar and no quadrupole file is
            given.
        """
        if not self.long_range:
            return None
        try:
            model = LongRangeModel.from_coupling(
                self.phonon_path, self.save_path, quadrupoles=self.quadrupoles_path
            )
        except ValueError:  # ph.x without epsil: no dielectric tensor, no long-range model
            return None
        return model if (model.is_polar() or model.has_quadrupoles) else None

    def degeneracy(self, electrons: Electrons) -> float:
        """Spin degeneracy: the configured value, else that of the coupling save."""
        return float(self.spin_degeneracy) if self.spin_degeneracy else electrons.spin_degeneracy

    def coupling_arguments(
        self, electrons: Electrons, *, symmetry: bool = True, cache: bool = True
    ) -> dict:
        """Keyword arguments of the dense-q drivers that describe the coarse coupling.

        Parameters
        ----------
        electrons : Electrons
            Of :meth:`electrons`.
        symmetry : bool, optional
            Include ``sym_rots`` (the drivers that fold the dense q-grid).
        cache : bool, optional
            Include ``cache_dir``/``cache_tag`` (the basis signature with ``pthr``).

        Returns
        -------
        dict
            ``coupling_source``, ``save_dir``, ``long_range``,
            ``allow_missing_long_range``, ``masses_amu``, ``tau_cryst``, ``species``,
            ``orbital_positions`` and, optionally, ``sym_rots``, ``cache_dir``,
            ``cache_tag``.
        """
        nscf = electrons.nscf
        kwargs = dict(
            coupling_source=self.coupling_source,
            save_dir=self.save_path,
            long_range=self.long_range_model(),
            allow_missing_long_range=not self.long_range,
            masses_amu=self.masses(nscf),
            tau_cryst=nscf['tau_cryst'],
            species=nscf['atom_names'],
            orbital_positions=nscf['orbital_positions'],
        )
        if symmetry:
            kwargs['sym_rots'] = nscf['s_cryst']
        if cache and self.cache_path:
            kwargs['cache_dir'] = self.cache_path
            kwargs['cache_tag'] = basis_tag(electrons.data_controller, self.pthr)
        return kwargs

    def positional_arguments(self, electrons: Electrons) -> tuple:
        """``(A, HRs, kpts_cryst, bg, at, alat, omega, coupling_dir, qgrid, coarse_grid)``.

        The positional arguments shared by ``carrier_scattering_dense_q``,
        ``phonon_assisted_absorption_dense_q`` and ``fan_migdal_self_energy``.
        """
        nscf = electrons.nscf
        return (
            electrons.projections,
            electrons.HRs,
            nscf['kpts_cryst'],
            nscf['bg'],
            nscf['at'],
            nscf['alat'],
            nscf['omega'],
            self.coupling_dir,
            self.qgrid,
            self.coarse_grid,
        )
