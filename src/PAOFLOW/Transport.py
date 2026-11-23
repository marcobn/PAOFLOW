from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController
    from PAOFLOW.transport.data import ConductorData, SmearingType
    from PAOFLOW.transport.partition.types import HamiltonianBlockPartition
    from PAOFLOW.transport.results import TransportResults


class Transport:
    """User-facing transport orchestrator with direct-argument APIs.

    Parameters
    ----------
    data_controller : DataController
        Shared PAOFLOW ``DataController`` used by transport preparation stages.

    Attributes
    ----------
    conductor_data : ConductorData or None
        Validated conductor input model populated by ``build_hamiltonian_blocks``.
    blc_blocks : dict[str, Any] or None
        Hamiltonian block operators populated by ``build_hamiltonian_blocks``.
    results : TransportResults or None
        Cached full-grid transport observables.
    """

    def __init__(self, data_controller: DataController) -> None:
        self.data_controller = data_controller
        self.conductor_data: ConductorData | None = None
        self.blc_blocks: dict[str, Any] | None = None
        self.results: TransportResults | None = None
        self._energy_grid_config: dict[str, Any] | None = None
        self._output_config: dict[str, Any] = {
            'output_dir': './',
            'work_dir': './',
            'postfix': '',
            'write_kdata': False,
            'write_green_function': False,
            'write_lead_self_energy': False,
        }
        self._transport_options_config: dict[str, Any] = {
            'formula': 'landauer',
        }
        self._partition: HamiltonianBlockPartition | None = None
        self._onsite_shift_config: dict[str, Any] | None = None
        self._lead_convergence_config: dict[str, Any] | None = None
        self._eigenchannel_config: dict[str, Any] | None = None
        self._surface_bands_config: dict[str, Any] | None = None
        self._bias_settings: Any = None
        self._bias_electrostatics: dict[str, Any] = {'kind': 'planar'}

    def configure_energy_grid(
        self,
        *,
        emin: float,
        emax: float,
        ne: int,
        delta: float,
        nk: list[int] | tuple[int, int] = (0, 0),
        smearing_type: SmearingType = 'lorentzian',
        delta_ratio: float = 5.0e-3,
        xmax: float = 25.0,
        ne_buffer: int = 1,
        energy_step: float = 0.001,
        nx_smear: int = 20000,
    ) -> None:
        """Configure the energy and k-grid integration settings.

        Must be called before any full-grid observable computation.

        Parameters
        ----------
        emin : float
            Minimum energy in eV.
        emax : float
            Maximum energy in eV.
        ne : int
            Number of energy points.
        delta : float
            Broadening parameter.
        nk : list[int] or tuple[int, int], optional
            Transverse k-mesh (the two directions perpendicular to transport). ``(0, 0)``
            (default) uses the transverse DFT mesh. It sets the mesh on which the Hamiltonian
            blocks are built, so a value other than the default must be given before
            :meth:`build_hamiltonian_blocks`.
        smearing_type : str, optional
            Smearing function: ``'lorentzian'`` (default), ``'gaussian'``,
            ``'fermi-dirac'``/``'fd'``, ``'methfessel-paxton'``/``'mp'``, or
            ``'marzari-vanderbilt'``/``'mv'``.
        delta_ratio : float, optional
            Adaptive-smearing ratio. Default ``5.0e-3``.
        xmax : float, optional
            Smearing cutoff in units of the broadening. Default ``25.0``.
        ne_buffer : int, optional
            Number of energy points processed per buffered chunk. Default ``1``.
        energy_step : float, optional
            Energy step used when building the smearing table. Default ``0.001``.
        nx_smear : int, optional
            Number of samples in the precomputed smearing table. Default ``20000``.
        """
        self._energy_grid_config = {
            'emin': emin,
            'emax': emax,
            'ne': ne,
            'delta': delta,
            'nk': list(nk),
            'smearing_type': smearing_type,
            'delta_ratio': delta_ratio,
            'xmax': xmax,
            'ne_buffer': ne_buffer,
            'energy_step': energy_step,
            'nx_smear': nx_smear,
        }
        if self.conductor_data is not None:
            built_nk = self.conductor_data.get_runtime_data().nk_par
            if any(int(n) > 0 for n in nk) and [int(n) for n in nk] != [int(n) for n in built_nk]:
                raise ValueError(
                    'The transverse k-mesh %s was fixed when the Hamiltonian blocks were built; '
                    'call configure_energy_grid(nk=%s) before build_hamiltonian_blocks().'
                    % ([int(n) for n in built_nk], [int(n) for n in nk])
                )
            self.conductor_data.energy.emin = emin
            self.conductor_data.energy.emax = emax
            self.conductor_data.energy.ne = ne
            self.conductor_data.energy.delta = delta
            self.conductor_data.kpoint_grid.nk = list(nk)
            self.conductor_data.energy.smearing_type = smearing_type
            self.conductor_data.energy.delta_ratio = delta_ratio
            self.conductor_data.energy.xmax = xmax
            self.conductor_data.energy.ne_buffer = ne_buffer
            self.conductor_data.energy.energy_step = energy_step
            self.conductor_data.energy.nx_smear = nx_smear
        self.results = None

    def configure_outputs(
        self,
        *,
        output_dir: str = './',
        work_dir: str = './',
        postfix: str = '',
        write_kdata: bool = False,
    ) -> None:
        """Configure output directory, file postfix, and optional operator outputs.

        Parameters
        ----------
        output_dir : str, optional
            Directory where transport output files are written.
        work_dir : str, optional
            Working directory for transport assets.
        postfix : str, optional
            String appended to default transport file names.
        write_kdata : bool, optional
            If ``True``, write k-resolved transmission and DOS to separate files.
        """
        import PAOFLOW.transport.io.log_module as log

        self._output_config = {
            'output_dir': output_dir,
            'work_dir': work_dir,
            'postfix': postfix,
            'write_kdata': write_kdata,
        }
        if self.conductor_data is not None:
            self.conductor_data.file_names.output_dir = output_dir
            self.conductor_data.file_names.work_dir = work_dir
            self.conductor_data.file_names.postfix = postfix
            self.conductor_data.symmetry.write_kdata = write_kdata
            log.initialize_logger(
                self.data_controller,
                log_file_name=f'transport_conductor{postfix}.log',
            )
        self.results = None

    def configure_transport_options(
        self,
        *,
        formula: str = 'landauer',
        **options: Any,
    ) -> None:
        """Configure the conductance formula and other transport options.

        Parameters
        ----------
        formula : str, optional
            Conductance formula. Supported values: ``'landauer'``,
            ``'generalized'``. Default is ``'landauer'``.
        **options : Any
            Additional formula-specific options forwarded to ``ConductorData``.
        """
        self._transport_options_config = {'formula': formula, **options}
        if self.conductor_data is not None:
            self.conductor_data.conduct_formula = formula
        self.results = None

    def define_partition(
        self,
        *,
        central_atoms: str = 'ALL',
        central_layers: int | None = None,
        left_lead_layers: int | None = None,
        right_lead_layers: int | None = None,
        transport_direction: str,
        layer_tolerance: float = 1.0e-6,
    ) -> None:
        """Define a layer-based transport partition.

        The partition is resolved immediately from PAOFLOW atom/order metadata
        into Hamiltonian block dimensions and orbital selectors. Users specify
        physical layers along ``transport_direction`` instead of PAO indices.

        Parameters
        ----------
        central_atoms : str, optional
            Central region atom selector. Currently only ``'ALL'`` is supported.
        central_layers : int or None, optional
            If provided, use this many layers from the start of the transport
            direction as the central region. This is useful for lead-only bulk
            calculations.
        left_lead_layers, right_lead_layers : int or None, optional
            Number of layers at the start/end of the transport direction used
            to construct conductor lead and coupling blocks.
        transport_direction : {'x', 'y', 'z'}
            Direction along which transport is computed.
        layer_tolerance : float, optional
            Coordinate tolerance used to group atoms into layers.
        """
        from PAOFLOW.transport.partition import resolve_layer_partition

        self._partition = resolve_layer_partition(
            self.data_controller,
            central_atoms=central_atoms,
            central_layers=central_layers,
            left_lead_layers=left_lead_layers,
            right_lead_layers=right_lead_layers,
            transport_direction=transport_direction,
            layer_tolerance=layer_tolerance,
        )
        self.results = None

    def configure_onsite_shifts(
        self,
        *,
        shift_L: float = 0.0,
        shift_C: float = 0.0,
        shift_R: float = 0.0,
        shift_corr: float = 0.0,
    ) -> None:
        """Configure rigid on-site energy shifts for lead/conductor alignment.

        Parameters
        ----------
        shift_L, shift_C, shift_R : float, optional
            Rigid on-site energy shifts (eV) applied to the left-lead, conductor,
            and right-lead blocks respectively. Default ``0.0``.
        shift_corr : float, optional
            On-site energy shift (eV) applied to the correlation self-energy.
            Default ``0.0``.
        """
        from PAOFLOW.transport.conductor_orchestration import apply_onsite_shifts

        self._onsite_shift_config = {
            'shift_L': shift_L,
            'shift_C': shift_C,
            'shift_R': shift_R,
            'shift_corr': shift_corr,
        }
        if self.conductor_data is not None:
            apply_onsite_shifts(self.conductor_data, self._onsite_shift_config)
        self.results = None

    def configure_lead_convergence(
        self,
        *,
        niterx: int = 200,
        transfer_thr: float = 1.0e-7,
        nprint: int = 20,
        nfailx: int = 5,
        surface: bool = False,
    ) -> None:
        """Configure the lead surface-Green's-function transfer-matrix iteration.

        Parameters
        ----------
        niterx : int, optional
            Maximum number of lead transfer-matrix iterations. Default ``200``.
        transfer_thr : float, optional
            Convergence threshold for the transfer-matrix iteration. Default ``1.0e-7``.
        nprint : int, optional
            Iteration-progress print frequency. Default ``20``.
        nfailx : int, optional
            Maximum number of allowed convergence failures. Default ``5``.
        surface : bool, optional
            If ``True``, run in surface mode (lead surface Green's function only,
            skipping the right-lead self-energy). Default ``False``.
        """
        from PAOFLOW.transport.conductor_orchestration import apply_lead_convergence

        self._lead_convergence_config = {
            'niterx': niterx,
            'transfer_thr': transfer_thr,
            'nprint': nprint,
            'nfailx': nfailx,
            'surface': surface,
        }
        if self.conductor_data is not None:
            apply_lead_convergence(self.conductor_data, self._lead_convergence_config)
        self.results = None

    def configure_surface_bands(
        self,
        *,
        band_path: str | None = None,
        high_sym_points: dict[str, Any] | None = None,
        ibrav: int | None = None,
        dk: float = 0.01,
        nk_path: int | None = None,
    ) -> None:
        r"""Enable the surface-projected band structure and set its k-path.

        Switches transverse sampling from the uniform Monkhorst-Pack mesh to a
        surface-projected high-symmetry k-path and puts the conductor in surface
        mode, so the Green's function drops the right-lead self-energy and its
        spectral function
        :math:`A(k, E) = -\frac{1}{\pi}\mathrm{Im}\,\mathrm{Tr}\,G_s`
        is the surface-projected bulk band structure.

        Must be called **before** :meth:`build_hamiltonian_blocks`, like the
        other ``configure_*`` methods, because the k-path is built during
        Hamiltonian preparation.

        Parameters
        ----------
        band_path : str or None, optional
            High-symmetry path string, e.g. ``'gG-X'``. Choose segments lying in
            the surface plane (perpendicular to the transport direction). When
            ``None``, the default path for ``ibrav`` is used.
        high_sym_points : dict or None, optional
            Explicit label -> fractional coordinate mapping. When ``None``, the
            tabulated points for ``ibrav`` are used.
        ibrav : int or None, optional
            Quantum ESPRESSO Bravais lattice index used to resolve the default
            high-symmetry points. Required when the data controller has no
            ``ibrav`` (for example when the SCF used ``ibrav=0``).
        dk : float, optional
            k-point spacing along the path. Ignored when ``nk_path`` is given.
        nk_path : int or None, optional
            Target number of k-points along the path.

        Notes
        -----
        Surface bands need a genuine transverse ``R``-grid to disperse; this is
        supplied automatically from the DFT Monkhorst-Pack mesh.
        """
        self._surface_bands_config = {
            'surface_bands': True,
            'surface_band_path': band_path,
            'surface_high_sym_points': high_sym_points,
            'surface_ibrav': ibrav,
            'surface_dk': dk,
            'surface_nk_path': nk_path,
        }
        if self.conductor_data is not None:
            sb = self.conductor_data.surface_bands
            sb.enabled = True
            sb.band_path = band_path
            sb.high_sym_points = high_sym_points
            sb.ibrav = ibrav
            sb.dk = dk
            sb.nk_path = nk_path
            self.conductor_data.advanced.surface = True
        self.results = None

    def configure_eigenchannels(
        self,
        *,
        do_eigenchannels: bool = False,
        neigchnx: int = 200000,
        do_eigplot: bool = False,
        ie_eigplot: int = 0,
        ik_eigplot: int = 0,
    ) -> None:
        """Configure transmission eigenchannel decomposition and plotting.

        Parameters
        ----------
        do_eigenchannels : bool, optional
            If ``True``, compute transmission eigenchannels. Default ``False``.
        neigchnx : int, optional
            Maximum number of eigenchannels to retain. Default ``200000``.
        do_eigplot : bool, optional
            If ``True``, write eigenchannel data for a chosen (energy, k) point.
        ie_eigplot : int, optional
            Energy index to plot eigenchannels for (with ``do_eigplot``).
        ik_eigplot : int, optional
            k-point index to plot eigenchannels for (with ``do_eigplot``).
        """
        from PAOFLOW.transport.conductor_orchestration import apply_eigenchannels

        self._eigenchannel_config = {
            'do_eigenchannels': do_eigenchannels,
            'neigchnx': neigchnx,
            'do_eigplot': do_eigplot,
            'ie_eigplot': ie_eigplot,
            'ik_eigplot': ik_eigplot,
        }
        if self.conductor_data is not None:
            apply_eigenchannels(self.conductor_data, self._eigenchannel_config)
        self.results = None

    def build_hamiltonian_blocks(
        self,
        *,
        calculation_type: str = 'bulk',
        carriers: str = 'electrons',
        use_sym: bool = False,
        do_overlap_transformation: bool = False,
        ispin: int = 0,
        debug: bool = False,
        **block_options: Any,
    ) -> dict[str, Any]:
        """Build conductor Hamiltonian blocks from the defined partition.

        Accepts only parameters required to construct Hamiltonian/overlap
        block objects after :meth:`define_partition`. Energy-grid parameters
        (``emin``, ``emax``, ``ne``, ``delta``), output paths, and optional
        operator outputs must be configured separately via
        ``configure_energy_grid(...)``, ``configure_outputs(...)``, and
        ``configure_transport_options(...)``.

        Parameters
        ----------
        calculation_type : str, optional
            Calculation mode: ``'bulk'`` or ``'conductor'``.
        carriers : str, optional
            Carrier type (``'electrons'`` or ``'phonons'``).
        use_sym : bool, optional
            Apply time-reversal symmetry to reduce the k-point sampling.
        do_overlap_transformation : bool, optional
            Apply the overlap-matrix orthogonalisation transformation.
        ispin : int, optional
            Spin channel to use for spin-polarized inputs (``0``, ``1``, or ``2``).
        debug : bool, optional
            If ``True``, write human-inspectable debug artifacts during setup:
            the legacy ``.ham`` file, ``projectability.txt``, and (when overlap
            is enabled) ``kovp.txt``. Off by default.
        **block_options : Any
            Additional block-construction options forwarded to ``ConductorData``
            (for example self-energy file paths for generalized formulas).

        Returns
        -------
        dict[str, Any]
            Block-operator mapping used by conductor self-energy and
            Green-function calculations. Also stored as ``self.blc_blocks``.

        Notes
        -----
        A layer partition must be supplied beforehand via
        :meth:`define_partition`. Optional physics tuning is applied
        through :meth:`configure_onsite_shifts`, :meth:`configure_lead_convergence`,
        and :meth:`configure_eigenchannels`; energy/smearing via
        :meth:`configure_energy_grid` and outputs via :meth:`configure_outputs`.

        Sets ``self.conductor_data`` and ``self.blc_blocks`` as side effects.
        Calling this method a second time resets both for the new calculation.
        """
        import PAOFLOW.transport.io.log_module as log
        from PAOFLOW.transport.conductor_orchestration import (
            apply_eigenchannels,
            apply_lead_convergence,
            apply_onsite_shifts,
        )
        from PAOFLOW.transport.conductor_steps import (
            build_conductor_blocks,
            build_conductor_input_values,
            prepare_conductor_step_state,
        )

        if self._partition is None:
            raise RuntimeError('Call define_partition(...) before build_hamiltonian_blocks(...).')

        input_values = build_conductor_input_values(
            dimC=self._partition.dim_c,
            dimL=self._partition.dim_l,
            dimR=self._partition.dim_r,
            transport_direction=self._partition.transport_direction,
            calculation_type=calculation_type,
            carriers=carriers,
            formula=self._transport_options_config.get('formula', 'landauer'),
            work_dir=self._output_config.get('work_dir', './'),
            output_dir=self._output_config.get('output_dir', './'),
            postfix=self._output_config.get('postfix', ''),
            use_sym=use_sym,
            do_overlap_transformation=do_overlap_transformation,
            debug=debug,
            ispin=ispin,
            **self._partition.selectors,
            **block_options,
        )

        if self._energy_grid_config is not None:
            input_values.update(self._energy_grid_config)

        if self._surface_bands_config is not None:
            input_values.update(self._surface_bands_config)

        state = prepare_conductor_step_state(
            data_controller=self.data_controller,
            input_values=input_values,
        )
        log.initialize_logger(
            self.data_controller,
            log_file_name=f'transport_conductor{state.data.file_names.postfix}.log',
        )
        build_conductor_blocks(
            state=state,
            data_controller=self.data_controller,
        )

        write_green_function = self._output_config.get('write_green_function', False)
        write_lead_self_energy = self._output_config.get('write_lead_self_energy', False)
        state.data.symmetry.write_gf = write_green_function
        state.data.symmetry.write_lead_sgm = write_lead_self_energy
        state.data.symmetry.write_kdata = self._output_config.get('write_kdata', False)

        apply_onsite_shifts(state.data, self._onsite_shift_config)
        apply_lead_convergence(state.data, self._lead_convergence_config)
        apply_eigenchannels(state.data, self._eigenchannel_config)

        # Surface bands imply surface mode regardless of lead-convergence config.
        if state.data.surface_bands.enabled:
            state.data.advanced.surface = True

        self.conductor_data = state.data
        self.blc_blocks = state.blc_blocks
        self.results = None
        return state.blc_blocks

    def compute_leads_self_energy(
        self,
        *,
        write: bool = True,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> tuple[NDArray[np.complex128] | None, NDArray[np.complex128] | None]:
        """Compute full-grid lead self-energies and write XML outputs.

        Parameters
        ----------
        write : bool, optional
            If ``True`` (default), write real-space self-energies to
            ``lead_L_sgm.xml`` and ``lead_R_sgm.xml``.
        comm : MPI.Comm, optional
            MPI communicator. Default is ``MPI.COMM_WORLD``.

        Returns
        -------
        tuple[NDArray[np.complex128] or None, NDArray[np.complex128] or None]
            ``(self_energy_L, self_energy_R)`` in real space,
            shape ``(ne, nrtot_par, dimC, dimC)`` each, or ``(None, None)``
            if ``write`` is ``False``.
        """
        from PAOFLOW.transport.conductor_orchestration import (
            compute_full_grid_results,
            require_grid_config,
            require_hamiltonian_blocks,
        )
        from PAOFLOW.transport.conductor_pipeline import write_self_energy_results

        require_hamiltonian_blocks(self.conductor_data, self.blc_blocks)
        require_grid_config(self._energy_grid_config)
        if write:
            if self.conductor_data is not None:
                self.conductor_data.symmetry.write_lead_sgm = True
            if self.results is not None and self.results.self_energy_L is None:
                self.results = None
        self.results = compute_full_grid_results(
            conductor_data=self.conductor_data,
            blc_blocks=self.blc_blocks,
            energy_grid_config=self._energy_grid_config,
            cached_results=self.results,
            comm=comm,
        )
        results = self.results
        if write:
            write_self_energy_results(
                data=self.conductor_data,
                results=results,
                comm=comm,
            )
        return results.self_energy_L, results.self_energy_R

    def compute_greens_functions(
        self,
        *,
        write: bool = True,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> NDArray[np.complex128] | None:
        """Compute full-grid conductor Green's functions and write XML output.

        Parameters
        ----------
        write : bool, optional
            If ``True`` (default), write real-space Green's functions to
            ``greenf.xml``.
        comm : MPI.Comm, optional
            MPI communicator. Default is ``MPI.COMM_WORLD``.

        Returns
        -------
        NDArray[np.complex128] or None
            Real-space conductor Green's function,
            shape ``(ne, nrtot_par, dimC, dimC)``, or ``None`` if
            ``write`` is ``False``.
        """
        from PAOFLOW.transport.conductor_orchestration import (
            compute_full_grid_results,
            require_grid_config,
            require_hamiltonian_blocks,
        )
        from PAOFLOW.transport.conductor_pipeline import write_greens_function_results

        require_hamiltonian_blocks(self.conductor_data, self.blc_blocks)
        require_grid_config(self._energy_grid_config)
        if write:
            if self.conductor_data is not None:
                self.conductor_data.symmetry.write_gf = True
            if self.results is not None and self.results.green_functions is None:
                self.results = None
        self.results = compute_full_grid_results(
            conductor_data=self.conductor_data,
            blc_blocks=self.blc_blocks,
            energy_grid_config=self._energy_grid_config,
            cached_results=self.results,
            comm=comm,
        )
        results = self.results
        if write:
            write_greens_function_results(
                data=self.conductor_data,
                results=results,
                comm=comm,
            )
        return results.green_functions

    def compute_transmission(
        self,
        *,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> NDArray[np.float64]:
        """Compute full-grid transmission and write output files.

        Parameters
        ----------
        comm : MPI.Comm, optional
            MPI communicator. Default is ``MPI.COMM_WORLD``.

        Returns
        -------
        NDArray[np.float64]
            Transmission, shape ``(1 + neigchn, ne)``.

        Notes
        -----
        Writes ``conductance*.dat`` under the configured output directory.
        """
        from PAOFLOW.transport.conductor_orchestration import compute_full_grid_results
        from PAOFLOW.transport.conductor_pipeline import write_transmission_results

        self.results = compute_full_grid_results(
            conductor_data=self.conductor_data,
            blc_blocks=self.blc_blocks,
            energy_grid_config=self._energy_grid_config,
            cached_results=self.results,
            comm=comm,
        )
        results = self.results
        write_transmission_results(
            data=self.conductor_data,
            results=results,
            comm=comm,
        )
        return results.transmission

    def compute_dos(
        self,
        *,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> NDArray[np.float64]:
        """Compute full-grid DOS and write output files.

        Parameters
        ----------
        comm : MPI.Comm, optional
            MPI communicator. Default is ``MPI.COMM_WORLD``.

        Returns
        -------
        NDArray[np.float64]
            Density of states, shape ``(ne,)``.

        Notes
        -----
        Writes ``doscond*.dat`` under the configured output directory.
        """
        from PAOFLOW.transport.conductor_orchestration import compute_full_grid_results
        from PAOFLOW.transport.conductor_pipeline import write_dos_results

        self.results = compute_full_grid_results(
            conductor_data=self.conductor_data,
            blc_blocks=self.blc_blocks,
            energy_grid_config=self._energy_grid_config,
            cached_results=self.results,
            comm=comm,
        )
        results = self.results
        write_dos_results(
            data=self.conductor_data,
            results=results,
            comm=comm,
        )
        return results.dos

    def compute_surface_bands(
        self,
        *,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> NDArray[np.float64]:
        r"""Compute the surface-projected band structure and write output files.

        Requires :meth:`configure_surface_bands` to have been called before
        :meth:`build_hamiltonian_blocks`, so that transverse sampling follows the
        surface k-path rather than a uniform mesh.

        Parameters
        ----------
        comm : MPI.Comm, optional
            MPI communicator. Default is ``MPI.COMM_WORLD``.

        Returns
        -------
        NDArray[np.float64]
            Surface spectral function :math:`A(k, E)`, shape ``(ne, nkpts)``.

        Raises
        ------
        RuntimeError
            If surface-band mode was not configured before the Hamiltonian
            blocks were built.

        Notes
        -----
        Writes ``surfband*.dat`` (the ``(ne, nkpts)`` spectral map) along with
        ``surfband_egrid*.dat`` and ``surfband_kpath*.dat`` axis files under the
        configured output directory.
        """
        from PAOFLOW.transport.conductor_orchestration import compute_full_grid_results
        from PAOFLOW.transport.conductor_writers import write_surface_bands

        if self.conductor_data is None or not self.conductor_data.surface_bands.enabled:
            raise RuntimeError(
                'Call configure_surface_bands(...) before build_hamiltonian_blocks(...) '
                'to compute a surface band structure.'
            )

        self.results = compute_full_grid_results(
            conductor_data=self.conductor_data,
            blc_blocks=self.blc_blocks,
            energy_grid_config=self._energy_grid_config,
            cached_results=self.results,
            comm=comm,
        )
        results = self.results
        write_surface_bands(
            rank=comm.Get_rank(),
            data=self.conductor_data,
            dos_k=results.dos_k,
            egrid=results.energy_grid,
        )
        return results.dos_k

    def compute_current(
        self,
        *,
        bias_min: float,
        bias_max: float,
        nbias: int,
        mu_L: float,
        mu_R: float,
        sigma: float,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> NDArray[np.float64]:
        r"""Compute current vs bias from the in-memory transmission and write output.

        Requires that ``build_hamiltonian_blocks(...)``,
        ``configure_energy_grid(...)``, and ``configure_outputs(...)`` have
        already been called. Transmission is taken from the cached full-grid
        results; if not yet computed, the full-grid calculation runs here.

        Parameters
        ----------
        bias_min : float
            Minimum bias voltage in V.
        bias_max : float
            Maximum bias voltage in V (must be greater than ``bias_min``).
        nbias : int
            Number of bias points (must be positive).
        mu_L : float
            Left chemical-potential scaling coefficient; enters as
            :math:`\mu_L \cdot V`.
        mu_R : float
            Right chemical-potential scaling coefficient; enters as
            :math:`\mu_R \cdot V`.
        sigma : float
            Smearing width in eV for the Fermi-Dirac broadening (must be
            positive).
        comm : MPI.Comm, optional
            Communicator used for work distribution. Default is
            ``MPI.COMM_WORLD``.

        Returns
        -------
        NDArray[np.float64]
            Current values aligned with the internally generated bias grid,
            shape ``(nbias,)``.

        Notes
        -----
        The current is the Landauer integral of the zero-bias transmission between the two shifted
        Fermi functions,

        .. math::
            I(V) = \int dE\, T(E)\,[f(E - \mu_L V) - f(E - \mu_R V)],

        in units of :math:`2e/h` times eV per transverse cell, positive for electrons flowing from
        left to right (:math:`\mu_L V > \mu_R V`). This is the sign of
        ``compute_current_self_consistent``. Before 15 November 2026 the Fermi function was
        :math:`1 - f` and the sign was the opposite: swap ``mu_L`` and ``mu_R`` in scripts written
        for it.

        Writes ``current.dat`` as two columns ``V I`` under the configured
        output directory.
        """
        from PAOFLOW.transport.calculators.current import (
            build_bias_grid,
            compute_current_vs_bias,
        )
        from PAOFLOW.transport.conductor_orchestration import compute_full_grid_results
        from PAOFLOW.transport.io.write_data import write_current_results

        if nbias <= 0:
            raise ValueError(f'nbias must be positive, got {nbias}.')
        if bias_max <= bias_min:
            raise ValueError(f'bias_max ({bias_max}) must be greater than bias_min ({bias_min}).')
        if sigma <= 0:
            raise ValueError(f'sigma must be positive, got {sigma}.')
        if not (np.isfinite(mu_L) and np.isfinite(mu_R)):
            raise ValueError('mu_L and mu_R must be finite floats.')

        self.results = compute_full_grid_results(
            conductor_data=self.conductor_data,
            blc_blocks=self.blc_blocks,
            energy_grid_config=self._energy_grid_config,
            cached_results=self.results,
            comm=comm,
        )
        results = self.results
        energy_grid = results.energy_grid
        transmission = results.transmission[0]

        if len(energy_grid) != len(transmission):
            raise RuntimeError(
                f'Energy grid length ({len(energy_grid)}) does not match '
                f'transmission length ({len(transmission)}).'
            )

        bias_grid = build_bias_grid(bias_min, bias_max, nbias)
        currents = compute_current_vs_bias(
            egrid=energy_grid,
            transm=transmission,
            vgrid=bias_grid,
            mu_L=mu_L,
            mu_R=mu_R,
            sigma=sigma,
        )

        if comm.Get_rank() == 0:
            write_current_results(
                output_dir=self._output_config['output_dir'],
                bias_grid=bias_grid,
                currents=currents,
            )

        results.bias_grid = bias_grid
        results.current = currents
        return currents

    def configure_bias_self_consistency(
        self,
        *,
        temperature: float = 300.0,
        fermi_energy: float = 0.0,
        electrostatics: str = 'planar',
        charge_widths: float | dict[str, float] | None = None,
        potential_grid_spacing: float = 0.3,
        arc_points: int = 30,
        line_points: int = 40,
        fermi_poles: int = 10,
        window_spacing: float = 0.005,
        mixing: float = 0.5,
        history: int = 6,
        tolerance: float = 1.0e-3,
        max_iterations: int = 80,
        thomas_fermi_preconditioner: bool = True,
        verbose: bool = False,
    ) -> None:
        """Configure the finite-bias calculation of :meth:`compute_current_self_consistent`.

        Parameters
        ----------
        temperature : float, optional
            Electron temperature of the leads (K).
        fermi_energy : float, optional
            Fermi level of the leads at zero bias (eV); 0 for a PAO Hamiltonian referred to the Fermi level.
        electrostatics : {'planar', '3d'}, optional
            Poisson solver between the electrode planes, periodic across the transport direction:

            - ``'planar'``: the charge of every atomic layer is a sheet (junctions uniform across the transverse
              cell);
            - ``'3d'``: the charge of every atom is a Gaussian and the potential is three-dimensional (molecules,
              wires). The transverse lattice vectors must be perpendicular to the transport direction.
        charge_widths : float, dict or None, optional
            Gaussian widths of the atomic charges for ``'3d'`` (bohr): one value, one per species, or None to take
            every species from its pseudopotential (the width with the Hartree self-energy of the valence
            pseudo-atomic density, ``PAOFLOW.transport.negf.charge_model``). Species without a pseudopotential
            (tight-binding models) need a value.
        potential_grid_spacing : float, optional
            Spacing (bohr) of the grid of :math:`U(\\mathbf r)` written for ``potential_grid_biases`` of
            :meth:`compute_current_self_consistent`.
        arc_points, line_points, fermi_poles : int, optional
            Equilibrium contour: Gauss-Legendre points on the arc and on the line, Fermi poles enclosed. The
            defaults integrate a bound state to about 2e-7.
        window_spacing : float, optional
            Energy step in the bias window (eV).
        mixing : float, optional
            Anderson mixing parameter, applied to the preconditioned residual.
        history : int, optional
            Iterations kept by the Anderson mixing.
        tolerance : float, optional
            Convergence threshold on the change of the potential (eV).
        max_iterations : int, optional
            Iteration limit per bias.
        thomas_fermi_preconditioner : bool, optional
            Precondition the residual with the Thomas-Fermi response of the conductor (default): without it the
            charge of a metal sloshes unless ``mixing`` is a few hundredths.
        verbose : bool, optional
            Print one line per iteration.

        Raises
        ------
        ValueError
            For an unknown electrostatics or non-positive numerical settings.
        """
        from PAOFLOW.transport.negf.self_consistency import SelfConsistencySettings

        if electrostatics not in ('planar', '3d'):
            raise ValueError(f"electrostatics must be 'planar' or '3d', got {electrostatics!r}")
        if potential_grid_spacing <= 0:
            raise ValueError('potential_grid_spacing must be positive')
        if temperature <= 0 or window_spacing <= 0 or tolerance <= 0 or not 0 < mixing <= 1:
            raise ValueError(
                'temperature, window_spacing, tolerance must be positive and 0 < mixing <= 1'
            )
        self._bias_settings = SelfConsistencySettings(
            temperature=temperature,
            fermi_energy=fermi_energy,
            arc_points=arc_points,
            line_points=line_points,
            fermi_poles=fermi_poles,
            window_spacing=window_spacing,
            mixing=mixing,
            history=history,
            tolerance=tolerance,
            max_iterations=max_iterations,
            thomas_fermi_preconditioner=thomas_fermi_preconditioner,
            verbose=verbose,
        )
        self._bias_electrostatics = {
            'kind': electrostatics,
            'charge_widths': charge_widths,
            'grid_spacing': potential_grid_spacing,
        }

    def compute_current_self_consistent(
        self,
        *,
        bias: Any,
        mu_L: float = 0.5,
        mu_R: float = -0.5,
        profile: str = 'self_consistent',
        write: bool = True,
        potential_grid_biases: Any = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> Any:
        r"""Current, potential profile and charge of the junction at finite bias.

        At each bias the leads shift to :math:`\mu_{L,R} = E_F + c_{L,R}\,eV` and the conductor gets a potential
        energy on every atomic layer (``electrostatics='planar'`` in :meth:`configure_bias_self_consistency`) or
        on every atom (``'3d'``). With ``profile='self_consistent'`` it is the Hartree potential of the
        bias-induced charge (Poisson equation between the electrodes) and is iterated to self-consistency; with
        ``'linear'`` it is the Laplace potential of the electrodes (the ramp). The current is the Landauer integral
        of the bias-dependent transmission :math:`T(E;V)`.

        Parameters
        ----------
        bias : array_like
            Biases (V), solved in order (each starts from the previous one).
        mu_L, mu_R : float, optional
            Coefficients :math:`c_L`, :math:`c_R` (as in :meth:`compute_current`); default symmetric.
        profile : {'self_consistent', 'linear'}, optional
        write : bool, optional
            Write ``current_<profile><postfix>.dat`` (V, I, iterations, converged, residual),
            ``potential_<profile><postfix>.dat`` and ``charge_<profile><postfix>.dat`` (per layer or per atom,
            one column per bias), ``transmission_<profile><postfix>.dat`` (V, E, T) and, for ``'3d'``,
            ``potential_planar_average_<profile><postfix>.dat`` to the output directory.
        potential_grid_biases : sequence of float, optional
            For ``'3d'`` with ``write``: biases of the scan at which :math:`U(\mathbf r)` is written on a grid
            of the slab, ``potential_grid_<profile><postfix>_V<bias>.xsf`` (XCrySDen, VESTA).
        comm : MPI.Comm, optional
            The energy points are shared out over its ranks.

        Returns
        -------
        FiniteBiasResults
            :mod:`PAOFLOW.transport.negf.self_consistency`; also ``self.results.bias_scan``.

        Raises
        ------
        RuntimeError
            Before ``build_hamiltonian_blocks(calculation_type='conductor')``.
        NotImplementedError
            For spin-polarised inputs or non-orthonormal orbitals; for ``'3d'``, transverse lattice vectors not
            perpendicular to the transport direction.

        Notes
        -----
        The current is :math:`\int dE\, T(E;V)[f(E-\mu_L) - f(E-\mu_R)]` per transverse cell, in eV: units
        of :math:`2e/h` as :meth:`compute_current`. The zero-bias PAO Hamiltonian is the self-consistent
        reference: only the change of the electron numbers acts, and the exchange-correlation response is
        neglected (``docs/internals/negf-finite-bias.md``).
        """
        from PAOFLOW.transport.negf.density_matrix import open_conductor_from
        from PAOFLOW.transport.negf.self_consistency import (
            SelfConsistencySettings,
            check_orthonormal_conductor,
            finite_bias_scan,
        )

        if self.conductor_data is None or self.blc_blocks is None or self._partition is None:
            raise RuntimeError(
                'Call build_hamiltonian_blocks(...) before compute_current_self_consistent(...).'
            )
        if self._partition.dim_l is None or not self._partition.central_atoms:
            raise RuntimeError(
                'finite-bias transport needs a conductor partition: define_partition(..., left_lead_layers=..., '
                "right_lead_layers=...) and build_hamiltonian_blocks(calculation_type='conductor')"
            )
        arrays, attributes = self.data_controller.data_dicts()
        if int(attributes.get('nspin', 1)) != 1:
            raise NotImplementedError('finite-bias self-consistency for spin-polarised inputs')
        check_orthonormal_conductor(self.blc_blocks)
        settings = self._bias_settings or SelfConsistencySettings()
        model = self._electrostatic_model(comm)
        spin_factor = 1.0 if attributes.get('dftSO', False) else 2.0
        scan = finite_bias_scan(
            open_conductor_from(self.conductor_data, self.blc_blocks),
            model,
            settings,
            np.atleast_1d(np.asarray(bias, dtype=float)),
            mu_left_coefficient=mu_L,
            mu_right_coefficient=mu_R,
            profile=profile,
            spin_factor=spin_factor,
            comm=comm,
        )
        if scan.electrostatics == '3d':
            heights = np.linspace(
                model.slab.lower, model.slab.upper, 2 * int(model.slab.length / 0.1) + 1
            )
            scan.planar_average_coordinates = heights
            for point in scan.points:
                point.planar_average_potential = model.planar_average(
                    self._applied_charge(scan, point),
                    point.chemical_potential_left - settings.fermi_energy,
                    point.chemical_potential_right - settings.fermi_energy,
                    heights,
                )
        if write and comm.Get_rank() == 0:
            from PAOFLOW.transport.io.write_data import write_finite_bias_results

            postfix = self._output_config.get('postfix', '')
            write_finite_bias_results(
                output_dir=self._output_config['output_dir'], postfix=postfix, scan=scan
            )
            if potential_grid_biases is not None and scan.electrostatics == '3d':
                self._write_potential_grids(scan, model, potential_grid_biases, settings, postfix)
        if self.results is not None:
            self.results.bias_scan = scan
        return scan

    def _electrostatic_model(self, comm: MPI.Comm) -> Any:
        """The planar layers or the atomic charges of the conductor (``configure_bias_self_consistency``)."""
        from PAOFLOW.transport.negf.atomic_electrostatics import atomic_electrostatics
        from PAOFLOW.transport.negf.charge_model import atomic_charge_widths
        from PAOFLOW.transport.negf.electrostatics import planar_layers
        from PAOFLOW.transport.partition.directions import direction_index

        arrays, attributes = self.data_controller.data_dicts()
        atoms = self._partition.central_atoms
        axis = direction_index(self._partition.transport_direction)
        cell = np.asarray(arrays['a_vectors'], dtype=float) * float(attributes['alat'])
        if self._bias_electrostatics['kind'] == 'planar':
            return planar_layers(atoms, axis, cell)
        widths = atomic_charge_widths(
            [atom.element for atom in atoms],
            self._bias_electrostatics['charge_widths'],
            species=arrays.get('species'),
            attributes=attributes,
        )
        return atomic_electrostatics(atoms, axis, cell, widths, comm=comm)

    @staticmethod
    def _applied_charge(scan: Any, point: Any) -> Any:
        """The charge whose potential acts at a bias point: none for the linear ramp."""
        if scan.profile == 'linear':
            return np.zeros_like(point.group_charge)
        return point.group_charge

    def _write_potential_grids(
        self, scan: Any, model: Any, biases: Any, settings: Any, postfix: str
    ) -> None:
        """Write :math:`U(\\mathbf r)` at the requested biases of a three-dimensional scan (XSF)."""
        import os

        from PAOFLOW.writers.grid_xsf import write_grid_xsf

        for bias in np.atleast_1d(np.asarray(biases, dtype=float)):
            matches = [point for point in scan.points if abs(point.bias - bias) < 1e-9]
            if not matches:
                raise ValueError(f'potential_grid_biases: {bias} V is not a bias of the scan')
            point = matches[0]
            origin, spanning, values = model.potential_on_grid(
                self._applied_charge(scan, point),
                point.chemical_potential_left - settings.fermi_energy,
                point.chemical_potential_right - settings.fermi_energy,
                spacing=self._bias_electrostatics['grid_spacing'],
            )
            write_grid_xsf(
                os.path.join(
                    self._output_config['output_dir'],
                    f'potential_grid_{scan.profile}{postfix}_V{point.bias:+.4f}.xsf',
                ),
                origin,
                spanning,
                values,
                model.labels,
                model.coordinates,
                periodic_axes=(0, 1),
                name='potential_energy_eV',
            )
