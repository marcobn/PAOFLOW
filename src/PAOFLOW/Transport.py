from __future__ import annotations

from typing import Any

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

import PAOFLOW.transport.io.log_module as log
from PAOFLOW.DataController import DataController
from PAOFLOW.transport.calculators.current import (
    build_bias_grid,
    compute_current_vs_bias,
)
from PAOFLOW.transport.conductor_pipeline import (
    compute_conductor_results,
    write_dos_results,
    write_greens_function_results,
    write_self_energy_results,
    write_transmission_results,
)
from PAOFLOW.transport.conductor_steps import (
    ConductorStepState,
    build_conductor_blocks,
    build_conductor_input_values,
    compute_conductor_green,
    compute_conductor_self_energy,
    prepare_conductor_step_state,
)
from PAOFLOW.transport.data import ConductorData, SmearingType
from PAOFLOW.transport.grid.egrid import initialize_energy_grid
from PAOFLOW.transport.io.write_data import write_current_results
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
        self._conductor_state: ConductorStepState | None = None
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
            2D k-grid dimensions. Default is ``(0, 0)``.
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
            if self._conductor_state is not None:
                self._conductor_state.energy_grid = initialize_energy_grid(
                    emin=emin,
                    emax=emax,
                    ne=ne,
                    carriers=self.conductor_data.carriers,
                )
        self.results = None

    def configure_outputs(
        self,
        *,
        output_dir: str = './',
        work_dir: str = './',
        postfix: str = '',
        write_kdata: bool = False,
        write_green_function: bool = False,
        write_lead_self_energy: bool = False,
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
        write_green_function : bool, optional
            If ``True``, compute and write the real-space conductor Green's
            function to ``greenf.xml``. This increases memory usage proportional
            to ``ne * nrtot_par * dimC * dimC``.
        write_lead_self_energy : bool, optional
            If ``True``, compute and write real-space lead self-energies to
            ``lead_L_sgm.xml`` and ``lead_R_sgm.xml``. This increases memory
            usage proportional to ``ne * nrtot_par * dimC * dimC``.
        """
        self._output_config = {
            'output_dir': output_dir,
            'work_dir': work_dir,
            'postfix': postfix,
            'write_kdata': write_kdata,
            'write_green_function': write_green_function,
            'write_lead_self_energy': write_lead_self_energy,
        }
        if self.conductor_data is not None:
            self.conductor_data.file_names.output_dir = output_dir
            self.conductor_data.file_names.work_dir = work_dir
            self.conductor_data.file_names.postfix = postfix
            self.conductor_data.symmetry.write_kdata = write_kdata
            self.conductor_data.symmetry.write_gf = write_green_function
            self.conductor_data.symmetry.write_lead_sgm = write_lead_self_energy
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

    def build_hamiltonian_blocks(
        self,
        *,
        datafile_C: str,
        dimC: int,
        dimL: int | None = None,
        dimR: int | None = None,
        datafile_L: str | None = None,
        datafile_R: str | None = None,
        transport_direction: int = 1,
        calculation_type: str = 'bulk',
        carriers: str = 'electrons',
        use_sym: bool = False,
        do_overlap_transformation: bool = False,
        debug: bool = False,
        surface: bool = False,
        ispin: int = 0,
        niterx: int = 200,
        transfer_thr: float = 1.0e-7,
        nprint: int = 20,
        nfailx: int = 5,
        shift_L: float = 0.0,
        shift_C: float = 0.0,
        shift_R: float = 0.0,
        shift_corr: float = 0.0,
        do_eigenchannels: bool = False,
        neigchnx: int = 200000,
        do_eigplot: bool = False,
        ie_eigplot: int = 0,
        ik_eigplot: int = 0,
        H00_C: dict[str, Any] | None = None,
        H_CR: dict[str, Any] | None = None,
        H_CL: dict[str, Any] | None = None,
        H_LC: dict[str, Any] | None = None,
        H00_L: dict[str, Any] | None = None,
        H01_L: dict[str, Any] | None = None,
        H00_R: dict[str, Any] | None = None,
        H01_R: dict[str, Any] | None = None,
        **block_options: Any,
    ) -> dict[str, Any]:
        """Build conductor Hamiltonian blocks from direct arguments.

        Accepts only parameters required to construct Hamiltonian/overlap
        block objects and lead-device partitions.  Energy-grid parameters
        (``emin``, ``emax``, ``ne``, ``delta``), output paths, and optional
        operator outputs must be configured separately via
        ``configure_energy_grid(...)``, ``configure_outputs(...)``, and
        ``configure_transport_options(...)``.

        Parameters
        ----------
        datafile_C : str
            Path to the conductor Hamiltonian/projection input.
        dimC : int
            Conductor block dimension.
        dimL : int or None, optional
            Left lead block dimension. Leave as ``None`` for bulk mode.
        dimR : int or None, optional
            Right lead block dimension. Leave as ``None`` for bulk mode.
        datafile_L : str or None, optional
            Path to the left-lead input for non-bulk calculations.
        datafile_R : str or None, optional
            Path to the right-lead input for non-bulk calculations.
        transport_direction : int, optional
            Transport direction index in ``{1, 2, 3}``.
        calculation_type : str, optional
            Calculation mode: ``'bulk'`` or ``'conductor'``.
        carriers : str, optional
            Carrier type (``'electrons'`` or ``'phonons'``).
        use_sym : bool, optional
            Apply time-reversal symmetry to reduce the k-point sampling.
        do_overlap_transformation : bool, optional
            Apply the overlap-matrix orthogonalisation transformation.
        debug : bool, optional
            If ``True``, write human-inspectable debug artifacts during setup:
            the legacy ``.ham`` file, ``projectability.txt``, and (when overlap
            is enabled) ``kovp.txt``. Off by default.
        surface : bool, optional
            If ``True``, run in surface mode (lead surface Green's function only,
            skipping the right-lead self-energy). Default ``False``.
        ispin : int, optional
            Spin channel to use for spin-polarized inputs (``0``, ``1``, or ``2``).
        niterx : int, optional
            Maximum number of lead transfer-matrix iterations. Default ``200``.
        transfer_thr : float, optional
            Convergence threshold for the transfer-matrix iteration. Default ``1.0e-7``.
        nprint : int, optional
            Iteration-progress print frequency. Default ``20``.
        nfailx : int, optional
            Maximum number of allowed convergence failures. Default ``5``.
        shift_L, shift_C, shift_R : float, optional
            Rigid on-site energy shifts (eV) applied to the left lead, conductor,
            and right lead blocks respectively. Default ``0.0``.
        shift_corr : float, optional
            On-site energy shift (eV) applied to the correlation self-energy.
            Default ``0.0``.
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
        H00_C : dict or None, optional
            Row/column selectors for the conductor on-site block.
        H_CR : dict or None, optional
            Row/column selectors for the conductor–right-lead coupling block.
        H_CL : dict or None, optional
            Row/column selectors for the conductor–left-lead coupling block.
        H_LC : dict or None, optional
            Row/column selectors for the left-lead–conductor coupling block.
        H00_L : dict or None, optional
            Row/column selectors for the left-lead on-site block.
        H01_L : dict or None, optional
            Row/column selectors for the left-lead hopping block.
        H00_R : dict or None, optional
            Row/column selectors for the right-lead on-site block.
        H01_R : dict or None, optional
            Row/column selectors for the right-lead hopping block.
        **block_options : Any
            Additional block-construction options forwarded to ``ConductorData``
            (for example ``niterx``, ``transfer_thr``, or self-energy file paths
            for generalized formulas).

        Returns
        -------
        dict[str, Any]
            Block-operator mapping used by conductor self-energy and
            Green-function calculations. Also stored as ``self.blc_blocks``.

        Notes
        -----
        Sets ``self.conductor_data``, ``self.blc_blocks``, and
        ``self._conductor_state`` as side effects. Calling this method a second
        time resets all three for the new calculation.
        """
        hamiltonian_selectors: dict[str, Any] = {}
        for h_name, h_val in [
            ('H00_C', H00_C),
            ('H_CR', H_CR),
            ('H_CL', H_CL),
            ('H_LC', H_LC),
            ('H00_L', H00_L),
            ('H01_L', H01_L),
            ('H00_R', H00_R),
            ('H01_R', H01_R),
        ]:
            if h_val is not None:
                hamiltonian_selectors[h_name] = h_val

        input_values = build_conductor_input_values(
            datafile_C=datafile_C,
            dimC=dimC,
            dimL=dimL,
            dimR=dimR,
            datafile_L=datafile_L,
            datafile_R=datafile_R,
            transport_direction=transport_direction,
            calculation_type=calculation_type,
            carriers=carriers,
            formula=self._transport_options_config.get('formula', 'landauer'),
            work_dir=self._output_config.get('work_dir', './'),
            output_dir=self._output_config.get('output_dir', './'),
            postfix=self._output_config.get('postfix', ''),
            use_sym=use_sym,
            do_overlap_transformation=do_overlap_transformation,
            debug=debug,
            surface=surface,
            ispin=ispin,
            niterx=niterx,
            transfer_thr=transfer_thr,
            nprint=nprint,
            nfailx=nfailx,
            shift_L=shift_L,
            shift_C=shift_C,
            shift_R=shift_R,
            shift_corr=shift_corr,
            do_eigenchannels=do_eigenchannels,
            neigchnx=neigchnx,
            do_eigplot=do_eigplot,
            ie_eigplot=ie_eigplot,
            ik_eigplot=ik_eigplot,
            **hamiltonian_selectors,
            **block_options,
        )

        if self._energy_grid_config is not None:
            input_values.update(self._energy_grid_config)

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

        # Apply output flags configured before build_hamiltonian_blocks was called
        write_green_function = self._output_config.get('write_green_function', False)
        write_lead_self_energy = self._output_config.get('write_lead_self_energy', False)
        state.data.symmetry.write_gf = write_green_function
        state.data.symmetry.write_lead_sgm = write_lead_self_energy
        state.data.symmetry.write_kdata = self._output_config.get('write_kdata', False)

        self.conductor_data = state.data
        self.blc_blocks = state.blc_blocks
        self._conductor_state = state
        self.results = None
        return state.blc_blocks

    def _require_hamiltonian_blocks(self) -> None:
        if self.conductor_data is None or self.blc_blocks is None:
            raise RuntimeError('Call build_hamiltonian_blocks(...) before transport computations.')

    def _require_step_state(self) -> None:
        if self._conductor_state is None:
            raise RuntimeError('Call build_hamiltonian_blocks(...) before point calculations.')

    def _require_grid_config(self) -> None:
        if self._energy_grid_config is None:
            raise RuntimeError(
                'Call configure_energy_grid(...) before full-grid transport calculations.'
            )

    def _compute_full_grid_results(
        self,
        *,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> TransportResults:
        self._require_hamiltonian_blocks()
        self._require_grid_config()
        if self.results is None:
            self.results = compute_conductor_results(
                data=self.conductor_data,
                blc_blocks=self.blc_blocks,
                comm=comm,
            )
        return self.results

    def compute_self_energy_point(
        self,
        *,
        ie_g: int,
        ik: int,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], int]:
        """Compute lead self-energies for one ``(E, k)`` point."""
        self._require_step_state()
        return compute_conductor_self_energy(
            state=self._conductor_state,
            ie_g=ie_g,
            ik=ik,
        )

    def compute_greens_function_point(
        self,
        *,
        ik: int,
        sigma_L: NDArray[np.complex128] | None = None,
        sigma_R: NDArray[np.complex128] | None = None,
    ) -> NDArray[np.complex128]:
        """Compute conductor retarded Green's function for one k-point."""
        self._require_step_state()
        return compute_conductor_green(
            state=self._conductor_state,
            ik=ik,
            sigma_L=sigma_L,
            sigma_R=sigma_R,
        )

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
        self._require_hamiltonian_blocks()
        self._require_grid_config()
        if write:
            if self.conductor_data is not None:
                self.conductor_data.symmetry.write_lead_sgm = True
            if self.results is not None and self.results.self_energy_L is None:
                self.results = None
        results = self._compute_full_grid_results(comm=comm)
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
        self._require_hamiltonian_blocks()
        self._require_grid_config()
        if write:
            if self.conductor_data is not None:
                self.conductor_data.symmetry.write_gf = True
            if self.results is not None and self.results.green_functions is None:
                self.results = None
        results = self._compute_full_grid_results(comm=comm)
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
        results = self._compute_full_grid_results(comm=comm)
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
        results = self._compute_full_grid_results(comm=comm)
        write_dos_results(
            data=self.conductor_data,
            results=results,
            comm=comm,
        )
        return results.dos

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
        Writes ``current.dat`` as two columns ``V I`` under the configured
        output directory.
        """
        if nbias <= 0:
            raise ValueError(f'nbias must be positive, got {nbias}.')
        if bias_max <= bias_min:
            raise ValueError(f'bias_max ({bias_max}) must be greater than bias_min ({bias_min}).')
        if sigma <= 0:
            raise ValueError(f'sigma must be positive, got {sigma}.')
        if not (np.isfinite(mu_L) and np.isfinite(mu_R)):
            raise ValueError('mu_L and mu_R must be finite floats.')

        results = self._compute_full_grid_results(comm=comm)
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
