from __future__ import annotations

from typing import Any

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

import PAOFLOW.transport.io.log_module as log
from PAOFLOW.DataController import DataController
from PAOFLOW.transport.conductor_pipeline import run_conductor
from PAOFLOW.transport.current_pipeline import run_current_from_file
from PAOFLOW.transport.calculators.green import compute_conductor_green_function
from PAOFLOW.transport.calculators.leads_self_energy import build_self_energies_from_blocks
from PAOFLOW.transport.calculators.transmittance import evaluate_transmittance
from PAOFLOW.transport.hamiltonian.hamiltonian_setup import hamiltonian_setup
from PAOFLOW.transport.grid.egrid import initialize_energy_grid
from PAOFLOW.transport.io.input_parameters import ConductorData
from PAOFLOW.transport.utils.memusage import MemoryTracker
from PAOFLOW.transport.utils.timing import global_timing
from PAOFLOW.transport.workspace.prepare_data import (
    prepare_conductor_data,
    prepare_conductor_runtime,
)


class Transport:
    """User-facing transport orchestrator with direct-argument APIs.

    Parameters
    ----------
    data_controller : DataController
        Shared PAOFLOW ``DataController`` used by transport preparation stages.
    """

    def __init__(self, data_controller: DataController):
        self.data_controller = data_controller
        self._conductor_data: ConductorData | None = None
        self._conductor_blocks: dict[str, Any] | None = None
        self._memory_tracker: MemoryTracker | None = None
        self._energy_grid: NDArray[np.float64] | None = None
        self._last_sigma_L: NDArray[np.complex128] | None = None
        self._last_sigma_R: NDArray[np.complex128] | None = None
        self._last_gC: NDArray[np.complex128] | None = None
        self._last_ik: int | None = None

    def _build_conductor_input_values(
        self,
        *,
        datafile_C: str,
        dimC: int,
        dimL: int,
        dimR: int,
        datafile_L: str,
        datafile_R: str,
        emin: float,
        emax: float,
        ne: int,
        delta: float,
        nk: list[int] | tuple[int, int] = (0, 0),
        formula: str = 'landauer',
        transport_direction: int = 1,
        carriers: str = 'electrons',
        work_dir: str = './',
        output_dir: str = './',
        postfix: str = '',
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build ``ConductorData`` constructor inputs from direct arguments."""
        input_values: dict[str, Any] = {
            'datafile_C': datafile_C,
            'datafile_L': datafile_L,
            'datafile_R': datafile_R,
            'dimC': dimC,
            'dimL': dimL,
            'dimR': dimR,
            'emin': emin,
            'emax': emax,
            'ne': ne,
            'delta': delta,
            'nk': list(nk),
            'conduct_formula': formula,
            'transport_direction': transport_direction,
            'carriers': carriers,
            'work_dir': work_dir,
            'output_dir': output_dir,
            'postfix': postfix,
        }
        input_values.update(kwargs)
        return input_values

    def prepare(
        self,
        *,
        datafile_C: str,
        dimC: int,
        dimL: int,
        dimR: int,
        datafile_L: str,
        datafile_R: str,
        emin: float,
        emax: float,
        ne: int,
        delta: float,
        nk: list[int] | tuple[int, int] = (0, 0),
        formula: str = 'landauer',
        transport_direction: int = 1,
        carriers: str = 'electrons',
        work_dir: str = './',
        output_dir: str = './',
        postfix: str = '',
        **kwargs: Any,
    ) -> ConductorData:
        """Prepare validated conductor input and runtime metadata.

        Returns
        -------
        ConductorData
            Prepared conductor input model stored on this ``Transport`` instance.
        """
        input_values = self._build_conductor_input_values(
            datafile_C=datafile_C,
            dimC=dimC,
            dimL=dimL,
            dimR=dimR,
            datafile_L=datafile_L,
            datafile_R=datafile_R,
            emin=emin,
            emax=emax,
            ne=ne,
            delta=delta,
            nk=nk,
            formula=formula,
            transport_direction=transport_direction,
            carriers=carriers,
            work_dir=work_dir,
            output_dir=output_dir,
            postfix=postfix,
            **kwargs,
        )

        data = ConductorData(filename='<direct-arguments>', validate=True, **input_values)
        log.initialize_logger(
            self.data_controller,
            log_file_name=f'transport_conductor{data.file_names.postfix}.log',
        )
        memory_tracker = MemoryTracker()
        prepare_conductor_data(data, self.data_controller)

        self._conductor_data = data
        self._memory_tracker = memory_tracker
        self._conductor_blocks = None
        self._last_sigma_L = None
        self._last_sigma_R = None
        self._last_gC = None
        self._last_ik = None
        self._energy_grid = initialize_energy_grid(
            emin=data.energy.emin,
            emax=data.energy.emax,
            ne=data.energy.ne,
            carriers=data.carriers,
        )
        return data

    def build_blocks(self) -> dict[str, Any]:
        """Build Hamiltonian blocks for the prepared conductor workflow."""
        if self._conductor_data is None or self._memory_tracker is None:
            raise RuntimeError('Call prepare(...) before build_blocks().')
        hamiltonian_system = prepare_conductor_runtime(
            self._conductor_data,
            self.data_controller,
            self._memory_tracker,
        )
        self._conductor_blocks = hamiltonian_system.blocks
        return self._conductor_blocks

    def compute_self_energy(
        self,
        *,
        ie_g: int,
        ik: int,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], int]:
        """Compute lead self-energies for one ``(E, k)`` point."""
        if self._conductor_data is None or self._conductor_blocks is None:
            raise RuntimeError('Call prepare(...) and build_blocks() before compute_self_energy().')
        if self._energy_grid is None:
            raise RuntimeError(
                'Energy grid is unavailable. Call prepare(...) before compute_self_energy().'
            )

        hamiltonian_setup(
            ik=ik,
            ie_g=ie_g,
            egrid=self._energy_grid,
            shift_L=self._conductor_data.shift_L,
            shift_C=self._conductor_data.shift_C,
            shift_R=self._conductor_data.shift_R,
            shift_C_corr=getattr(self._conductor_data, 'shift_corr', 0.0),
            blc_blocks=self._conductor_blocks,
            ie_buff=1,
        )

        sigma_R, sigma_L, niter_R, niter_L = build_self_energies_from_blocks(
            blc_00R=self._conductor_blocks['blc_00R'].at_k(ik),
            blc_01R=self._conductor_blocks['blc_01R'].at_k(ik),
            blc_00L=self._conductor_blocks['blc_00L'].at_k(ik),
            blc_01L=self._conductor_blocks['blc_01L'].at_k(ik),
            blc_CR=self._conductor_blocks['blc_CR'].at_k(ik),
            blc_LC=self._conductor_blocks['blc_LC'].at_k(ik),
            leads_are_identical=self._conductor_data.advanced.leads_are_identical,
            delta=self._conductor_data.energy.delta,
            niterx=self._conductor_data.iteration.niterx,
            transfer_thr=self._conductor_data.iteration.transfer_thr,
            fail_counter=None,
            fail_limit=self._conductor_data.iteration.nfailx,
            verbose=False,
        )
        total_iterations = niter_R + (
            niter_L if not self._conductor_data.advanced.leads_are_identical else 0
        )
        self._last_sigma_L = sigma_L
        self._last_sigma_R = sigma_R
        self._last_ik = ik
        return sigma_L, sigma_R, total_iterations

    def compute_green_function(
        self,
        *,
        ik: int,
        sigma_L: NDArray[np.complex128] | None = None,
        sigma_R: NDArray[np.complex128] | None = None,
    ) -> NDArray[np.complex128]:
        """Compute conductor retarded Green's function for one k-point."""
        if self._conductor_data is None or self._conductor_blocks is None:
            raise RuntimeError(
                'Call prepare(...) and build_blocks() before compute_green_function().'
            )

        sigma_left = sigma_L if sigma_L is not None else self._last_sigma_L
        sigma_right = sigma_R if sigma_R is not None else self._last_sigma_R
        if sigma_left is None:
            raise RuntimeError(
                'sigma_L is required. Call compute_self_energy(...) first or pass sigma_L explicitly.'
            )

        g_c = compute_conductor_green_function(
            blc_00C=self._conductor_blocks['blc_00C'].at_k(ik),
            sigma_l=sigma_left,
            sigma_r=sigma_right if not self._conductor_data.advanced.surface else None,
            delta=self._conductor_data.energy.delta,
            surface=self._conductor_data.advanced.surface,
        )
        self._last_gC = g_c
        self._last_ik = ik
        return g_c

    def compute_transmission(
        self,
        *,
        gC: NDArray[np.complex128] | None = None,
        sigma_L: NDArray[np.complex128] | None = None,
        sigma_R: NDArray[np.complex128] | None = None,
        weighted: bool = False,
    ) -> NDArray[np.float64]:
        """Compute transmission channels from Green's function and self-energies."""
        if self._conductor_data is None:
            raise RuntimeError('Call prepare(...) before compute_transmission().')

        g_ret = gC if gC is not None else self._last_gC
        sigma_left = sigma_L if sigma_L is not None else self._last_sigma_L
        sigma_right = sigma_R if sigma_R is not None else self._last_sigma_R
        if g_ret is None or sigma_left is None or sigma_right is None:
            raise RuntimeError(
                'gC, sigma_L, and sigma_R are required. Compute self-energy and Green function first.'
            )

        gamma_L = 1j * (sigma_left - sigma_left.conj().T)
        gamma_R = 1j * (sigma_right - sigma_right.conj().T)
        channels, _ = evaluate_transmittance(
            gamma_L=gamma_L,
            gamma_R=gamma_R,
            G_ret=g_ret,
            formula=self._conductor_data.conduct_formula,
            do_eigenchannels=self._conductor_data.symmetry.do_eigenchannels,
            do_eigplot=False,
            sgm_corr=None,
            eta=self._conductor_data.energy.delta,
            S_overlap=None,
        )
        if weighted:
            if self._last_ik is None:
                raise RuntimeError('weighted=True requires a known k-point from previous calls.')
            return self._conductor_data._runtime.wk_par[self._last_ik] * channels
        return channels

    def compute_dos(
        self,
        *,
        gC: NDArray[np.complex128] | None = None,
        weighted: bool = False,
    ) -> float:
        """Compute DOS contribution from a conductor Green's function."""
        if self._conductor_data is None:
            raise RuntimeError('Call prepare(...) before compute_dos().')

        g_ret = gC if gC is not None else self._last_gC
        if g_ret is None:
            raise RuntimeError('gC is required. Call compute_green_function(...) first or pass gC.')

        dos_value = -np.sum(np.imag(np.diagonal(g_ret))) / np.pi
        if weighted:
            if self._last_ik is None:
                raise RuntimeError('weighted=True requires a known k-point from previous calls.')
            dos_value *= self._conductor_data._runtime.wk_par[self._last_ik]
        return float(dos_value)

    def conductor(
        self,
        *,
        datafile_C: str,
        dimC: int,
        dimL: int,
        dimR: int,
        datafile_L: str,
        datafile_R: str,
        emin: float,
        emax: float,
        ne: int,
        delta: float,
        nk: list[int] | tuple[int, int] = (0, 0),
        formula: str = 'landauer',
        transport_direction: int = 1,
        carriers: str = 'electrons',
        work_dir: str = './',
        output_dir: str = './',
        postfix: str = '',
        **kwargs: Any,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.complex128] | None,
        NDArray[np.complex128] | None,
        NDArray[np.complex128] | None,
        NDArray[np.float64],
    ]:
        """Run conductor transport from direct Python arguments.

        Parameters
        ----------
        datafile_C : str
            Path to the conductor Hamiltonian/projection input.
        dimC : int
            Conductor block dimension.
        dimL : int
            Left lead block dimension.
        dimR : int
            Right lead block dimension.
        datafile_L : str
            Path to the left-lead input.
        datafile_R : str
            Path to the right-lead input.
        emin : float
            Minimum energy in eV.
        emax : float
            Maximum energy in eV.
        ne : int
            Number of energy points.
        delta : float
            Broadening parameter.
        nk : list[int] or tuple[int, int], optional
            2D k-grid parameters for ``kpoint_grid.nk``.
        formula : str, optional
            Conductance formula. Default is ``'landauer'``.
        transport_direction : int, optional
            Transport direction index in ``{1, 2, 3}``.
        carriers : str, optional
            Carrier type (``'electrons'`` or ``'phonons'``).
        work_dir : str, optional
            Working directory for transport assets.
        output_dir : str, optional
            Output directory for generated transport files.
        postfix : str, optional
            Output postfix appended to default transport file names.
        **kwargs : Any
            Additional optional ``ConductorData`` fields (for example
            ``write_kdata``, ``write_gf``, ``niterx``, ``transfer_thr``, or
            self-energy file paths for generalized formulas).

        Returns
        -------
        tuple
            ``(conduct, dos, conduct_k, dos_k, gf_out, rsgmL_out, rsgmR_out, egrid)``
            returned by ``run_conductor``.
        """
        data = self.prepare(
            datafile_C=datafile_C,
            dimC=dimC,
            dimL=dimL,
            dimR=dimR,
            datafile_L=datafile_L,
            datafile_R=datafile_R,
            emin=emin,
            emax=emax,
            ne=ne,
            delta=delta,
            nk=nk,
            formula=formula,
            transport_direction=transport_direction,
            carriers=carriers,
            work_dir=work_dir,
            output_dir=output_dir,
            postfix=postfix,
            **kwargs,
        )
        blocks = self.build_blocks()
        results = run_conductor(data=data, blc_blocks=blocks, comm=MPI.COMM_WORLD)

        if MPI.COMM_WORLD.Get_rank() == 0:
            global_timing.report()
            if self._memory_tracker is not None:
                self._memory_tracker.report(include_real_memory=True)

        return results

    def current(
        self,
        *,
        filein: str,
        fileout: str,
        bias_min: float,
        bias_max: float,
        nbias: int,
        sigma: float,
        mu_L: float,
        mu_R: float,
    ) -> NDArray[np.float64]:
        """Run current-vs-bias calculation from direct Python arguments.

        Parameters
        ----------
        filein : str
            Input transmission file path.
        fileout : str
            Output path for current-vs-bias data.
        bias_min : float
            Minimum bias in volts.
        bias_max : float
            Maximum bias in volts.
        nbias : int
            Number of bias samples.
        sigma : float
            Smearing parameter used in Fermi occupations.
        mu_L : float
            Left chemical potential.
        mu_R : float
            Right chemical potential.

        Returns
        -------
        NDArray[np.float64]
            Current values aligned with the generated bias grid.
        """
        log.initialize_logger(self.data_controller, log_file_name='transport_current.log')

        current_data = {
            'fileout': fileout,
            'mu_L': mu_L,
            'mu_R': mu_R,
            'sigma': sigma,
        }
        return run_current_from_file(
            data=current_data,
            filein=filein,
            bias_min=bias_min,
            bias_max=bias_max,
            nbias=nbias,
        )
