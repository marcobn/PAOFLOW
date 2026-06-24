from __future__ import annotations

from typing import Any

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

import PAOFLOW.transport.io.log_module as log
from PAOFLOW.DataController import DataController
from PAOFLOW.transport.conductor_pipeline import run_conductor
from PAOFLOW.transport.conductor_steps import (
    ConductorStepState,
    build_conductor_blocks,
    build_conductor_input_values,
    compute_conductor_dos,
    compute_conductor_green,
    compute_conductor_self_energy,
    compute_conductor_transmission,
    prepare_conductor_step_state,
)
from PAOFLOW.transport.current_pipeline import run_current_from_file
from PAOFLOW.transport.io.input_parameters import ConductorData
from PAOFLOW.transport.utils.timing import global_timing


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
    """

    def __init__(self, data_controller: DataController) -> None:
        self.data_controller = data_controller
        self._conductor_state: ConductorStepState | None = None
        self.conductor_data: ConductorData | None = None
        self.blc_blocks: dict[str, Any] | None = None

    def build_hamiltonian_blocks(
        self,
        *,
        datafile_C: str,
        dimC: int,
        dimL: int | None = None,
        dimR: int | None = None,
        datafile_L: str | None = None,
        datafile_R: str | None = None,
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
        """Build conductor Hamiltonian blocks from direct arguments.

        Merges all setup operations into a single call: builds conductor
        input values, initializes staged state, sets up logging, and
        constructs the Hamiltonian block operators.

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
        dict[str, Any]
            Block-operator mapping used by conductor self-energy and
            Green-function calculations. Also stored as ``self.blc_blocks``.

        Notes
        -----
        Sets ``self.conductor_data``, ``self.blc_blocks``, and
        ``self._conductor_state`` as side effects. Calling this method a
        second time on the same instance resets all three for the new
        calculation.
        """
        input_values = build_conductor_input_values(
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
        self.conductor_data = state.data
        self.blc_blocks = state.blc_blocks
        self._conductor_state = state
        return state.blc_blocks

    def compute_self_energy(
        self,
        *,
        ie_g: int,
        ik: int,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], int]:
        """Compute lead self-energies for one ``(E, k)`` point.

        Returns
        -------
        tuple[NDArray[np.complex128], NDArray[np.complex128], int]
            ``(sigma_L, sigma_R, total_iterations)`` for the requested
            ``(ie_g, ik)`` pair.

        Raises
        ------
        RuntimeError
            If ``build_hamiltonian_blocks`` was not called first.
        """
        if self._conductor_state is None:
            raise RuntimeError('Call build_hamiltonian_blocks(...) before compute_self_energy().')
        return compute_conductor_self_energy(
            state=self._conductor_state,
            ie_g=ie_g,
            ik=ik,
        )

    def compute_green_function(
        self,
        *,
        ik: int,
        sigma_L: NDArray[np.complex128] | None = None,
        sigma_R: NDArray[np.complex128] | None = None,
    ) -> NDArray[np.complex128]:
        """Compute conductor retarded Green's function for one k-point.

        Returns
        -------
        NDArray[np.complex128]
            Retarded conductor Green's function at the selected k-point.

        Raises
        ------
        RuntimeError
            If ``build_hamiltonian_blocks`` was not called first.
        """
        if self._conductor_state is None:
            raise RuntimeError(
                'Call build_hamiltonian_blocks(...) before compute_green_function().'
            )
        return compute_conductor_green(
            state=self._conductor_state,
            ik=ik,
            sigma_L=sigma_L,
            sigma_R=sigma_R,
        )

    def compute_transmission(
        self,
        *,
        gC: NDArray[np.complex128] | None = None,
        sigma_L: NDArray[np.complex128] | None = None,
        sigma_R: NDArray[np.complex128] | None = None,
        weighted: bool = False,
    ) -> NDArray[np.float64]:
        """Compute transmission channels from Green's function and self-energies.

        Returns
        -------
        NDArray[np.float64]
            Total and optional channel-resolved transmission values.

        Raises
        ------
        RuntimeError
            If ``build_hamiltonian_blocks`` was not called first.
        """
        if self._conductor_state is None:
            raise RuntimeError('Call build_hamiltonian_blocks(...) before compute_transmission().')
        return compute_conductor_transmission(
            state=self._conductor_state,
            gC=gC,
            sigma_L=sigma_L,
            sigma_R=sigma_R,
            weighted=weighted,
        )

    def compute_dos(
        self,
        *,
        gC: NDArray[np.complex128] | None = None,
        weighted: bool = False,
    ) -> float:
        """Compute DOS contribution from a conductor Green's function.

        Returns
        -------
        float
            DOS contribution from the selected Green's function.

        Raises
        ------
        RuntimeError
            If ``build_hamiltonian_blocks`` was not called first.
        """
        if self._conductor_state is None:
            raise RuntimeError('Call build_hamiltonian_blocks(...) before compute_dos().')
        return compute_conductor_dos(state=self._conductor_state, gC=gC, weighted=weighted)

    def conductor(
        self,
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

        Compatibility wrapper around ``build_hamiltonian_blocks`` and
        ``run_conductor``.  All keyword arguments are forwarded to
        ``build_hamiltonian_blocks``; see that method for the full parameter
        list.

        Returns
        -------
        tuple
            ``(conduct, dos, conduct_k, dos_k, gf_out, rsgmL_out, rsgmR_out, egrid)``
            returned by ``run_conductor``.

        Notes
        -----
        Sets ``self.conductor_data`` and ``self.blc_blocks`` as side effects
        via ``build_hamiltonian_blocks``.
        """
        self.build_hamiltonian_blocks(**kwargs)
        results = run_conductor(
            data=self.conductor_data,
            blc_blocks=self.blc_blocks,
            comm=MPI.COMM_WORLD,
        )

        if MPI.COMM_WORLD.Get_rank() == 0:
            global_timing.report()
            if self._conductor_state is not None:
                self._conductor_state.memory_tracker.report(include_real_memory=True)

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
