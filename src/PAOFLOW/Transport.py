from __future__ import annotations

from typing import Any

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

import PAOFLOW.transport.io.log_module as log
from PAOFLOW.DataController import DataController
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
    compute_conductor_dos,
    compute_conductor_green,
    compute_conductor_self_energy,
    compute_conductor_transmission,
    prepare_conductor_step_state,
)
from PAOFLOW.transport.io.input_parameters import ConductorData
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
        Cached full-grid transport observables for staged workflow methods.
    """

    def __init__(self, data_controller: DataController) -> None:
        self.data_controller = data_controller
        self._conductor_state: ConductorStepState | None = None
        self.conductor_data: ConductorData | None = None
        self.blc_blocks: dict[str, Any] | None = None
        self.results: TransportResults | None = None

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
        self.results = None
        return state.blc_blocks

    def _require_hamiltonian_blocks(self) -> None:
        if self.conductor_data is None or self.blc_blocks is None:
            raise RuntimeError('Call build_hamiltonian_blocks(...) before transport computations.')

    def _require_step_state(self) -> None:
        if self._conductor_state is None:
            raise RuntimeError('Call build_hamiltonian_blocks(...) before point calculations.')

    def _compute_full_grid_results(
        self,
        *,
        comm: MPI.Comm = MPI.COMM_WORLD,
        require_green_functions: bool = False,
        require_self_energies: bool = False,
    ) -> TransportResults:
        self._require_hamiltonian_blocks()
        if require_green_functions or require_self_energies:
            if (
                not self.conductor_data.symmetry.write_gf
                or not self.conductor_data.symmetry.write_lead_sgm
            ):
                self.conductor_data.symmetry.write_gf = True
                self.conductor_data.symmetry.write_lead_sgm = True
                self.results = None
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

    def compute_transmission_point(
        self,
        *,
        gC: NDArray[np.complex128] | None = None,
        sigma_L: NDArray[np.complex128] | None = None,
        sigma_R: NDArray[np.complex128] | None = None,
        weighted: bool = False,
    ) -> NDArray[np.float64]:
        """Compute transmission channels for one selected point."""
        self._require_step_state()
        return compute_conductor_transmission(
            state=self._conductor_state,
            gC=gC,
            sigma_L=sigma_L,
            sigma_R=sigma_R,
            weighted=weighted,
        )

    def compute_dos_point(
        self,
        *,
        gC: NDArray[np.complex128] | None = None,
        weighted: bool = False,
    ) -> float:
        """Compute DOS contribution for one selected point."""
        self._require_step_state()
        return compute_conductor_dos(state=self._conductor_state, gC=gC, weighted=weighted)

    def compute_self_energy(
        self,
        *,
        write: bool = True,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> tuple[NDArray[np.complex128] | None, NDArray[np.complex128] | None]:
        """Compute full-grid lead self-energies and optionally write XML outputs."""
        results = self._compute_full_grid_results(
            comm=comm,
            require_self_energies=True,
        )
        if write:
            write_self_energy_results(
                data=self.conductor_data,
                results=results,
                comm=comm,
            )
        return results.self_energy_L, results.self_energy_R

    def compute_greens_function(
        self,
        *,
        write: bool = True,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> NDArray[np.complex128] | None:
        """Compute full-grid conductor Green's functions and optionally write XML output."""
        results = self._compute_full_grid_results(
            comm=comm,
            require_green_functions=True,
        )
        if write:
            write_greens_function_results(
                data=self.conductor_data,
                results=results,
                comm=comm,
            )
        return results.green_functions

    def compute_greens_functions(
        self,
        *,
        write: bool = True,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> NDArray[np.complex128] | None:
        """Alias of ``compute_greens_function`` with plural naming."""
        return self.compute_greens_function(write=write, comm=comm)

    def compute_transmission(
        self,
        *,
        write: bool = True,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> NDArray[np.float64]:
        """Compute full-grid transmission and optionally write output files."""
        results = self._compute_full_grid_results(comm=comm)
        if write:
            write_transmission_results(
                data=self.conductor_data,
                results=results,
                comm=comm,
            )
        return results.transmission

    def compute_dos(
        self,
        *,
        write: bool = True,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> NDArray[np.float64]:
        """Compute full-grid DOS and optionally write output files."""
        results = self._compute_full_grid_results(comm=comm)
        if write:
            write_dos_results(
                data=self.conductor_data,
                results=results,
                comm=comm,
            )
        return results.dos
