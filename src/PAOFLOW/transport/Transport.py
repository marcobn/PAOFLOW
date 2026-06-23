from __future__ import annotations

import numpy as np
from mpi4py import MPI

import PAOFLOW.transport.io.log_module as log
from PAOFLOW.DataController import DataController
from PAOFLOW.transport.calculators.current import (
    build_bias_grid,
    read_transmittance,
)
from PAOFLOW.transport.conductor_kpoint import compute_kpoint_conductor_quantities
from PAOFLOW.transport.conductor_pipeline import run_conductor
from PAOFLOW.transport.current_pipeline import run_current
from PAOFLOW.transport.grid.egrid import initialize_energy_grid
from PAOFLOW.transport.io.get_input_params import ConductorData
from PAOFLOW.transport.io.write_header import headered_function
from PAOFLOW.transport.utils.memusage import MemoryTracker
from PAOFLOW.transport.utils.timing import global_timing, timed_function
from PAOFLOW.transport.workspace.prepare_data import (
    prepare_conductor,
    prepare_conductor_runtime,
    prepare_current,
)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class ConductorCalculator:
    """
    Driver class for quantum transport calculations in a conductor geometry.

    This class encapsulates the workflow for computing retarded Green's functions,
    lead self-energies, conductance, and density of states (DOS) in a central conductor
    connected to left and right leads.
    """

    def __init__(
        self,
        data: ConductorData,
        *,
        blc_blocks: dict,
    ):
        """
        Initialize a ConductorCalculator.

        Parameters
        ----------
        data : ConductorData
            Input parameters and runtime metadata describing the conductor setup.
        blc_blocks : dict
            Dictionary of OperatorBlock objects holding Hamiltonian and overlap blocks.
        """
        self.data = data
        self.blc_blocks = blc_blocks
        self.vkpt_par3D = data._runtime.vkpt_par3D

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.runtime = data.get_runtime_data()
        self.dimC = data.dimC
        self.dimR = data.dimR
        self.dimL = data.dimL
        self.ne = data.energy.ne
        self.delta = data.energy.delta
        self.nkpts_par = int(self.runtime.nkpts_par)
        self.wk_par = data._runtime.wk_par

        self.ivr_par3D = self.runtime.ivr_par3D
        self.vr_par3D = 2 * np.pi * self.ivr_par3D.astype(np.float64)
        self.nrtot_par = int(self.runtime.nrtot_par)
        self.egrid = initialize_energy_grid(
            emin=data.energy.emin,
            emax=data.energy.emax,
            ne=data.energy.ne,
            carriers=data.carriers,
        )

    @timed_function('do_conductor')
    @headered_function('Frequency Loop')
    def run(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute the full conductor calculation.

        Returns
        -------
        conduct : ndarray of shape (nchannels, ne)
            Energy-resolved conductance, including eigenchannels if requested.
        dos : ndarray of shape (ne,)
            Density of states of the conductor.
        conduct_k : ndarray of shape (nchannels, nkpts_par, ne)
            k-resolved conductance values.
        dos_k : ndarray of shape (ne, nkpts_par)
            k-resolved density of states.

        Notes
        -----
        The main loop distributes energy grid points across MPI ranks.
        At each energy:

        1. Hamiltonian blocks are updated with ``hamiltonian_setup``.
        2. Lead self-energies are computed by iterative surface Green’s function methods.
        3. The conductor Green's function is constructed:

           ``G_C(E) = [ (E + iδ)I - H_C - Σ_L(E) - Σ_R(E) ]⁻¹``

        4. DOS is accumulated as:

           ``DOS(E) = -1/π · Im Tr[G_C(E)]``

        5. Conductance is evaluated via the Landauer formula:

           ``T(E) = Tr[ Γ_L G_C Γ_R G_C† ]``

           where ``Γ_{L/R} = i (Σ_{L/R} - Σ_{L/R}†)``.
        """
        (
            self.conduct,
            self.dos,
            self.conduct_k,
            self.dos_k,
            self.gf_out,
            self.rsgmL_out,
            self.rsgmR_out,
            self.egrid,
        ) = run_conductor(data=self.data, blc_blocks=self.blc_blocks, comm=self.comm)

    def process_kpoint(self, ie_g: int, ik: int):
        """
        Perform the calculation for one energy and one k-point.

        Parameters
        ----------
        ie_g : int
            Index of the energy point.
        ik : int
            Index of the k-point.

        Returns
        -------
        gC : ndarray of shape (dimC, dimC)
            Conductor Green’s function at this (E, k).
        sigma_L : ndarray
            Left lead self-energy.
        sigma_R : ndarray
            Right lead self-energy.
        niter_sum : int
            Total number of Sancho-Rubio iterations performed.
        """
        return compute_kpoint_conductor_quantities(
            data=self.data,
            blc_blocks=self.blc_blocks,
            egrid=self.egrid,
            delta=self.delta,
            ie_g=ie_g,
            ik=ik,
        )


class ConductorRunner:
    @classmethod
    def from_yaml(cls, yaml_file: str, data_controller: DataController) -> 'ConductorRunner':
        data = prepare_conductor(yaml_file, data_controller)
        postfix = data.file_names.postfix
        log.initialize_logger(data_controller, log_file_name=f'transport_conductor{postfix}.log')
        memory_tracker = MemoryTracker()

        ham_sys = prepare_conductor_runtime(data, data_controller, memory_tracker)

        calculator = ConductorCalculator(data=data, blc_blocks=ham_sys.blocks)

        return cls(calculator, memory_tracker)

    def __init__(self, calculator: ConductorCalculator, memory_tracker: MemoryTracker):
        self.calculator = calculator
        self.memory_tracker = memory_tracker

    def finalize(self):
        if self.calculator.rank == 0:
            global_timing.report()
            self.memory_tracker.report(include_real_memory=True)

    def run(self):
        self.calculator.run()
        self.finalize()


class CurrentCalculator:
    def __init__(self, data: dict):
        self.data = data
        self.vgrid = build_bias_grid(data['Vmin'], data['Vmax'], data['nV'])
        self.egrid, self.transm = read_transmittance(data['filein'])
        self.currents = None

    def run(self) -> None:
        self.currents = run_current(
            data=self.data,
            egrid=self.egrid,
            transm=self.transm,
            vgrid=self.vgrid,
        )


class CurrentRunner:
    @classmethod
    def from_yaml(cls, yaml_file: str, data_controller: DataController) -> 'CurrentRunner':
        log.initialize_logger(data_controller, log_file_name='transport_current.log')
        data = prepare_current(yaml_file)
        memory_tracker = MemoryTracker()

        calculator = CurrentCalculator(data)

        return cls(calculator, memory_tracker)

    def __init__(self, calculator: CurrentCalculator, memory_tracker: MemoryTracker):
        self.calculator = calculator
        self.memory_tracker = memory_tracker

    def finalize(self):
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            global_timing.report()
            self.memory_tracker.report(include_real_memory=True)

    def run(self):
        self.calculator.run()
        self.finalize()
