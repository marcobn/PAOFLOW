from __future__ import annotations

from pathlib import Path

import numpy as np
from mpi4py import MPI

import PAOFLOW.transport.io.log_module as log
from PAOFLOW.DataController import DataController
from PAOFLOW.transport.calculators.current import (
    build_bias_grid,
    read_transmittance,
)
from PAOFLOW.transport.conductor_energy_loop import (
    process_energy_point,
    reduce_conductor_results,
    transform_k_to_r_at_energy as transform_k_to_r_at_energy_point,
)
from PAOFLOW.transport.conductor_kpoint import compute_kpoint_conductor_quantities
from PAOFLOW.transport.conductor_observables import (
    accumulate_dos as accumulate_dos_contribution,
)
from PAOFLOW.transport.conductor_observables import (
    accumulate_transmission,
)
from PAOFLOW.transport.conductor_outputs import (
    initialize_conductor_outputs,
    initialize_kpoint_operator_buffers,
)
from PAOFLOW.transport.conductor_pipeline import run_conductor
from PAOFLOW.transport.conductor_writers import (
    write_conductor_operators,
    write_conductor_output,
)
from PAOFLOW.transport.current_pipeline import run_current
from PAOFLOW.transport.grid.egrid import initialize_energy_grid
from PAOFLOW.transport.io.get_input_params import ConductorData
from PAOFLOW.transport.io.write_header import headered_function
from PAOFLOW.transport.utils.memusage import MemoryTracker
from PAOFLOW.transport.utils.timing import global_timing, timed_function
from PAOFLOW.transport.workspace.prepare_data import (
    prepare_conductor,
    prepare_current,
    prepare_hamiltonian_blocks_and_leads,
    prepare_hamiltonian_system,
    prepare_kpoints,
    prepare_smearing,
    prepare_workspace,
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

    def initialize_outputs(self):
        """
        Allocate output arrays for conductance, DOS, and optional Green’s functions.

        Returns
        -------
        conduct : ndarray
            Total and eigenchannel conductance vs energy.
        dos : ndarray
            DOS vs energy.
        conduct_k : ndarray
            k-resolved conductance.
        dos_k : ndarray
            k-resolved DOS.
        """
        conduct, dos, conduct_k, dos_k, self.gf_out, self.rsgmL_out, self.rsgmR_out = (
            initialize_conductor_outputs(
                data=self.data,
                dimC=self.dimC,
                dimL=self.dimL,
                dimR=self.dimR,
                ne=self.ne,
                nkpts_par=self.nkpts_par,
                nrtot_par=self.nrtot_par,
            )
        )
        return conduct, dos, conduct_k, dos_k

    def process_energy(self, conduct, dos, conduct_k, dos_k, ie_g: int, ie_start: int, ie_end: int):
        """
        Perform all calculations for a single energy point.

        Returns
        -------
        conduct : ndarray
            Updated conductance.
        dos : ndarray
            Updated DOS.
        """
        process_energy_point(
            data=self.data,
            blc_blocks=self.blc_blocks,
            egrid=self.egrid,
            delta=self.delta,
            ie_g=ie_g,
            ie_start=ie_start,
            ie_end=ie_end,
            rank=self.rank,
            nkpts_par=self.nkpts_par,
            dimC=self.dimC,
            nrtot_par=self.nrtot_par,
            vkpt_par3D=self.vkpt_par3D,
            vr_par3D=self.vr_par3D,
            wk_par=self.wk_par,
            conduct=conduct,
            dos=dos,
            conduct_k=conduct_k,
            dos_k=dos_k,
            gf_out=self.gf_out,
            rsgmL_out=self.rsgmL_out,
            rsgmR_out=self.rsgmR_out,
        )
        return conduct, dos

    def initialize_k_dependent_operators(self):
        """
        Allocate temporary arrays for k-dependent Green's functions and self-energies.

        Returns
        -------
        gC_k : ndarray or None
            Conductor Green’s function at each k-point, if requested.
        sgmL_k : ndarray or None
            Left lead self-energy at each k-point, if requested.
        sgmR_k : ndarray or None
            Right lead self-energy at each k-point, if requested.
        """
        gC_k, sgmL_k, sgmR_k = initialize_kpoint_operator_buffers(
            data=self.data,
            nkpts_par=self.nkpts_par,
            dimC=self.dimC,
        )
        return gC_k, sgmL_k, sgmR_k

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

    def accumulate_dos(self, dos, dos_k, gC, ie_g, ik):
        """
        Accumulate DOS contributions from a given k-point.

        Notes
        -----
        The contribution from each k-point is weighted by its k-point weight:

        ``DOS(E) += -w_k / π · Im Tr[G_C(E, k)]``
        """
        accumulate_dos_contribution(dos, dos_k, gC, self.wk_par, ie_g, ik)

    def accumulate_conductance(self, conduct, conduct_k, gC, sigma_L, sigma_R, ie_g, ik):
        """
        Accumulate conductance contributions from a given k-point.

        Notes
        -----
        Transmission is computed using the Landauer expression:

        ``T(E, k) = Tr[ Γ_L G_C Γ_R G_C† ]``

        where ``Γ_{L/R} = i (Σ_{L/R} - Σ_{L/R}†)``.

        Eigenchannel decomposition is optionally performed by diagonalizing
        ``√Γ_L G_C Γ_R G_C† √Γ_L``.
        """
        accumulate_transmission(
            conduct,
            conduct_k,
            gC,
            sigma_L,
            sigma_R,
            self.wk_par,
            ie_g,
            ik,
            data=self.data,
            delta=self.delta,
            rank=self.rank,
            vkpt=self.vkpt_par3D,
        )

    def transform_k_to_r_at_energy(self, ie_g, gC_k, sgmL_k, sgmR_k):
        """
        Transform k-space Green’s functions and self-energies into real space
        for a given energy index.
        """
        transform_k_to_r_at_energy_point(
            data=self.data,
            ie_g=ie_g,
            gC_k=gC_k,
            sgmL_k=sgmL_k,
            sgmR_k=sgmR_k,
            gf_out=self.gf_out,
            rsgmL_out=self.rsgmL_out,
            rsgmR_out=self.rsgmR_out,
            nrtot_par=self.nrtot_par,
            vr_par3D=self.vr_par3D,
            vkpt_par3D=self.vkpt_par3D,
            wk_par=self.wk_par,
        )

    def reduce_results(self, conduct, dos, conduct_k, dos_k):
        """
        Collect results across MPI ranks by summing over all contributions.

        Notes
        -----
        Calls `MPI.Allreduce` to accumulate conductance, DOS, Green’s functions,
        and lead self-energies across all ranks.
        """
        reduce_conductor_results(
            comm=self.comm,
            data=self.data,
            conduct=conduct,
            dos=dos,
            conduct_k=conduct_k,
            dos_k=dos_k,
            gf_out=self.gf_out,
            rsgmL_out=self.rsgmL_out,
            rsgmR_out=self.rsgmR_out,
        )

    def write_operators(self):
        """
        Write operator data (Green’s functions, lead self-energies) to XML.

        Notes
        -----
        Uses `write_operator_xml` to replicate the Fortran IOTK format exactly.
        """
        write_conductor_operators(
            rank=self.rank,
            data=self.data,
            gf_out=self.gf_out,
            rsgmL_out=self.rsgmL_out,
            rsgmR_out=self.rsgmR_out,
            ivr_par3D=self.ivr_par3D,
            egrid=self.egrid,
            dimC=self.dimC,
        )

    def write_output(self):
        """
        Write final conductance and DOS results to disk.

        Notes
        -----
        - Writes `conductance.dat` and `doscond.dat` for total results.
        - Optionally writes k-resolved data per k-point.
        """
        write_conductor_output(
            rank=self.rank,
            data=self.data,
            conduct=self.conduct,
            dos=self.dos,
            conduct_k=self.conduct_k,
            dos_k=self.dos_k,
            egrid=self.egrid,
        )


class ConductorRunner:
    @classmethod
    def from_yaml(cls, yaml_file: str, data_controller: DataController) -> 'ConductorRunner':
        data = prepare_conductor(yaml_file, data_controller)
        postfix = data.file_names.postfix
        log.initialize_logger(data_controller, log_file_name=f'transport_conductor{postfix}.log')
        memory_tracker = MemoryTracker()

        _ = prepare_smearing(data, memory_tracker)
        _ = prepare_kpoints(data, memory_tracker)
        ham_sys = prepare_hamiltonian_system(data, memory_tracker)
        prepare_hamiltonian_blocks_and_leads(data, ham_sys, data_controller)
        _ = prepare_workspace(data, memory_tracker)

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

    def write_output(self) -> None:
        outpath = Path(self.data['fileout'])
        outpath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(outpath, np.column_stack([self.vgrid, self.currents]))
        log.log_rank0(f'Saved current vs bias to {outpath}')

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
