from __future__ import annotations

from mpi4py import MPI

from PAOFLOW import PAOFLOW
from PAOFLOW.Transport import Transport

comm = MPI.COMM_WORLD


def main() -> None:
    paoflow = PAOFLOW.PAOFLOW(
        savedir='output/qe/al.save',
        outputdir='output/paoflow',
        smearing='gauss',
        npool=1,
        verbose=True,
        save_overlaps=True,
    )

    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian(shift_type=1, expand_wedge=False)
    paoflow.projections()

    transport = Transport(paoflow.data_controller)

    transport.define_partition(
        central_atoms='ALL',
        left_lead_layers=5,
        right_lead_layers=5,
        transport_direction='z',
    )

    transport.build_hamiltonian_blocks(
        calculation_type='conductor',
        use_sym=False,
        do_overlap_transformation=False,
    )

    # --- optional physics tuning (compute-time; defaults shown, call before compute) ---
    # transport.configure_onsite_shifts(shift_L=0.0, shift_C=0.0, shift_R=0.0, shift_corr=0.0)
    # transport.configure_lead_convergence(niterx=200, transfer_thr=1.0e-7, nprint=20, nfailx=5, surface=False)
    # transport.configure_eigenchannels(do_eigenchannels=False, neigchnx=200000)

    transport.configure_energy_grid(
        emin=-7.0,
        emax=2.0,
        ne=6001,
        delta=0.0005,
    )

    transport.configure_outputs(
        output_dir='./output/paoflow',
        postfix='_defect',
    )
    transport.compute_leads_self_energy(write=True, comm=comm)
    transport.compute_greens_functions(write=True, comm=comm)
    transport.compute_transmission(write=True, comm=comm)
    transport.compute_dos(write=True, comm=comm)


if __name__ == '__main__':
    main()
