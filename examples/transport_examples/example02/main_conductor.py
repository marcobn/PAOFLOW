"""Bulk transport calculation example.

This example demonstrates a bulk transport calculation with Al system.
"""

from PAOFLOW import PAOFLOW
from PAOFLOW.Transport import Transport
from PAOFLOW.transport.conductor_pipeline import run_conductor
from PAOFLOW.transport.observables.broadening import compute_broadening_matrix


def main() -> None:
    # PAOFLOW setup
    paoflow = PAOFLOW.PAOFLOW(
        savedir='output/qe/alh.save',
        outputdir='output/paoflow',
        smearing='gauss',
        npool=1,
        verbose=True,
        save_overlaps=True,
    )

    # Read projections from QE output
    paoflow.read_atomic_proj_QE()

    # Projectability analysis
    paoflow.projectability()

    # Build PAO Hamiltonian
    paoflow.pao_hamiltonian(shift_type=1, expand_wedge=False)

    # Compute projections
    paoflow.projections()

    # Transport setup
    transport = Transport(paoflow.data_controller)

    # Build Hamiltonian blocks for bulk transport
    transport.build_hamiltonian_blocks(
        datafile_C='./output/qe/alh.save/atomic_proj.xml',
        dimC=41,
        emin=-7.0,
        emax=2.0,
        ne=9001,
        delta=0.0005,
        transport_direction=3,
        output_dir='./output/paoflow',
        postfix='_bulk',
        calculation_type='bulk',
        do_overlap_transformation=False,
        do_eigenchannels=True,
        neigchnx=4,
        do_eigplot=True,
        ie_eigplot=7001,
        ik_eigplot=1,
        write_gf=False,
        write_lead_sgm=False,
        use_sym=False,
        H00_C={'rows': 'ALL', 'cols': 'ALL'},
        H_CR={'rows': 'ALL', 'cols': 'ALL'},
    )

    # Inspect intermediate observables at one (energy, k-point).
    energy_index = 7001
    kpoint_index = 0
    sigma_L, sigma_R, _ = transport.compute_self_energy(ie_g=energy_index, ik=kpoint_index)
    gamma_L = compute_broadening_matrix(sigma_L)
    gamma_R = compute_broadening_matrix(sigma_R)
    gC = transport.compute_green_function(ik=kpoint_index, sigma_L=sigma_L, sigma_R=sigma_R)
    transmission = transport.compute_transmission(
        gC=gC,
        sigma_L=sigma_L,
        sigma_R=sigma_R,
        weighted=True,
    )
    dos = transport.compute_dos(gC=gC, weighted=True)

    # Run conductor calculation
    run_conductor(
        data=transport.conductor_data,
        blc_blocks=transport.blc_blocks,
    )


if __name__ == '__main__':
    main()
