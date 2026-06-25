from PAOFLOW import PAOFLOW
from PAOFLOW.Transport import Transport
from PAOFLOW.transport.conductor_pipeline import run_conductor
from PAOFLOW.transport.observables.broadening import compute_broadening_matrix


def main():
    paoflow = PAOFLOW.PAOFLOW(
        savedir='output/qe/al5.save',
        outputdir='output/paoflow',
        smearing='gauss',
        npool=1,
        verbose=True,
        save_overlaps=True,
    )

    paoflow.read_atomic_proj_QE()
    paoflow.projectability(pthr=0.95)
    paoflow.pao_hamiltonian(
        shift_type=1,
        expand_wedge=False,
    )
    paoflow.projections()

    transport = Transport(paoflow.data_controller)
    transport.build_hamiltonian_blocks(
        datafile_C='./output/qe/al5.save/atomic_proj.xml',
        dimC=20,
        emin=-7.0,
        emax=2.0,
        ne=9001,
        delta=0.0005,
        transport_direction=3,
        output_dir='./output/paoflow',
        postfix='_bulk',
        calculation_type='bulk',
        write_gf=True,
        write_lead_sgm=True,
        use_sym=False,
        do_overlap_transformation=False,
        H00_C={'rows': 'ALL', 'cols': 'ALL'},
        H_CR={'rows': 'ALL', 'cols': 'ALL'},
    )

    energy_index = 7001
    kpoint_index = 0
    sigma_L, sigma_R, _ = transport.compute_self_energy(ie_g=energy_index, ik=kpoint_index)
    _ = compute_broadening_matrix(sigma_L)
    _ = compute_broadening_matrix(sigma_R)
    gC = transport.compute_green_function(ik=kpoint_index, sigma_L=sigma_L, sigma_R=sigma_R)
    _ = transport.compute_transmission(gC=gC, sigma_L=sigma_L, sigma_R=sigma_R, weighted=True)
    _ = transport.compute_dos(gC=gC, weighted=True)

    run_conductor(
        data=transport.conductor_data,
        blc_blocks=transport.blc_blocks,
    )


if __name__ == '__main__':
    main()
