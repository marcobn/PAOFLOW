from PAOFLOW import PAOFLOW
from PAOFLOW.Transport import Transport


def main():
    paoflow = PAOFLOW.PAOFLOW(
        savedir='al5.save',
        outputdir='output',
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
        transport_direction=3,
        calculation_type='bulk',
        use_sym=False,
        do_overlap_transformation=False,
        H00_C={'rows': 'ALL', 'cols': 'ALL'},
        H_CR={'rows': 'ALL', 'cols': 'ALL'},
    )

    transport.configure_energy_grid(
        emin=-7.0,
        emax=2.0,
        ne=100,
        delta=0.0005,
    )

    transport.configure_outputs(
        output_dir='./output',
        postfix='_bulk',
    )

    transport.compute_leads_self_energy(write=True)
    transport.compute_greens_functions(write=True)
    transport.compute_transmission(write=True)
    transport.compute_dos(write=True)


if __name__ == '__main__':
    main()
