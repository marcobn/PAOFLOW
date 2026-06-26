from PAOFLOW import PAOFLOW
from PAOFLOW.Transport import Transport


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

    transport.compute_self_energy(write=True)
    transport.compute_greens_functions(write=True)
    transmission = transport.compute_transmission(write=True)
    dos = transport.compute_dos(write=True)
    print('Transmission shape:', transmission.shape)
    print('DOS shape:', dos.shape)


if __name__ == '__main__':
    main()
