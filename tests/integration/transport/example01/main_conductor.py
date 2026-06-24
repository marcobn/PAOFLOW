from mpi4py import MPI

from PAOFLOW import PAOFLOW
from PAOFLOW.Transport import Transport
from PAOFLOW.transport.conductor_pipeline import run_conductor

comm = MPI.COMM_WORLD


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
        emin=-7.0,
        emax=2.0,
        ne=100,
        delta=0.0005,
        transport_direction=3,
        output_dir='./output',
        postfix='_bulk',
        calculation_type='bulk',
        write_gf=True,
        write_lead_sgm=True,
        use_sym=False,
        do_overlap_transformation=False,
        H00_C={'rows': 'ALL', 'cols': 'ALL'},
        H_CR={'rows': 'ALL', 'cols': 'ALL'},
    )

    run_conductor(
        data=transport.conductor_data,
        blc_blocks=transport.blc_blocks,
    )


if __name__ == '__main__':
    main()
