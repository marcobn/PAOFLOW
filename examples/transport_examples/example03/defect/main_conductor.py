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

    transport.build_hamiltonian_blocks(
        datafile_C='./output/qe/al.save/atomic_proj.xml',
        datafile_L='./output/qe/al5.save/atomic_proj.xml',
        datafile_R='./output/qe/al5.save/atomic_proj.xml',
        dimC=52,
        dimL=20,
        dimR=20,
        emin=-7.0,
        emax=2.0,
        ne=6001,
        delta=0.0005,
        transport_direction=3,
        output_dir='./output/paoflow',
        postfix='_defect',
        calculation_type='conductor',
        do_overlap_transformation=False,
        write_gf=False,
        write_lead_sgm=False,
        use_sym=False,
        H00_C={'rows': 'ALL', 'cols': 'ALL'},
        H_CR={'rows': 'ALL', 'cols': '1-20'},
        H_LC={'rows': '33-52', 'cols': 'ALL'},
        H00_L={'rows': '1-20', 'cols': '1-20'},
        H01_L={'rows': '33-52', 'cols': '1-20'},
        H00_R={'rows': '1-20', 'cols': '1-20'},
        H01_R={'rows': '33-52', 'cols': '1-20'},
    )

    transport.compute_self_energy(write=True, comm=comm)
    transport.compute_greens_functions(write=True, comm=comm)
    transmission = transport.compute_transmission(write=True, comm=comm)
    dos = transport.compute_dos(write=True, comm=comm)
    print('Transmission shape:', transmission.shape)
    print('DOS shape:', dos.shape)


if __name__ == '__main__':
    main()
