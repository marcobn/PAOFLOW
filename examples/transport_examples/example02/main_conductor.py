"""Bulk transport calculation example.

This example demonstrates a bulk transport calculation with Al system.
"""

from PAOFLOW import PAOFLOW
from PAOFLOW.Transport import Transport


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

    # Compute and write transport observables by physics stage.
    transport.compute_self_energy(write=True)
    transport.compute_greens_functions(write=True)
    transmission = transport.compute_transmission(write=True)
    dos = transport.compute_dos(write=True)
    print('Transmission shape:', transmission.shape)
    print('DOS shape:', dos.shape)


if __name__ == '__main__':
    main()
