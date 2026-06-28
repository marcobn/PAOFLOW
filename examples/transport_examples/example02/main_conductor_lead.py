"""Lead/surface transport calculation example.

This example demonstrates a self-energy calculation for a single lead region,
useful for computing surface or interface properties.
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

    # Build Hamiltonian blocks for lead calculation
    transport.build_hamiltonian_blocks(
        datafile_C='./output/qe/alh.save/atomic_proj.xml',
        dimC=12,
        transport_direction=3,
        calculation_type='bulk',
        do_overlap_transformation=False,
        H00_C={'rows': '1-12', 'cols': '1-12'},
        H_CR={'rows': '30-41', 'cols': '1-12'},
    )

    transport.configure_energy_grid(
        emin=-7.0,
        emax=2.0,
        ne=1000,
        delta=0.0005,
    )

    transport.configure_outputs(
        output_dir='./output/paoflow',
        postfix='_lead',
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
