"""Bulk transport calculation example.

This example demonstrates a bulk transport calculation with Al system.
"""

from PAOFLOW import PAOFLOW
from PAOFLOW.Transport import Transport


def _run_case(case: str) -> None:
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

    if case == 'bulk':
        transport.define_blocks(
            H00_C={'rows': 'ALL', 'cols': 'ALL'},
            H_CR={'rows': 'ALL', 'cols': 'ALL'},
        )

        transport.build_hamiltonian_blocks(
            datafile_C='./output/qe/alh.save/atomic_proj.xml',
            dimC=41,
            transport_direction=3,
            calculation_type='bulk',
            use_sym=False,
            do_overlap_transformation=False,
        )

        transport.configure_eigenchannels(
            do_eigenchannels=True,
            neigchnx=4,
            do_eigplot=True,
            ie_eigplot=7001,
            ik_eigplot=1,
        )

        transport.configure_energy_grid(
            emin=-7.0,
            emax=2.0,
            ne=9001,
            delta=0.0005,
        )

        transport.configure_outputs(
            output_dir='./output/paoflow',
            postfix='_bulk',
        )

    elif case == 'lcr':
        transport.define_blocks(
            H00_C={'rows': '1-41', 'cols': '1-41'},
            H_CR={'rows': '1-41', 'cols': '1-12'},
            H_LC={'rows': '30-41', 'cols': '1-41'},
            H00_L={'rows': '1-12', 'cols': '1-12'},
            H01_L={'rows': '30-41', 'cols': '1-12'},
            H00_R={'rows': '1-12', 'cols': '1-12'},
            H01_R={'rows': '30-41', 'cols': '1-12'},
        )

        transport.build_hamiltonian_blocks(
            datafile_C='./output/qe/alh.save/atomic_proj.xml',
            datafile_L='./output/qe/alh.save/atomic_proj.xml',
            datafile_R='./output/qe/alh.save/atomic_proj.xml',
            dimC=41,
            dimL=12,
            dimR=12,
            transport_direction=3,
            calculation_type='conductor',
            do_overlap_transformation=False,
        )

        transport.configure_eigenchannels(
            do_eigenchannels=True,
            neigchnx=4,
            do_eigplot=True,
            ie_eigplot=7001,
            ik_eigplot=1,
        )

        transport.configure_energy_grid(
            emin=-7.0,
            emax=2.0,
            ne=9001,
            delta=0.0005,
        )

        transport.configure_outputs(
            output_dir='./output/paoflow',
            postfix='_lcr',
        )

    elif case == 'lead':
        transport.define_blocks(
            H00_C={'rows': '1-12', 'cols': '1-12'},
            H_CR={'rows': '30-41', 'cols': '1-12'},
        )

        transport.build_hamiltonian_blocks(
            datafile_C='./output/qe/alh.save/atomic_proj.xml',
            dimC=12,
            transport_direction=3,
            calculation_type='bulk',
            do_overlap_transformation=False,
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

    transport.compute_leads_self_energy(write=True)
    transport.compute_greens_functions(write=True)
    transport.compute_transmission()
    transport.compute_dos()
    if case == 'lcr':
        transport.compute_current(
            bias_min=-1.0,
            bias_max=1.0,
            nbias=100,
            mu_L=-0.5,
            mu_R=0.5,
            sigma=0.05,
        )


def main() -> None:
    selectors = ['bulk', 'lcr', 'lead']
    for selector in selectors:
        _run_case(selector)


if __name__ == '__main__':
    main()
