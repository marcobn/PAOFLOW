from PAOFLOW import PAOFLOW
from PAOFLOW.Transport import Transport


def _run_case(case: str) -> None:
    paoflow = PAOFLOW.PAOFLOW(
        savedir='alh.save',
        outputdir='output',
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
    if case == 'bulk':
        transport.define_partition(central_atoms='ALL', transport_direction='z')
        transport.build_hamiltonian_blocks(
            calculation_type='bulk',
            use_sym=False,
            do_overlap_transformation=False,
        )
        transport.configure_eigenchannels(
            do_eigenchannels=True,
            neigchnx=4,
            do_eigplot=True,
            ie_eigplot=50,
            ik_eigplot=0,
        )
        transport.configure_outputs(output_dir='./output', postfix='_bulk')
    elif case == 'lcr':
        transport.define_partition(
            central_atoms='ALL',
            left_lead_layers=3,
            right_lead_layers=3,
            transport_direction='z',
        )
        transport.build_hamiltonian_blocks(
            calculation_type='conductor',
            use_sym=False,
            do_overlap_transformation=False,
        )
        transport.configure_eigenchannels(
            do_eigenchannels=True,
            neigchnx=4,
            do_eigplot=True,
            ie_eigplot=50,
            ik_eigplot=0,
        )
        transport.configure_outputs(output_dir='./output', postfix='_lcr')
    elif case == 'lead':
        transport.define_partition(central_layers=3, transport_direction='z')
        transport.build_hamiltonian_blocks(
            calculation_type='bulk',
            use_sym=False,
            do_overlap_transformation=False,
        )
        transport.configure_outputs(output_dir='./output', postfix='_lead')
    else:
        raise ValueError(f'Unsupported conductor selector: {case}')

    transport.configure_energy_grid(
        emin=-7.0,
        emax=2.0,
        ne=100,
        delta=0.0005,
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
