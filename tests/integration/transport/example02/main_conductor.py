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
        transport.build_hamiltonian_blocks(
            datafile_C='./alh.save/atomic_proj.xml',
            dimC=41,
            emin=-7.0,
            emax=2.0,
            ne=100,
            delta=0.0005,
            transport_direction=3,
            output_dir='./output',
            postfix='_bulk',
            calculation_type='bulk',
            do_overlap_transformation=False,
            do_eigenchannels=True,
            neigchnx=4,
            do_eigplot=True,
            ie_eigplot=50,
            ik_eigplot=0,
            write_gf=False,
            write_lead_sgm=False,
            use_sym=False,
            H00_C={'rows': 'ALL', 'cols': 'ALL'},
            H_CR={'rows': 'ALL', 'cols': 'ALL'},
        )
    elif case == 'lcr':
        transport.build_hamiltonian_blocks(
            datafile_C='./alh.save/atomic_proj.xml',
            datafile_L='./alh.save/atomic_proj.xml',
            datafile_R='./alh.save/atomic_proj.xml',
            dimC=41,
            dimL=12,
            dimR=12,
            emin=-7.0,
            emax=2.0,
            ne=100,
            delta=0.0005,
            transport_direction=3,
            output_dir='./output',
            postfix='_lcr',
            calculation_type='conductor',
            do_overlap_transformation=False,
            do_eigenchannels=True,
            neigchnx=4,
            do_eigplot=True,
            ie_eigplot=50,
            ik_eigplot=0,
            use_sym=False,
            H00_C={'rows': '1-41', 'cols': '1-41'},
            H_CR={'rows': '1-41', 'cols': '1-12'},
            H_LC={'rows': '30-41', 'cols': '1-41'},
            H00_L={'rows': '1-12', 'cols': '1-12'},
            H01_L={'rows': '30-41', 'cols': '1-12'},
            H00_R={'rows': '1-12', 'cols': '1-12'},
            H01_R={'rows': '30-41', 'cols': '1-12'},
        )
    elif case == 'lead':
        transport.build_hamiltonian_blocks(
            datafile_C='./alh.save/atomic_proj.xml',
            dimC=12,
            emin=-7.0,
            emax=2.0,
            ne=100,
            delta=0.0005,
            transport_direction=3,
            output_dir='./output',
            postfix='_lead',
            calculation_type='bulk',
            do_overlap_transformation=False,
            use_sym=False,
            H00_C={'rows': '1-12', 'cols': '1-12'},
            H_CR={'rows': '30-41', 'cols': '1-12'},
        )
    else:
        raise ValueError(f'Unsupported conductor selector: {case}')

    transport.compute_self_energy(write=True)
    transport.compute_greens_functions(write=True)
    transport.compute_transmission(write=True)
    transport.compute_dos(write=True)


def main() -> None:
    selectors = ['bulk', 'lcr', 'lead']
    for selector in selectors:
        _run_case(selector)


if __name__ == '__main__':
    main()
