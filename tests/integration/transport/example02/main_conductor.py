import sys

from PAOFLOW import PAOFLOW
from PAOFLOW.Transport import Transport
from PAOFLOW.transport.conductor_pipeline import run_conductor
from PAOFLOW.transport.observables.broadening import compute_broadening_matrix


def main() -> None:
    input_name = sys.argv[1] if len(sys.argv) > 1 else 'conductor_bulk.yaml'

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
    if input_name.endswith('conductor_bulk.yaml'):
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
    elif input_name.endswith('conductor_lcr.yaml'):
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
    elif input_name.endswith('conductor_lead_Al.yaml'):
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
        raise ValueError(f'Unsupported conductor selector: {input_name}')

    energy_index = 50
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
