import sys

from mpi4py import MPI

from PAOFLOW import PAOFLOW
from PAOFLOW.Transport import Transport
from PAOFLOW.transport.conductor_pipeline import run_conductor

comm = MPI.COMM_WORLD

_CONDUCTOR_CONFIGS = {
    'conductor_bulk.yaml': {
        'datafile_C': './output/qe/alh.save/atomic_proj.xml',
        'dimC': 41,
        'emin': -7.0,
        'emax': 2.0,
        'ne': 9001,
        'delta': 0.0005,
        'transport_direction': 3,
        'output_dir': './output/paoflow',
        'postfix': '_bulk',
        'calculation_type': 'bulk',
        'do_overlap_transformation': False,
        'do_eigenchannels': True,
        'neigchnx': 4,
        'do_eigplot': True,
        'ie_eigplot': 7001,
        'ik_eigplot': 1,
        'write_gf': False,
        'write_lead_sgm': False,
        'use_sym': False,
        'H00_C': {'rows': 'ALL', 'cols': 'ALL'},
        'H_CR': {'rows': 'ALL', 'cols': 'ALL'},
    },
    'conductor_lcr.yaml': {
        'datafile_C': './output/qe/alh.save/atomic_proj.xml',
        'datafile_L': './output/qe/alh.save/atomic_proj.xml',
        'datafile_R': './output/qe/alh.save/atomic_proj.xml',
        'dimC': 41,
        'dimL': 12,
        'dimR': 12,
        'emin': -7.0,
        'emax': 2.0,
        'ne': 9001,
        'delta': 0.0005,
        'transport_direction': 3,
        'output_dir': './output/paoflow',
        'postfix': '_lcr',
        'calculation_type': 'conductor',
        'do_overlap_transformation': False,
        'do_eigenchannels': True,
        'neigchnx': 4,
        'do_eigplot': True,
        'ie_eigplot': 7001,
        'ik_eigplot': 1,
        'H00_C': {'rows': '1-41', 'cols': '1-41'},
        'H_CR': {'rows': '1-41', 'cols': '1-12'},
        'H_LC': {'rows': '30-41', 'cols': '1-41'},
        'H00_L': {'rows': '1-12', 'cols': '1-12'},
        'H01_L': {'rows': '30-41', 'cols': '1-12'},
        'H00_R': {'rows': '1-12', 'cols': '1-12'},
        'H01_R': {'rows': '30-41', 'cols': '1-12'},
    },
    'conductor_lead_Al.yaml': {
        'datafile_C': './output/qe/alh.save/atomic_proj.xml',
        'dimC': 12,
        'emin': -7.0,
        'emax': 2.0,
        'ne': 1000,
        'delta': 0.0005,
        'transport_direction': 3,
        'output_dir': './output/paoflow',
        'postfix': '_lead',
        'calculation_type': 'bulk',
        'do_overlap_transformation': False,
        'H00_C': {'rows': '1-12', 'cols': '1-12'},
        'H_CR': {'rows': '30-41', 'cols': '1-12'},
    },
}


def main() -> None:
    yaml_file = sys.argv[1] if len(sys.argv) > 1 else 'conductor_bulk.yaml'
    if yaml_file not in _CONDUCTOR_CONFIGS:
        raise ValueError(f'Unsupported conductor input selector: {yaml_file}')

    paoflow = PAOFLOW.PAOFLOW(
        savedir='output/qe/alh.save',
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
    transport.build_hamiltonian_blocks(**_CONDUCTOR_CONFIGS[yaml_file])
    run_conductor(
        data=transport.conductor_data,
        blc_blocks=transport.blc_blocks,
        comm=comm,
    )


if __name__ == '__main__':
    main()
