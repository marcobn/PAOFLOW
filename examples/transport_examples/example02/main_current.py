import sys

from PAOFLOW.transport.current_pipeline import run_current_from_file

_CURRENT_CONFIGS = {
    'current.yaml': {
        'filein': './output/paoflow/conductance_lcr.dat',
        'fileout': './output/paoflow/current.dat',
        'bias_min': -1.0,
        'bias_max': 1.0,
        'nbias': 1500,
        'sigma': 0.05,
        'mu_L': -0.5,
        'mu_R': 0.5,
    },
}


def main() -> None:
    yaml_file = sys.argv[1] if len(sys.argv) > 1 else 'current.yaml'
    if yaml_file not in _CURRENT_CONFIGS:
        raise ValueError(f'Unsupported current input selector: {yaml_file}')

    cfg = _CURRENT_CONFIGS[yaml_file]

    run_current_from_file(
        data={
            'fileout': cfg['fileout'],
            'mu_L': cfg['mu_L'],
            'mu_R': cfg['mu_R'],
            'sigma': cfg['sigma'],
        },
        filein=cfg['filein'],
        bias_min=cfg['bias_min'],
        bias_max=cfg['bias_max'],
        nbias=cfg['nbias'],
    )


if __name__ == '__main__':
    main()
