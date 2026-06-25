"""Current-vs-bias calculation from an existing conductance file."""

from PAOFLOW.transport.current_pipeline import run_current_from_file


def main() -> None:
    run_current_from_file(
        data={
            'fileout': './output/paoflow/current.dat',
            'mu_L': -0.5,
            'mu_R': 0.5,
            'sigma': 0.05,
        },
        filein='./output/paoflow/conductance_lcr.dat',
        bias_min=-1.0,
        bias_max=1.0,
        nbias=1500,
    )


if __name__ == '__main__':
    main()
