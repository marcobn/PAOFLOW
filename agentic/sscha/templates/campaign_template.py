#!/usr/bin/env python3
"""Drivers of an SSCHA campaign (template: copy to <study>/campaign.py).

    python campaign.py      # -> <RUN_ROOT>/main.sscha.py (one driver for all temperatures of the scan)

Writes the main.sscha.py that paoflow-gen (workflow sscha) would write, non-interactively, from the settings of
PLAN.md; then, in RUN_ROOT:

    python main.sscha.py harmonic   # only when HARMONIC_DYN does not exist
    python main.sscha.py run        # all temperatures (or --temperature T)
    python main.sscha.py status

Playbook stages 1-2: PAOFLOW/agentic/sscha/AGENT_WORKFLOW.md. Warm starts and later temperatures of the scan are
edits of the written driver (TEMPERATURES, START_FROM); the state of each temperature is on disk, so re-running
`run` continues every run where it stopped.
"""

from pathlib import Path

from PAOFLOW.gen.sscha_driver import build_sscha_script

RUN_ROOT = Path('<run folder: next to the scf save, in scratch on a cluster>')
CFG = {
    # the parent scf and its pseudopotentials (paths relative to RUN_ROOT or absolute)
    'savedir': '<prefix>.save',
    'prefix': '<prefix>',
    'pp_dir': '<pseudo dir or HERE>',
    'outputdir': 'output',
    # the ensembles
    'temperatures': [1000.0, 2000.0],  # bracket T_s first, refine with next_temperature
    'supercell': 2,  # must contain the unstable q of the harmonic run
    'kgrid': [6, 6, 6],  # supercell k-grid (None: scf grid / supercell)
    'electronic_temperature': None,  # None (parent occupations), 'ionic' (Fermi-Dirac at each T) or K
    'population_sizes': [100, 100, 200, 200, 300, 300],
    'hessian_size': 800,
    # the harmonic start (ph.x fildyn prefix; written by the harmonic phase on harmonic_qgrid if missing)
    'harmonic_dyn': 'harmonic/<prefix>.dyn',
    'harmonic_qgrid': [6, 6, 6],
    # launcher: 'local', 'slurm' or 'manual'
    'launcher': 'slurm',
    'mpi_qe': 'srun',
    'qe_pw': 'pw.x -nk 4',
    'qe_ph': 'ph.x -nk 4',
    'slurm_directives': [
        '#SBATCH -p <partition>',
        '#SBATCH -N 1',
        '#SBATCH -n <cores>',
        '#SBATCH -t 02:00:00',
    ],
    'slurm_step_directives': [
        '#SBATCH -p <partition>',
        '#SBATCH -N 1',
        '#SBATCH -n 4',
        '#SBATCH -t 02:00:00',
    ],
    'slurm_concurrent': 40,
    'env_setup': ['source <environment script loading QE and PAOFLOW>'],
    'python': 'python',
}


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    driver = RUN_ROOT / 'main.sscha.py'
    if driver.exists():
        raise SystemExit(
            '%s exists: edit it (TEMPERATURES, START_FROM) instead of regenerating' % driver
        )
    driver.write_text(build_sscha_script(CFG))
    print('wrote %s; next: cd %s && python main.sscha.py run' % (driver, RUN_ROOT))


if __name__ == '__main__':
    main()
