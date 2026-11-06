"""Write the case lists and the Slurm job arrays, and submit them.

``python submit.py`` takes no arguments: it reads ``CONFIG`` from
:mod:`config` and submits every case of every system of ``CONFIG['systems']``.
Each pipeline of a system (and its solver sweep) becomes one job array with one
task per size, all submitted at once; a size that fails is a data point, not
a reason to stop the ladder.  A system's ``cases.json``, job scripts,
``jobs.json`` and logs go into its ``systems/<name>/runs``.

Case directories are written in place, so empty ``runs/`` before submitting:
files of an earlier run would mix into the new ones.  Nothing runs after the
arrays; :mod:`finalize` and :mod:`collect` are run by hand if wanted.  Set
``CONFIG['slurm']['dry_run'] = True`` to write the scripts without calling
``sbatch``.
"""

import json
import os
import subprocess

from config import load_system, output_root
from ladder import build_cases, slurm_resources

HERE = os.path.dirname(os.path.abspath(__file__))


def group_cases(config: dict, name: str, system: dict) -> dict[str, list[dict]]:
    """``{group: [case, ...]}``: one group (job array) per pipeline."""
    groups = {}
    for case in build_cases(config, name, system):
        groups.setdefault(case['pipeline'], []).append(case)
    return groups


def check_savedirs(systems):
    """Raise unless every system's QE save directory exists."""
    missing = [
        '%s: %s' % (name, system['savedir'])
        for name, system in systems.items()
        if not os.path.isdir(system['savedir'])
    ]
    if missing:
        raise FileNotFoundError(
            'QE save directories not found:\n  %s\nThe Pt, Fe and Te save directories are '
            'not under version control: copy systems/<name>/*.save along with the benchmark, '
            'or extract them from the QE test assets (see the README).' % '\n  '.join(missing)
        )


def _sbatch_header(config, name, res, ntasks_array, log):
    lines = [
        '#!/bin/bash',
        '#SBATCH --job-name=%s' % name,
        '#SBATCH --nodes=%d' % res['nodes'],
        '#SBATCH --ntasks=%d' % res['ntasks'],
        '#SBATCH --cpus-per-task=%d' % res['cpus_per_task'],
        '#SBATCH --mem=%s' % res['mem'],
        '#SBATCH --time=%s' % res['time'],
        '#SBATCH --output=%s' % log,
    ]
    if ntasks_array is not None:
        limit = res.get('array_parallel')
        lines.append('#SBATCH --array=0-%d%s' % (ntasks_array - 1, '%%%d' % limit if limit else ''))
    for key in ('account', 'partition', 'qos'):
        if res.get(key):
            lines.append('#SBATCH --%s=%s' % (key, res[key]))
    if res.get('exclusive'):
        lines.append('#SBATCH --exclusive')
    lines += ['#SBATCH %s' % extra for extra in res.get('extra', [])]
    return lines


def write_scripts(config: dict, name: str, system: dict) -> dict[str, str]:
    """Write one system's ``cases.json`` and one job script per group.

    Returns
    -------
    dict
        ``{group: script path}``.
    """
    root = output_root(name)
    os.makedirs(os.path.join(root, 'logs'), exist_ok=True)
    groups = group_cases(config, name, system)
    with open(os.path.join(root, 'cases.json'), 'w') as f:
        json.dump({'groups': groups, 'config': config, 'system': system}, f, indent=1, default=list)

    scripts = {}
    for group, cases in groups.items():
        res = slurm_resources(config, system, cases[0])
        lines = _sbatch_header(
            config,
            'sps_%s_%s' % (name, group),
            res,
            len(cases),
            os.path.join(root, 'logs', group + '_%A_%a.out'),
        )
        lines += ['', 'set -uo pipefail']
        lines += res.get('setup', [])
        lines += [
            'export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}',
            'export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}',
            'export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}',
            # glibc otherwise keeps freed mid-size arrays in the heap: Te d = 2 held
            # 10.8 GB per rank after the mesh pass that nothing referenced
            'export MALLOC_MMAP_THRESHOLD_=131072',
            'export BENCH_OUTPUT_ROOT=%s' % root,
            'export BENCH_GROUP=%s' % group,
            'cd %s' % HERE,
            '%s %s run_case.py' % (res['launcher'], res['python']),
        ]
        path = os.path.join(root, 'job_%s.sh' % group)
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        scripts[group] = path
    return scripts


def _sbatch(args):
    out = subprocess.run(['sbatch', '--parsable'] + args, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError('sbatch failed: %s' % out.stderr.strip())
    return out.stdout.strip().split(';')[0]


def submit(config: dict) -> dict[str, dict[str, str]]:
    """Write everything and submit it; returns ``{system: {group: job id}}``."""
    systems = {name: load_system(name) for name in config['systems']}
    check_savedirs(systems)
    scripts = {name: write_scripts(config, name, system) for name, system in systems.items()}
    if config['slurm'].get('dry_run'):
        for name, paths in scripts.items():
            for group, path in paths.items():
                print('[dry run] %-4s %-20s %s' % (name, group, path))
        return {}
    jobs = {}
    for name, paths in scripts.items():
        jobs[name] = {}
        for group, path in paths.items():
            jobs[name][group] = _sbatch([path])
            print('submitted %-4s %-20s job %s' % (name, group, jobs[name][group]))
        with open(os.path.join(output_root(name), 'jobs.json'), 'w') as f:
            json.dump(jobs[name], f, indent=1)
    return jobs


if __name__ == '__main__':
    from config import CONFIG

    submit(CONFIG)
