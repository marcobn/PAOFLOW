"""Write the case lists and the Slurm job arrays, and submit them.

``python submit.py`` takes no arguments: it reads ``CONFIG`` from
:mod:`config` and submits every system of ``CONFIG['systems']``.  Each
pipeline of a system (and its solver sweep) becomes one job array with one
task per size, all submitted at once; a size that fails is a data point, not
a reason to stop the ladder.  A system's ``cases.json``, job scripts,
``jobs.json`` and logs go into its ``systems/<name>/runs``.  A last job, held
until every array has finished (``afterany``), runs :mod:`finalize` and
:mod:`collect` on every submitted system, to add the scheduler's verdict
(``OUT_OF_MEMORY``, ``TIMEOUT``, ``MaxRSS``) to each record and build the
tables.

Set ``CONFIG['slurm']['dry_run'] = True`` to write the scripts without
calling ``sbatch``.
"""

import json
import os
import subprocess

from config import load_system, output_root
from ladder import build_cases, slurm_resources

HERE = os.path.dirname(os.path.abspath(__file__))


def group_cases(config, name, system):
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


def write_scripts(config, name, system):
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
            os.path.join(root, 'logs', group + '_%a.out'),
        )
        lines += ['', 'set -uo pipefail']
        lines += res.get('setup', [])
        lines += [
            'export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}',
            'export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}',
            'export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}',
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


def write_finalize(config, names):
    """The finalize job: :mod:`finalize`, then :mod:`collect`, per system."""
    res = dict(
        config['slurm'], ntasks=1, cpus_per_task=1, mem='2G', time='00:30:00', exclusive=False
    )
    # the one job that spans the systems lives in the top-level runs/
    log_dir = os.path.join(HERE, 'runs')
    os.makedirs(log_dir, exist_ok=True)
    lines = _sbatch_header(config, 'sps_finalize', res, None, os.path.join(log_dir, 'finalize.out'))
    lines += ['', 'set -uo pipefail'] + res.get('setup', []) + ['cd %s' % HERE]
    for name in names:
        root = output_root(name)
        lines += [
            'export BENCH_OUTPUT_ROOT=%s' % root,
            '%s finalize.py' % res['python'],
            '%s collect.py' % res['python'],
        ]
    path = os.path.join(log_dir, 'job_finalize.sh')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def _sbatch(args):
    out = subprocess.run(['sbatch', '--parsable'] + args, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError('sbatch failed: %s' % out.stderr.strip())
    return out.stdout.strip().split(';')[0]


def submit(config):
    """Write everything and submit it; returns ``{system: {group: job id}}``."""
    systems = {name: load_system(name) for name in config['systems']}
    check_savedirs(systems)
    scripts = {name: write_scripts(config, name, system) for name, system in systems.items()}
    finalize = write_finalize(config, list(systems))
    if config['slurm'].get('dry_run'):
        for name, paths in scripts.items():
            for group, path in paths.items():
                print('[dry run] %-4s %-20s %s' % (name, group, path))
        print('[dry run] %-25s %s' % ('finalize', finalize))
        return {}
    jobs = {}
    for name, paths in scripts.items():
        jobs[name] = {}
        for group, path in paths.items():
            jobs[name][group] = _sbatch([path])
            print('submitted %-4s %-20s job %s' % (name, group, jobs[name][group]))
        with open(os.path.join(output_root(name), 'jobs.json'), 'w') as f:
            json.dump(jobs[name], f, indent=1)
    ids = [job for groups in jobs.values() for job in groups.values()]
    final = _sbatch(['--dependency=afterany:' + ':'.join(ids), finalize])
    print('submitted %-25s job %s (after all arrays)' % ('finalize', final))
    return jobs


if __name__ == '__main__':
    from config import CONFIG

    submit(CONFIG)
