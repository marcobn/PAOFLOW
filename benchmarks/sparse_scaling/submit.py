"""Write the case list and the Slurm job arrays, and submit them.

``python submit.py`` takes no arguments: it reads ``CONFIG`` from
:mod:`config`.  Each pipeline (and the solver sweep) becomes one job array
with one task per size, all submitted at once; a size that fails is a data
point, not a reason to stop the ladder.  A last job, held until every array
has finished (``afterany``), runs :mod:`finalize` to add the scheduler's
verdict (``OUT_OF_MEMORY``, ``TIMEOUT``, ``MaxRSS``) to each record.

Set ``CONFIG['slurm']['dry_run'] = True`` to write the scripts without
calling ``sbatch``.
"""

import json
import os
import subprocess

from ladder import build_cases, slurm_resources

HERE = os.path.dirname(os.path.abspath(__file__))


def group_cases(config):
    """``{group: [case, ...]}``: one group (job array) per pipeline."""
    groups = {}
    for case in build_cases(config):
        groups.setdefault(case['pipeline'], []).append(case)
    return groups


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


def write_scripts(config):
    """Write ``cases.json`` and one job script per group.

    Returns
    -------
    dict
        ``{group: script path}``, plus ``'_finalize'`` for the finalize job.
    """
    root = config['output_root']
    os.makedirs(os.path.join(root, 'logs'), exist_ok=True)
    groups = group_cases(config)
    with open(os.path.join(root, 'cases.json'), 'w') as f:
        json.dump({'groups': groups, 'config': config}, f, indent=1, default=list)

    scripts = {}
    for group, cases in groups.items():
        res = slurm_resources(config, cases[0])
        lines = _sbatch_header(
            config,
            'sps_' + group,
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

    res = dict(
        config['slurm'], ntasks=1, cpus_per_task=1, mem='2G', time='00:30:00', exclusive=False
    )
    lines = _sbatch_header(
        config, 'sps_finalize', res, None, os.path.join(root, 'logs', 'finalize.out')
    )
    lines += ['', 'set -uo pipefail'] + res.get('setup', [])
    lines += ['export BENCH_OUTPUT_ROOT=%s' % root, 'cd %s' % HERE]
    lines += ['%s finalize.py' % res['python'], '%s collect.py' % res['python']]
    path = os.path.join(root, 'job_finalize.sh')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    scripts['_finalize'] = path
    return scripts


def _sbatch(args):
    out = subprocess.run(['sbatch', '--parsable'] + args, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError('sbatch failed: %s' % out.stderr.strip())
    return out.stdout.strip().split(';')[0]


def submit(config):
    """Write everything and submit it; returns ``{group: job id}``."""
    scripts = write_scripts(config)
    root = config['output_root']
    if config['slurm'].get('dry_run'):
        for group, path in scripts.items():
            print('[dry run] %-20s %s' % (group, path))
        return {}
    jobs = {}
    for group, path in scripts.items():
        if group != '_finalize':
            jobs[group] = _sbatch([path])
            print('submitted %-20s job %s' % (group, jobs[group]))
    jobs['_finalize'] = _sbatch(
        ['--dependency=afterany:' + ':'.join(jobs.values()), scripts['_finalize']]
    )
    print('submitted %-20s job %s (after all arrays)' % ('finalize', jobs['_finalize']))
    with open(os.path.join(root, 'jobs.json'), 'w') as f:
        json.dump(jobs, f, indent=1)
    return jobs


if __name__ == '__main__':
    from config import CONFIG

    submit(CONFIG)
