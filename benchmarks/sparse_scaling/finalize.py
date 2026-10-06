"""Add the scheduler's verdict to every record after the arrays finish.

A job killed for memory or walltime cannot write its own record, so its
``record.json`` is left with ``status: 'running'`` (or is missing, if the
kill came before the first stage).  This reads ``sacct`` for every array
task and stores its ``State``, ``ExitCode``, ``MaxRSS`` and ``Elapsed``
under ``scheduler``; a record still marked running gets its status from
the state (``'oom'``, ``'timeout'``, ``'killed'``, or ``'not_run'`` when
sacct knows nothing of the task), and an error file left by a non-zero
rank is folded in.  Run by the dependent finalize job, or by
hand with ``python finalize.py``; it takes no arguments.
"""

import glob
import json
import os
import subprocess

STATUS_FROM_STATE = {
    'OUT_OF_MEMORY': 'oom',
    'TIMEOUT': 'timeout',
    'CANCELLED': 'cancelled',
    'NODE_FAIL': 'node_fail',
    'FAILED': 'killed',
    'PREEMPTED': 'preempted',
}


def _sacct(jobid):
    """``{task index: {State, ExitCode, MaxRSS (bytes), Elapsed}}`` of one array."""
    fmt = 'JobID,State,ExitCode,MaxRSS,Elapsed'
    out = subprocess.run(
        ['sacct', '-j', str(jobid), '--format=' + fmt, '--parsable2', '--noheader', '--units=K'],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise RuntimeError('sacct failed: %s' % out.stderr.strip())
    tasks = {}
    for line in out.stdout.splitlines():
        jid, state, exitcode, maxrss, elapsed = line.split('|')
        base = jid.split('.')[0]
        if '_' not in base or not base.split('_')[1].isdigit():
            # the array itself, or a still-pending range such as 123_[5-10]
            continue
        task = int(base.split('_')[1])
        entry = tasks.setdefault(
            task, {'State': None, 'ExitCode': None, 'MaxRSS': 0, 'Elapsed': None}
        )
        if '.' not in jid:
            # the allocation line carries the job's final state
            entry['State'] = state.split()[0]
            entry['ExitCode'] = exitcode
            entry['Elapsed'] = elapsed
        if maxrss:
            entry['MaxRSS'] = max(entry['MaxRSS'], int(float(maxrss.rstrip('K')) * 1024))
    return tasks


def finalize(root):
    with open(os.path.join(root, 'cases.json')) as f:
        groups = json.load(f)['groups']
    with open(os.path.join(root, 'jobs.json')) as f:
        jobs = json.load(f)
    for group, cases in groups.items():
        tasks = _sacct(jobs[group]) if group in jobs else {}
        for index, case in enumerate(cases):
            outdir = os.path.join(root, case['id'])
            os.makedirs(outdir, exist_ok=True)
            path = os.path.join(outdir, 'record.json')
            if os.path.exists(path):
                with open(path) as f:
                    record = json.load(f)
            else:
                record = {'case': case, 'status': 'running', 'stages': [], 'never_started': True}
            sched = tasks.get(index, {})
            record['scheduler'] = dict(sched, jobid='%s_%d' % (jobs.get(group), index))
            if record['status'] == 'running':
                errors = sorted(glob.glob(os.path.join(outdir, 'error_rank*.json')))
                if errors:
                    with open(errors[0]) as f:
                        record['error'] = json.load(f)
                    record['status'] = 'python_error'
                elif not sched:
                    record['status'] = 'not_run'
                else:
                    record['status'] = STATUS_FROM_STATE.get(sched.get('State'), 'killed')
            with open(path, 'w') as f:
                json.dump(record, f, indent=1)
            print('%-28s %-14s %s' % (case['id'], record['status'], sched.get('State')))


if __name__ == '__main__':
    from config import CONFIG

    finalize(os.environ.get('BENCH_OUTPUT_ROOT', CONFIG['output_root']))
