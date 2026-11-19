"""Merge every ``record.json`` of a system into flat tables for the analysis notebook.

``python collect.py`` (no arguments) writes, under each system's
``systems/<name>/runs`` (or only under ``BENCH_OUTPUT_ROOT`` when it is set):

``results.csv``
    One row per pipeline case: status and failure reason, sizes, stage
    times, peak memory, Hamiltonian memory and sparsity, solver dispatch,
    and the per-cell accuracy against the reference.
``sweep.csv``
    One row per (size, nev/n, kernel) of the solver sweep.

Accuracy: every per-primitive-cell curve of :mod:`observables` (DoS, PDOS,
transport, conductivity, doping, dielectric, effective masses, the Hall
conductivities and the linear responses) is compared,
as a relative L2 deviation on the range it shares with the other curve, with
(a) the ``reference`` case, the dense pipeline on the primitive cell without
doubling, in columns ``dev_<name>_vs_ref``, and (b) the same pipeline one
size down, in ``dev_<name>_vs_prev``, which separates a one-off jump from a
drift with size.
"""

import json
import os

import numpy as np
import pandas as pd

import observables

# the dense pipeline on the primitive cell, without doubling
REFERENCE_ID = 'reference_d00'

# the order of the t_<stage> columns; property stages not listed here (every
# property of the dense pipeline is a stage) follow in the order they ran
STAGE_ORDER = [
    'projections',
    'projectability',
    'pao_hamiltonian',
    'operators',
    'doubling',
    'disorder',
    'bands',
    'interpolated_hamiltonian',
    'window',
    'pao_eigh',
    'gradient_and_momenta',
    'adaptive_smearing',
    'mesh_and_properties',
    'solver_sweep',
    'finish',
]

# stages that build the Hamiltonian; every other stage but these and the
# band path (d = 0 only) is mesh work, summed per pipeline so the dense and the
# fused sparse stages compare like for like
SETUP = {'projections', 'projectability', 'pao_hamiltonian', 'operators', 'doubling', 'disorder'}
NOT_KSPACE = SETUP | {'bands', 'solver_sweep', 'finish'}


def load_records(root):
    records = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, 'record.json')
        if os.path.isfile(path):
            with open(path) as f:
                rec = json.load(f)
            rec['_dir'] = os.path.join(root, name)
            records.append(rec)
    return records


def _failure(rec):
    """Short reason of a failed case, or ``None``."""
    status = rec.get('status')
    if status == 'ok':
        return None
    if status == 'python_error':
        err = rec.get('error', {})
        return '%s: %s' % (err.get('type'), err.get('message', '').splitlines()[0][:160])
    running = [s['name'] for s in rec.get('stages', []) if s.get('status') != 'ok']
    where = (' in ' + running[-1]) if running else ''
    return '%s%s' % (status, where)


def pipeline_rows(records):
    rows, obs = [], {}
    # the base-cell size, to place cases that died before building H
    bases = [
        r['hamiltonian']['base']['nawf'] for r in records if 'base' in r.get('hamiltonian', {})
    ]
    base_any = bases[0] if bases else None
    for rec in records:
        case = rec['case']
        if case['kind'] != 'pipeline':
            continue
        ham = rec.get('hamiltonian', {})
        solver = rec.get('solver', {})
        mesh = rec.get('mesh', {})
        env = rec.get('env', {})
        stages = {s['name']: s for s in rec.get('stages', [])}
        sched = rec.get('scheduler', {})
        base_nawf = ham.get('base', {}).get('nawf', base_any)
        nawf = ham.get('nawf', base_nawf * 2 ** case['d'] if base_nawf else np.nan)
        row = {
            'system': (case.get('system') or {}).get('name', 'si'),
            'id': case['id'],
            'pipeline': case['pipeline'],
            'd': case['d'],
            'N': 2 ** case['d'],
            'nawf': nawf,
            'status': rec.get('status'),
            'failure': _failure(rec),
            'bond_order': case['sparse_config'].get('bond_order'),
            'disorder_meV': case.get('disorder_meV', 0.0),
            'mpi_ranks': env.get('mpi_ranks'),
            'threads': (env.get('threads') or {}).get('OMP_NUM_THREADS'),
            'npool': mesh.get('npool', case.get('npool')),
            'nkpnts': mesh.get('nkpnts'),
            'mesh_clamped': mesh.get('clamped'),
            'kernel': solver.get('kernel'),
            'nev': solver.get('nev', solver.get('bands_max')),
            'nev_fraction': solver.get('nev_fraction'),
            'nnz_bonds': ham.get('nnz_bonds'),
            'nnz_hk_per_row': ham.get('nnz_hk_per_row'),
            'density_hk': ham.get('density_hk'),
            'nR': ham.get('nR'),
            'H_bytes': ham.get('plan_bytes', ham.get('HRs_bytes')),
            'HRs_bytes': ham.get('HRs_bytes'),
            'plan_bytes': ham.get('plan_bytes'),
            'dense_hk_bytes': 16.0 * nawf**2,
            'eig_bound_eV': ham.get('eig_bound_eV'),
            'sched_state': sched.get('State'),
            'sched_maxrss': sched.get('MaxRSS'),
            'sched_elapsed': sched.get('Elapsed'),
            'node_mem_total': (env.get('meminfo') or {}).get('MemTotal'),
        }
        names = [n for n in STAGE_ORDER if n in stages]
        names += [n for n in stages if n not in STAGE_ORDER]
        for name in names:
            if 'seconds' in stages[name]:
                row['t_' + name] = stages[name]['seconds']
                row['peak_' + name] = stages[name].get('peak_rss_max')
        done = [s for s in rec.get('stages', []) if s.get('status') == 'ok']
        row['t_setup'] = sum(s['seconds'] for s in done if s['name'] in SETUP)
        row['t_kspace'] = sum(s['seconds'] for s in done if s['name'] not in NOT_KSPACE)
        row['t_total'] = sum(s.get('seconds', 0.0) for s in rec.get('stages', []))
        peaks = [s.get('peak_rss_max') for s in done if s.get('peak_rss_max')]
        row['peak_rss_max'] = max(peaks) if peaks else np.nan
        # per-k cost of the eigensolve: the mesh-pass module time (sparse) or
        # pao_eigh (dense), times the k-points one pool carries
        mods = {m['name']: m['seconds'] for m in rec.get('modules', [])}
        t_solve = mods.get('Sparse mesh pass', row.get('t_pao_eigh'))
        if t_solve is not None and row['nkpnts'] and row['npool']:
            row['t_solve'] = t_solve
            row['t_per_k'] = t_solve * row['npool'] / row['nkpnts']
        rows.append(row)
        if rec.get('status') == 'ok':
            obs[case['id']] = observables.load(rec['_dir'])
    df = pd.DataFrame(rows).sort_values(['pipeline', 'd']).reset_index(drop=True)

    ref = obs.get(REFERENCE_ID, {})
    devs = []
    for _, row in df.iterrows():
        mine = obs.get(row['id'], {})
        prev = obs.get('%s_d%02d' % (row['pipeline'], row['d'] - 1), {})
        dev = {}
        for name, curve in mine.items():
            if name in ref and row['id'] != REFERENCE_ID:
                dev['dev_%s_vs_ref' % name] = observables.deviation(curve, ref[name])
            if name in prev:
                dev['dev_%s_vs_prev' % name] = observables.deviation(curve, prev[name])
        devs.append(dev)
    return pd.concat([df, pd.DataFrame(devs, index=df.index)], axis=1)


def sweep_rows(records):
    rows = []
    for rec in records:
        if rec['case']['kind'] != 'solver_sweep':
            continue
        for r in rec.get('sweep', {}).get('results', []):
            for kernel in ('dense', 'sparse'):
                info = r.get(kernel, {})
                rows.append(
                    {
                        'd': rec['case']['d'],
                        'n': r['n'],
                        'nev': r['nev'],
                        'nev_fraction': r['nev_fraction'],
                        'kernel': kernel,
                        'seconds': info.get('seconds_median'),
                        'scratch_bytes': info.get('scratch_bytes'),
                        'note': info.get('skipped') or info.get('error'),
                        'status': rec.get('status'),
                    }
                )
    return pd.DataFrame(rows)


def collect(root):
    records = load_records(root)
    df = pipeline_rows(records)
    df.to_csv(os.path.join(root, 'results.csv'), index=False)
    sw = sweep_rows(records)
    sw.to_csv(os.path.join(root, 'sweep.csv'), index=False)
    print('%d pipeline cases, %d sweep rows -> %s' % (len(df), len(sw), root))
    return df, sw


if __name__ == '__main__':
    from config import all_systems, output_root

    if 'BENCH_OUTPUT_ROOT' in os.environ:
        collect(os.environ['BENCH_OUTPUT_ROOT'])
    else:
        for name in all_systems():
            if os.path.isdir(output_root(name)):
                print('== %s' % name)
                collect(output_root(name))
