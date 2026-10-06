"""Merge every ``record.json`` into flat tables for the analysis notebook.

``python collect.py`` (no arguments) writes, under the output root:

``results.csv``
    One row per pipeline case: status and failure reason, sizes, stage
    times, peak memory, Hamiltonian memory and sparsity, solver dispatch,
    and the per-cell accuracy against the reference.
``sweep.csv``
    One row per (size, nev/n, kernel) of the solver sweep.

Accuracy: the DoS of a cell doubled ``d`` times is per supercell, so it is
divided by ``N = 2**d`` to give the per-primitive-cell DoS; conductivity is
already intensive.  Each case is compared, on the energy range it shares
with the reference, with (a) the ``dense`` d = 0 run, the exact reference,
and (b) its own pipeline at d = 0, which isolates what doubling changes
from what the bond cutoff changes.
"""

import json
import os

import numpy as np
import pandas as pd

STAGE_ORDER = [
    'projections',
    'projectability',
    'pao_hamiltonian',
    'doubling',
    'disorder',
    'interpolated_hamiltonian',
    'window',
    'pao_eigh',
    'gradient_and_momenta',
    'adaptive_smearing',
    'dos',
    'transport',
    'mesh_and_properties',
    'solver_sweep',
    'finish',
]

# stages whose cost is the k-space work, summed per pipeline so the dense
# and the fused sparse stages compare like for like
KSPACE = {
    'interpolated_hamiltonian',
    'window',
    'pao_eigh',
    'gradient_and_momenta',
    'adaptive_smearing',
    'dos',
    'transport',
    'mesh_and_properties',
}
SETUP = {'projections', 'projectability', 'pao_hamiltonian', 'doubling', 'disorder'}


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


def _load_xy(path, ycol):
    data = np.loadtxt(path, ndmin=2)
    return data[:, 0], data[:, ycol]


def _deviation(e, y, e_ref, y_ref):
    """Relative L2 deviation of ``y(e)`` from ``y_ref`` on the shared range."""
    lo, hi = max(e.min(), e_ref.min()), min(e.max(), e_ref.max())
    mask = (e_ref >= lo) & (e_ref <= hi)
    if mask.sum() < 4:
        return np.nan
    yi = np.interp(e_ref[mask], e, y)
    norm = np.sqrt(np.mean(y_ref[mask] ** 2))
    return float(np.sqrt(np.mean((yi - y_ref[mask]) ** 2)) / norm) if norm > 0 else np.nan


def _observables(rec):
    """Per-cell DoS and sigma_xx of an ok case, or ``None`` each."""
    out = {}
    N = 2 ** int(rec['case']['d'])
    outputs = rec.get('outputs', {})
    if 'dosdk_0.dat' in outputs:
        e, g = _load_xy(os.path.join(rec['_dir'], outputs['dosdk_0.dat']), 1)
        out['dos'] = (e, g / N)
    if 'sigmagauss_0.dat' in outputs:
        # columns: T, mu, then the six tensor components (xx first)
        data = np.loadtxt(os.path.join(rec['_dir'], outputs['sigmagauss_0.dat']), ndmin=2)
        out['sigma'] = (data[:, 1], data[:, 2])
    return out


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
        for name in STAGE_ORDER:
            if name in stages and 'seconds' in stages[name]:
                row['t_' + name] = stages[name]['seconds']
                row['peak_' + name] = stages[name].get('peak_rss_max')
        done = [s for s in rec.get('stages', []) if s.get('status') == 'ok']
        row['t_setup'] = sum(s['seconds'] for s in done if s['name'] in SETUP)
        row['t_kspace'] = sum(s['seconds'] for s in done if s['name'] in KSPACE)
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
            obs[case['id']] = _observables(rec)
    df = pd.DataFrame(rows).sort_values(['pipeline', 'd']).reset_index(drop=True)

    ref = obs.get('dense_d00', {})
    for key in ('dos', 'sigma'):
        for ref_name in ('dense0', 'self0'):
            df['dev_%s_vs_%s' % (key, ref_name)] = np.nan
    for i, row in df.iterrows():
        mine = obs.get(row['id'])
        if not mine:
            continue
        own0 = obs.get('%s_d00' % row['pipeline'], {})
        for key in ('dos', 'sigma'):
            if key not in mine:
                continue
            if key in ref:
                df.loc[i, 'dev_%s_vs_dense0' % key] = _deviation(*mine[key], *ref[key])
            if key in own0:
                df.loc[i, 'dev_%s_vs_self0' % key] = _deviation(*mine[key], *own0[key])
    return df


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
    from config import CONFIG

    collect(os.environ.get('BENCH_OUTPUT_ROOT', CONFIG['output_root']))
