"""Expand the benchmark config into one plain dict per case."""

import copy


def doubling_pattern(d):
    """Doubling counts ``(nx, ny, nz)`` of size ``d``, spread over the axes in
    turn so every step doubles ``nawf``: 0 -> (0,0,0), 1 -> (1,0,0),
    2 -> (1,1,0), 3 -> (1,1,1), 4 -> (2,1,1), ..."""
    base, extra = divmod(int(d), 3)
    return tuple(base + (1 if axis < extra else 0) for axis in range(3))


def target_nfft(base_nfft, doublings, min_nfft):
    """The property mesh that keeps the physical k-point density of the base
    cell: ``base_nfft`` halved once per doubling along each axis."""
    return tuple(max(int(min_nfft), int(n) // 2**m) for n, m in zip(base_nfft, doublings))


def _merged(defaults, override):
    out = dict(defaults)
    out.update(override or {})
    return out


def build_cases(config):
    """Every case of ``config``, in submission order.

    Parameters
    ----------
    config : dict
        :data:`config.CONFIG`.

    Returns
    -------
    list of dict
        One self-contained dict per case: everything ``run_case`` needs, so a
        case can be re-run from ``cases.json`` alone.  ``id`` is unique and
        doubles as the directory name.
    """
    cases = []
    system = copy.deepcopy(config['system'])
    for name, pipe in config['pipelines'].items():
        if not pipe.get('enabled', True):
            continue
        for d in range(int(pipe['max_doublings']) + 1):
            doublings = doubling_pattern(d)
            cases.append(
                {
                    'id': '%s_d%02d' % (name, d),
                    'kind': 'pipeline',
                    'pipeline': name,
                    'sparse': bool(pipe['sparse']),
                    'sparse_config': copy.deepcopy(pipe.get('sparse_config', {})),
                    'd': d,
                    'doublings': doublings,
                    'nfft': target_nfft(config['base_nfft'], doublings, config['min_nfft']),
                    'system': system,
                    'dos': _merged(config['dos'], pipe.get('dos')),
                    'transport': _merged(config['transport'], pipe.get('transport')),
                    'disorder_meV': float(config['disorder_meV']) if pipe['sparse'] else 0.0,
                    'disorder_seed': int(config['disorder_seed']),
                    'npool': int(pipe['slurm']['npool']),
                }
            )
    sweep = config.get('solver_sweep', {})
    if sweep.get('enabled', False):
        for d in sweep['doublings']:
            cases.append(
                {
                    'id': 'sweep_d%02d' % d,
                    'kind': 'solver_sweep',
                    'pipeline': 'solver_sweep',
                    'sparse': True,
                    'sparse_config': {'bond_order': sweep['bond_order'], 'dense_n_max': 10**9},
                    'd': int(d),
                    'doublings': doubling_pattern(d),
                    'system': system,
                    'nev_fractions': list(sweep['nev_fractions']),
                    'nk_probe': int(sweep['nk_probe']),
                    'stop_ratio': float(sweep['stop_ratio']),
                    'disorder_meV': float(config['disorder_meV']),
                    'disorder_seed': int(config['disorder_seed']),
                    'npool': int(sweep['slurm']['npool']),
                }
            )
    ids = [c['id'] for c in cases]
    assert len(ids) == len(set(ids)), 'duplicate case ids'
    return cases


def slurm_resources(config, case):
    """The merged Slurm settings of the pipeline a case belongs to."""
    if case['kind'] == 'solver_sweep':
        own = config['solver_sweep']['slurm']
    else:
        own = config['pipelines'][case['pipeline']]['slurm']
    return _merged(config['slurm'], own)
