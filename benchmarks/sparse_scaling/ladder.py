"""Expand one system of the benchmark into one plain dict per case."""

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


def pipeline_properties(system, pipe):
    """``[[name, kwargs], ...]`` of a pipeline, in the order of
    ``system['properties']``, with its ``property_args`` applied."""
    wanted = set(pipe['properties'])
    unknown = wanted - set(system['properties'])
    if unknown:
        raise ValueError('properties not in SYSTEM["properties"]: %s' % sorted(unknown))
    overrides = pipe.get('property_args', {})
    return [
        [name, _merged(kwargs, overrides.get(name))]
        for name, kwargs in system['properties'].items()
        if name in wanted
    ]


def case_system(name, system):
    """The part of a system every case carries: what ``run_case`` and
    ``observables`` read back from ``cases.json`` and ``record.json``."""
    keys = ('label', 'savedir', 'projections', 'pthr', 'smearing', 'zeros', 'nonzero')
    out = {key: copy.deepcopy(system.get(key)) for key in keys}
    out['name'] = name
    return out


def build_cases(config, name, system):
    """Every case of one system, in submission order.

    Parameters
    ----------
    config : dict
        :data:`config.CONFIG`.
    name : str
        The system's folder name.
    system : dict
        Its :data:`SYSTEM` (:func:`config.load_system`).

    Returns
    -------
    list of dict
        One self-contained dict per case: everything ``run_case`` needs, so a
        case can be re-run from ``cases.json`` alone.  ``id`` is unique within
        the system and doubles as the directory name under its ``runs/``.
    """
    cases = []
    csys = case_system(name, system)
    for pname, pipe in system['pipelines'].items():
        if not pipe.get('enabled', True):
            continue
        for d in range(int(pipe['max_doublings']) + 1):
            doublings = doubling_pattern(d)
            cases.append(
                {
                    'id': '%s_d%02d' % (pname, d),
                    'kind': 'pipeline',
                    'pipeline': pname,
                    'sparse': bool(pipe['sparse']),
                    'sparse_config': copy.deepcopy(pipe.get('sparse_config', {})),
                    'd': d,
                    'doublings': doublings,
                    'nfft': target_nfft(system['base_nfft'], doublings, system['min_nfft']),
                    'system': csys,
                    'properties': pipeline_properties(system, pipe),
                    'band_curvature': bool(pipe.get('band_curvature', False)),
                    # the band path is folded by doubling: undoubled cell only
                    'bands': copy.deepcopy(system.get('bands'))
                    if pipe.get('bands') and d == 0
                    else None,
                    'disorder_meV': float(config['disorder_meV']) if pipe['sparse'] else 0.0,
                    'disorder_seed': int(config['disorder_seed']),
                    'npool': int(resources(config, pipe)['npool']),
                }
            )
    sweep = system.get('solver_sweep', {})
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
                    'system': csys,
                    'nev_fractions': list(sweep['nev_fractions']),
                    'nk_probe': int(sweep['nk_probe']),
                    'stop_ratio': float(sweep['stop_ratio']),
                    'disorder_meV': float(config['disorder_meV']),
                    'disorder_seed': int(config['disorder_seed']),
                    'npool': int(resources(config, sweep)['npool']),
                }
            )
    ids = [c['id'] for c in cases]
    assert len(ids) == len(set(ids)), 'duplicate case ids'
    return cases


def resources(config, pipe):
    """The resources of one pipeline (or the solver sweep):
    ``config['resources']`` with the pipeline's ``slurm`` applied."""
    return _merged(config['resources'], pipe.get('slurm'))


def slurm_resources(config, system, case):
    """The merged Slurm settings of the pipeline a case belongs to."""
    if case['kind'] == 'solver_sweep':
        own = system['solver_sweep']
    else:
        own = system['pipelines'][case['pipeline']]
    return _merged(config['slurm'], resources(config, own))
