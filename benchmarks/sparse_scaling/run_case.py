"""Run one benchmark case and write its ``record.json``.

Two kinds of case come out of :func:`ladder.build_cases`:

``'pipeline'``
    The example01 workflow on a doubled cell: QE projections ->
    ``pao_hamiltonian`` -> ``doubling_Hamiltonian`` -> property mesh ->
    ``dos`` + Boltzmann ``transport``, on the dense pipeline or the sparse
    one (from-the-bottom ``energy_window`` or ``interior_window``).
``'solver_sweep'``
    The doubled sparse H(k) at a few k-points, solved with LAPACK and with
    ARPACK for a range of ``nev/n``, to measure where the two cross
    (``dense_ratio``) and how much per-k scratch each kernel needs.

As a Slurm array task the module takes no arguments: the entry block reads
``SLURM_ARRAY_TASK_ID`` and the case list that ``submit`` wrote.  From
Python, call :func:`run_case` with a case dict.
"""

import json
import os
import time
import tracemalloc

import numpy as np

from recorder import Recorder


# ----------------------------------------------------------------------
# helpers on the PAOFLOW objects
# ----------------------------------------------------------------------


def _hook_module_times(pao, rec):
    """Copy PAOFLOW's own per-module timings (``report_module_time``) into
    the record, attributed to the benchmark stage that ran them."""
    original = pao.report_module_time

    def hooked(mname):
        dt = time.time() - pao.reset_time if pao.rank == 0 and pao.reset_time else None
        original(mname)
        if dt is not None:
            rec.record_module_time(mname, dt)

    pao.report_module_time = hooked


def sparse_h_metrics(H):
    """Size, sparsity and memory of a :class:`SparseHamiltonian`."""
    plan = H.plan
    plan_bytes = sum(int(v.data.nbytes + v.indices.nbytes + v.indptr.nbytes) for v in plan['V'])
    plan_bytes += sum(int(plan[k].nbytes) for k in ('indices', 'indptr', 'R_uniq', 'Rcart', 'dnm'))
    nnz_k = int(plan['indices'].size)
    return {
        'nawf': int(H.nawf),
        'nspin': int(H.nspin),
        'nnz_bonds': int(H.nnz),
        'density': float(H.density()),
        'nR': int(len(plan['R_uniq'])),
        'nnz_hk': nnz_k,
        'nnz_hk_per_row': nnz_k / float(H.nawf),
        'density_hk': nnz_k / float(H.nawf) ** 2,
        'plan_bytes': plan_bytes,
        'csr_hk_bytes': nnz_k * (16 + 4) + 4 * (H.nawf + 1),
        'dense_hk_bytes': 16 * H.nawf**2,
        'dense_HRs_equivalent_bytes': 16 * H.nawf**2 * int(np.prod(H.nk_grid)) * H.nspin,
        'eig_bound_eV': float(H.drop_report.get('eig_bound', 0.0)),
        'stats_line': H.stats_line(),
    }


def apply_onsite_disorder(H, sigma_eV, seed):
    """Add Gaussian random on-site energies to a compacted bond list.

    The on-site term of orbital ``i`` is the plan entry ``(i, i)`` at
    ``R = 0``; adding ``eps_i`` there shifts ``H(k)_{ii}`` by ``eps_i`` at
    every k and leaves ``dH/dk`` untouched (an on-site energy carries no
    phase).  The same draw is applied to every spin channel.  Touches the
    private assembly plan, which is acceptable in a benchmark only.
    """
    from scipy.sparse import csr_matrix

    plan = H.plan
    n = H.nawf
    r0 = np.flatnonzero(np.all(plan['R_uniq'] == 0, axis=1))
    if len(r0) != 1:
        raise RuntimeError('apply_onsite_disorder: no unique R = 0 in the plan')
    rows = np.repeat(np.arange(n), np.diff(plan['indptr']))
    diag = np.flatnonzero(rows == plan['indices'])
    if len(diag) != n:
        raise RuntimeError('apply_onsite_disorder: the H(k) pattern lacks some diagonal entries')
    eps = np.random.default_rng(seed).normal(0.0, sigma_eV, n)
    shape = plan['V'][0].shape
    D = csr_matrix((eps.astype(complex), (diag, np.full(n, r0[0]))), shape=shape)
    plan['V'] = [(V + D).tocsr() for V in plan['V']]
    return {'sigma_eV': sigma_eV, 'seed': seed, 'rms_eV': float(np.sqrt(np.mean(eps**2)))}


def _make_paoflow(case, outdir):
    from PAOFLOW.PAOFLOW import PAOFLOW

    system = case['system']
    return PAOFLOW(
        savedir=system['savedir'],
        outputdir=os.path.join(outdir, 'output'),
        smearing=system['smearing'],
        npool=case['npool'],
        verbose=False,
        sparse=case['sparse'],
        sparse_config=case['sparse_config'] if case['sparse'] else None,
    )


def _build_cell(case, outdir, rec):
    """Projections, H(R) and the doubled cell.  Returns the PAOFLOW object."""
    pao = _make_paoflow(case, outdir)
    _hook_module_times(pao, rec)
    arrays, attr = pao.data_controller.data_dicts()
    system = case['system']

    with rec.stage('projections', arrays):
        if system['projections'] is None:
            pao.read_atomic_proj_QE()
        else:
            pao.projections(**system['projections'])
    with rec.stage('projectability', arrays):
        pao.projectability(pthr=system['pthr'])
    with rec.stage('pao_hamiltonian', arrays):
        pao.pao_hamiltonian()
    rec.set(
        'hamiltonian',
        base={
            'nawf': int(attr['nawf']),
            'bnd': int(attr['bnd']),
            'nk_R': [int(attr['nk1']), int(attr['nk2']), int(attr['nk3'])],
            'nspin': int(attr['nspin']),
        },
    )

    nx, ny, nz = case['doublings']
    if case['sparse']:
        rec.set('hamiltonian', base_sparse=sparse_h_metrics(pao.sparse.H))
        if nx + ny + nz:
            proj = pao.sparse.H.project_doubling(nx, ny, nz)
            rec.set('hamiltonian', doubling_projection={k: proj[k] for k in proj})
    if nx + ny + nz:
        with rec.stage('doubling', arrays):
            pao.doubling_Hamiltonian(nx, ny, nz)
    if case['sparse']:
        if case['disorder_meV'] > 0.0:
            with rec.stage('disorder', arrays):
                info = apply_onsite_disorder(
                    pao.sparse.H, 1.0e-3 * case['disorder_meV'], case['disorder_seed']
                )
            rec.set('hamiltonian', disorder=info)
        rec.set('hamiltonian', **sparse_h_metrics(pao.sparse.H))
    else:
        rec.set(
            'hamiltonian',
            nawf=int(attr['nawf']),
            HRs_bytes=int(arrays['HRs'].nbytes) if 'HRs' in arrays else None,
            nk_R=[int(attr['nk1']), int(attr['nk2']), int(attr['nk3'])],
        )
    return pao


def _mesh(case, attr):
    """The property mesh of this case, clamped for the dense pipeline to the
    H(R) grid it can only interpolate up from."""
    target = [int(n) for n in case['nfft']]
    if case['sparse']:
        return target, False
    nk_R = [int(attr['nk1']), int(attr['nk2']), int(attr['nk3'])]
    used = [max(t, g) for t, g in zip(target, nk_R)]
    return used, used != target


def _solver_info(pao, case):
    """The per-k kernel the sparse mesh pass dispatched to."""
    from PAOFLOW.sparse.solver import INTERIOR_DENSE_N, select_hk_solver

    eng = pao.sparse
    cfg = eng.config
    attr = pao.data_controller.data_attributes
    n = eng.H.nawf
    if eng._interior is not None:
        small = cfg.hk_solver == 'auto' and n <= INTERIOR_DENSE_N
        kernel = 'dense' if cfg.hk_solver == 'dense' or small else 'sparse'
        return {
            'mode': 'interior',
            'window': list(eng._interior),
            'kernel': kernel,
            'reason': 'interior: dense below INTERIOR_DENSE_N = %d' % INTERIOR_DENSE_N,
        }
    nev = int(attr['bnd'])
    try:
        kernel, reason = select_hk_solver(n, nev, hk_solver=cfg.hk_solver, **cfg.limits)
    except NotImplementedError as exc:
        kernel, reason = 'refused', str(exc)[:500]
    return {
        'mode': 'energy_window' if eng._window is not None else 'bnd',
        'nev': nev,
        'nev_fraction': nev / float(n),
        'kernel': kernel,
        'reason': reason,
        'limits': cfg.limits,
    }


def _outputs(outdir):
    out = os.path.join(outdir, 'output')
    names = ['dosdk_0.dat', 'sigmagauss_0.dat', 'Seebeckgauss_0.dat', 'kappagauss_0.dat']
    return {n: os.path.join('output', n) for n in names if os.path.exists(os.path.join(out, n))}


# ----------------------------------------------------------------------
# case kinds
# ----------------------------------------------------------------------


def _run_pipeline(case, outdir, rec):
    pao = _build_cell(case, outdir, rec)
    arrays, attr = pao.data_controller.data_dicts()

    nfft, clamped = _mesh(case, attr)
    dos = dict(case['dos'])
    transport = dict(case['transport'])

    with rec.stage('interpolated_hamiltonian', arrays):
        pao.interpolated_hamiltonian(nfft1=nfft[0], nfft2=nfft[1], nfft3=nfft[2])
    rec.set(
        'mesh',
        nfft=nfft,
        nfft_target=list(case['nfft']),
        clamped=clamped,
        nkpnts=int(np.prod(nfft)),
        npool=int(attr['npool']),
    )

    if case['sparse']:
        with rec.stage('window', arrays):
            pao.sparse._ensure_window()
        rec.set('solver', **_solver_info(pao, case))
        with rec.stage('mesh_and_properties', arrays):
            with pao.sparse.fused():
                pao.dos(**dos)
                pao.transport(**transport)
        if pao.sparse._interior is not None:
            rec.set('solver', bands_max=int(attr['bnd']))
        rec.set('solver', skipped=[list(s) for s in pao.sparse._skipped])
    else:
        rec.set(
            'solver',
            mode='dense',
            kernel='dense',
            nev=int(attr['nawf']),
            reason='dense pipeline: full eigh per k',
        )
        with rec.stage('pao_eigh', arrays):
            pao.pao_eigh()
        with rec.stage('gradient_and_momenta', arrays):
            pao.gradient_and_momenta()
        with rec.stage('adaptive_smearing', arrays):
            pao.adaptive_smearing()
        with rec.stage('dos', arrays):
            pao.dos(**dos)
        with rec.stage('transport', arrays):
            pao.transport(**transport)

    with rec.stage('finish', arrays):
        pao.finish_execution()
    rec.set('outputs', **_outputs(outdir))


def _timed_solve(solve, measure_memory):
    """Wall time of one solve, and its peak traced allocation when asked.

    numpy, and the LAPACK work arrays scipy allocates through it, report to
    ``tracemalloc``, so the traced peak is the per-k scratch the kernel
    needs on top of the stored H(k).  It is measured in a separate call,
    because tracing slows the many small allocations of ARPACK's reverse
    communication loop.
    """
    t0 = time.perf_counter()
    solve()
    seconds = time.perf_counter() - t0
    peak = None
    if measure_memory:
        tracemalloc.start()
        solve()
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    return seconds, peak


def _run_sweep(case, outdir, rec):
    from PAOFLOW.sparse.solver import solve_lowest

    comm = rec.comm
    if comm.Get_size() != 1:
        raise RuntimeError('the solver sweep is serial: run it with one MPI rank')
    pao = _build_cell(case, outdir, rec)
    H = pao.sparse.H
    n = H.nawf
    big = 10**9
    rng = np.random.default_rng(case['disorder_seed'])
    kpts = rng.random((case['nk_probe'], 3))
    hks = [H.assemble_hk(k, sign=-1) for k in kpts]

    results = []
    arpack_stopped = None
    with rec.stage('solver_sweep'):
        for frac in case['nev_fractions']:
            nev = max(1, int(round(frac * n)))
            row = {'nev_fraction': frac, 'nev': nev, 'n': n}
            for kernel in ('dense', 'sparse'):
                if kernel == 'sparse':
                    if nev + 4 >= n - 1:
                        row['sparse'] = {'skipped': 'no Krylov room (nev + guard >= n - 1)'}
                        continue
                    if arpack_stopped is not None:
                        row['sparse'] = {'skipped': arpack_stopped}
                        continue
                times, peak = [], None
                for i, hk in enumerate(hks):
                    try:
                        dt, p = _timed_solve(
                            lambda hk=hk: solve_lowest(hk, nev, hk_solver=kernel, dense_n_max=big),
                            measure_memory=(i == 0),
                        )
                    except Exception as exc:
                        row[kernel] = {'error': '%s: %s' % (type(exc).__name__, str(exc)[:300])}
                        break
                    times.append(dt)
                    peak = p if p is not None else peak
                else:
                    row[kernel] = {
                        'seconds': times,
                        'seconds_median': float(np.median(times)),
                        'scratch_bytes': peak,
                    }
            d, s = row.get('dense', {}), row.get('sparse', {})
            if 'seconds_median' in d and 'seconds_median' in s:
                row['ratio_sparse_over_dense'] = s['seconds_median'] / d['seconds_median']
                if row['ratio_sparse_over_dense'] > case['stop_ratio']:
                    arpack_stopped = 'ARPACK already %.1fx slower than LAPACK at nev/n = %g' % (
                        row['ratio_sparse_over_dense'],
                        frac,
                    )
            elif 'error' in s:
                arpack_stopped = 'ARPACK failed at nev/n = %g' % frac
            results.append(row)
            rec.set('sweep', results=results, kpoints=kpts)


# ----------------------------------------------------------------------
# entry points
# ----------------------------------------------------------------------


def run_case(case, outdir):
    """Run one case into ``outdir`` (created) and return its record dict.

    A Python exception is recorded (``status: 'python_error'`` plus an
    ``error_rank<r>.json`` from the failing rank) and re-raised; with more
    than one MPI rank the job is then aborted, so the other ranks do not
    hang in a collective until the walltime.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        os.makedirs(outdir, exist_ok=True)
    comm.Barrier()
    rec = Recorder(os.path.join(outdir, 'record.json'), case, comm)
    try:
        if case['kind'] == 'pipeline':
            _run_pipeline(case, outdir, rec)
        elif case['kind'] == 'solver_sweep':
            _run_sweep(case, outdir, rec)
        else:
            raise ValueError('unknown case kind %r' % case['kind'])
    except Exception as exc:
        rec.fail(exc)
        if comm.Get_size() > 1:
            comm.Abort(1)
        raise
    rec.done()
    return rec.data


def load_cases(output_root):
    with open(os.path.join(output_root, 'cases.json')) as f:
        return json.load(f)


if __name__ == '__main__':
    # a Slurm array task: the case comes from cases.json, never from argv
    root = os.environ.get('BENCH_OUTPUT_ROOT')
    if root is None:
        from config import CONFIG

        root = CONFIG['output_root']
    listing = load_cases(root)
    task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    group = os.environ['BENCH_GROUP']
    case = listing['groups'][group][task]
    run_case(case, os.path.join(root, case['id']))
