//! PyO3 bindings: batched, `rayon`-parallel ERI evaluation.
//!
//! The basis crosses the FFI boundary as flat CSR-style NumPy arrays so that a
//! whole rank-local chunk of unique integral keys is evaluated in one call,
//! amortising the Python<->Rust overhead and releasing the GIL during compute.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::OnceLock;

use crate::eri::{contr_coulomb, Cgbf};

/// Optional dedicated rayon pool sized by `PAOFLOW_ACBN0_THREADS`.
///
/// When the variable is unset (or invalid / `<= 0`) the global rayon pool is
/// used, which itself honours `RAYON_NUM_THREADS`. Setting
/// `PAOFLOW_ACBN0_THREADS` lets callers cap intra-rank threads to avoid
/// oversubscription when launching many MPI ranks per node.
fn custom_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        std::env::var("PAOFLOW_ACBN0_THREADS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .map(|n| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .expect("failed to build PAOFLOW_ACBN0_THREADS rayon pool")
            })
    })
    .as_ref()
}

/// Evaluate `f(i)` for `i in 0..n` in parallel, honouring the optional
/// `PAOFLOW_ACBN0_THREADS` pool. The caller is expected to have released the
/// GIL already.
fn par_eval<F>(n: usize, f: F) -> Vec<f64>
where
    F: Fn(usize) -> f64 + Sync,
{
    let run = || (0..n).into_par_iter().map(&f).collect::<Vec<f64>>();
    match custom_pool() {
        Some(pool) => pool.install(run),
        None => run(),
    }
}

/// Reconstruct the contracted basis from flat CSR arrays.
fn build_basis(
    origins: &PyReadonlyArray2<f64>,
    prim_offsets: &PyReadonlyArray1<i64>,
    exps: &PyReadonlyArray1<f64>,
    coefs: &PyReadonlyArray1<f64>,
    norms: &PyReadonlyArray1<f64>,
    powers: &PyReadonlyArray2<i64>,
) -> Vec<Cgbf> {
    let origins = origins.as_array();
    let offsets = prim_offsets.as_array();
    let exps = exps.as_array();
    let coefs = coefs.as_array();
    let norms = norms.as_array();
    let powers = powers.as_array();

    let nbasis = origins.shape()[0];
    let mut basis = Vec::with_capacity(nbasis);
    for b in 0..nbasis {
        let start = offsets[b] as usize;
        let end = offsets[b + 1] as usize;
        let mut pexps = Vec::with_capacity(end - start);
        let mut pcoefs = Vec::with_capacity(end - start);
        let mut pnorms = Vec::with_capacity(end - start);
        let mut ppow = Vec::with_capacity(end - start);
        for p in start..end {
            pexps.push(exps[p]);
            pcoefs.push(coefs[p]);
            pnorms.push(norms[p]);
            ppow.push([powers[[p, 0]], powers[[p, 1]], powers[[p, 2]]]);
        }
        basis.push(Cgbf {
            origin: [origins[[b, 0]], origins[[b, 1]], origins[[b, 2]]],
            exps: pexps,
            coefs: pcoefs,
            norms: pnorms,
            powers: ppow,
        });
    }
    basis
}

fn keys_to_vec(keys: &PyReadonlyArray2<i64>) -> Vec<[usize; 4]> {
    let keys = keys.as_array();
    let nkeys = keys.shape()[0];
    (0..nkeys)
        .map(|r| {
            [
                keys[[r, 0]] as usize,
                keys[[r, 1]] as usize,
                keys[[r, 2]] as usize,
                keys[[r, 3]] as usize,
            ]
        })
        .collect()
}

/// On-site batched ERI: each key `(a, b, c, d)` indexes a single basis.
///
/// Returns one contracted integral `(ab|cd)` per key.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn eri_batch<'py>(
    py: Python<'py>,
    origins: PyReadonlyArray2<f64>,
    prim_offsets: PyReadonlyArray1<i64>,
    exps: PyReadonlyArray1<f64>,
    coefs: PyReadonlyArray1<f64>,
    norms: PyReadonlyArray1<f64>,
    powers: PyReadonlyArray2<i64>,
    keys: PyReadonlyArray2<i64>,
) -> Bound<'py, PyArray1<f64>> {
    let basis = build_basis(&origins, &prim_offsets, &exps, &coefs, &norms, &powers);
    let kv = keys_to_vec(&keys);

    let values = py.allow_threads(|| {
        par_eval(kv.len(), |i| {
            let k = &kv[i];
            contr_coulomb(&basis[k[0]], &basis[k[1]], &basis[k[2]], &basis[k[3]])
        })
    });

    values.into_pyarray(py)
}

/// Intersite batched ERI for eACBN0: each key `(i, k, j, l)` takes `i, k` from
/// basis I and `j, l` from basis J, returning `(ik|jl)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn eri_batch_2c<'py>(
    py: Python<'py>,
    origins_i: PyReadonlyArray2<f64>,
    prim_offsets_i: PyReadonlyArray1<i64>,
    exps_i: PyReadonlyArray1<f64>,
    coefs_i: PyReadonlyArray1<f64>,
    norms_i: PyReadonlyArray1<f64>,
    powers_i: PyReadonlyArray2<i64>,
    origins_j: PyReadonlyArray2<f64>,
    prim_offsets_j: PyReadonlyArray1<i64>,
    exps_j: PyReadonlyArray1<f64>,
    coefs_j: PyReadonlyArray1<f64>,
    norms_j: PyReadonlyArray1<f64>,
    powers_j: PyReadonlyArray2<i64>,
    keys: PyReadonlyArray2<i64>,
) -> Bound<'py, PyArray1<f64>> {
    let basis_i = build_basis(
        &origins_i,
        &prim_offsets_i,
        &exps_i,
        &coefs_i,
        &norms_i,
        &powers_i,
    );
    let basis_j = build_basis(
        &origins_j,
        &prim_offsets_j,
        &exps_j,
        &coefs_j,
        &norms_j,
        &powers_j,
    );
    let kv = keys_to_vec(&keys);

    let values = py.allow_threads(|| {
        par_eval(kv.len(), |i| {
            let k = &kv[i];
            contr_coulomb(
                &basis_i[k[0]],
                &basis_i[k[1]],
                &basis_j[k[2]],
                &basis_j[k[3]],
            )
        })
    });

    values.into_pyarray(py)
}

/// Number of threads the batched kernels will use (the `PAOFLOW_ACBN0_THREADS`
/// pool size when set, otherwise the global rayon pool size).
#[pyfunction]
fn thread_count() -> usize {
    match custom_pool() {
        Some(pool) => pool.current_num_threads(),
        None => rayon::current_num_threads(),
    }
}

/// The compiled extension module, importable as `paoflow_acbn0_rs`.
#[pymodule]
pub fn paoflow_acbn0_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(eri_batch, m)?)?;
    m.add_function(wrap_pyfunction!(eri_batch_2c, m)?)?;
    m.add_function(wrap_pyfunction!(thread_count, m)?)?;
    Ok(())
}
