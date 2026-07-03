//! PyO3 bindings: batched, `rayon`-parallel ERI evaluation.
//!
//! The basis crosses the FFI boundary as flat CSR-style NumPy arrays so that a
//! whole rank-local chunk of unique integral keys is evaluated in one call,
//! amortising the Python<->Rust overhead and releasing the GIL during compute.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::OnceLock;

use crate::acbn0::eri::{contr_coulomb, Cgbf};
use crate::epsilon::{eps, jdos};

/// Optional dedicated rayon pool sized by `PAOFLOW_RS_THREADS`.
///
/// When the variable is unset (or invalid / `<= 0`) the legacy
/// `PAOFLOW_ACBN0_THREADS` is consulted for backward compatibility, then the
/// global rayon pool is used (which itself honours `RAYON_NUM_THREADS`).
/// Setting either lets callers cap intra-rank threads to avoid oversubscription
/// when launching many MPI ranks per node.
fn custom_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        std::env::var("PAOFLOW_RS_THREADS")
            .or_else(|_| std::env::var("PAOFLOW_ACBN0_THREADS"))
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .map(|n| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .expect("failed to build PAOFLOW_RS_THREADS rayon pool")
            })
    })
    .as_ref()
}

/// Evaluate `f(i)` for `i in 0..n` in parallel, honouring the optional
/// `PAOFLOW_RS_THREADS` pool. The caller is expected to have released the
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
fn acbn0_eri_batch<'py>(
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
fn acbn0_eri_batch_2c<'py>(
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

/// Number of threads the batched kernels will use (the `PAOFLOW_RS_THREADS`
/// pool size when set, otherwise the global rayon pool size).
#[pyfunction]
fn thread_count() -> usize {
    match custom_pool() {
        Some(pool) => pool.current_num_threads(),
        None => rayon::current_num_threads(),
    }
}

/// Interband dielectric inner loop over the local k-slice.
///
/// `ek`/`fn_occ` are `(nk, nbnd)`; `pksp2` is `(nk, nbnd, nbnd)`. `deltakp2`
/// (adaptive width) and `fnf` (metal occupation) are optional. Returns
/// `(epsi, epsr, drude_weight)`; the caller scales and applies the Drude
/// profile.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (ek, fn_occ, pksp2, ene, intersmear, th0, th1, spin_factor, deltakp2=None, eta_floor=0.0, fnf=None))]
fn epsilon_eps_loop<'py>(
    py: Python<'py>,
    ek: PyReadonlyArray2<f64>,
    fn_occ: PyReadonlyArray2<f64>,
    pksp2: PyReadonlyArray3<f64>,
    ene: PyReadonlyArray1<f64>,
    intersmear: f64,
    th0: f64,
    th1: f64,
    spin_factor: f64,
    deltakp2: Option<PyReadonlyArray3<f64>>,
    eta_floor: f64,
    fnf: Option<PyReadonlyArray2<f64>>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64) {
    let ek = ek.as_array();
    let nk = ek.shape()[0];
    let nbnd = ek.shape()[1];
    let ek_v: Vec<f64> = ek.iter().copied().collect();
    let fn_v: Vec<f64> = fn_occ.as_array().iter().copied().collect();
    let pk_v: Vec<f64> = pksp2.as_array().iter().copied().collect();
    let ene_v: Vec<f64> = ene.as_array().iter().copied().collect();
    let dk_v = deltakp2.map(|a| a.as_array().iter().copied().collect::<Vec<f64>>());
    let fnf_v = fnf.map(|a| a.as_array().iter().copied().collect::<Vec<f64>>());

    let (epsi, epsr, drude) = py.allow_threads(|| {
        eps::eps_loop(
            &ek_v,
            &fn_v,
            &pk_v,
            nk,
            nbnd,
            &ene_v,
            intersmear,
            th0,
            th1,
            spin_factor,
            dk_v.as_deref(),
            eta_floor,
            fnf_v.as_deref(),
        )
    });
    (epsi.into_pyarray(py), epsr.into_pyarray(py), drude)
}

/// JDOS inner loop over the local k-slice. `smeartype` 0 = Gaussian, 1 =
/// Lorentzian. Returns `(jdos, count)` partials; the caller reduces and
/// normalises.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn epsilon_jdos_loop<'py>(
    py: Python<'py>,
    ek: PyReadonlyArray2<f64>,
    fn_occ: PyReadonlyArray2<f64>,
    kweights: PyReadonlyArray1<f64>,
    ene: PyReadonlyArray1<f64>,
    intersmear: f64,
    smeartype: u8,
) -> (Bound<'py, PyArray1<f64>>, f64) {
    let ek = ek.as_array();
    let nk = ek.shape()[0];
    let nbnd = ek.shape()[1];
    let ek_v: Vec<f64> = ek.iter().copied().collect();
    let fn_v: Vec<f64> = fn_occ.as_array().iter().copied().collect();
    let kw_v: Vec<f64> = kweights.as_array().iter().copied().collect();
    let ene_v: Vec<f64> = ene.as_array().iter().copied().collect();

    let (jd, count) = py.allow_threads(|| {
        jdos::jdos_loop(&ek_v, &fn_v, &kw_v, nk, nbnd, &ene_v, intersmear, smeartype)
    });
    (jd.into_pyarray(py), count)
}

/// The compiled extension module, importable as `paoflow_rs`.
#[pymodule]
pub fn paoflow_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(acbn0_eri_batch, m)?)?;
    m.add_function(wrap_pyfunction!(acbn0_eri_batch_2c, m)?)?;
    m.add_function(wrap_pyfunction!(epsilon_eps_loop, m)?)?;
    m.add_function(wrap_pyfunction!(epsilon_jdos_loop, m)?)?;
    m.add_function(wrap_pyfunction!(thread_count, m)?)?;
    Ok(())
}
