//! Joint density-of-states inner loop.
//!
//! Faithful port of the vectorised `jdos_loop` in `do_epsilon.py`. The caller
//! passes its rank-local k-slice; this returns the partial weighted JDOS and
//! the partial oscillator-strength `count`. Python performs the MPI reductions
//! and the final `nkpnts / count / spin_factor` normalisation.

use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::OnceLock;

fn pool() -> Option<&'static rayon::ThreadPool> {
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

/// `ek`/`fn_occ` are `(nk, nbnd)` row-major; `kweights` is `(nk,)`. `smeartype`
/// is 0 for Gaussian, 1 for Lorentzian. Returns `(jdos[ne], count)` partials.
#[allow(clippy::too_many_arguments)]
pub fn jdos_loop(
    ek: &[f64],
    fn_occ: &[f64],
    kweights: &[f64],
    nk: usize,
    nbnd: usize,
    ene: &[f64],
    intersmear: f64,
    smeartype: u8,
) -> (Vec<f64>, f64) {
    let ne = ene.len();
    let sqrt_pi = PI.sqrt();

    let per_k = |ik: usize| -> (Vec<f64>, f64) {
        let mut jdos = vec![0.0; ne];
        let mut count = 0.0;
        let kb = ik * nbnd;
        let wk = kweights[ik];
        for b2 in 0..nbnd {
            let e2 = ek[kb + b2];
            let f2 = fn_occ[kb + b2];
            for b1 in 0..nbnd {
                let f1 = fn_occ[kb + b1];
                let d = e2 - ek[kb + b1];
                if f1 <= 1.0e-4 || f2 >= 2.0 || d <= 1e-10 {
                    continue;
                }
                let f_nm = f1 - f2;
                count += f_nm;
                for j in 0..ne {
                    let kernel = if smeartype == 0 {
                        let x = (ene[j] - d) / intersmear;
                        (-x * x).exp() / intersmear / sqrt_pi
                    } else {
                        let dd = d - ene[j];
                        intersmear / (PI * (dd * dd + intersmear * intersmear))
                    };
                    jdos[j] += wk * f_nm * kernel;
                }
            }
        }
        (jdos, count)
    };

    let reduce = || {
        (0..nk).into_par_iter().map(per_k).reduce(
            || (vec![0.0; ne], 0.0),
            |mut a, b| {
                for j in 0..ne {
                    a.0[j] += b.0[j];
                }
                a.1 += b.1;
                a
            },
        )
    };

    match pool() {
        Some(pool) => pool.install(reduce),
        None => reduce(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_matches_pairs() {
        let nk = 1;
        let nbnd = 2;
        let ek = vec![0.0, 1.0];
        let fnocc = vec![1.0, 0.0];
        let kw = vec![1.0];
        let ene = vec![1.0];
        let (_, count) = jdos_loop(&ek, &fnocc, &kw, nk, nbnd, &ene, 0.1, 0);
        // pair b2=1,b1=0: f1=1>1e-4, f2=0<2, d=1>0 -> f_nm=1
        assert!((count - 1.0).abs() < 1e-12);
    }
}
