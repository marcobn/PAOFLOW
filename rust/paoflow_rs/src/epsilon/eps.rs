//! Interband dielectric-function inner loop (Kubo–Greenwood).
//!
//! Faithful port of the vectorised `eps_loop` in `do_epsilon.py`. Per local
//! k-point we accumulate the imaginary and real interband integrands over all
//! ordered band pairs `(n=b2, m=b1)` that survive the occupation mask, and the
//! diagonal metal (Drude) weight. The metal frequency profile itself is applied
//! once in Python after reduction, so this returns the scalar `drude_weight`.

use rayon::prelude::*;
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

/// Per-k partial result: (epsi[ne], epsr[ne], drude_weight).
struct Partial {
    epsi: Vec<f64>,
    epsr: Vec<f64>,
    drude: f64,
}

/// Inputs are flat row-major slices. `ek`/`fn_occ` are `(nk, nbnd)`; `pksp2`
/// (= `Re(P.T * Q)`, indexed `[b2, b1]`) is `(nk, nbnd, nbnd)`. `deltakp2`
/// (indexed `[b1, b2]`) is the optional adaptive width and `fnf` the optional
/// metal occupation derivative, both same shapes as above when present.
#[allow(clippy::too_many_arguments)]
pub fn eps_loop(
    ek: &[f64],
    fn_occ: &[f64],
    pksp2: &[f64],
    nk: usize,
    nbnd: usize,
    ene: &[f64],
    intersmear: f64,
    th0: f64,
    th1: f64,
    spin_factor: f64,
    deltakp2: Option<&[f64]>,
    eta_floor: f64,
    fnf: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>, f64) {
    let ne = ene.len();
    let ene2: Vec<f64> = ene.iter().map(|w| w * w).collect();

    let per_k = |ik: usize| -> Partial {
        let mut epsi = vec![0.0; ne];
        let mut epsr = vec![0.0; ne];
        let mut drude = 0.0;
        let kb = ik * nbnd;
        let kbb = ik * nbnd * nbnd;
        for b2 in 0..nbnd {
            let e2 = ek[kb + b2];
            let f2 = fn_occ[kb + b2];
            for b1 in 0..nbnd {
                if b1 == b2 {
                    continue;
                }
                let f1 = fn_occ[kb + b1];
                if (f2 - f1).abs() <= th0 || f1 <= th1 || f2 >= spin_factor {
                    continue;
                }
                let d = e2 - ek[kb + b1];
                let pk = pksp2[kbb + b2 * nbnd + b1];
                let eta = match deltakp2 {
                    Some(dk) => dk[kbb + b1 * nbnd + b2].max(eta_floor),
                    None => intersmear,
                };
                let pkf = pk * f1;
                let eta2 = eta * eta;
                for j in 0..ne {
                    let dm = d * d - ene2[j];
                    let denom = (dm * dm + eta2 * ene2[j]) * d;
                    let common = pkf / denom;
                    epsi[j] += common * eta * ene[j];
                    epsr[j] += common * dm;
                }
            }
        }
        if let Some(fnf) = fnf {
            for b in 0..nbnd {
                drude += pksp2[kbb + b * nbnd + b] * fnf[kb + b];
            }
        }
        Partial { epsi, epsr, drude }
    };

    let reduce = || {
        (0..nk).into_par_iter().map(per_k).reduce(
            || Partial {
                epsi: vec![0.0; ne],
                epsr: vec![0.0; ne],
                drude: 0.0,
            },
            |mut a, b| {
                for j in 0..ne {
                    a.epsi[j] += b.epsi[j];
                    a.epsr[j] += b.epsr[j];
                }
                a.drude += b.drude;
                a
            },
        )
    };

    let p = match pool() {
        Some(pool) => pool.install(reduce),
        None => reduce(),
    };
    (p.epsi, p.epsr, p.drude)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drude_only_when_fnf() {
        let nk = 1;
        let nbnd = 2;
        let ek = vec![0.0, 1.0];
        let fnocc = vec![2.0, 0.0];
        let pksp2 = vec![1.0, 0.5, 0.5, 2.0];
        let ene = vec![0.5, 1.0];
        let fnf = vec![0.3, 0.7];
        let (_, _, drude) = eps_loop(
            &ek,
            &fnocc,
            &pksp2,
            nk,
            nbnd,
            &ene,
            0.1,
            0.0,
            -1.0,
            2.0,
            None,
            0.0,
            Some(&fnf),
        );
        // diag pksp2 = [1,2]; drude = 1*0.3 + 2*0.7 = 1.7
        assert!((drude - 1.7).abs() < 1e-12);
    }
}
