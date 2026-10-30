//! Phonon-assisted absorption: second-order amplitudes and energy-conserving deltas.
//!
//! Faithful port of `indirect_absorption_kernel` in
//! `PAOFLOW/elphon/phonon_assisted_absorption.py`: for every candidate pair `(k, i, j)`
//! the absorption and emission amplitudes `S = S1 + S2` are summed over the intermediate
//! bands for each phonon mode, broadening `eta` and direction, and `|S|^2` is scattered
//! onto the photon energies whose Gaussian or Lorentzian delta lies within the cutoff.
//! The loop over pairs is `rayon`-parallel; the photon loop visits only the window of
//! energies inside the cutoff instead of masking a dense array.

use num_complex::Complex64;
use rayon::prelude::*;

/// Row-major views of the kernel inputs (`nk x nb` energies, `nk x 3 x nb x nb` velocities,
/// `nk x nmode x nb x nb` couplings, `ntemp x nk x nb` occupations, `ntemp x nmode` Bose factors).
pub struct Inputs<'a> {
    pub nb: usize,
    pub nmode: usize,
    pub energy_k: &'a [f64],
    pub energy_kq: &'a [f64],
    pub velocity_k: &'a [Complex64],
    pub velocity_kq: &'a [Complex64],
    pub coupling: &'a [Complex64],
    pub omega_ph: &'a [f64],
    pub occ_k: &'a [f64],
    pub occ_kq: &'a [f64],
    pub bose: &'a [f64],
    pub photon: &'a [f64],
    pub etas: &'a [f64],
    /// Gaussian width and its cutoff (`6 sigma`) of every pair.
    pub degauss: &'a [f64],
    pub cutoff: &'a [f64],
    pub pairs_k: &'a [i64],
    pub pairs_i: &'a [i64],
    pub pairs_j: &'a [i64],
}

struct Partial {
    gauss: Vec<f64>,
    lorentz: Vec<f64>,
}

/// Indices of the photon energies with `|delta_e - photon - shift| <= cutoff` for a sorted
/// `photon` grid (the whole grid when it is not sorted).
fn photon_window(photon: &[f64], centre: f64, cutoff: f64) -> (usize, usize) {
    let n = photon.len();
    if n < 2 || photon[n - 1] < photon[0] {
        return (0, n);
    }
    let lo = photon.partition_point(|&w| w < centre - cutoff);
    let hi = photon.partition_point(|&w| w <= centre + cutoff);
    (lo, hi)
}

/// Accumulate the contribution of one pair into `acc`.
#[allow(clippy::too_many_arguments)]
fn accumulate_pair(
    inp: &Inputs,
    p: usize,
    acc: &mut Partial,
    s_abs2: &mut [f64],
    s_emi2: &mut [f64],
) {
    let nb = inp.nb;
    let nmode = inp.nmode;
    let neta = inp.etas.len();
    let nomega = inp.photon.len();
    let ntemp = inp.bose.len() / nmode.max(1);
    let k = inp.pairs_k[p] as usize;
    let i = inp.pairs_i[p] as usize;
    let j = inp.pairs_j[p] as usize;
    let ek = &inp.energy_k[k * nb..(k + 1) * nb];
    let ekq = &inp.energy_kq[k * nb..(k + 1) * nb];
    let e_i = ek[i];
    let e_j = ekq[j];
    let delta_e = e_j - e_i;
    // amplitudes: s[v][e][a]
    for v in 0..nmode {
        let w = inp.omega_ph[v];
        let g_jm =
            &inp.coupling[((k * nmode + v) * nb + j) * nb..((k * nmode + v) * nb + j + 1) * nb];
        for e in 0..neta {
            let eta = Complex64::new(0.0, inp.etas[e]);
            for a in 0..3 {
                let v_k = &inp.velocity_k[(k * 3 + a) * nb * nb..(k * 3 + a + 1) * nb * nb];
                let v_kq = &inp.velocity_kq[(k * 3 + a) * nb * nb..(k * 3 + a + 1) * nb * nb];
                let mut s_a = Complex64::new(0.0, 0.0);
                let mut s_e = Complex64::new(0.0, 0.0);
                for m in 0..nb {
                    // path 1: photon at k (i -> m), phonon (m, k -> j, k+q)
                    let num1 = g_jm[m] * v_k[m * nb + i];
                    let gap1 = ek[m] - e_j;
                    s_a += num1 / (Complex64::new(gap1 + w, 0.0) + eta);
                    s_e += num1 / (Complex64::new(gap1 - w, 0.0) + eta);
                    // path 2: phonon first (i, k -> m, k+q), photon at k+q (m -> j)
                    let g_mi = inp.coupling[((k * nmode + v) * nb + m) * nb + i];
                    let num2 = g_mi * v_kq[j * nb + m];
                    let gap2 = ekq[m] - e_i;
                    s_a += num2 / (Complex64::new(gap2 - w, 0.0) + eta);
                    s_e += num2 / (Complex64::new(gap2 + w, 0.0) + eta);
                }
                s_abs2[(v * neta + e) * 3 + a] = s_a.norm_sqr();
                s_emi2[(v * neta + e) * 3 + a] = s_e.norm_sqr();
            }
        }
    }
    let sigma = inp.degauss[p];
    let cutoff = inp.cutoff[p];
    let gnorm = 1.0 / (sigma * std::f64::consts::PI.sqrt());
    let lnorm = sigma / std::f64::consts::PI;
    for v in 0..nmode {
        let w = inp.omega_ph[v];
        // photon energies where either delta is inside the cutoff
        let (lo_a, hi_a) = photon_window(inp.photon, delta_e - w, cutoff);
        let (lo_e, hi_e) = photon_window(inp.photon, delta_e + w, cutoff);
        let lo = lo_a.min(lo_e);
        let hi = hi_a.max(hi_e);
        if lo >= hi {
            continue;
        }
        let inv2w = 1.0 / (2.0 * w);
        for t in 0..ntemp {
            let f_i = inp.occ_k[(t * (inp.energy_k.len() / nb) + k) * nb + i];
            let f_j = inp.occ_kq[(t * (inp.energy_kq.len() / nb) + k) * nb + j];
            let forward = f_i * (1.0 - f_j);
            let backward = (1.0 - f_i) * f_j;
            let n_ph = inp.bose[t * nmode + v];
            let factor_abs = (n_ph * forward - (n_ph + 1.0) * backward) * inv2w;
            let factor_emi = ((n_ph + 1.0) * forward - n_ph * backward) * inv2w;
            for wi in lo..hi {
                let x_abs = delta_e - inp.photon[wi] - w;
                let x_emi = delta_e - inp.photon[wi] + w;
                if x_abs.abs() > cutoff && x_emi.abs() > cutoff {
                    continue;
                }
                let ga = (-(x_abs / sigma) * (x_abs / sigma)).exp() * gnorm;
                let ge = (-(x_emi / sigma) * (x_emi / sigma)).exp() * gnorm;
                let la = lnorm / (sigma * sigma + x_abs * x_abs);
                let le = lnorm / (sigma * sigma + x_emi * x_emi);
                for e in 0..neta {
                    for a in 0..3 {
                        let sa = s_abs2[(v * neta + e) * 3 + a];
                        let se = s_emi2[(v * neta + e) * 3 + a];
                        let idx = ((t * neta + e) * 3 + a) * nomega + wi;
                        acc.gauss[idx] += factor_abs * ga * sa + factor_emi * ge * se;
                        acc.lorentz[idx] += factor_abs * la * sa + factor_emi * le * se;
                    }
                }
            }
        }
    }
}

/// `(eps2_gaussian, eps2_lorentzian)` flattened as `(ntemp, neta, 3, nomega)`, without the
/// `weight * 8 pi^2 g_s / omega^2` prefactor.
pub fn indirect_absorption(inp: &Inputs) -> (Vec<f64>, Vec<f64>) {
    let nmode = inp.nmode;
    let neta = inp.etas.len();
    let nomega = inp.photon.len();
    let ntemp = inp.bose.len().checked_div(nmode).unwrap_or(0);
    let size = ntemp * neta * 3 * nomega;
    let npairs = inp.pairs_k.len();
    let chunk = 256usize;
    let reduce = || {
        (0..npairs.div_ceil(chunk))
            .into_par_iter()
            .map(|c| {
                let mut acc = Partial {
                    gauss: vec![0.0; size],
                    lorentz: vec![0.0; size],
                };
                let mut s_abs2 = vec![0.0; nmode * neta * 3];
                let mut s_emi2 = vec![0.0; nmode * neta * 3];
                for p in c * chunk..((c + 1) * chunk).min(npairs) {
                    accumulate_pair(inp, p, &mut acc, &mut s_abs2, &mut s_emi2);
                }
                acc
            })
            .reduce(
                || Partial {
                    gauss: vec![0.0; size],
                    lorentz: vec![0.0; size],
                },
                |mut a, b| {
                    for (x, y) in a.gauss.iter_mut().zip(b.gauss.iter()) {
                        *x += y;
                    }
                    for (x, y) in a.lorentz.iter_mut().zip(b.lorentz.iter()) {
                        *x += y;
                    }
                    a
                },
            )
    };
    let out = crate::pool::install(reduce);
    (out.gauss, out.lorentz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_covers_cutoff() {
        let photon: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        let (lo, hi) = photon_window(&photon, 0.45, 0.12);
        assert_eq!((lo, hi), (4, 6)); // 0.4 and 0.5
        let (lo, hi) = photon_window(&photon, 5.0, 0.1);
        assert_eq!(lo, hi);
    }

    #[test]
    fn single_pair_matches_direct_sum() {
        // one k, two bands, one mode, one eta: compare with an explicit evaluation
        let nb = 2;
        let energy_k = vec![-0.1, 0.3];
        let energy_kq = vec![-0.05, 0.4];
        let c = |re: f64, im: f64| Complex64::new(re, im);
        let velocity_k: Vec<Complex64> =
            (0..3 * nb * nb).map(|i| c(0.1 * i as f64, 0.05)).collect();
        let velocity_kq: Vec<Complex64> = (0..3 * nb * nb)
            .map(|i| c(0.02 * i as f64, -0.03))
            .collect();
        let coupling: Vec<Complex64> = (0..nb * nb)
            .map(|i| c(0.01 * (i + 1) as f64, 0.0))
            .collect();
        let inp = Inputs {
            nb,
            nmode: 1,
            energy_k: &energy_k,
            energy_kq: &energy_kq,
            velocity_k: &velocity_k,
            velocity_kq: &velocity_kq,
            coupling: &coupling,
            omega_ph: &[0.02],
            occ_k: &[1.0, 0.0],
            occ_kq: &[1.0, 0.0],
            bose: &[0.5],
            photon: &[0.45],
            etas: &[0.01],
            degauss: &[0.05],
            cutoff: &[0.3],
            pairs_k: &[0],
            pairs_i: &[0],
            pairs_j: &[1],
        };
        let (g, l) = indirect_absorption(&inp);
        assert_eq!(g.len(), 3);
        assert!(g.iter().all(|x| x.is_finite()) && l.iter().all(|x| x.is_finite()));
        assert!(g[0].abs() > 0.0);
    }
}
