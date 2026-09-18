//! Contracted four-centre two-electron Coulomb integrals (ERIs).
//!
//! Direct port of `coulomb_repulsion` and `contr_coulomb` from
//! `src/PAOFLOW/utils/pyints.py`, plus a [`Cgbf`] container and a batched
//! evaluator suitable for FFI + `rayon` parallelism.

use crate::acbn0::special::{b_array, fgamma};

#[inline]
fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

#[inline]
fn gaussian_product_center(a1: f64, a: &[f64; 3], a2: f64, b: &[f64; 3]) -> [f64; 3] {
    let g = a1 + a2;
    [
        (a1 * a[0] + a2 * b[0]) / g,
        (a1 * a[1] + a2 * b[1]) / g,
        (a1 * a[2] + a2 * b[2]) / g,
    ]
}

/// Single-primitive four-centre Coulomb repulsion integral (THO).
#[allow(clippy::too_many_arguments)]
pub fn coulomb_repulsion(
    axyz: &[f64; 3],
    anorm: f64,
    almn: &[i64; 3],
    aalpha: f64,
    bxyz: &[f64; 3],
    bnorm: f64,
    blmn: &[i64; 3],
    balpha: f64,
    cxyz: &[f64; 3],
    cnorm: f64,
    clmn: &[i64; 3],
    calpha: f64,
    dxyz: &[f64; 3],
    dnorm: f64,
    dlmn: &[i64; 3],
    dalpha: f64,
) -> f64 {
    let rab = dist(axyz, bxyz);
    let rcd = dist(cxyz, dxyz);
    let pxyz = gaussian_product_center(aalpha, axyz, balpha, bxyz);
    let qxyz = gaussian_product_center(calpha, cxyz, dalpha, dxyz);
    let rpq = dist(&pxyz, &qxyz);
    let gamma1 = aalpha + balpha;
    let gamma2 = calpha + dalpha;
    let delta = (1.0 / gamma1 + 1.0 / gamma2) / 4.0;

    let mut bx = [Vec::new(), Vec::new(), Vec::new()];
    for i in 0..3 {
        bx[i] = b_array(
            almn[i], blmn[i], clmn[i], dlmn[i], pxyz[i], axyz[i], bxyz[i], qxyz[i], cxyz[i],
            dxyz[i], gamma1, gamma2, delta,
        );
    }
    let (bxa, bya, bza) = (&bx[0], &bx[1], &bx[2]);

    let mut bsum = 0.0;
    let ni = almn[0] + blmn[0] + clmn[0] + dlmn[0];
    let nj = almn[1] + blmn[1] + clmn[1] + dlmn[1];
    let nk = almn[2] + blmn[2] + clmn[2] + dlmn[2];
    let arg = rpq / delta / 4.0;
    for i in 0..=ni {
        for j in 0..=nj {
            for k in 0..=nk {
                bsum +=
                    bxa[i as usize] * bya[j as usize] * bza[k as usize] * fgamma(i + j + k, arg);
            }
        }
    }

    let norm = anorm * bnorm * cnorm * dnorm;
    let g2 = gamma1 * gamma2 * (gamma1 + gamma2).sqrt();
    let e1 = (-aalpha * balpha * rab / gamma1).exp();
    let e2 = (-calpha * dalpha * rcd / gamma2).exp();
    2.0 * std::f64::consts::PI.powf(2.5) * e1 * e2 * bsum * norm / g2
}

/// A contracted Gaussian basis function.
#[derive(Clone, Debug)]
pub struct Cgbf {
    pub origin: [f64; 3],
    pub exps: Vec<f64>,
    pub coefs: Vec<f64>,
    pub norms: Vec<f64>,
    pub powers: Vec<[i64; 3]>,
}

/// Contracted four-centre Coulomb integral over four [`Cgbf`].
pub fn contr_coulomb(a: &Cgbf, b: &Cgbf, c: &Cgbf, d: &Cgbf) -> f64 {
    let mut jij = 0.0;
    for i in 0..a.exps.len() {
        for j in 0..b.exps.len() {
            for k in 0..c.exps.len() {
                for l in 0..d.exps.len() {
                    let incr = coulomb_repulsion(
                        &a.origin,
                        a.norms[i],
                        &a.powers[i],
                        a.exps[i],
                        &b.origin,
                        b.norms[j],
                        &b.powers[j],
                        b.exps[j],
                        &c.origin,
                        c.norms[k],
                        &c.powers[k],
                        c.exps[k],
                        &d.origin,
                        d.norms[l],
                        &d.powers[l],
                        d.exps[l],
                    );
                    jij += incr * a.coefs[i] * b.coefs[j] * c.coefs[k] * d.coefs[l];
                }
            }
        }
    }
    jij
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssss_is_positive() {
        let s = Cgbf {
            origin: [0.0, 0.0, 0.0],
            exps: vec![1.0],
            coefs: vec![1.0],
            norms: vec![1.0],
            powers: vec![[0, 0, 0]],
        };
        let v = contr_coulomb(&s, &s, &s, &s);
        assert!(v > 0.0, "(ss|ss) should be positive, got {v}");
    }
}
