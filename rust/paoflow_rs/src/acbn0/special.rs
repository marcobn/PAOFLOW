//! Special functions and Cartesian angular-momentum helpers for the
//! two-electron Coulomb integrals.
//!
//! This is a direct, numerically faithful port of the reference
//! implementation in `src/PAOFLOW/utils/pyints.py`. Every routine here is
//! validated to `<1e-12` relative error against the Python original via the
//! golden-value tests in `tests/golden.rs`.

const ITMAX: usize = 100;
const EPS: f64 = 3.0e-7;
const FPMIN: f64 = 1.0e-30;

/// Natural log of the Gamma function (Lanczos; Numerical Recipes 6.1).
pub fn gammln(x: f64) -> f64 {
    const COF: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.120_865_097_386_617_9e-2,
        -0.539_523_938_495_3e-5,
    ];
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    let mut ser = 1.000_000_000_190_015;
    let mut xt = x;
    for &c in COF.iter() {
        xt += 1.0;
        ser += c / xt;
    }
    -tmp + (2.506_628_274_631_000_5 * ser / x).ln()
}

/// Series representation of the incomplete Gamma function (NumRec 6.2).
/// Returns `(gamser, gln)`.
fn gser(a: f64, x: f64) -> (f64, f64) {
    let gln = gammln(a);
    if x < 0.0 {
        panic!("gser: x must be >= 0, got {x}");
    }
    if x == 0.0 {
        return (0.0, gln);
    }
    let mut ap = a;
    let mut delt = 1.0 / a;
    let mut tsum = delt;
    for _ in 0..ITMAX {
        ap += 1.0;
        delt *= x / ap;
        tsum += delt;
        if delt.abs() < tsum.abs() * EPS {
            break;
        }
    }
    let gamser = tsum * (-x + a * x.ln() - gln).exp();
    (gamser, gln)
}

/// Continued-fraction representation of the incomplete Gamma function
/// (NumRec 6.2). Returns `(gammcf, gln)`.
fn gcf(a: f64, x: f64) -> (f64, f64) {
    let gln = gammln(a);
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=ITMAX {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let delt = d * c;
        h *= delt;
        if (delt - 1.0).abs() < EPS {
            break;
        }
    }
    let gammcf = h * (-x + a * x.ln() - gln).exp();
    (gammcf, gln)
}

/// Regularised incomplete Gamma function P(a; x) (NumRec 6.2).
/// Returns `(p, gln)`.
fn gammp(a: f64, x: f64) -> (f64, f64) {
    if x <= 0.0 || a < 0.0 {
        panic!("gammp: require x > 0 and a >= 0, got x={x}, a={a}");
    }
    if x < a + 1.0 {
        gser(a, x)
    } else {
        let (gammcf, gln) = gcf(a, x);
        (1.0 - gammcf, gln)
    }
}

/// Unnormalised (lower) incomplete Gamma function gamma(a, x).
pub fn gamm_inc(a: f64, x: f64) -> f64 {
    let (gammap, gln) = gammp(a, x);
    gammap * gln.exp()
}

/// Boys function F_m(x) = gamma(m + 1/2, x) / (2 x^(m+1/2)).
pub fn fgamma(m: i64, x: f64) -> f64 {
    let mp5 = m as f64 + 0.5;
    let x = x.abs().max(1e-8);
    gamm_inc(mp5, x) / (2.0 * x.powf(mp5))
}

/// Exact factorials 0! .. 18! as f64 (all representable below 2^53, so these
/// are bit-identical to the accumulating loop for the angular-momentum range
/// encountered in practice, where arguments stay well under 18).
const FACT_TABLE: [f64; 19] = [
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5040.0,
    40320.0,
    362880.0,
    3628800.0,
    39916800.0,
    479001600.0,
    6227020800.0,
    87178291200.0,
    1307674368000.0,
    20922789888000.0,
    355687428096000.0,
    6402373705728000.0,
];

/// Factorial as f64 (matches Python's int -> float division semantics).
pub fn fact(i: i64) -> f64 {
    if i < 2 {
        1.0
    } else if (i as usize) < FACT_TABLE.len() {
        FACT_TABLE[i as usize]
    } else {
        let mut val = FACT_TABLE[FACT_TABLE.len() - 1];
        for k in (FACT_TABLE.len() as i64)..=i {
            val *= k as f64;
        }
        val
    }
}

/// fact(a) / fact(b) / fact(a - 2b).
pub fn fact_ratio(a: i64, b: i64) -> f64 {
    fact(a) / fact(b) / fact(a - 2 * b)
}

/// Binomial coefficient fact(a) / fact(b) / fact(a - b).
pub fn binomial(a: i64, b: i64) -> f64 {
    fact(a) / fact(b) / fact(a - b)
}

/// Binomial prefactor (Augspurger & Dykstra).
pub fn binomial_prefactor(s: i64, ia: i64, ib: i64, xpa: f64, xpb: f64) -> f64 {
    let mut bsum = 0.0;
    for t in 0..=s {
        if s - ia <= t && t <= ib {
            let sgn = xpa.powi((ia - s + t) as i32) * xpb.powi((ib - t) as i32);
            bsum += sgn * binomial(ia, s - t) * binomial(ib, t);
        }
    }
    bsum
}

/// THO B0 coefficient.
pub fn b0(i: i64, r: i64, g: f64) -> f64 {
    fact_ratio(i, r) * (4.0 * g).powi((r - i) as i32)
}

/// THO fB coefficient.
#[allow(clippy::too_many_arguments)]
pub fn f_b(i: i64, l1: i64, l2: i64, p: f64, a: f64, b: f64, r: i64, g: f64) -> f64 {
    binomial_prefactor(i, l1, l2, p - a, p - b) * b0(i, r, g)
}

/// THO eq. 2.22 Cartesian B-array along one axis.
#[allow(clippy::too_many_arguments)]
pub fn b_array(
    l1: i64,
    l2: i64,
    l3: i64,
    l4: i64,
    p: f64,
    a: f64,
    b: f64,
    q: f64,
    c: f64,
    d: f64,
    g1: f64,
    g2: f64,
    delta: f64,
) -> Vec<f64> {
    let ind_max = (l1 + l2 + l3 + l4 + 1) as usize;
    let mut bvec = vec![0.0_f64; ind_max];
    for i1 in 0..=(l1 + l2) {
        for i2 in 0..=(l3 + l4) {
            for r1 in 0..=(i1 / 2) {
                for r2 in 0..=(i2 / 2) {
                    let umax = (i1 + i2) / 2 - r1 - r2;
                    for u in 0..=umax {
                        let dm = i1 + i2 - 2 * (r1 + r2);
                        let ind = (dm - u) as usize;
                        let fr = fact_ratio(dm, u);
                        let pqp = (q - p).powi((dm - 2 * u) as i32);
                        let pdelta = delta.powi((dm - u) as i32);
                        let parity = i2 + u;
                        let sign = if parity % 2 == 0 { 1.0 } else { -1.0 };
                        let fb1 = f_b(i1, l1, l2, p, a, b, r1, g1);
                        let fb2 = f_b(i2, l3, l4, q, c, d, r2, g2);
                        bvec[ind] += sign * fb1 * fb2 * fr * pqp / pdelta;
                    }
                }
            }
        }
    }
    bvec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gammln_half_is_log_sqrt_pi() {
        // Gamma(1/2) = sqrt(pi)
        let got = gammln(0.5).exp();
        let want = std::f64::consts::PI.sqrt();
        assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
    }

    #[test]
    fn fact_basic() {
        assert_eq!(fact(0), 1.0);
        assert_eq!(fact(1), 1.0);
        assert_eq!(fact(5), 120.0);
    }

    #[test]
    fn fgamma_zero_limit() {
        // F_0(x->0) = 1
        let got = fgamma(0, 1e-8);
        assert!((got - 1.0).abs() < 1e-6, "got {got}");
    }
}
