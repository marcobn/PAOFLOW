# Kubo–Bastin linear response (`kubo_response`, `kubo_optical_response`)

PAOFLOW computes the conductivity, the anomalous, spin and orbital Hall conductivities, the
Rashba–Edelstein response and the spin–orbit torque from one formula: the Kubo–Bastin response at
constant broadening.
- `kubo_response` gives the static response as a function of the chemical potential.
- `kubo_optical_response` gives the frequency-dependent (Kubo–Greenwood) response.

Every older route is a limit of it: `anomalous_Hall`, `spin_Hall`, `orbital_Hall`, and the tutorial03
equations of `linear_response`.

- Code:
  - `response/kubo_kernels.py`: the kernels;
  - `response/kubo_operators.py`: observables, components, units and the charge sign;
  - `response/kubo_bastin.py`: the static k-loop, MPI reduction and writer;
  - `response/kubo_optical.py`: the optical one;
  - `PAOFLOW.kubo_response` and `PAOFLOW.kubo_optical_response`.
- Tests: `tests/unit/response/test_kubo_bastin.py`, `test_kubo_optical.py`, `test_kubo_legacy_parity.py`,
  `test_linear_response_fixes.py`.

## Convention (read this first)

1. **Formula:** the Kubo formula for the response of an observable $A$ to an electric field, with
   $\hbar v = \partial H/\partial k$ (PAOFLOW's `dHksp` is $-\partial H/\partial k$; the code takes care
   of it).
2. **Charge:** the carriers are electrons, $q = -|e|$.
   - **Charge current:** $\sigma = q^2 K$, positive on the diagonal. Same sign as Yao et al., PRL 92,
     037204 (2004).
   - **Everything else** (spin and orbital currents and densities, spin torque, custom operators):
     $\chi = (q/|e|)\,K = -K$. Same sign as Guo et al., PRL 100, 096401 (2008): Pt has a positive spin Hall
     conductivity, +2012 (ħ/e) S/cm here, equal to `spin_Hall`.
3. **Magnets:** the response is that of the spin configuration of the run. QE's positive
   `starting_magnetization` puts the electron spins along +z, so the magnetic moment points along −z.
   - Papers that quote the magnetization $M$ (Yao: "magnetization along [001]", experiments) mean the
     moment. bcc Fe from QE with `starting_magnetization` > 0 therefore gives $\sigma_{xy} < 0$ (−1175 S/cm in
     integration example04), which is Yao's +751 S/cm for the reversed moment.
   - To compare directly, use a negative `starting_magnetization` or flip the sign of the T-odd outputs.
4. **Older routes** (details in the table below):
   - `anomalous_Hall`, its `MCD*` files and tutorial03 Eq. 3 give $\sigma_{xy}$ for the moment along the
     run's spin axis, i.e. the opposite sign.
   - The Rashba–Edelstein routes of `linear_response` take `dHksp` (= −ħv) as the velocity, also the
     opposite sign.
   - Conductivity, spin Hall and orbital Hall agree.

Every `kubo_*` output file states this convention in its header.

## The formula

For the response of an observable $A$ to an electric field along $b$, coupled through
$\hbar v_b = \partial H/\partial k_b$, with a constant broadening $\eta = \hbar/2\tau$, the Bastin
formula at one k-point is

$$
\chi_{AB}(\mu) = -2\int d\varepsilon\, f(\varepsilon)\,
\mathrm{Im\,Tr}\big[A\,\partial_\varepsilon G^R(\varepsilon)\,\hbar v_b\,\delta(\varepsilon-H)\big],
\qquad G^R = (\varepsilon - H + i\eta)^{-1}.
$$

In the band basis, with $P_{nm} = A_{nm}(\hbar v_b)_{mn}$ and the normalised Lorentzian
$L_n(\varepsilon) = (\eta/\pi)/((\varepsilon-\varepsilon_n)^2+\eta^2)$, it splits exactly into

$$
\chi = \underbrace{\pi\sum_{nm}\mathrm{Re}P_{nm}\int(-f')\,L_nL_m}_{\text{surface}}
\;+\;\underbrace{2\sum_{nm}\mathrm{Im}P_{nm}\int f\,L_n\,
\mathrm{Re}\,(\varepsilon-\varepsilon_m+i\eta)^{-2}}_{\text{sea}} .
$$

The symmetric part of $P$ gives only the surface term, the antisymmetric part only the sea term. These are
the "true" Fermi-surface and Fermi-sea terms of Bonbien and Manchon. The split follows from
symmetrising the $\mathrm{Re}P$ part and integrating it by parts.

Terms of `kubo_response(terms=...)`:

| term | meaning |
|---|---|
| `total` | surface + sea, the full response at constant $\eta$ |
| `surface` | Fermi-surface term (tutorial03 Eq. 1 is this at $T = 0$) |
| `sea` | Fermi-sea term, exact at finite $\eta$ |
| `surface_intraband`, `surface_interband` | $n = m$ and $n \ne m$ parts of `surface` |
| `intrinsic` | the $\eta\to0$ sea, $2\sum_n f_n\sum_m \mathrm{Im}P_{nm}/(\varepsilon_{nm}^2+\delta^2)$, with the run's smearing at $T = 0$ or Fermi–Dirac occupations |

## Closed forms (`kubo_kernels.py`)

- **Surface at $T = 0$:** $\pi\sum_{nm}\mathrm{Re}P_{nm}L_n(\mu)L_m(\mu)$. Per k-point it is a
  quadratic form in the Lorentzian weights.
- **Sea at $T = 0$:** write $S_{nm}(\mu) = \int_{-\infty}^{\mu}L_n\,\mathrm{Re}(\varepsilon-\varepsilon_m+i\eta)^{-2}$
  with the poles $s_n = \varepsilon_n + i\eta$ and $q_m = \varepsilon_m - i\eta$. Partial fractions give

  $$
  S_{nm} = \frac{1}{2\pi}\,\mathrm{Im}\big[I(s_n,q_m) + I(s_n,\bar q_m)\big],\qquad
  I(a,b) = \frac{\ell(a)-\ell(b)}{(a-b)^2} + \frac{1}{(a-b)(\mu-b)},
  $$

  with $\ell(p) = \log(\mu-p) + i\pi\,\mathrm{sgn\,Im}\,p$.
  - The pair coefficients depend on $\varepsilon_n-\varepsilon_m$ only, and the $\mu$ dependence on single
    bands, so the cost is $O(n_b^2 + n_b n_\mu)$ per k-point.
  - For $|\varepsilon_n-\varepsilon_m| < \eta/4$ (degenerate and near-degenerate pairs) the second
    integral uses the series $I(b+d,b) = -\sum_j d^j/((j+2)(\mu-b)^{j+2})$.
  - As $\eta\to0$, $S_{nm}\to\theta(\mu-\varepsilon_n)/\varepsilon_{nm}^2$, the Berry-curvature form.
- **Finite temperature:** the formula is linear in $f$, and
  $f_T(\varepsilon-\mu) = \int d\mu'\,\theta(\mu'-\varepsilon)(-\partial_{\mu'}f_T(\mu'-\mu))$. Hence
  $\chi(\mu,T) = \int d\mu'\,(-\partial_{\mu'}f)\,\chi(\mu',0)$ is exact.
  - `kubo_bastin` evaluates the $T = 0$ kernels on a fine grid (spacing $\min(\eta,kT)/4$, $\pm36\,kT$
    around the window) and convolves after the k-sum.
  - Below $kT = \eta/100$ the Sommerfeld correction is under $2\times10^{-4}$, and the $T = 0$ kernels are used.
- **Checks:**
  - Against a brute-force quadrature of the Bastin integral, the kernels agree to $10^{-15}$ at $T = 0$ (random
    6-band matrices with a near-degenerate and a degenerate pair) and to $10^{-7}$ at $T > 0$.
  - The $\eta\to0$ sea equals $-\sum_n f_n\Omega_n$ to $10^{-4}$ at $\eta = 10^{-5}$.

## Observables, components, units (`kubo_operators.py`)

| observable | component | operator $A$ | unit (bulk; `twoD`) |
|---|---|---|---|
| `charge_current` | current, field (`'xy'`) | $\hbar v_a$ | S/cm; $e^2/\hbar$ |
| `spin_current` | spin, current, field (`'zxy'`) | $\{S_s,\hbar v_a\}/2$ | (ħ/e) S/cm; (ħ/e) $e^2/\hbar$ |
| `orbital_current` | l, current, field | $\{L_l,\hbar v_a\}/2$ | as spin current |
| `spin_density` | spin, field (`'yx'`) | $S_s$ | ħ cm/V per cell |
| `orbital_density` | l, field | $L_l$ | ħ cm/V per cell |
| `spin_torque` | spin, field | $i[H,S_s]/\hbar$ | $e\,a_0$ per cell (torkance) |
| `custom` | field (`'x'`) | `operator=` matrix | [O] $a_0$/V per cell |

- $S = \sigma/2$ and $L$ are in units of $\hbar$.
- A `projection` (as `shc_proj`) acts as $(PA+AP)/2$.
- Scalar runs (no spin–orbit, `nspin = 1`) count both spins (×2, as the Boltzmann routes do). The
  spin observables need a spinor basis.
- The scalar orbital operator (real harmonics of the shells) is built by `orbital_operator` since
  14 Nov 2026, so the orbital Hall and orbital Edelstein responses run without spin–orbit.

**Velocity sign.** PAOFLOW's `dHksp` is $-\partial H/\partial k$ for DFT runs and models alike. The
cause is the $e^{-ik\cdot R}$ convention of `HRs` together with the $+iR$ of `do_gradient`.
- Models: $H(k)$ equals the model's to $10^{-14}$.
- DFT (Pt): QE lists the full grid in PAOFLOW's order, and the band velocities from `pksp` are
  $-\partial E/\partial k$.

`kubo_operators.band_matrices` therefore uses $\hbar v = -$`dHksp`.
- This matters only for responses linear in $v$ (spin and orbital densities, torques, custom operators);
  products of two velocities do not see it.
- The degenerate subspaces are rotated with the `dHksp`-signed operators, the gauge of the older routes.
  The intraband/interband split then matches them.

## Frequency-dependent response (`kubo_optical_response`, `kubo_optical.py`)

The Kubo–Greenwood response at photon energy $\hbar\omega$ and one chemical potential, for the same
observables, components, units and charge sign:

$$
\chi_{AB}(\omega) = s_q\Big[-i\sum_{\varepsilon_{mn}\ne0}
\frac{(f_n-f_m)\,A_{nm}(\hbar v_b)_{mn}}{\varepsilon_{mn}(\varepsilon_{mn}-\hbar\omega-i\eta)}
+ \frac{i}{\hbar\omega+2i\eta}\sum_{\varepsilon_{mn}=0}(-f'_n)\,A_{nm}(\hbar v_b)_{mn}\Big],
\qquad \varepsilon_{mn} = \varepsilon_m-\varepsilon_n .
$$

- **The two terms:**
  - **Interband (length gauge):** the transitions, with $r^b_{mn} = -i(\hbar v_b)_{mn}/\varepsilon_{mn}$.
    $\mathrm{Re}\,\sigma_{aa}(\omega) \ge 0$ is the absorption. The Hall part is
    $\sum_{nm}(f_n-f_m)\,\mathrm{Im}P_{nm}/(\varepsilon_{mn}^2-(\hbar\omega+i\eta)^2)$, Eq. 7 of Yao et al.
  - **Intraband:** the Drude term of the bands at the Fermi level, with $\hbar/\tau = 2\eta$. Degenerate
    blocks (bands closer than `degenerate_threshold`) enter as a whole.
- **Occupations:** the run's smearing at $T = 0$ (adaptive Gaussian or Methfessel–Paxton; a step with a
  Lorentzian window otherwise), Fermi–Dirac at $T > 0$.
- **Output:** `kubo_optical_<observable>_<component>.dat` with ω and the Re/Im parts of `total`,
  `interband` and `intraband`; Results kind `kubo_optical`; `GPAO.plot_kubo_response`.
- **Limits (tests):**
  - at ω → 0 and η → 0 the interband Hall part is the `intrinsic` term of `kubo_response` (Haldane:
    $10^{-8}$, $2\pi\sigma_{xy} = 2.000$; Kane–Mele spin Hall: $10^{-6}$);
  - the intraband term at ω = 0 is the Boltzmann conductivity of tutorial03 Eq. 4 ($10^{-10}$, after its
    $10^{-4}$ regulariser is undone);
  - the Drude form $\sigma(0)\,2\eta/(2\eta - i\hbar\omega)$ holds;
  - the Hall part equals Yao's closed form ($10^{-12}$).
- **DFT check (bcc Fe, example04 data, 24³, η = 0.1 eV;** `validation/kubo_bastin/fe_optical/` in the
  workspace):
  - Re σ_xy(ω) = −950, −283, −35, +127, +31 S/cm at 0, 1, 2, 3, 4 eV: the shape of Yao's Fig. 5 with the
    opposite sign (moment convention, see above);
  - correlation with the legacy `MCDr_xy` output: −0.965. The two differ in the broadening (constant η here,
    adaptive pair widths there) and in the sign convention.
- **Cost:** $O(n_k n_b^2 n_\omega)$, like `do_ac_conductivity`. The k-points are streamed in blocks and
  reduced over MPI.

## Sign convention and the older routes

`kubo_response` is the Kubo–Bastin formula $K(A,\hbar v_b)$ with the electron charge $q = -|e|$:
- **Charge current:** $\sigma = q^2K = K$. This is Yao et al., PRL 92, 037204 (2004), Eqs. (4), (6), (7):
  $\sigma_{xy} = 2\sum_n f_n\sum_m\mathrm{Im}[v^x_{nm}v^y_{mn}]/\varepsilon_{nm}^2$, positive on the
  diagonal.
- **Every other observable:** $\chi = (q/|e|)\,K = -K$. This is Guo et al., PRL 100, 096401 (2008),
  Eq. (1): $\sigma^z_{xy} = (e/\hbar)\sum_n f_n\Omega^z_n$, with $e$ the (negative) electron charge. Guo
  states that the sign was "flipped … because the sign of the charge of the carriers is opposite" to the
  hole case. The spin Hall conductivity of Pt is positive.

**DFT checks (14 Nov 2026).**
- **Pt.** `kubo_response` and `spin_Hall` both give +2012 (ħ/e) S/cm at $E_F$ (8³ nscf, 24³
  interpolation), the sign of Guo.
  - The QE run with `nosym`/`noinv` (512 k-points from QE, no PAOFLOW unfolding) gives the same number to
    0.1 as the symmetry-reduced one.
  - QE writes $\langle\phi_i|\psi_n\rangle$ (`projwfc.f90`, `calbec(wfcatom, evc)`), and `build_Hks`
    assembles $\sum_n\langle\phi_i|\psi_n\rangle\varepsilon_n\langle\psi_n|\phi_j\rangle$. PAOFLOW's DFT
    $H(k)$ is QE's.
- **Fe.** With QE's majority spin along +z (`starting_magnetization` > 0; PAOFLOW's
  $\langle S_z\rangle = +1.10\,\hbar$), the path Berry curvature (`topology`, Yao's $\Omega^z$ of Eqs. 4–5)
  has $-\Omega^z = -1369$ bohr² near H on P→H. Yao's Fig. 2 shows a large positive $-\Omega^z$ spike
  there.
  - The computed state is therefore the time-reverse of Yao's. His "magnetization along [001]" is the
    magnetic moment, which is antiparallel to the electron spin.
  - `kubo_response` gives $\sigma_{xy} < 0$ for this state (−1175 S/cm in integration example04), the value
    of Yao's +751 for the reversed moment.
  - To compare with Yao directly, run QE with the moment along +z, i.e. a negative `starting_magnetization`,
    or flip the sign of the T-odd output.

The older routes compared with `kubo_response`, on the same arrays (tests, $10^{-10}$ or better):

| route | kubo term | sign |
|---|---|---|
| `linear_response('cond')` Eq. 1 | `surface` (η = γ) | same |
| `linear_response('cond', intraband)` Eq. 4 | `surface_intraband` as η → 0 (Boltzmann, τ = ħ/2η; adaptive δ) | same (5 %, different δ) |
| `linear_response('shc')` Eq. 3, `spin_Hall` | `intrinsic` (δ = 0.05) of `spin_current` | same |
| `linear_response('shc', t_odd)` Eq. 1 | `surface` of `spin_current` | same |
| `orbital_Hall` | `intrinsic` of `orbital_current` | same |
| `linear_response('ree')` Eq. 1 / `t_odd` Eq. 3 | `surface` / `intrinsic` (δ = 0.05) of `spin_density` | opposite: the old route takes `dHksp` (= −ħv) as the velocity |
| `linear_response('ahc')` Eq. 3, `anomalous_Hall` | `intrinsic` (δ = 0.05) of `charge_current` | opposite: the old route reports $\sigma_{xy}$ for the moment along the run's spin axis |
| `linear_response('ahc', full_chi2)` Eq. 2 | none (Ly's Fermi-sea kernel: sharp occupation of $n$) | same as `sea` |

- The old Eq. 3 / `do_Hall` routes share one Berry-curvature sum, $\sum_n f_n\Omega_n$ with
  $\Omega_n = -2\sum_m\mathrm{Im}[A_{nm}B_{mn}]/\varepsilon_{nm}^2$.
  - For the spin and orbital currents this is $q\,K$, the physical value.
  - For the charge current it is $-K$: $\sigma_{xy}$ of the time-reversed state. That is why `anomalous_Hall`
    reproduces Yao's sign for QE's "+z" magnetization.
- `full_chi2` flips the sign of the old AHC.
- On models the same relations hold (Haldane: $C = +1$ per spin from `kubo_response`, $-1$ from the old
  AHC routes).

## Comparison with Ly 2026 and py4mulas

O. Ly, J. Phys.: Condens. Matter **38**, 435802 (2026) writes every Kubo–Bastin term as
$\mathrm{Re}\sum_k\sum_{nm}A_{nm}B_{mn}K_{nm}$ with closed-form kernels; py4mulas 1.0.0 implements them.
Compared with the exact constant-$\eta$ kernels here, on the magnetic Rashba gas of Ly Eq. 17
(`validation/kubo_bastin` in the workspace):

- **Sign and normalisation:**
  - py4mulas returns $2\pi K$, i.e. $e^2/h$, with the response operator in the `alpha` slot and no charge
    sign. It equals `kubo_response` for the charge current and is minus it for the other observables.
  - Its η→0 Fermi sea equals the `intrinsic` charge Hall term to $2\times10^{-5}$.
  - An independent numpy Berry sum (bare $K$) equals the `intrinsic` term to $10^{-12}$, for the charge Hall
    and the spin density responses alike (the latter divided by the charge sign). The only exceptions are
    the two μ that sit exactly on a band level (the θ(0) convention).
- **Fermi sea at finite η:** Ly's kernel takes the occupation of level $n$ sharp,
  $f(\varepsilon_n)\,\mathrm{Re}(\varepsilon_{nm}+i\eta)^{-2}$ (tutorial03 Eq. 2). The exact sea
  broadens it with $L_n$. At η = 0.1 the two differ by 5 % of the peak.
- **Fermi surface:** Ly's kernel $-\eta f'(\varepsilon_m)/(\varepsilon_{nm}^2+\eta^2)$ is exact only
  for η → 0. At finite η it matches the exact surface term at η/2 (0.6 % on converged grids), so the
  τ ↔ η mapping is off by 2. For $kT \ll \eta$ it is badly off.
- py4mulas' `KuboBastinKernel(None)` ("full") has no Fermi-surface term; the full response is
  FermiSea + FermiSurface there.
- Typos in the paper: Eq. 11 vs B.2 (a stray $\varepsilon_{mn}$); Eq. 12 has $f(\varepsilon_m)$ where B.5
  has $f(\varepsilon_n)$.

## Validation (unit tests)

- Kernels against a brute-force quadrature at $T = 0$ and $T > 0$; the clean limit; the symmetric and
  antisymmetric separation.
- Same numbers as the tutorial03 routes, with the signs of the table.
- Physical limits:
  - Haldane: $2\pi\sigma_{xy} = 2.000$ ($e^2/\hbar$, both spins) from `total`, `sea` and `intrinsic`.
  - Kane–Mele without Rashba coupling: $2\pi\sigma^z_{xy} = +1.00$.
  - Kane–Mele (non-magnetic): no charge Hall response.
  - The intraband term at small η is the Boltzmann conductivity (5 %).
  - T → 0 limit; a custom operator reproduces the spin density.
- Studio: the generated Haldane driver gives $C = +1$ from `kubo_response`.

## Performance and memory

- The k-points are streamed in blocks: band matrices per k-point (`perturb_split`, as the older
  routes), kernels per block of k-points.
- No `(nk, nb, nb)` array is stored, and the k-sums are reduced over MPI at the end (no gathers).
- Timings (Rashba gas, 128², 161 μ, one core):

  | route | time |
  |---|---|
  | Eq. 3 | 0.22 s |
  | `intrinsic` | 0.24 s |
  | Eq. 1 | 0.21 s |
  | exact surface + sea | 0.51 s |
  | the same at 300 K (thermal grid) | 4.0 s |

## Not done (and why)

- **The Smrčka–Středa split and the overlap term (Bonbien–Manchon): skipped.**
  - Their totals are the same as `surface + sea`. The Smrčka–Středa terms mix Fermi-surface and Fermi-sea
    pieces, and the overlap term measures that mixing.
  - Their uses are sum-rule cross-checks and term-by-term comparison with papers that report the
    Smrčka–Středa or Freimuth I(a)/I(b)/II components (spin–orbit torques). The Bonbien–Manchon split
    implemented here is the physically clean one.
  - py4mulas has all of them if such a comparison is ever needed.
- **Closed ψ/ψ′ forms for $T > 0$:** they would replace the thermal grid of the static response.
- **The sparse backend.**
- **The exact constant-η Kubo–Bastin kernel at finite ω** (Lorentzian spectral functions instead of
  broadened transitions): same closed-form machinery as the static sea term, and a higher cost.

## Library fixes of the campaign (14 Nov 2026)

- `linear_response`:
  - reads `a_tensor` on its own;
  - reorders the `s_tensor` fallback from `spin_Hall`;
  - has a `delta` parameter (the regulariser of Eq. 3, which was fixed at 0.05).
- The Eq. 2/4/5 spin Hall branches read their own tensor.
- `emaxH` is no longer written back.
- Sheet (`twoD`) headers say $e^2/\hbar$.
- `anomalous_Hall(twoD=)`.
- Orbital Hall without spin–orbit.
- The optical grid on models (see above).
- Dead code removed.
- Berry-curvature, Fermi-surface and spin-current code shared between `do_Hall` and the Eq. 1/3/2-4-5
  modules.
- Frozen-number parity: 34 cases (`tests/unit/response/data/kubo_legacy_reference.npz`).
