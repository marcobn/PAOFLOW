# Non-linear optics: shift and injection currents, second-harmonic generation

`PAOFLOW.nonlinear_optics(nk, ...)` computes the second-order optical responses of the PAO
Hamiltonian: the shift current (bulk photovoltaic effect), the injection currents and the
second-harmonic susceptibility, on a dense k-grid interpolated from `HRs` without storing any
`(nk, nawf, nawf)` tensor. The engine lives in
[hamiltonian/bloch_derivatives.py](../../src/PAOFLOW/hamiltonian/bloch_derivatives.py)
(Bloch Hamiltonian and its first and second k-derivatives at arbitrary k) and
[response/nonlinear_optics.py](../../src/PAOFLOW/response/nonlinear_optics.py) (band geometry,
kernels, driver, writers). It implements the formulation of Wang, Liu, Kang, Gu, Xu and Duan,
Phys. Rev. B **96**, 115147 (2017) and Chen, Ye, Zou, Gu, Xu and Duan, Phys. Rev. B **105**,
075123 (2022), with the PAO basis replacing the Wannier functions.

## Workflow

```python
pf = PAOFLOW.PAOFLOW(savedir='GaAs.save', ...)
pf.read_atomic_proj_QE(); pf.projectability(); pf.pao_hamiltonian()
spectra = pf.nonlinear_optics(nk=(60, 60, 60), emax=4.0, eta=0.05, scissor=0.86,
                              components=['xyz'])
pf.finish_execution()
```

`nonlinear_optics` needs `HRs`, so it runs after `pao_hamiltonian()` (and `add_external_fields`)
and before `interpolated_hamiltonian()` / `pao_eigh()`, which release it. It writes two-column
files (photon energy in eV, value): `shift_<abc>.dat` and `shiftm_<abc>.dat` (µA/V²),
`injectc_<abc>.dat` and `injectm_<abc>.dat` (A V⁻² s⁻¹), `shgr_/shgi_/shga_<abc>.dat`
(Re, Im, |χ| in pm/V). With `twoD=True` the sheet responses are written instead (multiplied by
the slab height: nm µA/V², nm A V⁻² s⁻¹, pm²/V). `GPAO.plot_optical` knows the prefixes.
Tight-binding models (`PAOFLOW(model=...)`) work the same way, with the orbital centres taken
from the model's `tau`/`naw` (or passed as `orbital_positions`).

## Conventions

Units inside the kernels: energies eV, lengths bohr, k in 1/bohr, ħ = 1 (`v` is ħ**v** in
eV bohr, `r` in bohr, `r_{;a}` in bohr²); the SI prefactors are applied once at the end.

**Bloch Hamiltonian.** PAOFLOW stores `HRs` such that $H(\mathbf k) = \sum_{\mathbf n}
e^{-i\mathbf k\cdot\mathbf R_{\mathbf n}} H(\mathbf n)$ (lattice gauge), with the pair-resolved
Wigner–Seitz images of the real-space cells of `elphon.elph_bloch`. Derivatives are explicit sums
with the factors $-iR^a_{\mathbf n}$ and $-R^a_{\mathbf n}R^b_{\mathbf n}$. With the position operator
diagonal on the orbital centres $\boldsymbol\tau_i$ (the tight-binding approximation), the
velocity and the "curvature" operators of the atomic-gauge Hamiltonian, written in the lattice gauge,
are

$$
v_a = \partial_a H + i\,\Delta\tau_a\circ H,\qquad
w_{ab} = \partial_a\partial_b H + i\,\Delta\tau_b\circ\partial_a H + i\,\Delta\tau_a\circ\partial_b H
- \Delta\tau_a\Delta\tau_b\circ H,
$$

$(\Delta\tau_a)_{ij} = (\tau_j - \tau_i)_a$. Their band matrices $U^\dagger v_a U$,
$U^\dagger w_{ab} U$ (with $U$ the eigenvectors of the lattice-gauge $H$) are those of
$\partial_a H^A$ and $\partial_a\partial_b H^A$ in the eigenbasis of the atomic-gauge Hamiltonian;
the test `test_bloch_derivatives.py` checks both against the exact operators of hand-built models and
against finite differences.

**Band geometry** (`band_geometry`): $\omega_{nm} = E_n - E_m$, $r^a_{nm} = -i v^a_{nm}/\omega_{nm}$
for $|\omega_{nm}| >$ `degeneracy_tol` (zero otherwise), $\Delta^a_{nm} = v^a_{nn} - v^a_{mm}$, and the
generalized derivative from the sum rule over all `nawf` bands,

$$
r^b_{nm;a} = \frac{i}{\omega_{nm}}\Big[\frac{v^b_{nm}\Delta^a_{nm}}{\omega_{nm}} - w^{ab}_{nm}
+ \big[v^b, G^a\big]_{nm}\Big],\qquad G^a_{pm} = \frac{v^a_{pm}}{\omega_{pm}}\ (p\neq m),
$$

which is the sum rule of Wang *et al.* (their Eq. 5, including the $\langle n|\partial_a\partial_b H|m\rangle$
term that does not vanish in a tight-binding basis) after the $p = n, m$ terms are moved into the
commutator. The sign of the curvature term is pinned by
`test_generalized_derivative_matches_covariant_finite_differences`, which compares $r^b_{nm;a}$ with
the k-derivative of $r^b_{nm}$ in a parallel-transport gauge (Richardson-extrapolated central
differences). The PAO basis is finite, so this sum rule is exact within the model: no slow
band-convergence as in the plane-wave sum-rule route.

**Responses** (`nonlinear_optical_spectra`), with $g_s$ the spin factor, $\int[dk] \to
\frac{1}{N_k\Omega}\sum_k$, and $f_{nm} = f_n - f_m$:

| quantity | expression | file, unit |
|---|---|---|
| shift current, $J^a = \sigma^{abc}E^b(\omega)E^c(-\omega)$ | $\sigma^{abc} = \frac{\pi e^3 g_s}{\hbar^2}\int[dk]\sum_{nm} f_{nm}\,\mathrm{Im}[r^b_{mn}r^c_{nm;a} + r^c_{mn}r^b_{nm;a}]\,\delta(\omega_{nm}-\omega)$ | `shift_abc.dat`, µA/V² |
| magnetic shift current (T-odd), $J^a = \sigma_M^{abc}\mathrm{Im}[E^bE^{c*}]$ | $-\frac{\pi e^3 g_s}{\hbar^2}\int[dk]\sum f_{nm}\mathrm{Re}[r^b_{mn}r^c_{nm;a} - r^c_{mn}r^b_{nm;a}]\delta(\omega_{nm}-\omega)$ | `shiftm_abc.dat`, µA/V² |
| circular injection, $dJ^a/dt = \eta_C^{abc}\mathrm{Im}[E^bE^{c*}]$ | $\frac{2\pi e^3 g_s}{\hbar^2}\int[dk]\sum f_{mn}\Delta^a_{mn}\mathrm{Im}[r^b_{mn}r^c_{nm}]\delta(\omega_{nm}-\omega)$ | `injectc_abc.dat`, A V⁻² s⁻¹ |
| magnetic injection (T-odd), $dJ^a/dt = \eta_M^{abc}\mathrm{Re}[E^bE^{c*}]$ | $-\frac{2\pi e^3 g_s}{\hbar^2}\int[dk]\sum f_{mn}\Delta^a_{mn}\mathrm{Re}[r^b_{mn}r^c_{nm}]\delta(\omega_{nm}-\omega)$ | `injectm_abc.dat`, A V⁻² s⁻¹ |
| SHG, $P^a(2\omega) = \epsilon_0\chi^{abc}E^bE^c$ | Chen *et al.* Eqs. (C1)–(C3): the three-band term $\chi_e$ and the mixed term $\chi_i$ including the T-odd $-\Delta^a_{mn}g^{bc}_{nm}/(2\omega_{mn}^2(\omega_{mn}-\omega))$ | `shgr/shgi/shga_abc.dat`, pm/V |

The field convention is $E(t) = E(\omega)e^{-i\omega t} + E(-\omega)e^{i\omega t}$; for $b = c$ the
shift current is Eq. (2) of Wang *et al.* and twice the $\sigma^{abb}$ of Sipe and Shkrebtii (2000),
who write $J = 2\sigma EE^*$. The injection coefficients follow Eq. (9) of Chen *et al.*
($\eta_C \propto$ the two-band Berry curvature $\Omega^{bc}_{mn} = -2\mathrm{Im}[r^b_{mn}r^c_{nm}]$,
$\eta_M \propto$ the quantum metric $g^{bc}_{mn} = \mathrm{Re}[r^b_{mn}r^c_{nm}]$). The SI prefactors
per unit of the dimensionless k-sums are $\pi e^2/\hbar = 7.647\times10^{-4}$ A/V²,
$2\pi e^3/\hbar^2 = 2.324\times10^{12}$ A V⁻² s⁻¹ and $e/(2\epsilon_0) = 9048$ pm/V.

**δ-functions** are Gaussians (PAOFLOW's convention) or Lorentzians of width `eta`, or the adaptive
pair widths $a|\mathbf v_n - \mathbf v_m|\Delta k$ floored at `eta` (`adaptive_smearing=a`), evaluated
directly on the photon-energy grid for the pairs inside the window.

**SHG histograms.** Every term of $\chi_e + \chi_i$ has the form $W/(E - c\,\omega - i\eta)$ with $E$
a transition energy and $c \in \{1, 2\}$ (one common $i\eta$ in every denominator, the $1/\tau$ of Chen
*et al.*). The complex weights are accumulated per component into two histograms of $E$ (bin `shg_bin`,
default `eta/5`, linear two-point binning) and convolved once with the resonant denominators on the
photon grid. The cost per k-point is $O(n_{\rm awf}^3)$ for the three-band term and independent of the
number of photon energies; `test_shg_histograms_match_direct_evaluation` checks it against an explicit
evaluation.

**Scissor.** `scissor` shifts the conduction energies in every resonance energy and in the explicit
$1/\omega_{mn}$ factors, while $r$, $r_{;a}$ and $\Delta$ keep the DFT eigenvectors (the
"scissors with fixed matrix elements" of Nastos *et al.* 2005); the δ-spectra shift rigidly
(`test_scissor_shifts_the_delta_spectra_rigidly`).

**Occupations.** 0 K: a step by band index (`nelec`/$g_s$ bands) or by energy if `fermi_energy` is
given; finite `temperature`: Fermi–Dirac at `fermi_energy` (default the mid-gap).

**Degeneracies.** Pairs with $|\omega_{nm}| <$ `degeneracy_tol` are dropped from $r$ and from the band
sums (the U(1)-covariant formulation); the U(N)-covariant derivative of Chen *et al.* for degenerate
subspaces is not implemented.

**Spin.** $g_s = 2$ for `nspin = 1` without spin–orbit, 1 otherwise; with `nspin = 2` the two channels
are computed separately (`_0`, `_1` files) and summed.

**2D.** `twoD=True` multiplies by `slab_height` (default $|\mathbf a_3|$): sheet conductivities in
nm µA/V² and sheet susceptibilities in pm²/V.

## Validation (unit tests, `tests/unit/response/test_nonlinear_optics.py`)

| test | model | checks |
|---|---|---|
| velocity and curvature operators | random two-site model, honeycomb | $D^\dagger v D$, $D^\dagger w D$ equal the exact atomic-gauge operators and their finite differences |
| generalized derivative | random model without symmetries | sum rule vs parallel-transport finite differences, $10^{-5}$ relative |
| gauge invariance | random model | spectra unchanged under unitaries mixing orbitals on a site |
| Berry curvature | Haldane model | $-2\mathrm{Im}\sum_c r^x_{vc}r^y_{cv}$ integrates to the Chern number ($\pm1$, 0) |
| inversion | SSH / Rice–Mele chain | all responses vanish with inversion; odd under the inversion of the chain; invariant under a half-cell translation |
| D3h selection rules | gapped graphene | $\sigma^{yyy} = -\sigma^{yxx} = -\sigma^{xxy} = -\sigma^{xyx}$, $\sigma^{xxx} = 0$, no circular injection, no T-odd responses; the same for $\chi$ |
| time reversal | Haldane with strain | magnetic shift and injection currents appear with the flux and vanish without |
| scissor | gapped graphene | rigid shift of the δ-spectra |
| SHG histograms | random model | equal to the explicit evaluation |
| Kleinman symmetry | strained honeycomb | real $\chi(0)$, $\chi^{xxy} = \chi^{xyx}$ exactly, $\chi^{xxy} = \chi^{yxx}$ to $10^{-3}$ at 48² k |
| units and files | — | prefactors, component parsing, the writer (3D and 2D units) |

Literature targets for the examples: GaAs $\chi_{xyz}$ with a 0.86 eV scissor (Wang *et al.* 2017,
Fig. 4; Rashkeev *et al.* 1998; Nastos *et al.* 2005) and $\sigma_{xyz}$ (Ibañez-Azpiroz *et al.* 2018);
monolayer GeS shift current and SHG (Wang *et al.* 2017, Fig. 1; Rangel *et al.* 2017); the
Rice–Mele shift vector (Fregoso, Morimoto and Moore 2017). The examples are in
`examples/nlo_examples/` (`Graphene`, `GaAs`, `GeS`; the graphene one is also in `examples/TBmodel_examples/Graphene_nonlinear`),
with the report `examples/nlo_examples/REPORT.md`.

## Performance

Per k-point: 10 complex matrix products (nws × nawf²) for $H$, $\partial H$, $\partial^2 H$, one
`eigh`, the band rotations and the $O(n_{\rm awf}^3)$ sum rule and three-band contraction. For nawf = 26
on a 12³ coarse grid (≈ 2·10³ Wigner–Seitz cells) this is a few ms per k-point per rank; k-points are
split over the MPI ranks (`comm.Allreduce` at the end) and processed in blocks of `kblock`. Knobs:
`components` (the three-band term is evaluated per component), `hopping_cutoff` (drop negligible
cells), `second_harmonic=False` when only the δ-spectra are needed.

## Practical notes from the GaAs validation

- **Degeneracy tolerance.** DFT-derived PAO Hamiltonians have many nearly degenerate pairs
  (semicore and ghost bands). With `degeneracy_tol = 1e-4` eV the symmetry-forbidden
  $\sigma^{xxx}$ of GaAs came out at 30 % of $\sigma^{xyz}$; with 10 meV (the default) the
  leakage is 0.8 % and the allowed components are unchanged. Analytic models can use `1e-4`.
- **Symmetry of the PAO Hamiltonian.** The Hamiltonian expanded from a symmetry-reduced nscf
  k-set is exactly symmetric at the grid points that are rotation images of each other but its
  real-space images are symmetric only to about 1 meV, which the $1/\omega$ factors of the
  second-order responses amplify: equivalent components ($\chi_{xyz}$ vs $\chi_{yzx}$) differed by
  9 % on a 16³ nscf grid. Quote such differences as the symmetry error bar, or run the nscf with
  `nosym = .true., noinv = .true.` on the full grid (the examples do).
- **GaAs numbers** (symmetric 16³ nscf, 40³ dense grid, scissor 0.86 eV, η = 0.05 eV):
  $|\chi_{xyz}(0)| = 110$ pm/V, 341 pm/V at 1.5 eV, peak 610 pm/V at 2.63 eV, against
  100–170 pm/V and 400–600 pm/V in the literature (Nastos 2005, Wang 2017, experiment
  $2d_{14} \approx 170$ pm/V); $\sigma_{xyz}$ reaches $-30$ µA/V² at 3.9 eV. The T-odd responses and
  the circular injection are $10^{-5}$ of their natural scale.
- **GeS numbers** (monolayer, nosym 18×18×1 nscf, 200² dense grid, η = 0.05 eV, PBE gap 1.68 eV,
  sheet units with the 18 Å cell): shift-current peaks $\sigma^{xxx} = 46$, $\sigma^{xyy} = 48$,
  $\sigma^{yxy} = 26$ nm µA/V² (about 75 µA/V² for a 0.6 nm layer, the order of magnitude of Rangel
  *et al.* 2017), the mirror-forbidden $yyy$ and $xxy$ below $10^{-8}$, T-odd files identically zero.
  The repository BASIS tables projected the Ge 3d bands at 0.90 and gave a gapless PAO Hamiltonian;
  the tables generated from the pseudopotentials (`paoflow-genbasis-ps`, preset `standard`) give 26
  bands above 0.95.
- **Basis.** The 26-orbital Ga/As basis (no 4D/5D/6S/6P) projects the Ga 3d semicore bands
  poorly (projectability 0.45) and cannot be used; the 62-orbital configuration of example10
  (18 bands above 0.95) is the working one. Cost at nawf = 62, 40³ k, three components with
  SHG: 15 min on 8 ranks.
- **Memory.** The δ-kernels are evaluated in pair blocks of about 16 MB and the three-band SHG
  term in k-chunks of about 2·10⁶ triples; the Wigner–Seitz images of `HRs` (nws × nawf² complex)
  are the largest resident array (300 MB for GaAs at 16³).

## Not implemented (follow-ups)

Position dipoles beyond the tight-binding approximation (`position_dipoles.py` in $r$ and $r_{;a}$),
the U(N)-covariant derivative for degenerate bands, k-resolved integrand maps, symmetry reduction of
the k-grid, and a Rust kernel for the post-`eigh` algebra.

## References

- C. Wang, X. Liu, L. Kang, B.-L. Gu, Y. Xu and W. Duan, *First-principles calculation of nonlinear optical responses by Wannier interpolation*, Phys. Rev. B **96**, 115147 (2017).
- H. Chen, M. Ye, N. Zou, B.-L. Gu, Y. Xu and W. Duan, *Basic formulation and first-principles implementation of nonlinear magneto-optical effects*, Phys. Rev. B **105**, 075123 (2022).
- J. E. Sipe and A. I. Shkrebtii, *Second-order optical response in semiconductors*, Phys. Rev. B **61**, 5337 (2000).
- C. Aversa and J. E. Sipe, Phys. Rev. B **52**, 14636 (1995); S. N. Rashkeev, W. R. L. Lambrecht and B. Segall, Phys. Rev. B **57**, 3905 (1998); F. Nastos, B. Olejnik, K. Schwarz and J. E. Sipe, Phys. Rev. B **72**, 045223 (2005).
- J. Ibañez-Azpiroz, S. S. Tsirkin and I. Souza, *Ab initio calculation of the shift photocurrent by Wannier interpolation*, Phys. Rev. B **97**, 245143 (2018).
- B. M. Fregoso, T. Morimoto and J. E. Moore, Phys. Rev. B **96**, 075421 (2017); T. Rangel *et al.*, Phys. Rev. Lett. **119**, 067402 (2017).
