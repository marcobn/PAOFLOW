# Non-linear optics, phase 2: position dipoles, degenerate bands, symmetry reduction

*Plan written on October 27 2026 after the phase-1 validation
([nonlinear-optics.md](nonlinear-optics.md), `examples/nlo_examples/REPORT.md`). Status: planned, not
implemented. Three independent work packages, in the order they should be done.*

## Starting point

`response/nonlinear_optics.py` evaluates, per k-block, the band energies, the band-basis
velocity `v` and curvature `w` (from `hamiltonian/bloch_derivatives.py`), the interband position
$r^a_{nm} = -i v^a_{nm}/\omega_{nm}$, the generalized derivative

$$
r^b_{nm;a} = \frac{i}{\omega_{nm}}\Big[\frac{v^b_{nm}\Delta^a_{nm}}{\omega_{nm}} - w^{ab}_{nm}
+ [v^b, G^a]_{nm}\Big],\qquad G^a_{pq} = \frac{v^a_{pq}}{\omega_{pq}}\ (|\omega_{pq}| > \text{tol}),\ 0\ \text{otherwise},
$$

and the shift / injection δ-kernels and SHG histograms on the full dense grid. The phase-1
validation left three gaps: (1) the position operator is diagonal on the orbital centres
(tight-binding approximation), although the Löwdin position dipoles of
`hamiltonian/position_dipoles.py` are available and already used in the velocity of the linear
optics and the e-ph routes; (2) pairs closer than `degeneracy_tol` are simply dropped, which is
the U(N)-covariant treatment only if the dropped pairs form consistent clusters; (3) every
k-point of the dense grid is computed, and the symmetry of the crystal is neither exploited nor
enforced. (The 2–5 % spread between equivalent components of the first GaAs runs was a problem of the
internal projections with PAW pseudopotentials, fixed with the augmented metric of
`projection/augmentation.py`, see `examples/nlo_examples/REPORT.md` §3.2. What remains with an
exactly symmetric Hamiltonian is the gauge of exactly degenerate bands on the Λ lines in the SHG
and injection kernels, 10 % / 2 % between χ_xyz and χ_yzx at 40³ and a non-zero circular
injection current in T_d, while the shift current is exact to 1e-10: that is C2, and C3 would
average it, not remove it.)

## C1. Position dipoles in r and r_{;a}

**Physics.** In the Löwdin-orthonormal PAO Bloch basis the position operator is
$\hat r = \boldsymbol\tau + \mathbf d(\mathbf k)$, with $\mathbf d(\mathbf k)$ the Hermitian
dipole matrix of `position_dipole_operator` (lattice gauge, bohr, `(nb, 3, nawf, nawf)`). Two
consequences, already used elsewhere in PAOFLOW: the velocity gains $i[H, \mathbf d]$
(`dipole_velocity_term`), and the Berry connection of the Bloch states gains the band-basis
dipole $D = U^\dagger \mathbf d U$. For the second-order responses:

- $r^a_{nm} = -i v^a_{nm}/\omega_{nm}$ with the **full** velocity gives
  $r^{TB,a}_{nm} + D^a_{nm}$ automatically ($i[H,d]_{nm} = i\omega_{nm}D_{nm}$): no change in
  `band_geometry` beyond the velocity. $\Delta^a_{nm}$ is unchanged ($i[H,d]$ has no diagonal).
- The generalized derivative needs the k-derivative of $D$. With the sum rule of the tight-binding
  part written with $v^{TB}$ and $G^{TB}$ (the Hamiltonian is unchanged, so $w$ and $\partial H$ are
  the tight-binding ones), the derivation of phase 1 extended to $r = r^{TB} + D$ gives, after
  the cancellation of the gauge-dependent $A_{nn}$ terms,

  $$
  r^b_{nm;a} = r^{TB,b}_{nm;a} + (U^\dagger\partial_a d^b\,U)_{nm} + [G^{TB,a}, D^b]_{nm}
  - i\,(D^a_{nn} - D^a_{mm})\,r^b_{nm},
  $$

  where $r^b_{nm}$ in the last term is the full position matrix and $G^{TB,a}$ has zero
  diagonal (the $p = n, m$ terms of the band sums are included in the commutator). The sign and
  the three terms are to be pinned by the finite-difference test below.
- $\partial_a \mathbf d(\mathbf k)$: pass 1 by central differences of the callable (six extra
  evaluations of the Bloch sums per block, step $10^{-4}$ bohr$^{-1}$, Richardson optional);
  pass 2 analytic, by extending `assemble_position_kspace` with $\partial_{\mathbf k} X$ and
  $\partial^2_{\mathbf k} S$ (the tables already hold $\Delta\mathbf R$) and
  `overlap_sqrt_and_derivative` with $\partial^2 S^{1/2}$ (second divided differences of the
  eigendecomposition), then $\partial_a$ of the Löwdin formula term by term.

**Code.**
- `band_geometry(energies, U, velocity_orb, curvature_orb, tol, dipoles=None, dipole_gradient=None)`:
  keeps `velocity` (full) and `velocity_tb` (for $G^{TB}$ and the tight-binding sum rule); adds
  the three dipole terms to `position_derivative` when `dipoles` is given.
- `nonlinear_optical_spectra(..., position_dipoles=None, dipole_gradient=None)`: callables
  `K_cryst -> d` and `K_cryst -> ∂d` (`(nb, 3, 3, nawf, nawf)`, `[a, b]` = $\partial_a d^b$);
  `dipole_gradient=None` with `position_dipoles` set falls back to central differences inside
  the driver.
- `PAOFLOW.nonlinear_optics(..., position_dipoles=False, dipole_level='full')`: builds the
  operator with `position_dipole_operator(data_controller, bg, alat, level=...)` (scalar-relativistic
  only, as today) and stores `attr['nonlinear_optics']['position_dipoles']`.
- Memory: the dipole tables are built once; per block the cost is the Bloch sums (comparable
  to the Hamiltonian derivatives) plus 6 more for the finite differences.

**Tests** (`tests/unit/response/test_nonlinear_optics.py`, `_nonlinear_models.py`):
- a synthetic dipole matrix $d(\mathbf R)$ on the random two-site model (Hermitian, short-range,
  with on-site and inter-site blocks) → analytic $d(\mathbf k)$ and $\partial d(\mathbf k)$;
- finite-difference test of $r^b_{nm;a}$ with dipoles: parallel-transport gauge of $U$, full
  Berry connection $A_{nn} = D_{nn}$ in that gauge, Richardson central differences, $10^{-5}$;
- invariances: $d \to d + c\,\mathbb 1$ (a uniform shift) changes nothing; $d \to 0$ reproduces
  phase 1 to $10^{-14}$; gauge invariance under orbital unitaries with $d$ transformed accordingly;
- the D3h relations of the honeycomb model with an artificial symmetric $d$ (selection rules
  survive);
- analytic vs finite-difference $\partial d$ on the real `position_dipole_operator` of the Si
  example save (integration asset) at a few k-points.

**Validation runs.** GaAs 40³ with and without dipoles (χ_xyz(0): phase 1 gives 110 pm/V against
166–180 measured; Wang *et al.* 2017, Fig. 3(d), report that the off-diagonal position matrix
changes the shift current noticeably); GeS 200² with and without; cost ratio; the report and the
docs tables gain a "dipoles" column.

## C2. U(N)-covariant treatment of degenerate bands

**Finding.** Repeating the phase-1 derivation with degenerate clusters $S_n \ni n$, $S_m \ni m$
and the U(N)-covariant derivative of Chen *et al.* (2022),
$D^a[r^b]_{nm} = \partial_a r^b_{nm} - i\sum_{\lambda\in S_n} A^a_{n\lambda} r^b_{\lambda m}
+ i\sum_{\lambda\in S_m} r^b_{n\lambda}A^a_{\lambda m}$, the intra-cluster connections cancel
exactly as the diagonal ones did, and the result is **the same sum rule** with $G^a$ zeroed
inside the clusters (not only on the diagonal): the present `pair_mask` is that mask when the
dropped pairs form clusters. The responses then sum over pairs and triples in *different*
clusters, which is Chen's $\sum_{\mu\nu}$ structure. What the present code lacks is (i) cluster
transitivity (a pairwise threshold can keep $n$–$p$ while dropping $n$–$m$ and $m$–$p$), (ii) the
covariance for *nearly* degenerate clusters, where $1/\omega_{pq}$ varies inside a cluster, and
(iii) evidence that the responses are insensitive to the cluster radius.

**Code.**
- `band_clusters(energies, tol)` → cluster labels per k (connected components of consecutive
  gaps below `tol`); `pair_mask = label_n != label_m`.
- `band_geometry(..., cluster='mean')`: optional replacement of the energies by the cluster
  means in every $1/\omega$ factor (exact covariance), the raw energies kept for the resonances.
- Occupations: a cluster straddling the Fermi level (metal, or $T > 0$) gets the mean occupation
  in $f_{nm}$ (document; the responses of metals need the intraband terms that are out of scope).
- `degeneracy_tol` documented as the cluster radius; default stays 10 meV.

**Tests.** Exact degeneracy: a model with a symmetry-enforced two-fold degenerate pair (the
honeycomb model at the K point, or a random model with two identical orbitals) → spectra
invariant under random U(2) rotations inside the pair ($10^{-12}$) and under `tol` between
1 and 50 meV; near degeneracy: a Rice–Mele chain with a 2 meV splitting → `cluster='mean'` vs
raw differ by $O(\text{tol}/\omega)$. GaAs: leakage into $\sigma^{xxx}$ and the allowed
components versus tol = 1, 5, 10, 30, 100 meV (table in the docs; the phase-1 numbers are
33 % at 0.1 meV and 0.8 % at 10 meV).

## C3. Symmetry reduction of the k-grid and symmetrisation of the tensors

**Design.** A third-rank polar tensor obeys $T^{abc} = \sum R_{aa'}R_{bb'}R_{cc'}T^{a'b'c'}$ for
every point-group operation. The full-grid sum equals the symmetrised sum over the irreducible
wedge: $\sum_{\mathbf k\in BZ} T(\mathbf k) = \mathrm{Sym}_G\big[\sum_{\mathbf k\in IBZ} w_{\mathbf k}\,T(\mathbf k)\big]$
with $w_{\mathbf k}$ the orbit size, because the symmetrisation is linear. So: reduce the grid,
accumulate the orbit-weighted contributions of **all components** of the closure of the requested
ones, and symmetrise the final spectra once. Time reversal maps $\mathbf k \to -\mathbf k$ with
$T(-\mathbf k) = T(\mathbf k)$ for the T-even real responses, $-T$ for the T-odd ones (which
then vanish identically) and the complex conjugate for the SHG histogram weights; it is applied
as an extra orbit doubling with conjugated weights when the Hamiltonian is time-reversal
symmetric (real `HRs` up to a tolerance).

**Operations.** From the QE XML (`arrays['sym_rot']`, `sym_shift`, `sym_TR`, read by
`read_QE_xml`): crystal-coordinate matrices acting on k as in `pao_sym.find_equiv_k`;
Cartesian form for the tensor through `elphon.symmetry.frac_to_cartesian_rotation` (convention
to be pinned by the test "energies at $W\mathbf k$ equal energies at $\mathbf k$ on the grid").
For tight-binding models and VASP runs: the generators of `PAOFLOW.agentic.nlo.symmetry`
(moved to a library module `utils/point_groups.py`, the agentic module importing from it) by
name, or an explicit list of Cartesian matrices; spglib stays optional.

**Code.**
- `utils/point_groups.py`: `point_group_operations`, `symmetrize_rank3`, `forbidden_components`,
  `independent_relations`, `component_closure(components, ops)` (the orbit of the requested
  components), `reduce_k_grid(nk, ops_cryst, time_reversal) -> (k_ibz, weights, orbit_map)`;
  `PAOFLOW.agentic.nlo.symmetry` becomes a thin wrapper.
- `nonlinear_optical_spectra(..., symmetry_ops_cryst=None, symmetry_ops_cart=None, time_reversal=False)`:
  reduces the grid, weights the blocks, computes the closure of `components`, symmetrises the
  final tensors, and returns both the symmetrised spectra and the pre-symmetrisation
  `symmetry_error` (the spread removed by the projection, the phase-1 diagnostic) in the result.
- `PAOFLOW.nonlinear_optics(..., symmetry=True)`: default on when `sym_rot` exists; prints the
  reduction factor (24 for GaAs: 64000 → ≈ 2700 points at 40³) and the symmetry error.
- Writers unchanged; `attr['nonlinear_optics']['symmetry_error']`.

**Tests.** Honeycomb model (D3h, 12 operations): IBZ-reduced vs full 18² grid identical to
$10^{-12}$ for every kind; random model with no symmetry: reduction is the identity;
`component_closure(['xyz'], T_d)` = the six permutations; a full-grid run symmetrised equals
the IBZ run; time reversal on the Rice–Mele chain (real `HRs`) halves the grid and reproduces
the full result; GaAs 40³: full vs IBZ run, identical after symmetrisation, wall time ÷ 20.

## Order, effort and deliverables

| package | effort | deliverable |
|---|---|---|
| C3 symmetry | 1 day | 20× faster GaAs, exact tensor symmetry, the symmetry error as a printed diagnostic |
| C2 clusters | 0.5 day | transitive clusters, `cluster='mean'`, the tolerance study in the docs |
| C1 dipoles | 1.5 days (+1 for the analytic gradient) | `position_dipoles=True`, GaAs / GeS with and without dipoles in the report |

C3 first because it makes the C1/C2 validation runs cheap and projects out whatever residual
asymmetry is left (the PAW-projection error bar itself is removed at the source by norm-conserving
pseudopotentials or the `projwfc.x` projections, which the C1/C2 validation runs will use). Each package ships with its tests, a docs paragraph in
[nonlinear-optics.md](nonlinear-optics.md), the `examples/nlo_examples` drivers updated, the
report tables extended, and the `agentic/nlo` lessons where a pitfall was found.
