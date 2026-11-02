# The finite-difference electron-phonon vertex: fixed basis, history and validation

Status: 1 November 2026. The fixed-basis option is implemented (`fixed_basis=True`, default) in
`elphon/dvscf_fd.py`, `elphon/do_elphon.py`, `elphon/do_gkq.py` and `PAOFLOW.electron_phonon`, and validated
against the native DFPT vertex on Pb. This page records what the route is, why the July 2026 version failed,
and the numbers.

## 1. The route

`PAOFLOW.electron_phonon` is two-phase. Phase 1 writes one `pw.x` scf input per displaced supercell (phonopy
displacements of the reference primitive cell, `displacement_mode='symmetry'` or `'cartesian'`, `is_plusminus`
for central differences). Phase 2 rebuilds the PAO Hamiltonian of every displaced cell and forms
$dV_{\kappa\alpha}=[H^{\rm SC}(+\delta)-H^{\rm SC}(-\delta)]/2\delta$ in the supercell PAO basis (eV/bohr).
`do_gkq.assemble_eph_tensor` expands the symmetry-reduced directions (`symmetry.py`, Wigner-D), folds the
supercell derivative to the primitive real-space tensor $g(\kappa\alpha;\mathbf R_e,\mathbf R_p)$ (`fold.py`,
diagonal supercells) and, if asked, enforces the translational sum rule. `gkq.py`/`eph_kq.py` contract it with
the phonopy polarisations and evaluate an isotropic Eliashberg function on the commensurate q-grid.

## 2. Why the basis must not move

Let $\tilde\phi_i$ be the Löwdin orbitals of the reference cell. In a fixed orthonormal basis the matrix of any
operator differentiates term by term, so $\tfrac{d}{du}\langle\tilde\phi_i|\hat H(u)|\tilde\phi_j\rangle=
\langle\tilde\phi_i|\partial_uV_{\rm KS}|\tilde\phi_j\rangle$: the DFPT vertex in the PAO gauge, the object
`dfpt_vertex.py` builds as $A^\dagger d\,A$. PAOFLOW's reconstruction $\sum_{n\in{\rm good}}\tilde A_n\varepsilon_n
\tilde A_n^\dagger+\eta(1-P)$ adds two things: expanding the displaced eigenvectors to first order in the fixed
basis, $d\tilde A_n=\sum_{m\ne n}\tilde A_mV_{mn}/(\varepsilon_n-\varepsilon_m)$, the energy denominators cancel
pairwise inside the good set and

$$P\,\frac{dH^{\rm PAO}}{du}\,P=\sum_{m,n\in{\rm good}}\tilde A_mV_{mn}\tilde A_n^\dagger=P\,(A^\dagger d\,A)\,P ,$$

exactly, up to the perturbative mixing with bands outside the good set and the $O(\delta^2)$ error. The
$\eta\,\partial_uP$ term is removed by the projection on the reference good subspace (`project_good_subspace`,
default); it is not removed by the fixed basis.

If instead the orbitals follow the atoms, the finite difference is the derivative of the hopping parameters of
a moving basis: the physical vertex plus the basis-rotation commutator $[H,dU]$ with elements
$(\varepsilon_n-\varepsilon_m)\langle m|dU|n\rangle$ (zero on the diagonal by Hellmann–Feynman, nonzero off it)
plus the $\eta$ term. That quantity is what a Slater–Koster-type electron-phonon model differentiates; it is not
$\langle\psi_m|\partial_uV|\psi_n\rangle$.

Implementation: `build_supercell_HRs(positions=...)` overrides `arry['tau']` before `projections()`, so the atomic
Bloch sums of a displaced save are centred at the reference positions; `compute_dV(fixed_basis=True)` passes the
reference `tau` (from `return_symmetry_inputs`) to every displaced build. Both $\pm$ cells are projected on the
same reference good subspace as before. Note that in a fixed basis a rigid translation of the crystal does change
the Hamiltonian matrix, so `enforce_asr` must stay off for the fixed-basis tensor.

## 3. History (July 2026; Dropbox wiki `PAOFLOW.wiki/{Electron-Phonon-Coupling,Elphon_module,Elphon_modified_plani}.md`)

The route was written on 1–3 July 2026 with the basis following the atoms and validated on fcc Al only: the
internal checks passed (lossless fold, Hermitian $g$, cubic isotropy, q-convergence), but $\lambda_{\rm Al}=1.79$
against the measured 0.42; the diagonal (eigenvalue-pinned) part gave 0.43–0.48 and the overshoot sat in the
off-diagonal elements. The good-subspace projection brought $\lambda$ to 0.48 and the roadmap noted that a
cross-check against a DFPT deformation potential remained to be done. On 6 July the Pb comparison with the
EPW-school QE example failed ("α²F does not follow the phonon DOS") and the plan pivoted to the QE-coupling route
(now the native DFPT vertex); the finite-difference matrix elements were deferred and never resumed. The
repository docs page was rewritten on 23 August for the new route, so only the module-map row survived here.

## 4. Validation on Pb (1 November 2026, `Workspace/PAOFLOW/validation/Pb_fd_fixed_basis`, music)

Setup: fcc Pb, ONCVPSP, 60 Ry, m-p 0.04 Ry. DFPT reference: scf 12³, ph.x on the 2³ q-grid, nscf on the explicit
12³ list with 24 bands, native vertex on 12³/6³ → the 2³ q-grid cache (18 orbitals, standard basis). Finite
differences: 2×2×2 supercell (8 atoms), Cartesian ±0.06 bohr (6 cells + reference), 6³ k, `nosym`, 192 bands.
The primitive PAO run keeps 9 of 24 bands (5d, 6s, 6p; η = 4.8 eV above E_F), the supercell 73 of 192.

The supercell derivative was unfolded to the primitive $d_{ij}(\mathbf k,\mathbf q)$ on the 12³ k / 2³ q grids
(bra at $\mathbf k+\mathbf q$ with $e^{-i\mathbf k'\cdot\mathbf T_a}$, ket at $\mathbf k$ with
$e^{+i\mathbf k\cdot\mathbf T_b}$; the unfolded reference Hamiltonian is block-diagonal to 1e-4) and compared
with the DFPT PAO vertex $d(\mathbf k,\mathbf q)=\sum e^{+2\pi i(\mathbf k\cdot\mathbf R_e+\mathbf q\cdot\mathbf R_p)}
g(\mathbf R_e,\mathbf R_p)$ (Ry/bohr × 13.6057).

| quantity (x displacement, all 1728 k × 8 q) | value |
|---|---|
| band elements $\lvert d_{mn}\rvert$ between reference eigenstates within 0.5 eV of E_F (1168 blocks): fixed-basis FD vs DFPT restricted to the same good bands | rel. rms difference **0.0012**, correlation 1.0000 |
| same vs DFPT with all 24 bands | 0.031 (leakage of the non-good bands into the PAO vertex) |
| same vs the emulated finite difference (DFPT band vertex + PAOFLOW's construction, 24-band space) | 0.014 |
| same, **moving basis** vs fixed basis | 0.051 (correlation 0.998) |
| whole good-subspace blocks, Frobenius: fixed FD vs DFPT (both projected on the good subspace) | 0.17 (states between the two η cut-offs and the 24-band truncation) |
| whole good-subspace blocks: moving vs fixed basis | 0.73 |
| share of the raw FD norm in the η-complement sector | 79 % (removed by the projection) |

Reading: on the physically relevant block the fixed-basis finite difference and the DFPT vertex are the same
object to 0.1 %. The moving basis is off by 5 % at the Fermi level and by 70 % over the good subspace, i.e.
its error grows with the energy separation of the pair, as the commutator term says; the July Al overshoot was
dominated by the η-complement artefact that the projection removes. The 3 % difference between the DFPT vertex
with all 24 bands and with the good bands only is a property of the DFPT PAO route itself (the projectability
factor noted in `Electron-Phonon-Coupling.md`).

End-to-end, through the production dense-q driver (`eliashberg.py --nk 24 --nq 2`, σ = 0.05 eV, the FD vertex
written in the cache layout $g(\mathbf R_e,\mathbf R_p)=N^{-1}\sum_{\mathbf k\mathbf q}e^{-2\pi i(\mathbf k\cdot\mathbf R_e+\mathbf q\cdot\mathbf R_p)}d(\mathbf k,\mathbf q)$,
convention checked to 5e-15 against the band vertex):

| vertex | λ (24³ k / 2³ q) | ω_log (meV) |
|---|---|---|
| DFPT, all 24 bands (reference) | 1.660 | 7.02 |
| fixed-basis FD, raw | **1.702** | 7.03 |
| fixed-basis FD, projected on the good subspace | 2.202 | 7.07 |
| DFPT projected on the good subspace | 2.131 | 7.06 |

The raw fixed-basis vertex reproduces the DFPT λ to 2.6 %, the level of the band-element difference above.
The good-subspace projection, applied to **either** vertex before the Fourier interpolation, inflates λ by about
30 %: the projector is not a smooth function of k (bands enter and leave the good set at the η cut-off), so the
projected vertex interpolates badly. The projection is therefore off by default with the fixed basis
(`project_good_subspace=None` → `not fixed_basis`); the η term of the raw derivative lives in the complement
sector and does not couple the dense eigenstates near the Fermi level. (At the coarse grid itself, nk = 12, the
double delta captures only a few states at σ = 0.05 eV and all four vertices give λ ≈ 0.15–0.17.)

Benchmark on the same cells (pw.x 7.5, 216 k, 192 bands): CPU build 32 ranks 5 min 32 s, GPU build (RTX PRO
6000) 1 rank 5 min 09 s; the GPU wins only for larger cells.

## 5. What this enables

The vertex of an arbitrary reference structure without DFPT: thermally displaced cells (special displacement
method, `special-displacement-plan.md` §9), molecular-dynamics snapshots, alloys, and systems where the native
vertex is not available (PAW/US with the internal projector, hybrids, DFT+U, VASP). Still to do for the
supercell use: the arbitrary-reference mode without fold, the Γ-mode handling in the Eliashberg drivers (all
modes at Γ are zeroed today), the sparse vertex layout beyond ~30 atoms, and the random-displacement
least-squares extraction that replaces the 6N Cartesian runs.

## Questions and answers (1 November 2026)

Three clarifications written during the zero-point test on Pb (`Workspace/PAOFLOW/validation/Pb_fd_fixed_basis`,
`scripts/fd_zg/`), kept here for reference.

### How the finite-difference calculation works, which cells are which

**The idea.** The electron-phonon vertex is the derivative of the electronic Hamiltonian with respect to an atomic
displacement, dH/du. DFPT computes that derivative analytically. The finite-difference route computes it numerically:
take a cell, move one atom by a small step, recompute the electrons, and divide the change of the Hamiltonian by the
step.

**The three kinds of cells.**

1. **The reference cell** is the geometry whose vertex you want, with no atom moved. In the validated test it is the
   ideal Pb crystal as a 2×2×2 supercell (8 atoms, all on lattice sites). In the zero-point test it is the ZG cell at
   T = 0 (8 atoms, each displaced by its zero-point amount). One scf of it is run, and its PAO Hamiltonian H₀ is built.
2. **The displaced cells** are copies of the reference cell with one atom moved by +0.06 bohr or −0.06 bohr along x,
   y or z. Each is a separate scf. In the ideal supercell, symmetry makes all 8 atoms equivalent, so only one atom is
   displaced: 6 runs. In the ZG cell nothing is equivalent, so every atom is displaced: 8 atoms × 3 directions × 2
   signs = 48 runs.
3. **The primitive cell** appears only at the end, in the ideal test: the vertex of the supercell is translated back
   into the language of the 1-atom crystal, q-points and all, so that the usual dense-grid Eliashberg machinery can
   consume it.

**The derivative, and why the basis is "fixed".** For each (atom, direction), dV = [H(+0.06) − H(−0.06)] / 0.12. The
subtlety is that the PAO Hamiltonian is written in a basis of atomic orbitals, and if the orbitals moved with the atom,
the difference would mix the physical response with a basis-change term. So all displaced cells are projected onto
orbitals centred at the *reference* positions. Then the difference is the matrix of dV/du in one fixed basis, which is
what DFPT gives too. This was checked on the ideal cell to 0.1 % on the band elements near E_F.

**The completion part, which is the trap.** A PAO Hamiltonian describes the good bands exactly and replaces everything
else by a constant η (6.2 eV here) times a projector onto the "rest". That rest is not physical and its derivative is
large: 79 % of the norm of dV. In the ideal cell it does no harm because it is the same for every atom and stays away
from E_F. The good-subspace projection removes it where it matters.

**Where the two tests diverge.**

- *Ideal cell:* 6 runs → one dV → by translation symmetry it is also the response of the other 7 atoms → folded to the
  primitive cell's (k, q) representation on the 2³ q-grid → written as the coarse vertex → the standard driver
  interpolates and gives λ = 1.70 against DFPT 1.66.
- *Zero-point cell:* 48 runs → 8 independent dV's, one per atom → summed with the phases e^{iq·T} to form the response
  to a phonon-like pattern at each q → same fold. The sum is exact (verified on synthetic data), but each atom's dV now
  carries its own, different completion part, so the 8 contributions add incoherently and the folded vertex is wrong
  (λ 0.73). The physical part is fine: evaluated between the cell's own eigenstates near E_F, atom by atom and q by q,
  it agrees with the ideal cell to 2 to 5 %.

**What that means.** For a disordered reference, stop at the supercell: use the derivatives in the band basis of the
reference cell itself at q = Γ, with the cell's phonons, and never route them through the primitive orbital-basis
cache. The ideal-cell fold remains valid for crystals.

### The supercell-native route in detail

**Why the supercell is enough.** A supercell of N atoms has 3N phonon modes at its own Γ point, and those modes are
exactly the phonons of the underlying crystal at the N q-points commensurate with the cell (for 2×2×2, the 2³ q-grid).
Likewise its k-points fold the primitive k and k+q onto the same supercell k. So "q = Γ of the supercell" already
contains every (k, q) pair that the coarse grid of the crystal would give. For an ideal crystal the two descriptions
are the same information written differently; for a disordered cell only the supercell description exists, because
there is no primitive cell to fold to.

**The band basis of the reference cell.** Diagonalise the PAO Hamiltonian H₀ of the reference cell (ideal or ZG) at
each of its supercell k-points: eigenstates |m, k⟩ with energies ε_m(k). These are the actual electronic states of that
structure, disorder included. Then take the matrix elements of the 3N per-atom derivatives between those states,

```
d_{κα}(m, n; k) = ⟨m, k| dV_{κα} |n, k⟩,
```

keeping only states within a window around E_F (about 1 eV). Two things happen automatically here. The completion
part η·Q, which wrecked the fold, drops out: between good states its first-order matrix elements vanish, and the window
keeps us far from the completion edge where the step mixes good and complement states. And the disorder is treated
exactly, because the states are those of the disordered cell. This is the quantity computed by
`scripts/fd_zg/diag_zg10.py`, which agrees with the ideal cell to 2 to 5 % at q ≠ 0.

**The cell's phonons.** The same 48 displaced runs contain the forces on all atoms. Finite differences of those forces
give the 3N×3N force-constant matrix of the reference cell: for the ZG cell these are the zero-point (or thermal)
renormalised force constants, which is the first step of the anharmonic SDM loop. Diagonalising it with the masses
gives the modes ν at Γ, frequencies ω_ν and eigenvectors e_{κα}^ν. Alternatively, for a first test, one can fold the
harmonic DFPT phonons of the crystal to the supercell Γ; that isolates the effect of the disorder on the electronic
vertex from its effect on the phonons.

**Putting them together.** The mode-resolved vertex is the rotation of the per-atom derivatives onto the modes,

```
g_ν(m, n; k) = Σ_{κα} e_{κα}^ν / sqrt(2 M_κ ω_ν) · d_{κα}(m, n; k),
```

and the Eliashberg function follows as in the crystal case, with k summed over the supercell grid and the δ-functions
smeared or made adaptive:

```
α²F(ω) = (1/N_F) Σ_{k,m,n,ν} |g_ν(m,n;k)|² δ(ε_m(k)) δ(ε_n(k)) δ(ω − ω_ν).
```

λ, ω_log, Tc, transport and the ARPES self-energy are then the usual functionals of α²F. For the ideal 2×2×2 cell this
must reproduce the primitive calculation on the 2³ q / 12³ k grid, which is the consistency check to run first.

**What is lost and what has to be built.** Lost: q-interpolation. The coarse 2³ sampling is what the cell gives, and
denser q means a bigger cell (hence the 64-atom cells; the 3³ pathology of Pb was exactly a sampling problem). The
k-sum can still be made denser by interpolating the supercell Hamiltonian in k, the same PAO interpolation used for
crystals. To build: the force-constant extraction from the displaced runs, the Γ-mode handling in the Eliashberg
drivers (today every mode at Γ is zeroed, only the three acoustic ones should be), and the dense-k interpolation for
the supercell. The diagnostic script already does the vertex part.

**Why the ideal-cell fold stays.** For a crystal, translating the 8 atoms' responses into the primitive (k, q) language
is exact, and the completion part is identical for every atom, so it interpolates smoothly and never reaches E_F. That
is why the validated fixed-basis numbers (λ 1.70 vs 1.66) stand. The fold only breaks when the cell is not a crystal.

### "Any disordered cell has an undisplaced counterpart: can one use its finite differences directly?"

**The undisplaced counterpart is just the crystal.** Its finite-difference vertex is the derivative at the equilibrium
geometry, the same object DFPT computes, and it is validated (λ 1.70 vs 1.66). Combined with the harmonic phonons and
the Bose factors it gives the whole standard theory: α²F, λ, Tc, transport, the kink. That is what the production
pipeline already does, with DFPT instead of finite differences. For a crystal the finite-difference route adds nothing
except independence from ph.x.

**What the displaced cell adds, and when it is needed.** The vertex at the thermal geometry is
dV(u_T) = dV(0) + (second derivatives)·u_T + …, so running the finite differences at the ZG geometry measures how the
coupling itself changes with the disorder, beyond the Bose-factor temperature dependence of the harmonic theory. That
is the only reason to displace atoms of an already-disordered cell. First measurement: +2 to +5 % on Pb's Fermi-window
vertex at zero point. For a crystal at low temperature this is a small correction and the counterpart suffices. It
stops being a correction, and the counterpart stops existing in any useful sense, for alloys, strongly anharmonic
systems and MD snapshots, which is the target of section 9 of the SDM plan.

**By goal:**

- Thermal band shifts and widths: the ZG scf alone, no finite differences at all (the thermal driver), or perturbation
  theory with the crystal vertex.
- α²F and everything downstream at the harmonic level: the crystal, by DFPT or by finite differences of the undisplaced
  supercell.
- Coupling renormalised by disorder, anharmonicity or alloying: finite differences at the disordered geometry,
  consumed in that cell's band basis.

**Why the crystal's machinery cannot be borrowed for the third case.** The fold to the primitive cell uses the crystal's
Bloch states and silently relies on every atom's derivative carrying the same completion part η·dQ/du, which then
cancels or stays away from E_F. The derivatives of the disordered cell each carry their own completion part, because
the poorly projected states of that cell are different. Projecting them with the crystal's projector does not help: the
contamination sits in the operators, not in the basis they are written in. The eigenstates of the disordered cell near
E_F are the one representation in which the contamination drops out at first order, and that is the band basis of the
supercell-native route.

**What the counterpart is still used for.** As the reference of the comparison, both for the electrons (the
undisplaced ZG cell in the thermal driver removes the supercell offset) and for the vertex (the ideal cell's
derivatives against which the +2 to +5 % was measured); and as the consistency check of the supercell-native code,
which run on the ideal 2×2×2 cell must reproduce the primitive λ on the 2³ q / 12³ k grid before it is trusted on a
disordered cell.
