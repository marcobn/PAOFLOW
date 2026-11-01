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
