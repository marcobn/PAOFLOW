# Self-consistent charge in Landauer transport at finite bias: plan

Status: implemented on 15 November 2026 with the recommended decisions (D1 planar, D2 Hartree only, D3 mu_L/mu_R coefficients, D4 linear ramp included, D5 Al | Si | Al); see [negf-finite-bias](negf-finite-bias.md). The 3D Poisson solver (plan: [negf-3d-electrostatics-plan](negf-3d-electrostatics-plan.md)), the on-site U and spin polarisation (P5) remain open.

## 1. Where the Transport module is today

`Transport.compute_current` integrates the **zero-bias** transmission between two shifted Fermi functions
(`transport/calculators/current.py`):

$$
I(V) = \frac{2e}{h}\int dE\; T_0(E)\,\bigl[f(E-\mu_L) - f(E-\mu_R)\bigr],\qquad \mu_{L,R} = \text{mu\_L}\cdot V,\ \text{mu\_R}\cdot V .
$$

The Hamiltonian does not see the bias. No electron density is computed, there is no potential drop, and T does
not depend on V. This is exact for the linear conductance (dI/dV at V → 0). It is a fair approximation for metals with
weak scattering, and wrong whenever charge piles up at a barrier, an interface or a molecule (tunnel junctions,
Schottky contacts, resonant tunnelling, molecular junctions).

### What already exists and can be reused

| need | existing piece | where |
|---|---|---|
| G, Σ_L, Σ_R, Γ at one (E, k∥) | `compute_kpoint_self_energies`, `compute_kpoint_green` | `transport/conductor_kpoint.py` |
| complex energies z = E + iη | `delta` enters as (E + iδ)S − H in the decimation and in G_C, so `delta = η` gives G(z) | `calculators/transfer.py`, `green.py` |
| rigid shifts of L, C, R | `configure_onsite_shifts(shift_L, shift_C, shift_R)`: aux = (E − shift)S − H | `hamiltonian/hamiltonian_setup.py` |
| an energy-dependent self-energy in C | `block.sgm` (correlation self-energy) and the Ferretti `'generalized'` formula | `hamiltonian_setup.py`, `transmittance.py` |
| transverse momenta | the HRs grid and `nk` (fixed 15 Nov 2026) | `parsers/atmproj_tools.py` |
| screening region | `'conductor'` mode: C is the whole supercell, its end layers being lead material | `partition/layers.py` |
| atom positions per orbital | `AtomOrbitals(position, first_orbital, last_orbital)` | `partition/types.py`, `orbital_inventory.py` |
| MPI | energies split over ranks, every rank loops over all k∥ | `conductor_pipeline.compute_conductor_results` |
| mixing | `anderson_fixed_point(map, trial, nsiter, conv_thr, mix, nhist)` | `elphon/migdal_eliashberg.py` |

### Missing

- The density matrix from NEGF (G<), with the equilibrium contour integral and the non-equilibrium window integral.
- Electrostatics: the potential from the bias-induced charge, with the leads as boundary conditions.
- The self-consistency loop, with mixing and continuation in V.
- T(E; V) with the bias-dependent Hamiltonian, and the finite-bias Landauer current.
- Leads under bias. With `leads_are_identical` (set by `prepare_data.check_leads_are_identical`) the left self-energy
  reuses the right lead's transfer matrices, which is wrong once `shift_L != shift_R`. It must be switched off, or the
  left lead evaluated at E − μ_L, under bias.
- The partition does not keep the positions of the C orbitals (needed for Poisson).

## 2. Physics and approach

PAOFLOW has no DFT engine. The PAO Hamiltonian H₀ of the DFT run at V = 0 is the equilibrium self-consistent
reference, and the bias adds the **Hartree response to the bias-induced density change**. This is SCC-DFTB / gDFTB
logic on an *ab initio* tight-binding Hamiltonian (Pecchia and Di Carlo).

$$
H(V) = H_0 + \mathrm{diag}(\phi_i),\qquad \delta n_i = n_i(V) - n_i(0),
\qquad \nabla^2\phi = -4\pi\,\delta\rho[\delta n],\qquad \phi\big|_{L} = \mu_L/e,\ \phi\big|_{R} = \mu_R/e .
$$

- The basis is orthonormal (Löwdin PAOs), so a local potential is diagonal and the occupations are `Re ρ_ii`. The
  coupling blocks LC and CR are unchanged.
- Both n(V) and n(0) come from the same open-system NEGF calculation. The error of the PAO projection in the absolute
  charge cancels in δn, and V = 0 returns H₀ exactly.
- The dielectric constant in Poisson is 1: the valence electrons in the PAO basis screen explicitly. That is
  Hartree/RPA screening. The exchange-correlation response is neglected (option D2: an on-site U δq term).
- The leads are charge-neutral metals. Their bands shift rigidly with their electrochemical potentials
  (`shift_L = μ_L`, `shift_R = μ_R`). The buffer layers of C must screen the potential to these values; this is
  checked by flatness.

### Density matrix

Following Brandbyge et al., PRB 65, 165401 (2002), with ω_n = (2n+1)πk_BT:

$$
\rho = \rho_{\rm eq}(\mu_R) + \rho_{\rm neq},\qquad
\rho_{\rm eq} = -\frac{1}{\pi}\,\mathrm{Im}\Bigl[\int_C dz\,G(z)f(z-\mu_R) - 2\pi i\,k_BT\sum_{n<N_p} G(\mu_R+i\omega_n)\Bigr],
$$
$$
\rho_{\rm neq} = \frac{1}{2\pi}\int dE\;G(E)\,\Gamma_L(E)\,G^\dagger(E)\,\bigl[f(E-\mu_L)-f(E-\mu_R)\bigr],
$$

summed over k∥ with their weights.
- **Equilibrium part.** It needs 30–60 points on a semicircle from below the bottom of the C spectrum, at height
  2N_pπk_BT, plus the N_p Fermi poles. These are the Matsubara frequencies of the EPC lecture box. At large Im z,
  G is smooth and the decimation converges in a few steps.
- **Non-equilibrium part.** It needs real energies in the bias window ± 10 k_BT, with a small δ. Only the diagonal of
  ρ is accumulated, so memory stays O(dim C).
- States of C that are bound outside the lead bands are filled by the equilibrium contour (μ_R). This is the known
  NEGF caveat for bound states in the window.

### Electrostatics

- **E1, planar (MVP).** Sheet charges σ_l = e δq_l / A per layer, and a 1D Poisson equation along the transport axis
  with Dirichlet conditions at the two ends of C. The potential is piecewise linear, from cumulative sums of the
  charges, and φ_i = φ(z_atom). It costs nothing and is exact in planar average for laterally uniform junctions
  (Al | Si | Al, metal | oxide | metal).
- **E2, 3D.**
  - Atom-centred Gaussian charges; their widths come from covalent radii or U.
  - The Poisson solve is an FFT in the transverse plane (periodic) and a tridiagonal or sine transform along z
    (Dirichlet).
  - φ_i is the potential averaged over each orbital's Gaussian.
  - This is needed for molecules, nanowires and point contacts.
- **E3, SCC-DFTB γ functional** with 2D Ewald sums and image planes. It fits Dirichlet electrodes poorly; not
  recommended.

### Self-consistency and current

- **Self-consistency.**
  - The loop is φ → H(V) → ρ → δq → φ, with Anderson mixing on φ. `anderson_fixed_point` moves to `utils/mixing.py`
    unchanged.
  - Convergence: max|Δφ| < 1 meV and the total δQ.
  - Continuation in V: each bias starts from the previous one, and the first from the linear ramp.
  - Warn on hysteresis (resonant tunnelling bistability).
- **Current.** T(E; V) is computed with H(V) and the shifted leads on a grid covering the window, and then
  $I(V) = \frac{2e}{h}\int T(E;V)\,[f_L - f_R]\,dE$.
- **Outputs:**
  - I(V) and dI/dV;
  - φ per layer (or atom);
  - δq per layer;
  - T(E; V);
  - SCF iterations and residual.

## 3. API (main.py reads as physics)

```python
t = Transport(paoflow.data_controller)
t.define_partition(central_atoms='ALL', left_lead_layers=4, right_lead_layers=4, transport_direction='z')
t.configure_energy_grid(emin=-3.0, emax=3.0, ne=601, delta=1e-3, nk=(6, 6))
t.build_hamiltonian_blocks(calculation_type='conductor', use_sym=False, do_overlap_transformation=False)
t.configure_bias_self_consistency(electrostatics='planar', temperature=300.0, contour_points=40,
                                  window_spacing=0.005, mixing=0.3, tolerance=1e-3, max_iterations=60)
results = t.compute_current_self_consistent(bias=np.linspace(0.0, 1.0, 11), mu_L=0.5, mu_R=-0.5)
```

`compute_current` (rigid shift) stays as it is. An optional non-self-consistent `profile='linear'` (ramp across C)
gives the textbook intermediate step.

Backend: new package `transport/negf/`, with descriptive names and no `do_*`:
- `bias_hamiltonian.py`: the potential on the C blocks, the lead shifts and lead identity under bias;
- `contour.py`: contour points and weights, Fermi poles, the window grid;
- `density_matrix.py`: k∥-summed occupations from the two integrals, MPI over energy points, reusing the per-point
  kernels with complex z;
- `electrostatics.py`: planar Poisson (E1), later the 3D solver (E2);
- `self_consistency.py`: the loop, mixing, continuation, convergence report.

`TransportResults` gets bias-resolved fields, and the partition keeps the z of every C orbital.

## 4. Validation (each item a unit test unless marked)

1. **Complex z.** The decimation G(z) equals direct inversion of a long finite chain at complex z.
2. **V = 0.** The contour density equals the real-axis density (small δ, dense grid) to 1e-6. In the lead buffer
   layers it equals the band filling of the bulk lead.
3. **Gauge.** A common shift of μ_L, μ_R and φ leaves the current unchanged.
4. **Symmetric junction.** I(−V) = −I(V), and φ is antisymmetric.
5. **Linear response.** dI/dV at V → 0 equals the rigid-shift G(0), because the profile enters the current at
   O(V²).
6. **Metallic leads.** Σ δq → 0 and φ is flat in the buffer layers as the buffer grows.
7. **Datta single level with charging energy U** (D2 option): the analytic self-consistent I–V with broadened steps.
8. **Cubic one-band metal with an impurity plane** (transport lecture). The potential drops at the scatterer, with
   the residual-resistivity dipole (Landauer 1957) and Friedel oscillations.
9. **Tunnel barrier** (Al | Si | Al supercell of the lecture; manual run). The voltage drops across the barrier, with
   a charge dipole of capacitor size.
10. **MPI.** 1 and 3 ranks give identical results.

## 5. Cost

The Al | Si | Al lecture case has dim C = 182 and a 6×6 transverse mesh (36 k∥). One self-consistent iteration
evaluates about 250 energy points (40 contour points plus the window at 1 V and 5 meV) × 36 k∥, about 9000 kernel
calls. Each call is the decimation (≈ 20 small inversions) plus one 182³ inversion, a few ms. That makes:
- 20–45 s per iteration, serial;
- about 20 iterations per bias, so 7–15 min per bias;
- 1–2.5 h for 10 biases, serial; MPI over energies divides this almost linearly.

Model tests run in seconds.

## 6. Phases (agentic estimates)

| phase | content | estimate |
|---|---|---|
| P0 | lead identity under shifts, a complex-z entry to the kernels, C orbital positions in the partition; tests | 1 h |
| P1 | bias-dependent Hamiltonian, T(E; V), finite-bias current; the non-self-consistent linear-ramp option | 1.5 h |
| P2 | density matrix (contour + window, MPI); validations 1–2 | 2–3 h |
| P3 | planar Poisson, self-consistent loop, mixing, continuation, outputs; validations 3–6, 8 | 2 h |
| P4 | Al \| Si \| Al demo (8 ranks), docs/internals note, transport_tutorial section "Finite bias" (replaces the caveat) | 2–3 h |
| P5 (later) | 3D Poisson (E2), on-site U, spin polarisation, non-orthogonal overlaps | 3 h |

## 7. Limits

- H₀ comes from the DFT run at V = 0. The geometry is fixed and the exchange-correlation response is neglected
  (Hartree only, unless D2).
- The buffer layers of C must screen the field. If the potential is not flat there, a longer supercell (a new QE run)
  is needed.
- Resonances inside the bias window need a fine window grid; adaptive refinement is a later option.
- Bound states in C inside the window: their occupation is not fixed by the leads (NEGF caveat).
- Depletion lengths of doped semiconductors exceed any supercell. Schottky physics with doping is out of scope.

## 8. Decisions

- **D1 Electrostatics first.** Planar 1D Poisson (recommended; the planar junctions of the lecture) or the 3D grid
  solver from the start.
- **D2 Exchange-correlation response.** Hartree only (recommended), or with an on-site U δq term (U from ACBN0 or an
  input).
- **D3 Bias convention.** Keep `mu_L`, `mu_R` coefficients as in `compute_current` (recommended; ±V/2 by default).
- **D4 Non-self-consistent linear-ramp option.** Include it (recommended: cheap, and it separates the effect of the
  profile from that of self-consistency).
- **D5 Demo system.** The Al | Si | Al supercell of the transport lecture (recommended: it exists, with leads
  validated), or a new one, such as Au | molecule | Au, which needs new QE runs.

References:
- M. Brandbyge et al., PRB 65, 165401 (2002);
- J. Taylor, H. Guo, J. Wang, PRB 63, 245407 (2001);
- A. Pecchia, A. Di Carlo, Rep. Prog. Phys. 67, 1497 (2004); A. Pecchia et al., New J. Phys. 10, 065022 (2008);
- T. Ozaki, PRB 75, 035123 (2007) (Fermi-function poles);
- S. Datta, *Quantum Transport: Atom to Transistor* (2005) (self-consistent field with charging);
- A. Ferretti et al., PRL 94, 116802 (2005);
- R. Landauer, IBM J. Res. Dev. 1, 223 (1957).
