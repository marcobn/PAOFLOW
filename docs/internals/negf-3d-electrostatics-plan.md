# Three-dimensional electrostatics for finite-bias transport (molecules, wires): plan

Status: implemented on 15 November 2026 with the recommended decisions; see
[negf-3d-electrostatics](negf-3d-electrostatics.md). The kernel became semi-analytic (in-plane sums exact, sine
series along z) instead of a 3D grid solve; the grid is used only for the XSF output. It extends [negf-finite-bias](negf-finite-bias.md), whose planar
Poisson solver averages the bias-induced charge over the transverse cell. Decisions are in Sec. 8.

## 1. Why the planar solver is not enough

`compute_current_self_consistent` adds to H₀ the potential energy U of the bias-induced charge δn = n(V) − n(0). The
planar solver spreads the δq of each atomic layer uniformly over the transverse cell (area A) and solves
U″ = −4πδn between the electrode planes. That is exact in planar average for a junction uniform across the cell
(Al | Si | Al). It is wrong when the charge is localised in the plane:

- **Molecule between slabs.** The δq of a molecule sits on a few atoms, not on a sheet. The planar kernel
  (4π/A)(z_< − a)(b − z_>)/(b − a) depends on the arbitrary transverse cell. It misses the on-site Hartree energy of
  the molecule (its charging energy) and the lateral screening by the electrodes.
- **Wire.** Transport example02 (Al wire with a bridging H) has a 12-bohr transverse cell, mostly vacuum. The planar
  model would turn the wire into a charged plate of 144 bohr².
- **Gates.** A gate is a boundary condition off the transport axis: not representable in planar average.

The self-consistency loop does not depend on the geometry. From the electrostatics it uses only three things:
- `layer_of_orbital`, the map from orbitals to charge groups;
- the external potential at the groups (the ramp);
- the kernel P (eV per electron), with U = U_ext + P δq. P also builds the Thomas–Fermi preconditioner
  (1 + Pχ₀)⁻¹.

A 3D solver therefore replaces the layers by atoms and the planar kernel by a 3D one. Nothing else in
`transport/negf/` changes.

## 2. Physics and approach

### Charge model

The bias-induced electron number on atom A is δq_A = Σ_{i∈A} δn_i (Löwdin occupations: the PAO basis is
orthonormal). It is spread over a normalised spherical Gaussian:

$$
\delta n(\mathbf r) = \sum_A \delta q_A\, g_A(\mathbf r - \mathbf R_A),\qquad
g_A(\mathbf r) = (2\pi\sigma_A^2)^{-3/2}\, e^{-r^2/2\sigma_A^2}.
$$

- Point charges would give an infinite on-site term. The width σ_A sets the on-site Hartree energy: two Gaussians of
  width σ interact at zero distance with 1/(√π σ) hartree, which is 15.4 eV for σ = 1 bohr.
- **Width, parameter-free.** Match the Hartree self-energy of the atom's valence PAO density:
  σ_A = 1/(√π J_A), with J_A = ∫∫ρ_A ρ_A/|r − r′| and ρ_A = Σ_l w_l |R_l(r)|²/4π.
  - R_l comes from the radial tables of the run's basis: `atomic_basis` for internal projections, the UPF
    `PP_PSWFC` on the `projwfc.x` route.
  - w_l are the zero-bias shell occupations.
- The response is what is modelled: the equilibrium charge transfer (the molecule–metal dipole at V = 0) is already
  in H₀. The Gaussian shape only shapes δn.
- **Matrix elements.** The potential on atom A is the average of U over g_A, applied to all its orbitals:
  H = H₀ + ½{U, S} as now (diagonal for the orthonormal basis).

### Poisson problem

The electron potential energy obeys ∇²U = −4π δn (hartree units), as in the planar solver:
- transverse cell periodic, with in-plane vectors a₁, a₂ ⊥ transport axis (they may be skewed: hexagonal (111)
  slabs);
- z ∈ [a, b], with Dirichlet U(a) = μ_L − E_F, U(b) = μ_R − E_F at the electrode planes, half a layer spacing
  outside C, exactly as `PlanarLayers`.

U = U_ext + U_H. The Laplace part U_ext is the linear ramp for periodic transverse boundaries; with gates it is a
Laplace solution. U_H has homogeneous Dirichlet conditions and is diagonal in the mixed basis of in-plane plane waves
times sines along z:

$$
U_H(\mathbf r) = \sum_{\mathbf G_\parallel}\sum_{n\ge1} \frac{4\pi\,\delta n_{\mathbf G_\parallel n}}{|\mathbf G_\parallel|^2 + k_n^2}\,
e^{i\mathbf G_\parallel\cdot\mathbf r_\parallel}\,\sin k_n(z-a),\qquad k_n = \frac{n\pi}{b-a}.
$$

- **Solver.** FFT on the in-plane fractional grid, DST-I along z (`scipy.fft`), with spacing about 0.3 bohr.
  Cost O(N log N); about 0.1 s for 10⁶ points.
- **Planar limit.** The G∥ = 0 term is exactly the planar kernel smoothed along z by the Gaussians. With
  σ_z → 0 it equals `poisson_kernel`. The G∥ ≠ 0 terms are the in-plane structure. They decay as e^{−|G∥||z − z′|}
  along z.
- **Lateral screening.** Between grounded plates a distance d = b − a apart, the potential of a charge decays
  laterally as K₀(πρ/d) ~ e^{−πρ/d}. Periodic images of a molecule or wire therefore decouple once the transverse
  cell exceeds about d. Report this length so that users can check their cell.

### Kernel between atoms

The charges live on N_A atoms with fixed shapes, so the response is a dense N_A × N_A matrix, computed once per
geometry:

$$
P_{AB} = \int d^3r\; g_A(\mathbf r - \mathbf R_A)\; U_H[g_B](\mathbf r)\quad(\text{eV per electron}),\qquad
U_A = U_{{\rm ext},A} + \sum_B P_{AB}\,\delta q_B .
$$

- P is symmetric positive definite by construction (Galerkin projection with the same shapes). It is built from one
  spectral solve per atom.
- Each g_B is sampled on the grid and renormalised to unit charge. This corrects both the sampling and the part of
  the tail cut by the electrode planes, which those planes screen anyway.
- ⟨g_A|U_B⟩ is evaluated on a local box of ±5σ_A, so memory stays O(N_grid).
- Columns are distributed over MPI ranks, then reduced with one allreduce.
- The loop is unchanged: Anderson mixing of U_A, continuation in V, and the Thomas–Fermi preconditioner
  (1 + P diag χ₀)⁻¹, with χ₀ now per atom (Σ_{i∈A} −Im G_ii(E_F + iη)/π).

### Outputs

- Per-atom U and δq at every bias: element, x, y, z, U, δq (the analogue of the planar files).
- The planar average of U(r) along z, directly comparable with the planar solver.
- Optionally U(r) on the grid as XSF (DATAGRID_3D) at chosen biases, for VESTA or XCrySDen.

## 3. API (main.py reads as physics)

```python
t = Transport(paoflow.data_controller)
t.define_partition(central_atoms='ALL', left_lead_layers=4, right_lead_layers=4, transport_direction='z')
t.configure_energy_grid(emin=-3.0, emax=3.0, ne=601, delta=1e-3, nk=(4, 4))
t.build_hamiltonian_blocks(calculation_type='conductor', use_sym=False, do_overlap_transformation=False)
t.configure_bias_self_consistency(
    electrostatics='3d',      # 'planar' (default) or '3d'
    grid_spacing=0.3,         # bohr
    charge_widths=None,       # None: from the PAO valence densities; or {'Au': 1.9, 'S': 1.1} (bohr)
    temperature=300.0, window_spacing=0.005, mixing=0.5, tolerance=1e-3,
)
scan = t.compute_current_self_consistent(bias=np.linspace(0.0, 1.0, 11), mu_L=0.5, mu_R=-0.5,
                                         potential_grid_biases=[0.5])   # XSF of U(r) at 0.5 V
```

### Backend

- **Generic interface** in `electrostatics.py`: a protocol `ElectrostaticModel` with
  - `group_of_orbital` (dimC,) and `n_groups`;
  - `external_potential(potential_left, potential_right)`, shape (n_groups,);
  - `kernel()`, shape (n_groups, n_groups);
  - one `electrostatic_potential(charge, model, U_L, U_R)` = external + kernel @ charge.
- `PlanarLayers` implements the protocol (groups = layers; ramp; `poisson_kernel`); bit-identical outputs.
- **New** `transport/negf/poisson_3d.py`: the in-plane FFT × DST-I solver and the Laplace part.
- **New** `transport/negf/charge_model.py`: the Gaussian shapes and the widths from the radial tables.
- **New** `AtomicElectrostatics`, the 3D implementation (groups = atoms).
- `finite_bias_scan` and `ConductorPotential` take any `ElectrostaticModel`.
- `BiasPointResult` gets `group_potential` and `group_charge`. `layer_potential` and `layer_charge` stay as aliases,
  so the lecture notebook and the existing scripts are unchanged.
- `FiniteBiasResults` keeps `layer_coordinates` for planar and adds `group_positions` (n_groups, 3) and
  `group_labels`.
- XSF writer: a small `write_grid_xsf(path, cell, origin, grid)` in `writers/`. The existing `write2xsf` takes a data
  controller.

## 4. Validation (each item a unit test unless marked)

1. **Refactor.** The 11 planar tests and a stored planar scan are bit-identical.
2. **Solver.** A single in-plane-wave × sine mode is solved to machine precision. A Gaussian with σ ≥ 3h converges
   as the grid is refined.
3. **Image charges.** A Gaussian between two grounded plates in a wide cell: P_AA and P_AB against the alternating
   image series of erf(r/2σ)/r, to 1e-3. Lateral decay e^{−πρ/d}.
4. **Planar limit.**
   - The G∥ = 0 part with σ_z → 0 equals `poisson_kernel`.
   - A one-atom-per-layer crystal differs from the planar solver only by the in-plane Madelung term (quantified).
5. **Single level with charging** (Datta, *Quantum Transport: Atom to Transistor*, ch. 1 and 3).
   - Set-up: one site between two tight-binding chains (analytic Σ(E)).
   - Independent reference: the scalar self-consistent solution with the same η (from the Laplace part) and
     U₀ = P_AA.
   - N(V), U(V) and I(V) agree to 1e-6: the steps of the I–V broadened by charging.
6. **Symmetric molecule and wire models.**
   - I(−V) = −I(V), and U is antisymmetric under the mirror.
   - Linear response equals ∫T(−f′).
   - MPI: 1 and 3 ranks agree.
7. **Al | Si | Al, 3D against planar** (manual; no QE: the lecture Hamiltonian). The fraction of V across the Si plane
   (planar 0.157–0.159) and the current (planar to 0.2 %) should move only by the in-plane Madelung term.
8. **Al–H–Al wire, transport example02** (manual; saves in the transport assets, no QE).
   - Planar against 3D: where the voltage drops and the charge on H.
   - The planar model is expected to fail here.
9. **Molecular junction demo** (manual; needs QE, D4). Voltage division η, level alignment under bias and the I–V.

## 5. Cost

- **Kernel.** N_A spectral solves once per geometry.
  - Au(111) 3×3 slab junction with benzenedithiol (about 120 atoms; grid 54 × 54 × 250 at 0.3 bohr, 0.7 M points):
    about 10 s serial, less on ranks.
  - The chain demo (about 25 atoms): about 1 s.
- **Per iteration.** P δq is negligible. The cost stays that of the NEGF density, as for the planar solver.
  - Au(111) junction: dim C ≈ 1500, 16 k∥ and about 300 energies, which is about 500 s per iteration serial. That is a
    few minutes per bias on 64 ranks: a Snellius job.
  - Chain demo: dim C ≈ 150, seconds per iteration.

## 6. Phases (agentic estimates; compute time on top)

| phase | content | estimate |
|---|---|---|
| P1 | `ElectrostaticModel` protocol; `PlanarLayers` implements it; group fields with layer aliases; test 1 (bit-identical) | 30 min |
| P2 | `poisson_3d.py`, `charge_model.py` (widths from the radial tables, per-species override), kernel; tests 2–4 | 1.5 h |
| P3 | Transport plumbing (`electrostatics='3d'`), atom-level preconditioner, outputs (per-atom, planar average, XSF); tests 5–6 | 1 h |
| P4 | validations 7–8 (no QE), REPORT.md in `validation/negf_3d_electrostatics/`, docs note; optional lecture subsection (single level with charging) | 1–1.5 h + minutes of compute |
| P5 | molecular junction demo (D4): QE, projection with BASIS_PS (pthr 0.95), scan, report | chain: about 1 h on the Mac Studio; Au(111): Snellius |
| P6 (later) | gates (box solver: DST or DCT on transverse faces, Γ-only k∥), on-site exchange-correlation term, atomic dipoles, Dirichlet regions for wire leads, dielectric media | — |

## 7. Limits

- **Spherical charges.** Atoms polarise only by charge transfer between atoms. The field does not mix s and p on the
  same atom.
  - A later option: atomic dipoles, with ∇U(R_A)·⟨φ_i|r − R_A|φ_j⟩ in H and p_A in the Poisson source. PAOFLOW
    already computes the position-dipole matrix elements.
- **Electrode planes.** Dirichlet planes across the whole transverse cell are exact for slab electrodes. For wire
  leads (example02) they are an approximation: the fringing field ends on the plane instead of on the lead wire.
  - Mitigation: long buffers of lead wire in C.
  - Later (P6): Dirichlet regions on the lead atoms.
- **Periodic transverse cell.** It holds an array of molecules or wires. It is isolated only when the cell exceeds
  about d (Sec. 2).
- **Mean field.** Hartree only, ε = 1, fixed geometry, H₀ from the V = 0 run, as in the planar solver.
  - No Coulomb blockade: charging is a single-particle shift.
  - No image-charge renormalisation of molecular levels (a correlation effect).
- **Basis and spin.** Orthonormal basis and spin-unpolarised (or spinor) inputs only, as now.

## 8. Decisions

- **D1 Charge shape.** One Gaussian per atom with the parameter-free width from the PAO valence density (recommended),
  with a per-species override. Shell-resolved groups (Au 5d and 6s separately) come later; the interface allows them,
  since a group is any set of orbitals.
- **D2 Transverse boundary.** Periodic (FFT) now (recommended: it matches the k∥ mesh and slab electrodes). Box
  boundaries (Neumann or Dirichlet faces) come with the gates in P6.
- **D3 Exchange-correlation.** Hartree only (recommended, as for planar), or an on-site term U_A^xc δq_A per species,
  from an input or from ACBN0, added to the diagonal of P.
- **D4 Molecular demo.**
  - (a) Au chain | benzenedithiol | Au chain: wire leads, Γ-only k∥, QE on the Mac Studio in under an hour
    (recommended).
  - (b) Au(111) 3×3 | benzenedithiol | Au(111): slab leads; production cost on Snellius (shared budget).
  - (c) No new QE: Al | Si | Al and the example02 wire only.
- **D5 Gates.** Later (P6, recommended), or now together with the box solver.
