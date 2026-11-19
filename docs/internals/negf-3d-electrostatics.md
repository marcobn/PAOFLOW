# Three-dimensional electrostatics for finite-bias transport

Implemented on 15 November 2026 from the plan in [negf-3d-electrostatics-plan](negf-3d-electrostatics-plan.md),
with the recommended decisions:
- D1: one Gaussian per atom, width from the pseudo-atomic valence density, per-species override;
- D2: periodic transverse boundaries;
- D3: Hartree only;
- D4: Au chain | benzenedithiolate | Au chain as the molecular demo;
- D5: gates later.

It extends [negf-finite-bias](negf-finite-bias.md): the self-consistent loop is unchanged, only the electrostatics
behind it is new.

## Use

```python
t.build_hamiltonian_blocks(calculation_type='conductor', use_sym=False, do_overlap_transformation=False)
t.configure_bias_self_consistency(electrostatics='3d',        # 'planar' (default) or '3d'
                                  charge_widths=None,         # None: from the UPF files; a float; or {'Au': 1.9}
                                  temperature=300.0, window_spacing=0.005, tolerance=1e-3)
scan = t.compute_current_self_consistent(bias=[0.0, 0.5, 1.0], potential_grid_biases=[1.0])
```

The scan is a `FiniteBiasResults`, as for the planar electrostatics, with:
- `group_coordinates` of shape (n_atoms, 3) and `group_labels` (elements);
- per point, `group_potential` and `group_charge` (per atom), and `planar_average_potential` on
  `planar_average_coordinates`.

The `layer_*` names stay as aliases of the `group_*` fields. `paoflow-gen` and the Studio write this
script (workflow `landauer`, `main.landauer.py`). Output files:
- `potential_` and `charge_<profile><postfix>.dat`: per atom (index, element, x, y, z, then one column per bias);
- `potential_planar_average_<profile><postfix>.dat`;
- `potential_grid_<profile><postfix>_V<bias>.xsf`: U(r) on a grid of the slab, for XCrySDen or VESTA.

The transverse lattice vectors must be perpendicular to the transport axis.

## What it computes

**Interface.** The loop sees the electrostatics only through `ElectrostaticModel` (`electrostatics.py`):
- the charge group of every orbital;
- the Laplace potential of the electrodes on the groups;
- the kernel P, so that U_g = U^ext_g + Σ_h P_gh δq_h.

`PlanarLayers` (groups = layers) and `AtomicElectrostatics` (groups = atoms) implement it. The refactor that
introduced it left the planar scans bit-identical.

**Charge model** (`charge_model.py`).
- δq_A spreads over a normalised Gaussian of width σ_A.
- The width matches the Hartree self-energy J of the occupation-weighted valence pseudo-atomic density (UPF
  `PP_PSWFC`): σ = 1/(√π J). Hydrogen 1s gives J = 5/8 exactly.
- Values: Al 1.76 bohr (on-site 8.7 eV), Si 1.54 bohr (10.0 eV).

**Solver** (`poisson_3d.py`).
- The slab between the electrode planes (those of the planar model, half a layer spacing outside C) is periodic
  across the transverse cell.
- The potential is expanded in in-plane plane waves times sines along z, which is diagonal for ∇².
- A Gaussian has an analytic in-plane factor e^{−G²σ²/2} (periodic images included exactly). Its profile along z is
  sampled on a grid of σ_min/8 and sine-transformed (DST-I).
- Near a plane the profile is the Gaussian minus its mirror images, renormalised to one electron. It vanishes at
  the planes, so the sine series converges spectrally. A hard cut converged only to 2e-3.
- P_AB = (4π/A) Σ_G e^{iG·(R_A − R_B)} e^{−G²(σ_A² + σ_B²)/2} Σ_n γ_A(n)γ_B(n)/(G² + k_n²). The sum is grouped in
  shells of |G| and computed on rank 0, then broadcast. It takes seconds even for hundreds of atoms.
- The G = 0 term is the planar kernel smoothed by the profiles. Its diagonal is lower by exactly (4π/A)σ/√π (the
  self-energy of the Gaussian profile).

**Hamiltonian.** U_A acts on all the orbitals of atom A: H = H₀ + ½{U, S}, as for the layers. The Thomas-Fermi
preconditioner uses the per-atom χ₀.

## Validation

Model tests (`tests/unit/transport/test_finite_bias_3d.py`, 7 tests, 13 s):

| check | result |
|---|---|
| Hartree self-energy of H 1s | 5/8 to 1e-9 |
| two Gaussians far apart between plates against the Fourier-Bessel point-charge Green's function | 4e-6 |
| on-site kernel against 1/(√π σ) plus the image series | 3e-9 |
| G = 0 part against `poisson_kernel` minus the profile self-energy | 1e-5 eV |
| skewed cell with overlapping images against an FFT × DST solution on a real-space grid | 1.5e-4 (kernel), 4e-5 eV (potential, planar average) |
| Datta single level with charging (custom `ElectrostaticModel`; level between two chains, U₀ = 1 eV) against the scalar self-consistent solution | U, δq to 1e-6; I to 1.4e-4 |
| chain with a barrier, `electrostatics='3d'` through the Transport API | I(−V) = −I(V) to 1e-5, mirror symmetry, linear response 2e-3, outputs, XSF planes at ±V/2 |
| MPI, 1 against 3 ranks | 3e-17 (I), 2e-15 eV (U) |

DFT junctions (reports and scripts in `~/Workspace/PAOFLOW/validation/negf_3d_electrostatics/`):

**Al | Si plane | Al** (the 14-layer lecture Hamiltonian, 6 × 6 transverse mesh, 8 ranks, 503 s): 3D against planar.

| V | I 3D / planar (2e/h·eV per cell) | fraction of V across Si, 3D / planar | max \|ΔU\| |
|---|---|---|---|
| 0.10 | 0.13313 / 0.13314 | 0.155 / 0.157 | 3.8 meV |
| 0.25 | 0.33903 / 0.33909 | 0.155 / 0.159 | 5.8 meV |
| 0.50 | 0.65921 / 0.65930 | 0.112 / 0.117 | 9.1 meV |

For a junction uniform across the cell the 3D solver reproduces the planar one. It differs only by the profile
smearing and the in-plane structure. δq is up to 1.8 times larger, because the smeared diagonal is softer.

**Al wire with a bridging H** (transport example02 saves; 11 atoms, 12-bohr transverse cell, 4 ranks, 5 s).
- Planar and 3D agree: I to 1e-4, U to 0.01 V per volt, δq to 30 %. The wire screens strongly (Pχ₀ ≫ 1), so local
  neutrality fixes the potential whatever the kernel.
- A band edge of the wire 0.45 eV below E_F makes the current saturate: 0.32 at 0.5 V against 0.64 for the rigid
  shift.

**Au chain | benzene-1,4-dithiolate | Au chain** (26 atoms, 12 Å transverse cell, c = 48.6 Å; QE 48 min and the
scan 213 s on 16 ranks).
- **Zero bias.** T(E_F) = 0.0087. The d_π band edge of the Au-chain leads lies 0.12 eV below E_F.
- **Rigid shift.** It overestimates the current 10–200 times: once the leads shift, their channels no longer
  overlap.
- **Screening.** The Au chains are flat at ±V/2: the bias drops on the molecule and its contacts.
- **Asymmetric pinning.** The molecule follows the lead with the higher chemical potential and charges from it
  (mirror symmetric under V → −V to 1 meV).
- **Planar against 3D.** The potentials on the atoms agree (the metals pin them), but the planar model puts 2.5
  times more charge on the molecule and 0.20–0.30 of V across it, against 0.14–0.21 in 3D. The currents differ by
  2–16 %.
- **Au width.** Changing it from 0.87 to 2.0 bohr moves U by at most 21 meV.

## Finding: the conductor partition and non-bulk end layers

In `calculation_type='conductor'` both leads take their H₀₀ from the **left** end layers of the supercell
(`partition/layers.py`: `H00_R` = `H00_L`), and the couplings come from the periodic boundary. This is correct only
when the end layers are bulk-like.

In example02 they are not: the on-site energies run −2.07, −2.026, −2.018 eV towards the H. The right lead, built
from the left block, has its orbitals in the opposite order to the mirror of the left side. The zero-bias occupations
of mirrored atoms then differ by up to 0.02 electrons, and the bias-induced potential is not mirror-antisymmetric
(0.13 against −0.17 V per volt at the ends), with planar and 3D electrostatics alike. The conductor block itself is
mirror-symmetric to 9e-5 eV.

Use enough buffer layers that the lead layers are bulk-like. A check that compares the H₀₀ of the left and right end
layers would catch this.

## Limits

- **Spherical charges.** Spherical Gaussians per atom: no intra-atomic polarisation (s–p mixing by the field).
- **Electrode planes.** Dirichlet planes across the whole cell are exact for slab electrodes and an approximation for
  wire leads.
- **Periodic cell.** It holds an array of molecules or wires. The lateral images decouple once the cell exceeds the
  distance between the planes.
- **Mean field.** Hartree only, ε = 1, fixed geometry; no Coulomb blockade, no image-charge renormalisation of
  molecular levels.
- **Basis and spin.** Orthonormal basis, spin-unpolarised or spinor inputs.

## Files

- `transport/negf/`:
  - `electrostatics.py`: `ElectrostaticModel`, `group_sums`, `electrostatic_potential`;
  - `atomic_electrostatics.py`;
  - `poisson_3d.py`;
  - `charge_model.py`.
- `writers/grid_xsf.py`.
- `Transport.configure_bias_self_consistency(electrostatics='3d', charge_widths=..., potential_grid_spacing=...)`,
  `compute_current_self_consistent(potential_grid_biases=...)`.
- Tests: `tests/unit/transport/test_finite_bias_3d.py`.
