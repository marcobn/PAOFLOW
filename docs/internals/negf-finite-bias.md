# Finite-bias Landauer transport with a self-consistent potential

Implemented on 15 November 2026 from the plan in [negf-finite-bias-plan](negf-finite-bias-plan.md). The plan's
decisions were taken as recommended:
- planar electrostatics first;
- Hartree response only;
- the `mu_L`/`mu_R` coefficients of `compute_current`;
- a linear-ramp option;
- Al | Si | Al as the demo.

## Use

```python
t = Transport(paoflow.data_controller)
t.define_partition(central_atoms='ALL', left_lead_layers=4, right_lead_layers=4, transport_direction='z')
t.configure_energy_grid(emin=-0.6, emax=0.6, ne=241, delta=1e-3, nk=(6, 6))
t.build_hamiltonian_blocks(calculation_type='conductor', use_sym=False, do_overlap_transformation=False)
t.configure_bias_self_consistency(temperature=300.0, window_spacing=0.01, mixing=0.15, tolerance=1e-3)
scan = t.compute_current_self_consistent(bias=[0.0, 0.1, 0.25, 0.5], mu_L=0.5, mu_R=-0.5,
                                         profile='self_consistent')   # or 'linear'
```

The scan returns `FiniteBiasResults` (`transport/negf/self_consistency.py`), with I, U per layer, δq per layer,
T(E; V), iterations, convergence and residual for every bias. It writes the following files to the output directory:
- `current_<profile><postfix>.dat`;
- `potential_…`;
- `charge_…`;
- `transmission_…`.

The conductor Hamiltonian is restored at the end.

## What it computes

The zero-bias PAO Hamiltonian H₀ is the self-consistent reference. At a bias V:
- the leads shift rigidly to μ_{L,R} = E_F + c_{L,R}·eV (`shift_L`, `shift_R` of the kernels);
- the conductor gets a potential energy U on every atomic layer, H = H₀ + ½{U, S} (= H₀ + diag U for the orthonormal
  PAO basis);
- with `profile='self_consistent'`, U = U_ramp + v_H[δn] with δn = n(V) − n(0). Both densities come from the same
  open-system Green's functions, so the error of the projection in the absolute charge cancels and V = 0 returns H₀.
  The exchange-correlation response is neglected.

### Density

`density_matrix.open_system_density`. Two exact decompositions of the same density:
- ρ^L = ρ_eq(μ_R) + Δ_L, with Δ_L = (1/2π)∫ G Γ_L G† (f_L − f_R);
- ρ^R = ρ_eq(μ_L) + Δ_R, with Δ_R = (1/2π)∫ G Γ_R G† (f_R − f_L).

They are combined orbital by orbital with w_L = Δ_R²/(Δ_L² + Δ_R²) (Brandbyge et al., PRB 65, 165401 (2002)).
- The real-axis error of each decomposition grows with its Δ, so the more accurate one gets more weight.
- States bound inside the bias window (coupled to neither lead) get (f_L + f_R)/2.

The combination is essential. With ρ^L alone, the split-off states of two repulsive barriers on a 5×5 transverse mesh
broke the mirror symmetry of a symmetric junction by 8e-3 electrons, independently of the grid and of δ. With the
weighted combination the symmetry holds to 1e-15.

### Equilibrium part

`contour.equilibrium_contour`.
- A circular arc from below the spectrum to (μ − 10 k_BT, h), then the line Im z = h = 2N_pπk_BT to μ + 40 k_BT,
  where f is the real Fermi function, plus the N_p Fermi poles μ + i(2n+1)πk_BT with weight −2πi k_BT.
- The defaults (30 arc points, 40 line points, 10 poles) integrate a bound state to 2e-7 at 300 K. With 20 line points
  the error was 8.5e-5: the line must resolve the Fermi step.

### Non-equilibrium part and current

- **Window.** Real energies in [min μ − 10 k_BT, max μ + 10 k_BT] (trapezoid, `window_spacing`), with the broadening
  δ of `configure_energy_grid`.
- **Green's functions at complex z.** The kernels form (E + iδ)S − H from E·S − H and `delta`, both in the lead
  decimation and in G_C, so z = E + iη is evaluated with delta = η (`density_matrix.conductor_green_function`).
- **Current.** I = ∫ T(E; V) [f_L − f_R] dE, in eV, per transverse cell: units of 2e/h, as `compute_current`.
  Multiply by G₀ = 77.48 µS per volt.

### Electrostatics

`electrostatics.planar_potential`.
- The layers of the conductor (atoms within 1e-3 bohr along the transport axis) carry sheet charges δq_l per
  transverse cell.
- In Hartree units U″ = −4π δn, with Dirichlet values U = μ_L − E_F and μ_R − E_F at the electrode planes, half a
  layer spacing outside the first and the last layer:

  U(z) = U_L + (U_R − U_L)(z − a)/(b − a) + (4π/A) Σ_m δq_m (z_< − a)(b − z_>)/(b − a).

- ε = 1: the PAO valence electrons screen explicitly (Hartree/RPA level).

### Iteration

- Anderson (Pulay) mixing of U per layer (`anderson_update`), with an absolute tolerance on max|U_out − U_in| (eV).
- Each bias starts from the previous solution, rescaled to the new bias.
- **Thomas-Fermi preconditioner (default).**
  - In a metal the map U → U_ramp + P δq(U) has the Jacobian J ≈ −P χ₀, where P is the planar Poisson kernel
    (`poisson_kernel`) and χ₀ the density of states of each layer at E_F.
  - For Al (A = 29 bohr²) P is about 45 eV per electron and layer, so P χ₀ is about 30. Plain or Anderson mixing then
    needs β < 2/(1 + Pχ₀) ≈ 0.05; with β = 0.15 the Al | Si | Al scan diverged at 0.25 V (max|ΔU| from 4 eV to
    194 eV in one step).
  - The residual is preconditioned with (1 + P χ₀)⁻¹, exact for a linear Thomas-Fermi response. χ₀ comes from
    −Im G_ii(E_F + iη)/π with η = max(πk_BT, δ) (`local_density_of_states`, one complex energy, distributed over k∥).
  - On the cubic double barrier it reaches the same solution in 7–8 iterations instead of 10–13 (default mixing 0.5).

## Validation

**Cubic one-band model** (t = −1 eV, a = 5 bohr, 8 sites along z, leads of 2 sites, `tests/unit/transport/test_finite_bias.py`):

| check | result |
|---|---|
| open perfect crystal at V = 0, μ = 0 and −1 eV | occupations = infinite crystal (exact finite-T band filling) to 2e-7 |
| two barriers on sites 3, 4: n_l(+V) vs n_{7−l}(−V) | 1e-15 (both decompositions); 8e-3 with ρ^L alone (5×5 mesh) |
| self-consistent scan, V = ±0.01, ±0.1, ±0.3 V | I(+V) = −I(−V) to 1e-8; U and δq mirror-antisymmetric to 3e-6 eV and 8e-7 |
| convergence | 7–10 preconditioned Anderson iterations to 1e-4 eV (10–13 without the preconditioner) |
| linear response, V = 0.005 V | I/V = ∫T(−f′)dE of the zero-bias T to 2e-3 |
| T(E) in the window vs `compute_transmission` | identical to 1e-6 |
| MPI: 1 and 3 ranks | 4e-14, same iterations |

**Al | Si plane | Al** (PAO Hamiltonian of the transport lecture: 14 layers, 182 orbitals, leads of 4 layers, 6×6
transverse mesh, δ = 1 meV, 300 K; 515 s on 8 ranks; full report in
`~/Workspace/PAOFLOW/validation/negf_finite_bias/REPORT.md`):

- **Zero bias.**
  - The open conductor holds 43.004 electrons (13 Al × 3 + Si × 4 = 43).
  - T(E_F) = 1.3152 of M = 1.5762 lead channels, so t = 0.834 per channel. Landauer–Büttiker puts 1 − t = 0.166 of V
    across the Si plane.

| V | I rigid / ramp / self-consistent (2e/h·eV per cell) | iterations | fraction of V across the Si plane, ramp / self-consistent | Σδq ramp / self-consistent |
|---|---|---|---|---|
| 0.10 | 0.13373 / 0.13305 / 0.13314 | 8 | 0.571 / 0.157 | +0.043 / −0.0001 |
| 0.25 | 0.33995 / 0.33924 / 0.33909 | 12 | 0.571 / 0.159 | +0.070 / −0.0001 |
| 0.50 | 0.68773 / 0.66052 / 0.65930 | 14 | 0.571 / 0.117 | −0.025 / −0.0026 |

- **Voltage distribution.** The self-consistent potential reproduces the Landauer–Büttiker split (0.157–0.159
  against 0.166 across the scatterer; the contacts take 0.84, against t = 0.834).
- **The ramp** leaves up to 0.07 electrons per layer unscreened.
- **Currents.** Self-consistent and ramp agree to 0.2 %; both lie below the rigid shift by 0.4 % (0.1 V) to 4.1 %
  (0.5 V).
- **Convergence.** Without the preconditioner the scan diverged at 0.25 V.

## Bugs found and fixed on the way

- **`leads_are_identical` under shifts.** The left self-energy reused the right lead's transfer matrices, built
  with `shift_R`. With `shift_L != shift_R` (a bias, or `configure_onsite_shifts`) the left lead sat at the right
  lead's energy. `compute_kpoint_self_energies` now shares the matrices only when the shifts are equal; a unit test
  covers it.
- **`compute_current` truncated the Fermi window** at μ ± (σ + 3 dE). On a fine energy grid (dE ≪ σ = k_BT) it kept
  tanh(1/2) ≈ 0.46 to 0.57 of the current: 0.0024 instead of 0.0044 for the double barrier at 0.01 V, a 1 meV grid and
  300 K. The window now extends 10σ + 3dE and is clipped to the grid instead of skipping the bias point. The
  integration reference of example02 (0.09 eV grid) moves well within the 1 % tolerance, and the test passes.
- **Sign of `compute_current`, documented, not changed.** Its `fermi_dirac` is 1/(e^{−(E−μ)/σ}+1) = 1 − f, so it
  returns −∫T[f_L − f_R]: negative for μ_L > μ_R, against its own docstring formula. Changing it would invert the
  reference `current.dat` of the integration assets. `compute_current_self_consistent` returns the formula's sign.
- **Partition.** `HamiltonianBlockPartition` now keeps `central_atoms`, in the orbital order of the conductor block,
  for the layers.

## Limits

- Hartree only: no exchange-correlation response.
- Planar average: no in-plane structure (molecules, wires: a 3D Poisson solver is the next step).
- Orthonormal orbitals and spin-unpolarised (or spinor) inputs: a `NotImplementedError` otherwise.
- The buffer layers of the conductor must screen the field. Check that U is flat there; if it is not, use a longer
  supercell.
- Bound states in the window get (f_L + f_R)/2: steady-state NEGF does not fix their occupation.
- Resonances narrower than `window_spacing` need a finer window.

## Files

- `transport/negf/`: `contour.py`, `density_matrix.py`, `electrostatics.py`, `self_consistency.py`.
- `Transport.configure_bias_self_consistency`, `Transport.compute_current_self_consistent`.
- `transport/io/write_data.write_finite_bias_results`.
- Tests: `tests/unit/transport/test_finite_bias.py` (11).
