# The self-consistent spin response

On 16 November 2026 PAOFLOW gained the exchange-correlation response of the electron spin to a magnetic field. Before,
a Zeeman term on the PAO Hamiltonian (`magnetic_field`) gave only the bare Pauli and Van Vleck response. The Stoner
enhancement of a nearly ferromagnetic metal was missing; in Pd it is a factor of six.

There are two routes, and both are validated against fixed-spin-moment calculations in Quantum ESPRESSO:

- an on-site LSDA mean field on the PAO shells;
- the LSDA spin kernel on the real-space grid of the crystal.

The tests are:
- `tests/unit/hamiltonian/test_stoner.py`
- `tests/unit/response/test_spin_response.py`
- `tests/unit/gen/test_spin_response_hook.py`

The production runs are in `~/Workspace/PAOFLOW/validation/spin_response_2026-11-16`.

## What PAOFLOW computes

| method | what it is |
|---|---|
| `stoner_parameters(density)` | Stoner integrals I_gg′ of the PAO shells (eV per μ_B) |
| `self_consistent_spin(field, nk, ...)` | the on-site LSDA mean field, self-consistent on a k-grid; with `seed`, the ordered moment of a Stoner instability |
| `spin_susceptibility(field, nk, methods)` | χ = dM/dh per cell by `'bare'`, `'stoner'` and `'alda'`, written to `spin_susceptibility.dat` |
| `response.spin_response.static_spin_susceptibility` | the grid route, `'alda'` above |

The property also appears in `paoflow-gen` and in the Studio, under the name "Spin susceptibility". It writes a
`run_spin_response()` block into `main.py`. The example is `examples/qe_examples/example18` (Pd).

### Conventions

- **Susceptibility.**
  - M is the spin moment per cell, n↑ − n↓ along the field, in μ_B.
  - h = g_s μ_B B/2 is half the Zeeman splitting.
  - χ = dM/dh is in μ_B/eV per cell. The bare value is the Pauli term χ₀ = 2N(E_F), with N per spin.
  - This is the χ of a fixed-spin-moment calculation, M/h with h = (μ↑ − μ↓)/2.
- **Spin kernel.** For a small magnetization m, the LSDA potentials are v_σ = v̄ + σ f_s(n) m, with
  f_s = [(4/9)ε_x + α_c]/n: Slater exchange and the PW92 spin stiffness (`basis_gen.xc.lsda_spin_kernel`). The kernel is
  negative.
- **Stoner integrals.**
  - I_gg′ = −2∫r²dr R_g² R_g′² f_s(n)/4π. The splitting of shell g is Δ_g = Σ_g′ I_gg′ m_g′.
  - The Stoner criterion is I·N(E_F) > 1, and in the linear regime χ = χ₀/(1 − Iχ₀/2).
  - n is a valence density plus the pseudo-core density. With `density='crystal'`, the default when the save holds
    `charge-density.dat`, the valence part is QE's density averaged spherically about the atom. With
    `density='atomic'`, it is the atomic density of the pseudopotential.
  - The free-atom tail vanishes between the atoms, where f_s ∝ n^(−2/3) diverges. It gives a diffuse shell (Pd 5D)
    I = 1.66 eV; the crystal density gives 0.02 eV. The d shells change by 2 %.
- **Effective I.** One number compares the methods, I_eff = 2(1/χ₀ − 1/χ). It is negative-χ safe: an unstable
  paramagnet has χ < 0 and I_eff > 2/χ₀.

### The on-site mean field

H(k) + μ_B(**L** + g_s**S**)·**B** − Σ_g **h**_g·**S**_g with **h**_g = Σ_g′ I_gg′(**m**_g′ − **m**⁰_g′).

- **m**_g = 2⟨**S**_g⟩ of the occupied states. They are Löwdin populations in the spinor basis.
- **m**⁰ are the moments without field, so a magnetic DFT Hamiltonian keeps its own exchange splitting.
- Spin-degenerate runs are rewritten in spinors (`spinor_basis`). Collinear runs (nspin = 2) take the field along z.
- Ad hoc spin-orbit runs (`adhoc_spin_orbit`) work as they are. On Pt at 16³, λ = 0 reproduces the spinor result to
  10⁻⁵, and λ_d = 0.6 eV gives χ₀ = 1.86 and χ = 3.09. Fully relativistic QE runs are not tested yet.
- H(k) uses the pair-resolved Wigner–Seitz sums of the optics and the grid response, so the bare χ₀ of the two routes
  agree on any grid.
- **The loop.**
  - It starts with 30 linear-mixing iterations, then switches to Anderson mixing.
  - At convergence a perturbation of 10⁻³ μ_B is iterated. If it grows, the solution is a Stoner instability (the
    paramagnet of bcc Fe), and the loop restarts from it with linear mixing, which is repelled by unstable fixed points.
  - `linear=True` (used by `spin_susceptibility`) skips both steps and stays on the solution connected to **m**⁰. That
    is the linear response, negative across an instability.

### The grid response

The second route solves δm = χ₀[h + f_s(n₀)δm] on the real-space grid of the crystal.

- χ₀ is the static q = 0 response of the PAO bands. Interband pairs take (f_m − f_n)/(ε_m − ε_n). The diagonal and
  degenerate pairs take the smeared f′, as in QE.
- n₀ is QE's valence density (or the PAO one) plus the core.
- GMRES solves for δm(**r**) on the periodic parts of the PAO Bloch functions (`coupled_crystal.build_crystal_channel`,
  `grid_factor = 2`).
- The ground state is spin symmetric, so μ does not move at first order.

## Validation

### Models and kernels

| check | result |
|---|---|
| f_s against the second derivative of the full PW92 LSDA energy in m | 10⁻⁵, n from 10⁻³ to 1 bohr⁻³ |
| smeared occupations against their derivatives (MV, Gaussian, FD) | 10⁻⁷ |
| one-band cubic model, χ against 2N/(1 − IN) for IN = 0, 0.5, 0.9 (spinor and collinear) | 2×10⁻³ (finite field) |
| IN = 1.2, linear mode | the negative 2N/(1 − IN) |
| IN = 1.3, from m = 0 without field | the stable ferromagnet, self-consistent to 10⁻⁸ |
| Gaussian shell in a uniform density | I in closed form, zero between atoms |
| grid solver on a synthetic metal | the dense (1 − χ₀f)⁻¹χ₀ to 10⁻⁸; the Pauli term without kernel |
| core density on the grid | holds 4π∫r²ρ_core of every atom to 10⁻⁸ |
| crystal density about an atom | ρ₀ + 2ρ_G j₀(Gr) for one Fourier shell |

### Pd, Pt and Fe against QE fixed spin moments

**Settings.**
- QE 7.4.1, LDA (PZ), norm-conserving PBE pseudopotentials (`nc-sr-05`), ecutwfc 90 Ry, Marzari–Vanderbilt
  smearing of 0.02 Ry, experimental lattice constants.
- Scf on a 24³ grid; fixed-spin-moment ladders at 24³; nscf at 12³ for PAOFLOW.
- PAOFLOW: the standard internal basis (BASIS_PS), projectability 0.95, the same smearing.

The QE χ comes from E(M) − E(0) = M²/2χ at small M; the splitting of the two Fermi levels agrees to 1 %. Below,
χ₀ is PAOFLOW's at 24³; on the 12³ nscf grid it equals 2N(E_F) of QE's eigenvalues to five digits (Pd 2.3725).

| | χ₀ | on-site Stoner χ (I_eff) | grid ALDA I_eff (12³, 16³) | QE χ (I_eff) |
|---|---|---|---|---|
| Pd | 2.262 | 5.00 (0.48) | 0.770, 0.765 | 14.3 (0.74) |
| Pt | 1.864 | 3.13 (0.43) | 0.682, 0.668 | 4.59 (0.64) |
| Fe (paramagnet) | 3.970 | −5.59 (0.86) | 1.26–1.31 (8³–16³) | −3.05 (1.16) |

χ is in μ_B/eV, I_eff in eV.

- **Grid ALDA.** It reproduces the exchange of the full LSDA calculation. I_eff is 3 % (Pd), 5 % (Pt) and 9 % (Fe)
  above QE's.
- **χ itself.** Pd sits at Iχ₀/2 ≈ 0.86, so χ amplifies every difference in χ₀. The grid χ goes from 27 at 12³ to
  15.8 at 16³ while I_eff stays put, so converge the k-grid before quoting χ.
- **Fe's moment.** The seeded on-site mean field gives 2.11 μ_B (12³–24³: 2.11–2.14). QE's E(M) has its minimum at
  2.15 μ_B.
- **Pt.** Scalar relativistic here. With spin-orbit coupling only the on-site route applies.

### Why the on-site model is low

The on-site I_eff is about 2/3 of the full one for all three metals. Janak's formula, I = −2∫f_s γ²/(∫γ)², with γ the
Fermi-level spin density on the grid, locates the missing exchange (Pd and Fe, 8³):

| γ | Pd | Fe |
|---|---|---|
| shell model (atomic radial functions, spherical shells) | 0.50 | 0.88 |
| on-site products of the Löwdin PAOs only | 0.59 | 0.97 |
| full γ, with the bond (off-diagonal) products | 0.69 | 1.10 |
| self-consistent shape of δm (grid ALDA) | 0.77 | 1.27 |

- **Not the missing pieces.** Resolving the d shell orbital by orbital (the angular factors ∫Y²_m Y²_m′) adds 2 % (Pd)
  and 13 % (Fe). The crystal density instead of the atomic one moves the d integrals by 2 %.
- **What is missing.** The bond part of the magnetization density, between the atoms where f_s is large, and the
  self-consistent change of its shape.
- **Use of the on-site model.** Trends, spin-orbit runs and ordered moments. The grid route gives the numbers.

## Limits

- **Uniform field only.** The response is static and at q = 0. The finite-q and dynamical response (paramagnons, spin
  waves) needs a k, k + q channel and is not implemented.
- **Grid route.** It needs a spin-degenerate run (nspin = 1, no spin-orbit), the internal projections and
  norm-conserving pseudopotentials: the augmentation charges of US/PAW are not in the grid density. Memory is about
  10 MB per k-point for a 4d metal, and the periodic parts are cached up to 8 GB per rank.
- **Kernel.** It is the LSDA spin kernel whatever the functional of the DFT run. For a GGA run the comparison with a
  fixed-spin-moment calculation needs the same LDA (the QE references above use `input_dft = 'pz'`).
- **Orbital moments.** The on-site route includes the orbital Zeeman term (`orbital=True`) but no orbital
  polarization.

## Code map

| file | content |
|---|---|
| `hamiltonian/stoner.py` | shells, densities, Stoner matrix, occupations, `bloch_hamiltonians`, `solve_spin_response` |
| `response/spin_response.py` | core density on the grid, QE density on the grid, `solve_static_spin_response`, `static_spin_susceptibility` |
| `basis_gen/xc.py` | `lsda_spin_kernel` |
| `PAOFLOW.py` | `stoner_parameters`, `self_consistent_spin`, `spin_susceptibility` |
| `gen/questions/spin_response.py` | the paoflow-gen / Studio property |
