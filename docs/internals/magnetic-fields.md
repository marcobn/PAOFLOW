# Magnetic fields, orbital moments and the spin-orbit kernels

On 16 November 2026 the magnetic-field support of PAOFLOW was investigated and completed. Before, the only coupling
was a spin Zeeman term in `add_external_fields(Bfield=...)`, which had never been tested. The investigation also found
defects in the ad hoc spin-orbit kernels and in `doubling_Hamiltonian`. This note records:
- the coupling PAOFLOW now provides, and its conventions;
- what was wrong;
- how each piece was validated.

Every validation is a unit test:
- `tests/unit/hamiltonian/test_magnetic_field.py`
- `tests/unit/hamiltonian/test_peierls.py`
- `tests/unit/hamiltonian/test_spin_orbit_builders.py`
- `tests/unit/response/test_orbital_moment.py`

## The coupling of the PAO Hamiltonian to a uniform field

| term | method | what it is |
|---|---|---|
| spin and atomic orbital Zeeman | `magnetic_field(field)` | μ_B (**L** + g_s **S**)·**B** on the on-site blocks, **B** in tesla |
| exchange-like Zeeman | `add_external_fields(Bfield=...)` | −**B**·**S** on site, **B** an energy in eV |
| orbital motion (vector potential) | `peierls_field(field or flux_quanta)` | Peierls phases on every hopping, Landau gauge |
| orbital moment of Bloch states | `orbital_moment()` | **m**_n(**k**), μ_B: the itinerant part of the band g-factor |
| orbital magnetization | `orbital_magnetization(emin, emax, ne)` | **M**(μ) per cell, modern theory |
| spinor basis | `spinor_basis()` | doubles a spin-degenerate or collinear run to block-ordered spinors |

All the Hamiltonian terms act on `HRs`. Call them before `interpolated_hamiltonian`; each call adds its own field.

### Conventions

- **Sign.** `magnetic_field` and `peierls_field` use the electron's physical sign (charge −e).
  - The moment −μ_B(**L** + g_s**S**) lines up with **B**.
  - A state circulating counter-clockwise about **B** rises in energy.
  - The orbital moment of a state is m_n = −∂E_n/∂B.
- **`add_external_fields(Bfield)`** keeps its historical convention: −**B**·**S** lowers the spin parallel to **B**. It is
  an exchange field. A physical field **B**_T corresponds to `Bfield` = −g_s μ_B **B**_T, which is 0.116 meV/T for
  g_s = 2.
- **Units.** μ_B = 5.7883818060×10⁻⁵ eV/T. Φ₀ = h/e = 1.47685×10⁶ T·bohr². Orbital moments use
  (e/ħ)·eV·bohr² = μ_B/13.6057.
- **Operators.**
  - `Sj` is σ/2.
  - `Lj` comes from `orbital_operator`:
    - fully relativistic runs: J − S in the |j m_j⟩ basis;
    - ad hoc spin-orbit runs: real harmonics in block order;
    - scalar runs: real harmonics.
  - Models carry no orbital labels. Pass `orbital_angular_momentum=(Lx, Ly, Lz)` or use `orbital=False`.
- **Spin.**
  - A spin-degenerate run is doubled to spinors automatically by `magnetic_field`.
  - A collinear run (`nspin = 2`) takes only B_z, as ∓g_s μ_B B_z/2 on its two channels. Call `spinor_basis()` first
    for a transverse field.
- **Peierls gauge.**
  - (e/ħ)**A**·d**r** = 2πf s_p ds_q in crystal coordinates. f is the flux through the cell **a**_p × **a**_q in
    units of Φ₀, and the field points along the third lattice vector.
  - The hopping ⟨i,0|H|j,**R**⟩ takes the phase 2πf[(s_p,i + s_p,j + R_p)(s_q,j + R_q − s_q,i)/2 − R_p s_q,j]. The
    last term is the magnetic translation.
  - A cell periodic along **a**_p must hold an integer number of flux quanta: a magnetic supercell built with
    `doubling_Hamiltonian`. A ribbon or flake (open along **a**_p, `cutting_Hamiltonian`) takes any field.
  - Realistic laboratory fields in bulk need supercells of order 10⁴ cells, which is work for the sparse backend.
- **Orbital moment.** m^z_n = (e/ħ) Σ_{m∉D(n)} Im(V^x_nm V^y_mn)/(ε_m − ε_n), with the velocity of
  `gradient_and_momenta`, D_nm term included. States degenerate with n are left out: inside a degenerate group the
  per-band values depend on the basis, but their sum does not.
- **Magnetization.** **M**(μ) = (g/N_k) Σ f_n [**m**_n + (e/ħ)(μ − ε_n)**Ω**_n], with g = 2 for spin-degenerate
  bands. In the gap of a Chern insulator, dM/dμ = (e/h)C × area (Středa).

### What is not included

- **No self-consistent response of the density in `magnetic_field`.** The Zeeman term on a DFT Hamiltonian gives the
  bare Pauli and Van Vleck response. The exchange-correlation (Stoner) enhancement, a factor 2.5–7 in Pt and Pd, comes
  from `self_consistent_spin` and `spin_susceptibility`: see [the spin response](spin-response.md).
- **Interpolation artefact.** The zero-padding interpolation of an even `HRs` grid splits Kramers pairs off the grid
  points by a few meV (5 meV in Pt). That is more than any laboratory Zeeman energy, so use grid points or the
  Wigner–Seitz sums for g-factors.
- **Aliased boundary block.** On an even grid, the `HRs` block at N/2 aliases ±N/2. It takes the Peierls phase of
  R = −N/2. Model hopping lists leave that block empty.

## Validation

| check | result |
|---|---|
| Landé g_J of p and d spin-orbit multiplets, along z and an oblique axis | exact: 5×10⁻¹² eV at 0.1 T (the test asserts 10⁻¹⁰) |
| `Sj` of fully relativistic runs (Pt) | spin algebra to 10⁻¹⁶; H(**R**) time-reversal invariant to 2×10⁻¹¹ eV with Θ = e^{−iπJ_y}(−1)^l K |
| bcc Fe (fully relativistic), 12³ grid | spin moment 2.194 μ_B, orbital 0.0564 μ_B, parallel |
| square lattice, 1/3 flux per plaquette along +z | H(**k**) equals the Harper matrix to 7×10⁻¹⁶; Hofstadter Chern numbers (1, −2, 1) with C = (1/2π)∫Ω, Ω = ∇×𝒜, 𝒜 = i⟨u\|∇u⟩ (the Kubo curvature of `berry_phase`); the lattice plaquette sum of Fukui–Hatsugai–Suzuki, (1/2π)Σ arg ∏⟨u\|u′⟩, is −C: (−1, 2, −1) |
| six-site ring | levels 2t cos(2π(ℓ + Φ/Φ₀)/6) to 10⁻¹²; ℓ = +1 rises in B along +z |
| graphene ribbon, 32 cells, 1/150 flux quanta per cell | n = 1 Landau level 0.6 % below √2 ħv_F/l_B (64 cells, 1/600: 0.1 %; the 0.4 % quoted before came from the 1 % histogram bins of the test) |
| gapped graphene | m(K) = (ħv_F)²/Δ (2.9798 μ_B), equal in both bands, reversed at K′ |
| ring, Bloch formula against Peierls | m_n = −∂E_n/∂B for every state, to all digits |
| Haldane model, Středa | dM/dμ in the gap equals g·ΔN·Ω/(Φ₀μ_B), with ΔN = −1 counted in an 8-cell Peierls supercell |
| MoS2 three-band model | exciton valley g = −3.19: atomic part −4.00, Bloch part +0.81 (measured: about −4) |

## Defects fixed

1. **`add_external_fields` re-applied stored fields.** Every call re-applied the fields stored by earlier calls: a
   Zeeman term followed by a Hubbard-only call doubled the splitting. Each call now applies only its own fields; the
   arrays `Efield`, `Bfield` and `HubbardU` hold the totals. It also builds `Sj` when missing, rejects
   spin-degenerate runs with a clear message, and treats `nspin = 2` as ∓B_z/2.
2. **Non-Hermitian ad hoc spin-orbit kernels.** `soc_d_spd` and `soc_d_ssppd` set only the upper triangle of part of
   the d block, at every quantization axis; `soc_d_sspd` did the same off the z axis.
   - The upper triangle was right, so PAOFLOW's upper-triangle eigensolvers (bands, eigenvectors) were never
     affected. The example07 references are unchanged.
   - Every routine that uses the whole matrix saw a non-Hermitian H: numpy's default eigensolvers, Green's functions,
     products with H. That covers nonlinear optics, coupled response, `find_weyl_points`, `mirror_chern_number`, NEGF
     transport, ARPES and electron-phonon.
   - `soc_p_ps` was Hermitian but not L·S off the z axis.
   - All four now place the generic L·S block, `_shell_soc_block`. It is equal to the independent `soc_d_spds` and to
     every p kernel.
3. **`doubling_Hamiltonian` and model positions.**
   - The replica of a model's `tau` (alat units) was shifted in bohr.
   - `Dnm` was extended block-diagonally, zero between the replicas. The D_nm velocity term then had a band-diagonal
     part, up to 2.3 eV·bohr against a scale of 4.65 in doubled gapped graphene. That spoiled band velocities, and
     everything built on them, in doubled cells.
   - `Dnm` is now [[D, D − Δ], [D + Δ, D]], with Δ the replica's offset. The replica's `tau` is shifted in its own
     units, and the orbital centres are stored in `orbital_centres`.
   - The sparse kernel `sparse/doubling.double_axis` had copied the old convention and was corrected the same way.
4. **Orbital centres.** `hamiltonian/orbital_centres.orbital_centres_bohr` serves the electric field, the Zak phase
   and the Peierls phases. Its sources, in order: the stored `orbital_centres`, the projection basis, `tau` per
   orbital, then `tau` per atom with the orbital counts. The model loaders mark their `tau` with
   `tau_units = 'alat'`.

## Code map

| file | content |
|---|---|
| `hamiltonian/magnetic_field.py` | `spinor_basis`, `onsite_magnetic_field`, the constants |
| `hamiltonian/peierls.py` | `flux_quanta_per_cell`, `peierls_phases` |
| `hamiltonian/orbital_centres.py` | `orbital_centres_bohr` |
| `hamiltonian/add_ext_field.py` | electric field, exchange-like Zeeman, Hubbard corrections |
| `response/orbital_moment.py` | `moment_and_curvature`, `magnetization_density` |
| `hamiltonian/do_spin_orbit.py` | the ad hoc L·S kernels (`_placed_shell_block`) |
| `hamiltonian/do_doubling.py`, `sparse/doubling.py` | supercells (`Dnm`, `tau`, `orbital_centres`) |
