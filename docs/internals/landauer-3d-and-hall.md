# Landauer transport in 3D, band curvature and the Hall sign

On 15 November 2026 the formulas of the Boltzmann and Landauer modules were checked against closed-form models (the
one-band simple-cubic metal, an impurity plane) and against the band structures of bulk Al and Si. The checks were
made while writing the transport lecture (`PAOFLOW.Lectures/transport_tutorial`). They found six problems, all fixed
with unit tests. This note records what was wrong, what changed, and how each fix was validated. Landauer results
for bulk crystals, Hall coefficients and effective masses obtained before that date should be recomputed.

## Landauer transport (`Transport`)

### The transverse momentum

**Before.** `transport/parsers/atmproj_tools.parse_atomic_proj` hard-coded the real-space grid as `(1, 1, 4)`
("TODO: confirm hardcoded grid"). The transverse mesh was Γ only and the transverse hoppings were dropped. This is
correct for a wire isolated in vacuum, which is what all the transport examples are (1×1×8 cells). For a bulk crystal
it is wrong: the simple-cubic band, which spans [−6, 6] eV, transmitted one channel only for |E| < 1.9 eV, the band of
an isolated chain.

**Now.**
- The real-space grid is the mesh of `HRs`.
- The transverse k-mesh is the one passed to `configure_energy_grid(nk=(n1, n2))`. Its default is the transverse DFT
  mesh. `nk` must come before `build_hamiltonian_blocks`; a different `nk` given later raises `ValueError`.
- In the expanded-wedge case and for models (`PAOFLOW(model=...)`, which has no `kpnts`/`kpnts_wght`/`Hks`), the
  k-points are the mesh in FFT order, H(k) is `np.fft.fftn(HRs, axes=(2, 3, 4))`, and the reciprocal vectors come from
  `a_vectors`. This also covers a Hamiltonian after `doubling_Hamiltonian`, whose stored `Hks` belong to the
  original cell.
- The path with `expand_wedge=False` is unchanged.

Supporting changes:
- `transport/workspace/prepare_data.py` passes the requested `nk` to `initialize_meshsize`.
- `transport/parsers/read_matrix.py` loops over the transverse lattice vectors (`ivr_par`), not over the k-points.
  The two counts differ when `nk != nr`; the old loop raised `IndexError`. The R lookup and the block copy are
  vectorised.
- `transport/io/write_data.populate_real_space_hamiltonian` builds H(R) with one matrix product of the weighted
  phases and the stacked H(k). It replaces the Python triple loop of `compute_rham`. On the Al 12×12×8 mesh the set-up
  went from 273 s to seconds.

### Complex lead blocks

`transport/hamiltonian/hamiltonian_setup.py` stored the on-site blocks as `(E S − H00)^*`, and only the transfer
matrices undid it (`t11 = t11.T` in `calculators/transfer.py`). The surface and conductor Green's functions therefore
used `E − H00^*`. That is right for the real symmetric blocks of a wire at Γ, and wrong for the complex blocks of any
k∥ ≠ 0.

Every block is now stored as `E S − H`, and the transpose in the transfer matrices is gone. Bulk Al [001] on a 4×4 k∥
mesh:

| E (eV) | −6 | −3 | 0 | 3 |
|---|---|---|---|---|
| T, before | 0.31 | 0.88 | 0.70 | 0.81 |
| T, now | 0.80 | 1.23 | 2.02 | 1.95 |
| band crossings of the lead | 0.81 | 1.25 | 2.06 | 2.06 |

Removing the transpose alone gives a different wrong answer: both changes are needed.

### Internal projections and `doubling_Hamiltonian`

- `transport/partition/orbital_inventory.py`: the number of orbitals per atom came from the pseudo-orbitals of the UPF
  file. That count is wrong for the internal `standard`/`extended` bases (Al: "4 != 13" in `define_partition`). It is
  now read from `atomic_basis`, matching each record's position to `tau`, and also handles the cells repeated by
  `doubling_Hamiltonian`. The UPF count and the equal split remain the fallbacks for the `projwfc.x` route.
- `hamiltonian/do_doubling.py`: `b_vectors` stayed at the original cell (a3·b3 = 2 after doubling along z), which put
  every later k off its grid. It is now recomputed with the other post-doubling attributes.

### Validation

| case | reference | result |
|---|---|---|
| simple-cubic bulk, any transverse mesh | exact channel count | max error 5×10⁻⁴ |
| cubic model doubled along x (complex H00) | 2 × the single cell | 4×10⁻³ |
| impurity plane, conductor mode | analytic: average over k∥ of (4 − ε²)/(4 − ε² + V²) | 1.7×10⁻³ (0.5 meV broadening) |
| Al [001], doubled principal layer, 8×8 k∥ | band crossings of the PAO bands | max 0.259, mean 0.058 |
| Si [001], 8×8 k∥ | band crossings | max 0.342, mean 0.084 |

- The Al T(E_F) is 1.659 channels per transverse cell, against 1.703 crossings and a Boltzmann mode count of 1.590.
- The 1D transport integration examples are unchanged.
- Tests:
  - `tests/unit/transport/test_bulk_transmission_3d.py` covers the cubic bulk with nk 0 and 8, the late-`nk` guard,
    complex blocks (this test fails on the old code), doubling `b_vectors`, and the impurity plane.
  - `tests/unit/transport/test_orbital_inventory.py`.

## Band curvature

`gradient_and_momenta(band_curvature=True)` (`hamiltonian/do_d2Hd2k.py`) degenerates the velocity matrix to get band
curvatures in degenerate subspaces. It grouped bands by equal velocity components alone, so two bands with
`v_a = 0` at the same k counted as degenerate even with different energies. This happens wherever a velocity
component vanishes by symmetry or at an extremum, Γ included. `eigh` then reassigned the sorted curvatures.

In the one-band cubic model with a flat remote band at +12 eV, d²E/dk_x² of the band was 0 wherever k_x = 0, and
the flat band received the +50 eV bohr². The new `energy_and_velocity_degenerate_groups` intersects the velocity
groups with the energy groups. This affected `effective_mass()` and the Hall tensor.

## Hall and Nernst

`boltzmann/do_transport.py` evaluates the weak-field Hall tensor of Madsen and Singh (Comput. Phys. Commun. 175, 67
(2006), Eq. 13):

$$
\sigma_{abc} = e^3 \sum \tau^2\, \varepsilon_{cuv}\, v_a v_v M^{-1}_{bu} \left(-\frac{\partial f}{\partial E}\right), \qquad
R_{abc} = -[\sigma^{-1}]_{\alpha b}\, \sigma_{\alpha\beta c}\, [\sigma^{-1}]_{a\beta} .
$$

The code dropped the factor e³ < 0 of `L0_hall`, so R_H was positive for electrons. The one-band model at 0.01
electrons per cell gave +1.04×10⁻⁸ m³/C against −1/(ne) = −1.16×10⁻⁸.

It now computes `R_hall = inv(L0_unconverted) @ L0_hall @ inv(L0_unconverted)`, without the leading minus. The minus
of the formula and e³ < 0 cancel, so R_H is negative for electrons. Against 1/(ne) on the cubic model:
- electrons: R_H/(1/ne) = 0.975;
- holes: +1.114×10⁻⁸ against +1.156×10⁻⁸.

The Nernst tensor `N = R_H ν + σ⁻¹ ν_Hall`, with ν = σS = −L1, carried the same sign twice, so its values are
unchanged; the code now writes it in that form. For one parabolic band with constant τ, N vanishes (Sondheimer). The
residual is 3.2 / 1.4 / 0.69 % at 0.01 / 0.003 / 0.001 electrons per cell, scaling with the non-parabolicity.

Tests: `tests/unit/boltzmann/test_hall_and_curvature.py` covers:
- the degenerate-groups helper;
- the curvature not swapped with the flat band;
- R_H in the parabolic limit for electrons and holes on a 40³ grid, within 15 %.

## Two numerical traps (no code change)

- **Kernel.** With adaptive smearing (`smearing='gauss'`), `transport()` tabulates the transport distribution on its
  own μ grid `(emin, emax, ne)` and convolves it with the Fermi window on that grid. A spacing well above k_BT spoils
  the moments. On Al at 300 K, a 100 meV spacing gives L/L₀ = 1.116; 20, 10 or 5 meV give 1.000. Keep the spacing
  below k_BT, or use the thermal kernel (`smearing=None`).
- **Energy zero of the τ models.** The built-in relaxation-time models (`do_tau_models.py`) and `TauModel` receive
  |E| from the energy zero of the run. That is a band edge only for the valence band of a semiconductor. A
  `ScatteringChannel` receives signed energies and can measure them from any edge.
