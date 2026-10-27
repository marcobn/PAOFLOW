# ARPES simulation (`PAOFLOW.arpes`)

`PAOFLOW.arpes` simulates angle-resolved photoemission from the PAO Hamiltonian: dipole
matrix elements of the PAO Bloch states with a plane-wave final state, the spectral function
with an optional band- and k-resolved self-energy (electron-phonon, from
`PAOFLOW.elphon.self_energy`), the Fermi factor and the experimental resolution. The
formalism follows **chinook** (R. P. Day, B. Zwartsenberg, I. S. Elfimov and A. Damascelli,
npj Quantum Materials 4, 54 (2019), MIT licence; see `THIRD_PARTY_NOTICES.md`), re-implemented
in the real-harmonic PAO basis and vectorised over k-points and bands. Please cite chinook when
publishing results obtained with this module. The design note is
[arpes-chinook-plan.md](arpes-chinook-plan.md).

## Workflow

```python
from PAOFLOW import PAOFLOW
from PAOFLOW.arpes import ArpesExperiment, ConstantSelfEnergy

pf = PAOFLOW.PAOFLOW(savedir='si.save', outputdir='output')
pf.projections(basispath='BASIS_PS', configuration='standard')
pf.projectability(pthr=0.95)
pf.pao_hamiltonian()

experiment = ArpesExperiment(
    photon_ev=21.2, work_function_ev=4.6, inner_potential_ev=12.0, temperature_k=30.0,
    polarization=(1.0, 0.0, 0.0), energy=(-3.0, 0.2, 400),
    kx=(-1.5, 1.5, 151), ky=(-1.5, 1.5, 151), resolution=(0.02, 0.01),
)
arpes_map = pf.arpes(experiment, self_energy=ConstantSelfEnergy(0.03))
arpes_map.save('output/arpes.npz')
pf.plotter().plot_arpes(arpes_map, 'arpes_band_map.png', cut='band_map', value=0.0)
```

`ArpesExperiment` fields (eV, 1/Å, radians): `photon_ev`, `work_function_ev`,
`inner_potential_ev` (free-electron $k_z$ from the photon energy; `None` keeps `kz`),
`temperature_k`, `polarization` (Cartesian in the sample frame, complex for circular light),
`energy=(min, max, n)`, `kx`/`ky=(min, max, n)` or an explicit `kpoints (N, 3)` (a path),
`azimuth`, `resolution=(dE, dk)` FWHM, `escape_depth` (slab depths), `radial` (`'pao'`,
`'slater'`, `'hydrogenic'` or a `RadialIntegrals`), `phase_shifts`, `fixed_radial`,
`position_phase`, `kz_at_binding_energy`, `window_margin_ev`, `kblock`.

The result, `IntensityMap`, holds `kpoints`, `w`, `intensity` (broadened), `raw`, the window
`bands` with their `energies` and matrix-element `weights`, the mesh `grid`/`axes`, and the
settings in `metadata`; `as_grid()`, `energy_distribution_curve(ik)`,
`momentum_distribution_curve(E)`, `save(path)`.

## Formalism

For $|n\mathbf k\rangle = \sum_i c_{i,n\mathbf k}|\phi_i,\mathbf k\rangle$ (cell-periodic
gauge, `BlochHamiltonian`: $H(\mathbf k) = \sum_n W_n H(R_n)e^{-2\pi i\mathbf k\cdot n}$ on the
Wigner–Seitz images of the FFT cells) and a photoelectron $\mathbf k_f$ with
$|\mathbf k_f|^2 = 2m(h\nu + \varepsilon_{n\mathbf k} - W)/\hbar^2$, $\mathbf k_{f\parallel} =
\mathbf k_\parallel$,

$$
M^\alpha_{n\mathbf k} = \sum_i c_{i,n\mathbf k}\,\sigma_i\,e^{-d_i/2\lambda}\,
e^{-i(\mathbf k_f-\mathbf k)\cdot\boldsymbol\tau_i}\,s_i\sum_{l'=l_i\pm1} B^{l'}_i(|\mathbf k_f|)
\sum_{m'} G^\alpha_{l_i m_i,l'm'}\,Y_{l'm'}(\hat{\mathbf k}_f),
\qquad
B^{l'}_i(k) = i^{l'}\!\int_0^\infty\! dr\,\chi_i(r)\,r^2 j_{l'}(kr),
$$

with $\chi_i = rR_i$ the PAO radial table of the orbital (`arrays['basis'][i]['wfc']`), $G$
the real Gaunt coefficients of the dipole operator ($x, y, z$ are real $l = 1$ harmonics;
`hamiltonian/_two_center.real_gaunt_coefficient`), $Y_{l'm'}$ the real harmonics, $s_i$ the
sign of QE's harmonic relative to the standard one (`harmonics.qe_harmonic`), $\sigma_i$ a
cross-section factor and $d_i/\lambda$ the depth over the escape depth. The gauge phase is
chinook's atomic-gauge convention made explicit; it is unity for $\mathbf k_f = \mathbf k$ and
off by default (`position_phase=False`). The intensity is

$$
I(\mathbf k,\omega) = \sum_n\Big|\sum_\alpha\epsilon_\alpha M^\alpha_{n\mathbf k}\Big|^2
A_{n\mathbf k}(\omega)\,f(\omega,T)\otimes g_{\Delta E}\otimes g_{\Delta k},\qquad
A_{n\mathbf k}(\omega) = -\frac1\pi\mathrm{Im}\frac{1}{\omega-\varepsilon_{n\mathbf k}-\Sigma_{n\mathbf k}(\omega)+i0^+}.
$$

Module map: `basis.py` (`Orbital`, `PhotoemissionBasis.from_data_controller`,
`with_atomic_radial`, `with_depths`), `atomic_radial.py` (Slater and hydrogenic fallbacks,
ported from chinook with the physical Bohr radius and a corrected normalisation),
`radial.py` (`radial_integral`, `RadialIntegrals` tables with phase shifts and fixed values),
`harmonics.py` (QE order, dipole Gaunt tables), `kinematics.py` ($|k_f|$, $k_z$, meshes,
emission angles), `bloch.py`, `matrix_elements.py`, `spectral.py` (`spectral_function`,
`assemble_intensity`, `ConstantSelfEnergy`, `GridSelfEnergy`, `kramers_kronig_self_energy`,
`gaussian_broadening`), `experiment.py` (`ArpesExperiment`, `simulate`, `IntensityMap`).

## Self-energies

The self-energy protocol is a constant, an array `(N, nb, nw)` aligned with the window bands,
or a callable `f(w, energies, kpoints, bands)` evaluated per k-block. Sources:

- `ConstantSelfEnergy(gamma)`; `GridSelfEnergy(w, sigma)` for a tabulated $\Sigma(\omega)$;
  `kramers_kronig_self_energy(w, im_sigma)` builds the real part of a model;
- `PAOFLOW.elphon.self_energy.eliashberg_self_energy(omega, a2f, w, T)`: the isotropic
  Engelsberg–Schrieffer form from $\alpha^2F$ of `eliashberg_dense_q` (or the per-state
  $\alpha^2F_{n\mathbf k}$ of the anisotropic coupling), with the kink and the mass enhancement
  $\lambda = -\mathrm{Re}\,\Sigma'(0)$ (`mass_enhancement`);
- `PAOFLOW.elphon.self_energy.fan_migdal_self_energy(...)`: the band-resolved Fan–Migdal
  $\Sigma_{n\mathbf k}(\omega, T)$ at arbitrary k-points (the ARPES path or mesh) from the
  dense-q vertex of `prepare_dense_vertex`, intermediate states diagonalised at every
  $\mathbf k+\mathbf q$ (`elph_bloch.bloch_states_at`), Lorentzian width `eta_ev`, full dense
  q-grid, MPI over q; `TabulatedSelfEnergy.from_result(result, kpoints)` adapts it to the
  protocol (k-points matched by position, bands by index).

## Examples

`examples/elphon_example/Pb/arpes.py` (metal: the kink from the isotropic α²F, the
state-resolved form with adaptive widths), `examples/elphon_example/MgB2/arpes.py` (the
σ-band E₂g kink from the anisotropic Fermi-surface coupling, one self-energy per irreducible
state), `examples/elphon_example/Si/arpes.py` (semiconductor: band-resolved Fan–Migdal
linewidths versus temperature and the dressed band maps); each README has the numbers.

## Slabs

`build_slab(bloch, basis, nlayers, axis)` stacks `nlayers` cells of the bulk Hamiltonian along
a lattice vector with open boundaries (hoppings from the Wigner–Seitz images of `HRs`, zero
beyond their range) and returns the two-dimensional `BlochHamiltonian` of the slab and the
basis of every layer with the `depth` below the top surface, for the escape-depth attenuation
of `ArpesExperiment(escape_depth=...)`. The stacking vector should be the surface normal
(build a supercell first for a general Miller index). The self-energy of slab states is not
computed by the dense-q driver; use the isotropic form or the bulk one at the projected
momentum.

Metals: the state-resolved Fan–Migdal self-energy on an affordable q-grid is noisy with a
fixed width far below the level spacing $|v_F|\Delta q$ of the grid (Pb: 5 meV against
0.3 eV at 18³), and the adaptive widths (`adaptive_smearing=`, group velocities by finite
differences at $\mathbf k+\mathbf q$) that smooth it are then far larger than the phonon
energies; the kink of a metal is therefore resolved by the isotropic or anisotropic Eliashberg
form, the band-resolved form being the tool for semiconductors and for the
energy-dependent linewidths over wide windows (the Si example).

Spinor (fully relativistic) PAO bases are supported: each |l j m_j⟩ PAO function is expanded
on the real harmonics and the two spin components (`harmonics.spinor_components`, from
`clebsch_jm_matrix` and `calc_ylmg_complex_0`), the radial table of its j is used, and the
matrix elements get a spin index (`Orbital.components`); a closed shell gives the same emission
in both representations (test).

Not yet covered: the Debye–Waller term; the self-energy of slab states.

## Validation

`tests/unit/arpes` (26 tests): the QE harmonic order against `calc_ylmg`, orthonormality of
the real harmonics, the dipole Gaunt sum rule $\sum_{\alpha l'm'}G^2 = 1$ and the parity
selection rule, the radial integral of hydrogen 1s against quadrature, table interpolation,
phase shifts and fixed values, Slater's rules (C 2p: 3.25, Na 3s: 2.2) and normalisation, the
Bloch Hamiltonian against the FFT grid, isotropic emission of an $s$ orbital and of a closed
$p$ shell, the gauge phase and the escape-depth attenuation, spinor components, kinematics
(free-electron $k_z$, azimuth, evanescent states), the spectral function normalisation, the
intensity assembly against a literal state loop, Kramers–Kronig of a Lorentzian, the
resolution broadening, mesh and path simulations with the inner potential, and the
polarisation selection rule at normal emission, the slab of a cubic $s$ band against the open
chain, the spinor p shell against the scalar one, and **parity with chinook** on its
square-lattice Sb 5p tutorial model (`data/chinook_square_lattice.npz`, produced by
`data/chinook_square_lattice_reference.py` with chinook 1.1.3 in a throwaway environment):
eigenvalues to 1e-8 eV and the matrix-element weights of the linear polarisations to 2e-4
(chinook rounds its hopping vectors to 1e-4 Å), up to one global constant; for circular light
chinook's spherical components label the helicities the other way round, so its (x + iy)
weights equal PAOFLOW's (x − iy) ones. `tests/unit/elphon/test_self_energy.py`:
the Einstein-model mass enhancement and $\mathrm{Im}\,\Sigma$ plateau at $T = 0$ and finite $T$,
the Fan–Migdal kernel against a literal sum and its on-shell identity with the SERTA sum with
a Lorentzian delta, and the tabulated adapter.
