# Wigner-Seitz interpolation of the PAO Hamiltonian

`HRs` holds the PAO Hamiltonian on the `N1 x N2 x N3` grid of lattice vectors that is Fourier-conjugate to the DFT
mesh, and it is periodic on the supercell `N1 a1 x N2 a2 x N3 a3`. On the mesh every Fourier sum of `HRs` gives the
same `H(k)`. Off the mesh each grid index must first be given a lattice vector, and the choice matters whenever
`HRs` has not decayed at the edge of the supercell. That is typical of the diffuse box states of the
`standard`/`extended` presets on 8³–12³ meshes.

Since 15 November 2026 the two public interpolation steps take an `interpolation` argument:

| method | default (unchanged) | new options |
|---|---|---|
| `bands(..., interpolation=...)` | `'fft_grid'`: the FFT grid as stored, `n_i in [-N_i/2, N_i/2)` (`band_loop_H`) | `'wigner_seitz'`, `'wigner_seitz_pairs'` |
| `interpolated_hamiltonian(..., interpolation=...)` | `'zero_padding'`: the parallelepiped with the Nyquist planes split (`utils/zero_pad.py`) | `'wigner_seitz'`, `'wigner_seitz_pairs'` |

`'wigner_seitz_pairs'` (added later on 15 November 2026) chooses the images orbital pair by orbital pair, by the
length of the bond; see [Pair-resolved images](#pair-resolved-images).

With `'wigner_seitz'` every grid index `m` goes to its shortest image `n = m + N s` (Cartesian length) in the
supercell lattice. Images of equal length share the weight `1/deg`, as in Wannier interpolation and QE `wsweight`:

$$
H(\mathbf k) = \sum_{\mathbf n\in \mathrm{WS}} w_{\mathbf n}\, e^{-2\pi i\,\mathbf k\cdot\mathbf n}\,
\mathtt{HRs}[\mathbf n \bmod N].
$$

The code is in `src/PAOFLOW/hamiltonian/wigner_seitz.py`:
- `wigner_seitz_images` finds the images;
- `wigner_seitz_hamiltonian_k` evaluates the sum at any `k`;
- `fine_grid_placement` and `wigner_seitz_fine_grid_array` handle the fine grid;
- `check_fine_grid_holds_images` raises when the fine grid is too small.

The image search works per grid index (`N³ (2 span + 1)³` candidates), so its memory grows linearly with the grid.
It takes 7 ms for a 12³ mesh and 60 ms for 24³. It returns the same images and weights as
`elphon.elph_bloch._ws_lattice`, which keeps a full candidate box (about 0.9 GB for a 24³ mesh). The two are
identical for fcc, bcc, sc, hexagonal and triclinic cells on 4³, 5×6×3 and 8³ grids (unit test).

## The fine grid costs one FFT, as the zero padding

On a uniform grid `k_j = j/M` the phase `exp(-2 pi i k_j . n)` depends on `n` only modulo `M`. The Wigner-Seitz
interpolant on that grid is therefore one forward FFT per orbital pair, of the weighted hoppings placed at `n mod M`.
`do_double_grid` does exactly that in place of `zero_pad`; the rest of the step (MPI scatter over orbital pairs,
FFT, reshaping) is shared.

**Gradients.** `gradient_and_momenta`, the band curvature and the Berry routines rebuild the fine-grid real-space
array by inverse FFT and multiply it by the fine-grid vectors (`get_R_grid_fft`, range `-M/2 <= n < M/2`). They
differentiate the Wigner-Seitz interpolant exactly only if every image sits at its own vector on the fine grid.
`check_fine_grid_holds_images` enforces this and raises otherwise, naming the smallest grid that works. The images
reach `|n_i|` of:
- `3N/4` for fcc and bcc;
- `2N/3` in-plane for hexagonal;
- `N/2` for simple cubic.

The fine grid must therefore exceed about `1.5 N` for fcc/bcc (13 for an 8³ mesh). PAOFLOW's default doubling
always satisfies this.

## Validation (Si, PseudoDojo NC PBE, internal projections on `BASIS_PS`, lecture `basics_tutorial`)

- **Against the reference evaluator.** `bands(interpolation='wigner_seitz')` equals `bloch_terms_from_HRs` +
  `hamiltonian_block` (cell-based Wigner-Seitz) to 1.4e-13 eV. The Wigner-Seitz grid of
  `interpolated_hamiltonian` (8³ → 16³) equals the direct sum to 5e-14 eV.
- **On the DFT mesh.** It agrees with zero padding to 4e-14 eV.
- **Velocities.** `pksp_nn` equals −∂E/∂k of the Wigner-Seitz bands to 2.4e-4 eV bohr: 930 non-degenerate
  finite-difference checks, at the accuracy of the difference.
- **MPI.** Serial and `mpirun -np 4` give identical bands (both routes) and DOS (3e-16).

Bands along L–Γ–X–W–K–Γ against a QE band run, largest valence-band error, 12³ mesh, from `bands()`:

| basis | `fft_grid` | `wigner_seitz` |
|---|---|---|
| minimal | 24.8 meV | 12.1 meV |
| standard | 56.1 meV | 30.1 meV |
| extended | 39.6 meV | 15.4 meV |

Grid interpolation from coarse meshes. The PAO Hamiltonians of the 4³ and 8³ meshes are interpolated to 24³ and
compared, on the 1728 points of the 12³ sub-grid, with the PAO Hamiltonian of the 12³ DFT mesh. Largest errors:

| basis, mesh | zero padding: valence / bands 5–8 | Wigner-Seitz: valence / bands 5–8 |
|---|---|---|
| minimal, 4³ | 1048.4 / 580.9 meV | 372.7 / 435.9 meV |
| minimal, 8³ | 112.1 / 90.1 meV | 39.6 / 67.4 meV |
| standard, 4³ | 3846.2 / 1416.5 meV | 1430.6 / 578.7 meV |
| standard, 8³ | 263.4 / 275.5 meV | 80.0 / 294.0 meV |
| extended, 4³ | 5986.8 / 1275.8 meV | 2370.5 / 742.9 meV |
| extended, 8³ | 305.7 / 116.0 meV | 117.1 / 50.1 meV |

Other quantities:
- **Si gap** on the 24³ grid from the 12³ mesh: 0.569 eV with Wigner-Seitz, 0.565 eV with zero padding (QE band
  path 0.568).
- **Si longitudinal mass** at the conduction-band minimum (finite differences of the same interpolants, 12³ mesh):
  QE 0.957. Wigner-Seitz gives 1.022 (standard) and 0.893 (extended); zero padding gives 0.421 and 0.852.

## Performance

`interpolated_hamiltonian`, serial, Apple M2 Ultra, Si 12³ mesh, best of two runs:

| basis (`nawf`) | 12³ → | zero padding | Wigner-Seitz |
|---|---|---|---|
| standard (26) | 24³ | 0.15 s | 0.16 s |
| standard (26) | 36³ | 0.51 s | 0.50 s |
| standard (26) | 48³ | 1.38 s | 1.36 s |
| extended (44) | 24³ | 0.40 s | 0.39 s |
| extended (44) | 36³ | 1.29 s | 1.25 s |

Building the images and their fine-grid positions takes 7–8 ms. The two routes cost the same: one FFT per orbital
pair dominates both. On a band path the Wigner-Seitz sum runs over the images (1957 for an fcc 12³ mesh) instead of
the 1728 grid vectors.

## Why the default band path is less accurate

The FFT-grid sum of `bands()` takes the Nyquist planes of an even mesh on one side only. Off the mesh its `H(k)` is
therefore not Hermitian, and `bands_calc` diagonalises its upper triangle (`eigh(lower=False)`). The lower triangle
gives eigenvalues that differ by up to 92 meV on the Si path (extended basis, 12³ mesh). Both the zero padding and
the Wigner-Seitz sum are exactly Hermitian.

## Pair-resolved images

`HRs[n]_ij` holds the hopping from orbital `i` in the home cell to orbital `j` in the cell `-n`. The bond it
describes is `d = (-n + tau_j - tau_i) A`. The cell-based assignment picks the image of `n` by `|n A|` alone. With
several atoms per cell, that choice ignores the atoms, and it breaks the space-group operations that move one atom
onto another. In diamond these are the fourfold rotations and the {100} mirrors, which come with a fractional
translation.

`interpolation='wigner_seitz_pairs'` picks, for every pair of sites `(I, J)`, the images of shortest `|d|`. These
are `wigner_seitz_images(..., separation_offset=(tau_J - tau_I) A)`, the criterion of wannier90 `use_ws_distance`,
EPW `use_ws` and `elph_bloch._ws_lattice_pairs(sign=-1)`.

`wigner_seitz_pair_images` stores the union of the cells with one weight per site pair. A path evaluation
(`wigner_seitz_pair_hamiltonian_k`) is therefore one product of the phases with the pair-weighted hoppings. On the
fine grid every orbital-pair row of the scattered `HRs` gets the images of its site pair, found through
`scatter_full_indices`, so the route is one FFT per orbital pair, as before.

The orbital centres come from `orbital_centres_for_pairs`:
- a QE run after `projections` (`arrays['basis']`);
- a model (`arrays['tau', 'naw']`).

It raises when there is not one centre per orbital. That is the case after `doubling_Hamiltonian` and
`adhoc_spin_orbit`, which do not update the stored centres.

Si, standard basis, 12³ mesh, the same runs as above. The largest |E(Rk) − E(k)| is over 200 random k in the valence
bands:

| | path valence / bands 5–8 | C3 [111] | C4 z | mirror x |
|---|---|---|---|---|
| `wigner_seitz` | 30.1 / 91.0 meV | 2.3e-11 eV | 11 meV | 8.5 meV |
| `wigner_seitz_pairs` | 32.7 / 91.5 meV | 2.1e-11 eV | 2.3e-11 eV | 1.0e-11 eV |

The pair-resolved bands keep the full space group to machine precision; they are not more accurate on the path.
Earlier checks on the extended basis gave 24.3 against 15.4 meV for the valence bands.

Cost:
- the pair route's union holds 2191 cells, against 1957 for the cell route;
- `interpolated_hamiltonian` to 24³ takes 0.19 s, against 0.16 s for the cell route;
- the fine grid must hold the slightly longer bond images: 7³ for an fcc 4³ mesh, and the check raises otherwise;
- serial, `mpirun -np 3` and 2 pools give the same eigenvalues on the fine grid (1e-14 eV).

The single-site case reduces to the cell-based route exactly (unit test).

## Not covered

- The band paths of `do_berry_phase`, `do_topology`, `do_berry_curvature` and `do_eigh` keep their own FFT-grid sums.
- `wigner_seitz_pairs` after `doubling_Hamiltonian` or `adhoc_spin_orbit` (no centre per orbital): use
  `'wigner_seitz'`.
- `elphon.elph_bloch._ws_lattice` with `span=2` returns non-shortest images for fcc/bcc meshes with one point
  along an axis (6×6×1); `span=4` fixes it. The new search is converged at `span=2`.

Tests: `tests/unit/hamiltonian/test_wigner_seitz_interpolation.py` (38; 11 for the pair-resolved route, on a
diamond-structure s model whose aliased `HRs` is exactly symmetric).
