# Bi<sub>2</sub>Se<sub>3</sub>(0001) surface spectral function

QE → PAOFLOW → NEGF surface Green's function. This is the PAOFLOW analogue of the
[WannierTools Bi<sub>2</sub>Se<sub>3</sub> example](https://www.wanniertools.org/examples/bi2se3/),
which builds the same quantity from VASP + Wannier90. The difference is only in
how the tight-binding Hamiltonian is obtained: here the QE spin-orbit bands are
projected onto pseudo-atomic orbitals, so there is no Wannierisation step and the
basis is fixed by the pseudopotentials. Everything downstream — principal-layer
partition, transfer-matrix surface Green's function, spectral function — is the
same construction.

The observable is

```
G_s(k, E) = [ (E + iδ) S − H_00 − Σ_L ]⁻¹        A(k, E) = −(1/π) Im Tr G_s
```

with `Σ_L` the self-energy of the semi-infinite stack below the exposed face,
obtained by transfer-matrix iteration. The right-lead self-energy is dropped,
which is what makes the system a semi-infinite crystal with one surface rather
than a two-terminal conductor.

**Expected result:** a single Dirac cone at Γ̄ inside the ~0.3 eV bulk gap, and
no other spectral weight in the gap. Where the crossing falls relative to
E<sub>F</sub> = 0 is not a prediction of this calculation — for a gapped system
the smeared Fermi level is pinned only to *somewhere* inside the gap, so the cone
may come out slightly above or below zero. The cone itself is absent from the
bulk bands and absent from any slab thin enough for the two faces to hybridise;
it exists here only because the calculation is genuinely semi-infinite.

## Running

```bash
export PW_EXEC=/path/to/pw.x PP_EXEC=/path/to/projwfc.x NPROC=64
./job.sh
```

`job.sh` regenerates the QE inputs, runs scf → nscf → projwfc, runs the bulk-band
sanity check, runs the NEGF sweep, and plots. To drive the steps by hand:

| file | role |
|---|---|
| `build_inputs.py` | derives the structure and writes `scf.in` / `nscf.in`. **Edit parameters here, not in the `.in` files** |
| `scf.in`, `nscf.in` | generated; committed so the example is runnable without the generator |
| `proj.in` | `projwfc.x` with `lwrite_overlaps` |
| `check_pao_bands.py` | bulk PAO bands — cheap sanity gate, run before `main.py` |
| `main.py` | the surface-spectrum calculation |
| `plot_surface_bands.py` | heatmap from the three `surfband_surf*` files |

Pseudopotentials are the fully-relativistic PseudoDojo `nc-fr-04` set already in
the repo at `../../../PSEUDOS/nc-fr-04_pbe_standard/`; nothing needs downloading.

## Three choices that are load-bearing

Getting any of these wrong produces a plausible-looking spectrum with no Dirac
cone, so they are worth understanding before changing anything.

**1. The 15-atom hexagonal cell, not the 5-atom rhombohedral primitive cell.**
The transport partition stacks principal layers along the third lattice vector
and Fourier-transforms over `a1`/`a2` as the in-plane periodicity. That requires
`a3` to be the surface normal and `a1`, `a2` to lie in the surface plane. In the
rhombohedral setting of R-3m all three vectors carry the same *z* component, so
neither condition holds and the "in-plane" R-vectors would walk out of the
surface plane. The hexagonal conventional cell is the smallest cell that
satisfies both; it is three quintuple layers tall.

**2. The origin is shifted so the cell boundary lands in the van der Waals gap.**
The surface Green's function terminates the stack at the cell boundary. With the
standard Wyckoff origin, *z* = 0 sits at the centre of a quintuple layer, so the
calculation would model a surface cleaved *through* a QL — dangling bonds, no
topological state. `build_inputs.py` locates the widest interlayer spacing
(2.58 Å, the vdW gap) and moves it onto the boundary. Its printout shows the
resulting stacking; the three `<-- van der Waals gap` markers should be evenly
spaced with one of them at the cell boundary.

**3. `nk_z = 3` in the nscf mesh.** The transport code builds `H_00` from
`R_z = 0` and the principal-layer coupling `H_01` from `R_z = 1`, and silently
discards everything beyond. With `nk_z = 3` the R<sub>z</sub> grid is exactly
{−1, 0, 1}, so nothing is thrown away and the principal-layer approximation is
*exact* for this Hamiltonian. `nk_z = 5` would generate `R_z = ±2` blocks that
are then dropped — an uncontrolled error, not a refinement. Leave it at 3.

## Cost and knobs

The nscf step dominates: 15 atoms, spin-orbit, 300 bands, 108 k-points with
symmetry off, 80 Ry norm-conserving. Everything after it is minutes.

* **In-plane k-mesh** (`K_NSCF` in `build_inputs.py`, default `6 6 3`). This sets
  the transverse real-space range of the PAO Hamiltonian: 6×6 gives R-vectors out
  to ~3a ≈ 12 Å. It is the knob for a smoother in-plane dispersion — 9×9 or 12×12
  cost (N/6)² more in the nscf step and nothing at all in the NEGF step. Only the
  *z* count is fixed (see above).
* **Cutoffs** (`ECUTWFC` / `ECUTRHO`, default 80/320 Ry). The PseudoDojo NC
  potentials for Bi and Se are hard; 80 Ry is above the high-accuracy hint for
  both, but run a convergence test before quoting numbers.
* **Broadening** (`delta` in `main.py`, default 0.01 eV) sets the linewidth of the
  surface state. Smaller sharpens the cone but slows the transfer-matrix
  iteration; raise `niterx` alongside it.
* **Energy window** (default ±1 eV, 401 points). Energies are already referenced
  to E<sub>F</sub> = 0 — PAOFLOW subtracts the SCF Fermi level on read, so do not
  subtract the value printed in `nscf.out`.

## Limitation worth knowing

The trace runs over all 270 orbitals of the principal layer, i.e. over all three
quintuple layers, not just the outermost one. WannierTools projects onto the top
layer, which suppresses the bulk continuum. Here the continuum comes out roughly
three times brighter than in the WannierTools figure. The gap region is
unaffected — nothing else lives there — so the Dirac cone is still the only
feature in it, but a direct side-by-side comparison of continuum intensities is
not meaningful.

## Output

Written to `output/paoflow/`:

* `surfband_surf.dat` — the (n<sub>E</sub> × n<sub>k</sub>) spectral map, one row
  per energy
* `surfband_egrid_surf.dat` — energy axis
* `surfband_kpath_surf.dat` — `index  kdist  label`, label set at K̄, Γ̄, M̄
* `surfband_surf.png` — the heatmap

Note that `.gitignore` ignores `examples/**/*.png`, so the figure must be
force-added or copied out to reach a manuscript.

## References

* Structure: Nakajima, *J. Phys. Chem. Solids* **24**, 479 (1963).
* Topological surface state: Zhang *et al.*, *Nat. Phys.* **5**, 438 (2009);
  Xia *et al.*, *Nat. Phys.* **5**, 398 (2009).
* Method: Section 7.3 of the PAOFLOW paper; see also the Fe(001) and Si(001)
  surface examples in `../example04/` and `../example04_Si/`.
