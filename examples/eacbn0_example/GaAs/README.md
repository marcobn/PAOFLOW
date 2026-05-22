# eACBN0 on zincblende GaAs

Reproduces the eACBN0 (extended ACBN0) on-site U + intersite V
calculation of Lee & Son, *Phys. Rev. Research* **2**, 043410 (2020),
Table II, for zincblende GaAs.

## Workflow

`main.py` runs three sequential calculations on the same primitive
cell and overlays the PAOFLOW band structures:

1. **bare DFT** — plain PBE, no Hubbard correction.
2. **DFT+U**   — ACBN0 self-consistent on-site U on Ga-4s, Ga-4p,
                 As-4s, As-4p.
3. **DFT+U+V** — eACBN0 joint self-consistent loop adding intersite V
                 on all Ga–As bonds within 2.6 Å (the four nearest
                 neighbours at d = a√3/4 ≈ 2.45 Å).

After the U+V loop converges, the on-site U on **both** Ga-4s and
As-4s manifolds is zeroed and DFT is re-run once before the final band
plot. This follows Lee-Son: *"on-site interactions for s orbitals
were neglected"*, since the 4s shells are too delocalized for a
meaningful atomic-projection +U penalty. The V_ss / V_sp / V_pp
channels remain active.

The PAOFLOW band plots produced inside `main.py` cover only the bands
with high enough projectability (essentially the valence + a few
conduction bands). For a clean comparison that includes the full
conduction manifold, two helper scripts run a full-resolution QE bands
calculation:

- `run_bands.py` — QE `bands` calculation on top of the converged U+V
  SCF (reuses the existing `GaAs.save`); parses the gap.
- `run_bands_bare.py` — bare-PBE counterpart with prefix `GaAs_bare`
  (separate `bare/` outdir, so the converged U+V `GaAs.save` is
  untouched).
- `plot_bands.py` — overlays both bands and reports Γ-Γ, Γ→L, Γ→X
  gaps for each.

## Required QE settings

- `pseudo_dir` containing the Ga and As pseudopotentials
- `nosym = .true.`, `noinv = .true.` (mandatory for HUBBARD V)
- ortho-atomic Hubbard projectors:
  ```python
  ACBN0(..., projection='ortho-atomic')
  eACBN0(..., projection='ortho-atomic')
  ```
  This is critical — `(atomic)` projectors give U/V values ~30–50%
  smaller than the ortho-atomic ones used by Lee-Son.

### Note on the HUBBARD V card and cross-species duplicates

For mixed-species V channels (e.g. Ga-4s × As-4p and Ga-4p × As-4s)
QE's `card_hubbard` check considers the manifold name stripped of the
species prefix, so both reduce to atom-pair {1,2} with manifold-set
{4s,4p} and are rejected as duplicates. PAOFLOW's ACBN0 driver
detects this and emits a single averaged V entry per `(atom_pair,
unordered manifold-l pair)`. For GaAs that means four ACBN0-computed
channels collapse to three QE-legal `V` lines:

```
V Ga-4s As-4s 1 2  <V_ss>
V Ga-4s As-4p 1 2  (V_sp + V_ps) / 2
V Ga-4p As-4p 1 2  <V_pp>
```

## Convergence

| Quantity              | Value          |
|-----------------------|----------------|
| `ecutwfc` / `ecutrho` | 60 / 600 Ry    |
| SCF k-grid            | 8×8×8          |
| NSCF k-grid           | 16×16×16       |
| `nbnd` (NSCF)         | 32             |
| Mixing (eACBN0 loop)  | 0.7            |
| Convergence threshold | 0.05 eV        |
| V cutoff              | 2.6 Å          |

## Reference comparison

Lee-Son PRR 2020 Table II ("This work" / eACBN0 column) for GaAs:

| Quantity     | This work          | Lee-Son           | Δ      |
|--------------|--------------------|-------------------|--------|
| U_p (Ga-4p)  | 0.53 eV            | 0.37 eV           | +44%   |
| U_p (As-4p)  | 2.39 eV            | 1.88 eV           | +27%   |
| V_ss         | 1.56 eV¹           | 0.91 eV           | +72%   |
| V_sp         | 1.13 eV¹           | 1.26 eV           | −10%   |
| V_ps         | (folded into V_sp) | 0.80 eV           |        |
| V_pp         | 2.04 eV¹           | 1.75 eV           | +17%   |

¹ averaged over the two symmetric (i↔j) ACBN0 entries; V_sp is
additionally averaged with V_ps because QE's HUBBARD V card cannot
distinguish (Ga-4s × As-4p) from (Ga-4p × As-4s).

Trends match Lee-Son: V_pp > V_ss > V_sp, As-4p U ≫ Ga-4p U.

Band gaps from the QE bands calculation on top of the converged U+V
SCF (zeroed s-U):

| Transition  | This work | Experiment (0 K) |
|-------------|-----------|------------------|
| Γ-Γ direct  | 1.25 eV   | 1.52 eV          |
| Γ→L         | 1.43 eV   | 1.82 eV          |
| Γ→X         | 1.81 eV   | 1.98 eV          |

Bare PBE on this cell gives ~0.5 eV at Γ — the eACBN0 correction
roughly doubles the gap.

## Files

- `main.py` — driver (three-stage ACBN0 / eACBN0 run + PAOFLOW band plot)
- `run_bands.py` — QE bands.x on the converged U+V SCF
- `run_bands_bare.py` — QE scf + bands without HUBBARD card
- `plot_bands.py` — overlay bare PBE vs eACBN0 (U+V) bands, report gaps
- `GaAs.scf.in`, `GaAs.nscf.in`, `GaAs.projwfc.in` — QE input templates
- `GaAs.bands.in`, `GaAs.bandsx.in` — bands-mode templates (U+V)
- `Ga.pbe-dn-kjpaw_psl.1.0.0.UPF` — PAW Ga pseudo (3d10 in valence)
- `As.pbe-n-kjpaw_psl.1.0.0.UPF`  — PAW As pseudo
- `tmp/`, `bare/` — generated QE/PAOFLOW output (created on run)
