# eACBN0 on diamond Si

Reproduces the eACBN0 (extended ACBN0) on-site U + intersite V
calculation of Lee & Son, *Phys. Rev. Research* **2**, 043410 (2020),
Table II, for diamond silicon.

## Workflow

`main.py` runs three sequential calculations on the same primitive
cell and overlays the band structures:

1. **bare DFT** — plain PBE, no Hubbard correction.
2. **DFT+U**   — ACBN0 self-consistent on-site U on Si-3s and Si-3p.
3. **DFT+U+V** — eACBN0 joint self-consistent loop adding intersite V
                 on all Si–Si bonds within 2.6 Å (the four nearest
                 neighbours at d = a√3/4 ≈ 2.35 Å).

After the U+V loop converges, the on-site U on the Si-3s manifold is
**zeroed** (`e.uVals['Si-3s'] = 0.0`) and DFT is re-run once before
the final band plot. This follows Lee-Son: *"on-site interactions for
s orbitals were neglected"*, since the 3s shell is too delocalized for
a meaningful atomic-projection +U penalty. The V_ss / V_sp / V_pp
channels remain active.

## Required QE settings

- `pseudo_dir` containing the Si pseudopotential
- `nosym = .true.`, `noinv = .true.` (mandatory for HUBBARD V)
- ortho-atomic Hubbard projectors:
  ```python
  ACBN0(..., projection='ortho-atomic')
  eACBN0(..., projection='ortho-atomic')
  ```
  This is critical — `(atomic)` projectors give U/V values ~30–50%
  smaller than the ortho-atomic ones used by Lee-Son.

## Convergence

| Quantity              | Value          |
|-----------------------|----------------|
| `ecutwfc`             | 50 Ry          |
| SCF k-grid            | 8×8×8          |
| NSCF k-grid           | 16×16×16       |
| `nbnd` (NSCF)         | 18             |
| Mixing (eACBN0 loop)  | 0.7            |
| Convergence threshold | 0.05 eV        |
| V cutoff              | 2.6 Å          |

## Reference comparison

Lee-Son PRR 2020 Table II (eACBN0 column) for diamond Si:

| Quantity | This work | Lee-Son | Δ      |
|----------|-----------|---------|--------|
| U_p      | 3.00 eV   | 3.50 eV | −14%   |
| V_ss     | 1.25 eV   | 0.90 eV | +39%   |
| V_sp     | 0.66 eV   | 0.72 eV | −8%    |
| V_pp     | 1.66 eV   | 1.85 eV | −10%   |
| gap      | 1.23 eV   | 1.36 eV | −10%   |

Ordering V_pp > V_ss > V_sp matches Lee-Son exactly. Residual gap is
attributable to pseudopotential differences (Troullier-Martins NC here
vs PseudoDoJo / GBRV in the paper) and minor differences in the
Gaussian fits to the PP_PSWFC radial functions.

## Files

- `main.py` — driver
- `Si.scf.in`, `Si.nscf.in`, `Si.projwfc.in` — QE input templates
- `Si.pbe-tm-new-gipaw-v2.1.UPF` — Troullier-Martins NC pseudopotential
- `tmp/` — generated QE/PAOFLOW output (created on run)
