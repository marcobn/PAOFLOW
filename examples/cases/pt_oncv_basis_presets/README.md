# Pt (fully relativistic ONCV): minimal, standard and extended projection bases

Fcc Pt with spin-orbit coupling (noncollinear, `lspinorb`) and a fully relativistic norm-conserving
ONCV pseudopotential (`Pt_ONCV_PBE_fr.upf`). The basis generator writes j-resolved radial functions
for every channel with l ≥ 1 (`6P_j1.dat` for j = 1/2, `6P_j3.dat` for j = 3/2, ...), and PAOFLOW
builds the spinor basis from them. The reference for the ultrasoft case `pt_uspp_basis_presets`.

`main.py` builds the PAO Hamiltonian three times, once for each string preset of
`projections(configuration=...)`, and overlays the band structures:

- `minimal`: the pseudo-atomic wavefunctions (`PP_PSWFC`) of the UPF;
- `standard`: the valence shells plus the next missing angular momentum and the (n+1) shells
  (`PAOFLOW.inputs.basis_presets.standard_augmentation`), from the radial tables generated from the UPF;
- `extended`: `standard` with a rule-based set of polarization shells (`extended_augmentation`).

The radial tables `BASIS_PS/Pt/` are generated at the first run from the UPF of the case by
`PAOFLOW.basis_gen.generate_basis_for_pseudo` (the function behind `paoflow-genbasis-ps`), not taken
from the all-electron `BASIS/` of the repository.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials (from `PSEUDOS/library`) and lists the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example pt_oncv_basis_presets` copies the case into `./pt_oncv_basis_presets` and prints
  the steps; run them in that folder.

## Steps

1. Self-consistent run: `pw.x -in scf.in`
2. Non-self-consistent run on the full k grid: `pw.x -in nscf.in`
3. `mpirun -np 4 python main.py`: for each preset, projections → projectability (0.95) →
   `pao_hamiltonian` → bands (fcc path, 400 points); writes `output_<preset>/bands_0.dat` and the overlay
   `bands_minimal_vs_standard_vs_extended.png`.

QE: 40 / 467 Ry, Marzari-Vanderbilt smearing 0.02 Ry, scf on 6×6×6, nscf on 12×12×12. The nscf
run with spin-orbit coupling is the longest step.

Pseudopotential (from `PSEUDOS/library`): `Pt_ONCV_PBE_fr.upf`.

## What to look at

- The table printed at the end: the number of orbitals `nawf` of each basis and the number of bands
  above the projectability threshold 0.95 (the bands the PAO Hamiltonian keeps).
- The overlay: the presets agree on the bands they all span; the larger bases follow the conduction
  bands higher up.

Former location: `examples/recipes/basis_presets/Pt_ONCV`.
