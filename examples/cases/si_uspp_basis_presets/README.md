# Si (ultrasoft): minimal, standard and extended projection bases

Bulk Si with a scalar-relativistic ultrasoft pseudopotential (`Si.pbe-n-rrkjus_psl.1.0.0.UPF`, with
nonlinear core correction). The basis generator takes its augmented branch: the radial equation is
the generalized eigenproblem $H u = \varepsilon S u$ with the augmentation overlap
$S = 1 + \sum_{ij} q_{ij} |\beta_i\rangle\langle\beta_j|$ of `PP_AUGMENTATION/PP_Q`, and the core
charge enters the exchange-correlation potential. Compare with the norm-conserving case
`si_oncv_basis_presets` (the same atom; the lattice parameters differ, 10.2076 and 10.5995 bohr).

`main.py` builds the PAO Hamiltonian three times, once for each string preset of
`projections(configuration=...)`, and overlays the band structures:

- `minimal`: the pseudo-atomic wavefunctions (`PP_PSWFC`) of the UPF;
- `standard`: the valence shells plus the next missing angular momentum and the (n+1) shells
  (`PAOFLOW.inputs.basis_presets.standard_augmentation`), from the radial tables generated from the UPF;
- `extended`: `standard` with a rule-based set of polarization shells (`extended_augmentation`).

The radial tables `BASIS_PS/Si/` are generated at the first run from the UPF of the case by
`PAOFLOW.basis_gen.generate_basis_for_pseudo` (the function behind `paoflow-genbasis-ps`), not taken
from the all-electron `BASIS/` of the repository.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials (from `PSEUDOS/library`) and lists the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example si_uspp_basis_presets` copies the case into `./si_uspp_basis_presets` and prints
  the steps; run them in that folder.

## Steps

1. Self-consistent run: `pw.x -in scf.in`
2. Non-self-consistent run on the full k grid: `pw.x -in nscf.in`
3. `mpirun -np 4 python main.py`: for each preset, projections → projectability (0.95) →
   `pao_hamiltonian` → bands (fcc path, 400 points); writes `output_<preset>/bands_0.dat` and the overlay
   `bands_minimal_vs_standard_vs_extended.png`.

QE: 45 / 360 Ry, scf on 8×8×8, nscf on 12×12×12 with 24 bands. A few minutes on 4 cores for all
steps.

Pseudopotential (from `PSEUDOS/library`): `Si.pbe-n-rrkjus_psl.1.0.0.UPF`.

## What to look at

- The table printed at the end: the number of orbitals `nawf` of each basis and the number of bands
  above the projectability threshold 0.95 (the bands the PAO Hamiltonian keeps).
- The overlay: the presets agree on the bands they all span; the larger bases follow the conduction
  bands higher up.

Former location: `examples/recipes/basis_presets/Si_USPP`.
