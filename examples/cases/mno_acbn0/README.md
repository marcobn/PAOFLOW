# MnO (antiferromagnetic): ACBN0 Hubbard U

Self-consistent on-site U Hubbard parameters from the PAO density (ACBN0 runs pw.x and projwfc.x itself), then the corrected bands.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials, fills the **acbn0** form and lists the runs below in the
  Runs tab.
- **Command line**: `paoflow-gen --example mno_acbn0` copies the case into `./mno_acbn0` and writes the
  drivers; then run the steps below in that folder.

## Steps

1. PAOFLOW: `python main.acbn0.py` runs pw.x and projwfc.x itself until U (and V) converge, then the bands

Pseudopotentials (from `PSEUDOS/library`): `Mn_ONCV_PBE_sr.upf`, `O_ONCV_PBE_sr.upf`.

## Notes from the original example (`examples/acbn0_examples/MnO`)

ACBN0 self-consistent U on antiferromagnetic MnO.

AFM-II rocksalt MnO requires two Mn sublattice species (MnA, MnB)
so that QE can initialise opposite starting_magnetization values.
Both sublattices receive the same self-consistent U.

The Hubbard occupations come from PAOFLOW's internal projection on the
UPF pseudo-atomic orbitals, so no projwfc.in template is needed.
