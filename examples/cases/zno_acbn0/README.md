# ZnO: ACBN0 Hubbard U

Self-consistent on-site U Hubbard parameters from the PAO density (ACBN0 runs pw.x and projwfc.x itself), then the corrected bands.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials, fills the **acbn0** form and lists the runs below in the
  Runs tab.
- **Command line**: `paoflow-gen --example zno_acbn0` copies the case into `./zno_acbn0` and writes the
  drivers; then run the steps below in that folder.

## Steps

1. PAOFLOW: `python main.acbn0.py` runs pw.x and projwfc.x itself until U (and V) converge, then the bands

Pseudopotentials (from `PSEUDOS/library`): `Zn.pbe-n-nc.UPF`, `O.pbe-n-nc.UPF`.

## Notes from the original example (`examples/acbn0_examples/ZnO`)

ACBN0 self-consistent U on wurtzite ZnO.

Runs the ACBN0 self-consistent loop to determine on-site Hubbard U
corrections for Zn-3d and O-2p in wurtzite ZnO.
