# Projection tests

Small QE saves that test what the model Hamiltonians of `tests/unit/closed_forms` cannot: the QE
readers, the two projection routes and the PAO Hamiltonian. No reference output is stored; the
tests compare PAOFLOW with the DFT eigenvalues of the saves themselves.

| system | what it covers |
|---|---|
| `si` | norm-conserving pseudopotential; projwfc.x and the internal `minimal`, `standard` bases |
| `gaas_paw` | PAW: the augmented metric of the internal projections |
| `pt_fr` | fully relativistic ultrasoft: the spinor basis, Kramers pairs |
| `pt_lsda` | collinear magnet (`nspin = 2`): both spin channels |
| `bi_bilayer` | 2D layer, fully relativistic PAW |

Checks (`test_projections.py`):

- the PAO bands equal the DFT eigenvalues at the k-points the PAO grid and the save share, for the
  projectable bands below E_F + 1 eV (projwfc.x, internal `minimal` and `standard`);
- the internal `minimal` basis (the pseudo-atomic orbitals of the UPF, which projwfc.x reads too)
  gives the projwfc.x bands;
- with spin-orbit coupling and inversion every band is a Kramers pair.

## Assets

Each folder holds `scf.in`, `nscf.in` and `proj.in` (pseudopotentials from `PSEUDOS/library`, coarse
k grids). `.github/assets_generation/projections/build_assets.sh [OUT.tar.gz]` runs pw.x and
projwfc.x on them in a scratch folder, keeps what PAOFLOW reads (`data-file-schema.xml`, the
wavefunctions, `atomic_proj.xml`, the UPFs) and packs `projection_test_assets.tar.gz`: a few
minutes on a laptop, no HPC run.

```bash
QE_BIN=/path/to/qe/bin MPI="mpirun -np 4" .github/assets_generation/projections/build_assets.sh
PAOFLOW_PROJECTION_ASSET_ARCHIVE=$PWD/.github/assets_generation/projections/_assets/projection_test_assets.tar.gz \
  pytest tests/integration/projections -m integration
```

CI downloads the release asset (`PAOFLOW_PROJECTION_ASSET_URL`, `PAOFLOW_PROJECTION_ASSET_SHA256` in
`.github/workflows/ci.yaml`). Without either variable the tests skip.
