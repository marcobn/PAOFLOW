# Electron-phonon integration tests

`test_dfpt_coupling_parity.py` computes the native coarse vertex of
`PAOFLOW.elphon.dfpt_vertex` on the Si tutorial data (EPW tutorial 06: PBE, 20 Ry,
6³ k, 3³ q, 12 bands) and compares it with EPW's `epmatq` stored in the asset, at a
Γ point and at a rotated star member with an umklapp vector, plus the
$q \leftrightarrow -q$ hermiticity.

`test_irreducible_wedge.py` reads `si/si_ibz.save`, a second nscf of the same scf run
with `K_POINTS automatic 6 6 6 0 0 0` (16 irreducible points), rotates its Bloch
functions to the 216-point grid (`PAOFLOW.elphon.wavefunction_symmetry`) and compares
them, the PAO Hamiltonian and the PAO-basis vertex with the explicit full-grid nscf.

`test_long_range_sic.py` runs the polar (Fröhlich) treatment on 3C-SiC
(`sic/`: nscf 3³ with 8 bands at 40 Ry, `ph.x` 3³ with `epsil`, the internal PAO basis
tables and EPW's printed vertex with `lpolar` on a few fine (k, q)): the dense path with
the long-range model reproduces the DFPT vertex at a coarse point, the LO coupling
diverges as $1/|q|$ only with the model, the interpolated vertex agrees with EPW, and
the LO-TO splitting follows the dielectric data.

The asset tarball (`elphon_test_assets.tar.gz`, ~33 MB: `si/si.save` with the
wavefunctions, `si/si_ibz.save`, `si/save` with the `ph.x` outputs, `si/reference/epmatq_subset.npz`,
and the `sic/` tree)
is built by `.github/assets_generation/elphon/build_tar.sh` and resolved like the QE
assets, through `PAOFLOW_ELPHON_ASSET_ARCHIVE` (a local file) or
`PAOFLOW_ELPHON_ASSET_URL` + `PAOFLOW_ELPHON_ASSET_SHA256` + `PAOFLOW_ELPHON_ASSET_VERSION`
(a release asset).  Without them the tests skip.

```bash
PAOFLOW_ELPHON_ASSET_ARCHIVE=/path/to/elphon_test_assets.tar.gz pytest -m integration tests/integration/elphon
```
