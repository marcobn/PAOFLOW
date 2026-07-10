# Electron–phonon examples

Example scripts for the PAOFLOW el-ph module (`PAOFLOW.elphon`). See the
full write-up in `PAOFLOW.wiki/Elphon_module.md`.

## `example_pao_from_qe_coupling.py` — PAO route (recommended)

The pseudo-atomic-orbital (Agapito & Bernardi, *Phys. Rev. B* **97**, 235146 (2018))
route. Reads QE's **full** coarse-grid coupling `el_ph_mat` (which already
contains the bare local, bare nonlocal and induced parts, plus any NLCC /
ultrasoft augmentation), rotates it into the PAOFLOW pseudo-atomic-orbital (PAO) gauge,
and Wigner–Seitz interpolates electrons + vertex to a dense grid for α²F, λ,
ω_log and Tc. **No potential reconstruction** — so NLCC and ultrasoft pseudos
work with no extra effort. Pb (9³→18³): λ ≈ 1.7, ω_log ≈ 56 K, Tc ≈ 8 K.

```bash
ELPH_BASE=/path/to/exercise1 conda run -n work python example_pao_from_qe_coupling.py
```

Required inputs: a coarse `pw.x` nscf save on the full k-grid (`nosym`, `noinv`,
`nbnd > nawf`), the `elph_dir/elphmat.<iq>.dat` dumps from the PAOFLOW-patched
`ph.x` (`electron_phonon='interpolated'`, `PAOFLOW_DUMP_ONLY=1`), and the
`*.dyn*` files.

## `example_qe_dfpt_properties.py` — Route 1 (working)

Reads QE's already-computed coarse-grid coupling (`*.fc` + `a2Fmatdyn.NN`),
interpolates to a dense q-grid with PAOFLOW's Wigner–Seitz generalized-Fourier
interpolation, and computes α²F, λ, ω_log and Tc. Reproduces QE
`matdyn`/`lambda` essentially exactly (Pb: λ ≈ 1.34, ω_log ≈ 65.6 K).

```bash
ELPH_DATA=/path/to/qe/fc/files conda run -n work python example_qe_dfpt_properties.py
```

Required inputs: the phonon force constants (`q2r.x`) and the coupling force
constants (`matdyn.x` with `la2F=.true.`), one `a2Fmatdyn.NN` per smearing.

## Environment

Run in the `work` conda env (editable `src` tree). Units: `HRs`/eigenvalues in
eV; QE smearing and ω in Ry.
```
