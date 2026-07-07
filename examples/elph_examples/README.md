# Electron–phonon examples

Two example scripts for the PAOFLOW el-ph module (`PAOFLOW.elphon`). See the
full write-up in `PAOFLOW.wiki/Elphon_module.md`.

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

## `example_epw_like_interpolation.py` — Route 2 (reference / debugging)

Reconstructs the full DFPT perturbation from the QE coarse-grid `dvscf`
(`induced + bare local + bare nonlocal`), builds the Bloch vertex, rotates it
into the PAO gauge, and interpolates electrons + coupling to a dense grid.
Produces the correct **structure** (α²F follows the phonon DOS) but the coupling
magnitude |g|² is currently ~2.5× too large for Pb (delicate bare/induced
cancellation — see `Elphon_module.md` §6). Provided as a reference
implementation and debugging harness, not a production result.

```bash
ELPH_BASE=/path/to/exercise1_epw conda run -n work python example_epw_like_interpolation.py
```

Required inputs: a coarse `pw.x` nscf save on the **full** k-grid (`nosym`,
`noinv`, `nbnd > nawf`, wavefunctions saved), the `ph.x` DFPT output on the same
k-grid (`fildvscf`, `phsave/patterns.*.xml`), the `*.dyn*` files, and the
pseudopotential UPF.

## Environment

Run in the `work` conda env (editable `src` tree). Units: `HRs`/eigenvalues in
eV; QE smearing and ω in Ry.
```
