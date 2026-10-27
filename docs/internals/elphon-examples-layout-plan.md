# Electron-phonon examples and generator: reorganisation plan

Status (October 27 2026): **implemented** as alternative A (the thin drivers over
`PAOFLOW.elphon.workflows`), with the generator upgraded. What exists now:

- `src/PAOFLOW/elphon/workflows/`: `ElphonSystem` (`system.py`), `superconductivity.py`,
  `optics.py`, `transport.py`, `arpes.py`, `phonons.py`, `coupling.py`; `load_system` finds the
  nearest `system.py` above a driver.
- `src/PAOFLOW/gen/elphon_templates/`: the drivers of every workflow folder (`phonons/`,
  `coupling/`, `superconductivity/`, `optics/`, `transport/`, `arpes/`, `qe/kmesh.py`);
  `src/PAOFLOW/gen/elphon_system.py`: `paoflow-gen elphon` (prompts, `render_system_py`,
  `qe_inputs`, `render_run_qe`, `write_system_folder`). The old string templates of
  `paoflow_driver.py` (2.7k lines) are gone, as are the `ahc`/`elphmat` menu entries.
- `examples/elphon/{Si,Pb,MgB2,SiC}`: `system.py` (rendered by the generator), `qe/`
  (phonon, nscf, optics inputs, `run_qe.sh`, `kmesh.py`), the workflow folders with copies of
  the templates, READMEs with references, the SiC report. `examples/elphon_example/` is a
  pointer README.
- Tests: `tests/unit/elphon/test_workflows.py`, `tests/unit/gen/test_elphon_system.py` (the
  examples must equal the templates; every driver parses `--help`; the ME, resistivity and
  k-mesh templates run on synthetic data), `tests/integration/elphon/test_example_workflows.py`
  (each workflow on the Si asset).
- Numerics: the old drivers and the new ones were run on the same data with identical outputs
  (max relative difference ≤ 7e-15): Pb 12³/12³ Eliashberg and ARPES and MgB₂ anisotropic ME on
  the Workspace EPW data, SiC coupling path and 12³ mobility on the SiC run, Si 12³/6³
  phonon-assisted absorption and 12³ mobility on the 6³/3³ integration asset.

The original proposal follows.

## What we have

`examples/elphon_example/{Si,Pb,MgB2,SiC}` grew one script per property on top of a flat
layout: each system has `phonon/` (scf, ph.x) and `nscf/` (nscf, EPW inputs) and, next to
them, `main.py`, `me.py`, `me_aniso.py`, `transport.py`, `mobility.py`, `arpes.py`,
`optics_eps.py`, `emissivity_from_absorption.py`, `quadrupoles.py`, `coupling.py`,
`phonons.py`, `plot*.py`, 3.5k lines in all. Three problems came up during the last weeks of
work:

1. **The configuration is duplicated.** `SAVEDIR`, `PHONON_DIR`, `COARSE_GRID`, `QGRID`,
   `MASSES_AMU`, `NELEC`, the basis path and the long-range choices live at the top of every
   script (`Si/mobility.py` repeats `Si/main.py`; only the `arpes.py` drivers import them).
   Copies in the users' run directories drift: today a Si run failed twice because the copied
   `mobility.py` carried a 3³ q-grid while the data were 6³.
2. **`main.py` is a multiplexer.** `Si/main.py` (611 lines) is phonon-assisted absorption,
   emissivity, eACBN0 electrons and scissor handling in one file; `Pb/main.py` is Eliashberg
   plus a transport flag. A user who wants one property has to read the others.
3. **The expensive part is recomputed.** Every script rebuilds the coarse vertex
   $g(R_e, R_p)$ from the DFPT files (minutes to tens of minutes) although `coupling_cache`
   exists; nothing shares a cache directory across properties.
4. **The generator lags the examples.** `paoflow-gen elphon` (`gen/paoflow_driver.py`, a
   6.2k-line script of string templates) emits an EPW-source `main.elphon.py` /
   `me.elphon.py` / `transport.elphon.py` / `plot.elphon.py` in the layout of
   `examples/elphon_epw_example`, which no longer exists. It knows nothing of the native DFPT
   source, the irreducible-wedge nscf, the long-range and quadrupole terms, adaptive smearing
   or ARPES. Two template sets (examples and generator) cannot be kept in step by hand.

## Proposal: one system folder, shared assets, one subfolder per workflow

```
examples/elphon/
  README.md                       conventions, how a system folder is built, how to add one
  Si/
    README.md                     the system, the QE steps, what every workflow shows, the numbers
    REPORT.md                     (where a report exists)
    system.py                     THE configuration: prefix, paths, grids, masses, nelec, basis,
                                  long-range/quadrupole choice, pao_electronic_structure()
    qe/                           scf.in  ph.in  nscf.in  run_qe.sh  [epw.in write_ukk.py kmesh.py]
                                  (outputs: <prefix>.save, _ph0/, dyn files; git-ignored)
    cache/                        coarse vertex g(R_e, R_p) and dense-electron caches (git-ignored)
    phonons/      phonons.py      q2r + matdyn in PAOFLOW, LO-TO, dispersion figure
    coupling/     coupling.py quadrupoles.py   vertex checks, long-range term, quadrupoles -> quadrupole.fmt
    superconductivity/  eliashberg.py me.py me_aniso.py plot_me.py
    optics/       absorption.py (phonon-assisted + direct), dielectric.py, emissivity.py, plot.py
    transport/    mobility.py (semiconductors) | resistivity.py (metals: rho, Drude, emissivity), plot_*
    arpes/        arpes.py
    figures/
  Pb/ MgB2/ SiC/ (same shape; only the workflows that make sense for the system)
```

- **One configuration per system.** `system.py` is the only place with values; every
  workflow script starts with `from paoflow_elphon_example import system` (a two-line
  bootstrap: `sys.path.insert(0, parent)` + `import system`), or, see the alternative below,
  with `system = ElphonSystem.from_file(Path(__file__).parents[1] / 'system.py')`. The Workspace
  run directories then hold a copy of `system.py` and symlinks to `qe/`, nothing else.
- **Workflows are independent and self-describing.** A user opens `transport/` and finds the
  mobility driver, its plot and its README paragraph; nothing else is needed. The
  dependencies between workflows are data files with fixed names (`coupling/quadrupole.fmt`,
  `superconductivity/output/eliashberg.npz` read by `arpes.py`, `optics/output/epsilon.npz`
  read by the emissivity and the metal transport), documented in the README table.
- **Shared caches.** All workflows pass `cache_dir=system.CACHE` to `prepare_dense_vertex`
  and `precompute_dense_electrons`; the coarse vertex is built once per system and the
  (basis, pthr, nbnd) tag guards against stale entries. This is the main efficiency gain: the
  Pb vertex build is 2 min, the Si 12³/6³ one about 10 min, repeated today by every script.
- **Outputs stay local to the workflow** (`transport/output/`), so reruns of one property never
  touch another.

### Alternative A: the examples become thin drivers over a library workflow layer

Move the orchestration that is now copied into every driver (electronic structure from the
save, the long-range model, vertex preparation with the cache, the carrier statistics, the
output writers) into `src/PAOFLOW/elphon/workflows/`:

| library object | replaces |
|---|---|
| `ElphonSystem` (dataclass: prefix, save, phonon dir, grids, masses, nelec, basis, pthr, long_range, quadrupoles, cache) with `electrons()` and `coupling()` | the 100-line preamble of every script |
| `superconductivity(system, nk, nq, sigma, transport=False, fs_coupling=False)` | `Pb/main.py`, `MgB2/main.py` |
| `phonon_assisted_optics(system, ...)`, `thermal_emissivity(system, ...)` | `Si/main.py` |
| `carrier_mobility(system, ...)`, `metal_resistivity(system, ...)` | `mobility.py`, `transport.py` |
| `arpes_map(system, experiment, self_energy=...)` | the three `arpes.py` |

The example scripts shrink to argparse plus one call (20–40 lines each), the generator emits
exactly those scripts, and the examples are the generator's templates. The physics stays in
the procedural backends (`eliashberg_dense_q`, `carrier_scattering_dense_q`, ...); the
workflow layer only wires inputs, caches and outputs, which keeps the rule that `main.py`
reads as a physics workflow. This is the alternative I recommend: it removes the drift
between examples and generator at its root, and the per-system folders remain exactly as
proposed above. Cost: about one day more than the pure reorganisation.

### Alternative B: one `main.py --property` per system

Fewer files, one entry point. It hides the independent workflows and keeps the 600-line
multiplexers; it is what the generator does today and what the reorganisation is meant to
retire. Not recommended.

## The generator

`paoflow-gen elphon` should produce **the system folder**, not a script:

1. prompts (as today) for the `pw.x` input, prefix, pseudopotentials, grids, masses; new
   prompts: coupling source (`dfpt` default, `epw` optional), nscf on the irreducible wedge
   (default) or the explicit full list (EPW), long-range term (automatic from `epsil`),
   quadrupoles (`quadrupoles.py` workflow), and a **multi-select of workflows**
   (phonons, coupling, superconductivity, optics, transport, arpes);
2. writes `qe/` (scf, ph.x with `epsil` and `fildvscf`, nscf, `run_qe.sh`; EPW inputs only
   for the EPW source), `system.py` from the prompts, and copies the selected workflow
   folders from `examples/elphon/_template/` (the same files the examples use, parameters
   only in `system.py`);
3. drops the string templates of `paoflow_driver.py` for the electron-phonon workflow
   (the ~1.5k lines that build `main.elphon.py` and friends) and the reference to
   `examples/elphon_epw_example`.

With alternative A the templates are the thin drivers; with the pure reorganisation they are
the example scripts themselves. Either way, `paoflow-gen elphon` and
`examples/elphon/<system>` cannot drift again.

## Tests and CI

- The elphon integration asset (`si/`, `sic/`) is reshaped to the `qe/` layout, so every
  workflow driver runs on it unchanged with small grids; a smoke test
  `tests/integration/elphon/test_example_workflows.py` runs each workflow with `--nk 6 --nq 6`
  (seconds to a minute each) and checks the output files. Today the examples are not
  exercised by CI at all.
- The asset builder in `.github/assets_generation/elphon/` reuses `qe/run_qe.sh` of the
  example instead of its own copies.

## Migration

1. Create `examples/elphon/` with the layout above; `examples/elphon_example` becomes a
   one-line README pointing to it (removed one release later).
2. Write `system.py` for Si, Pb, MgB₂ and SiC from the current constants; split `Si/main.py`
   into `optics/absorption.py` and `optics/emissivity.py`; move the rest unchanged.
3. (Alternative A) add `elphon/workflows/` and cut the drivers down; keep the numerical
   output bit-identical (the drivers call the same backends with the same arguments; the
   existing REPORT numbers are the regression reference).
4. Shared `cache/` wired into every driver.
5. Docs: `Electron-Phonon-Coupling.md` (module map, CLI section, example references),
   `cli-generators.md`, `arpes.md`, the four READMEs and REPORTs, `CLAUDE.md` tree.
6. Generator: system-folder output, workflow multi-select, DFPT default, template copy.
7. CI smoke test on the asset; Workspace run directories re-created as `system.py` +
   symlinks to the data.

Effort: 1.5 days for the layout, `system.py`, the splits and the docs; 1 day for the workflow
layer and the caches (alternative A); 1 day for the generator; half a day for the CI smoke
test. Numerical outputs do not change.
