---
name: CLAUDE.md
description: "General description of the PAOFLOW codebase and high level instructions for claude code"
user-invocable: true
---

1. **Python environment** 
    Run all python code in the "conda -n work" environment

2. **Existing instructions**
    Follow the instructions in the folder .agents/skills when working on docstrings and feature-design:
    - `.agents/skills/docstring/SKILL.md`: NumPy-style docstrings, type hints on every new or edited function, RST `.. math::` equations in `Notes`
    - `.agents/skills/feature-design/SKILL.md`: API and module design rules (summarized in section 5)
    - `.agents/skills/sparse-backend/SKILLS.md`: design notes and memory contract for `src/PAOFLOW/sparse/`. Read it before touching the sparse backend or `SparsePAOFLOW.py`

3. **What PAOFLOW is**
    PAOFLOW (version 3.0.0, GPL-3.0-or-later) is a Python package for post-processing DFT calculations. It projects Kohn–Sham states from Quantum ESPRESSO (or VASP) onto a Projected Atomic Orbital (PAO) basis to build *ab initio* tight-binding Hamiltonians. From these it computes bands, DOS/PDOS, Fermi surfaces, Boltzmann transport, Berry-phase/topology, spin/anomalous Hall, dielectric response, Green's-function (Landauer) transport, phonons, electron–phonon coupling and ACBN0 Hubbard U/V.
    - Build: `flit_core` (`pyproject.toml`), Python >= 3.10, core dependencies are numpy, scipy, mpi4py and pandas
    - Optional extras: `graphics`, `fermisurface`, `transport`, `pyskeaf`, `qo_analysis`, `phonon`, `weyl_search`, `fast` (numba), `dev`, `docs`
    - Parallelism: MPI via `mpi4py`, with k-points split across ranks and pools. Scripts are run as `mpirun -np N python main.py`
    - Console scripts: `pyskeaf`, `paoflow-genbasis-ps`, `paoflow-gen-qe`, `paoflow-gen`, `fermi-plotter`

4. **Folder tree**
    ```
    src/PAOFLOW/
      PAOFLOW.py          Main orchestrator class PAOFLOW (user-facing physics methods)
      SparsePAOFLOW.py    Sparse-backend driver that mirrors the PAOFLOW API for large doubled cells
      Transport.py        Green's-function / Landauer transport orchestrator (class Transport)
      DataController.py   Central store: data_arrays / data_attributes dicts, MPI bcast/scatter, file I/O
      ACBN0.py            ACBN0 / eACBN0 self-consistent Hubbard U and intersite V
      GPAO.py             Plotting helper class (bands, DOS, Berry, transport, optics, phonons)
      ErrorHandler.py     Error reporting
      inputs/             QE XML, VASP, UPF, PAOFLOW XML input readers; basis presets; input_default.xml
      projection/         Atomic-wfc projections, projectability, ortho/non-ortho, minimal basis
      hamiltonian/        H(k)/H(R) construction, doubling, SOC, external fields, gradients, momenta,
                          nonlocal velocity, symmetry expansion (pao_sym), two-center integrals
      spectrum/           Bands, eigh, DOS/PDOS, Fermi energy, adaptive smearing, effective mass
      topology/           Berry phase, Fermi surface, Weyl search, spin texture, Z2/pfaffian
      response/           Dielectric tensor, (spin/anomalous) Hall, Rashba–Edelstein, IPR, linear response
      boltzmann/          Boltzmann transport tensors, doping, relaxation-time (tau) models
      transport/          Backend for Transport.py: calculators/ (Green's functions, leads, transmittance,
                          current), grid/, hamiltonian/, partition/, parsers/, io/, smearing/, workspace/
      sparse/             Sparse bond-list Hamiltonian, iterative solver, bands, DOS/PDOS, doubling
      phonon/             Phonopy interface, Born charges, IR/Raman, vibrational dielectric, QHA
      elphon/             Electron–phonon coupling (g(k,q), dvscf, QE elph I/O, folding, symmetry)
      models/             Tight-binding models, Slater–Koster fitting, band unfolding, surface projection
      pyskeaf/            Python SKEAF: de Haas–van Alphen extremal Fermi-surface orbits (CLI + adapter)
      agentic/            Library code of the agentic workflows (agentic/ folder below): qo_analysis/ (pyskeaf
                          orbits vs quantum oscillations; PAOFLOW.qo_analysis is a back-compat shim),
                          thermo_optics/ (T-dependent optics vs experimental data of any form: datasets with
                          provenance, PDF/raster digitising, metrics, plots)
      basis_gen/          Generates atomic-orbital basis from pseudopotentials (paoflow-genbasis-ps)
      gen/                Generators: QE input from AFLOW/C2DB (paoflow-gen-qe), main.py driver (paoflow-gen),
                          fermi_plotter
      graphics/           Color and plotting utilities
      writers/            BXSF / XSF / BoltzTraP2 writers
      utils/              Constants, MPI communication, FFT grids, smearing, pyints (Gaussian integrals)
      acbn0_native/, epsilon_native/   Optional bridges to the Rust extension, with pure-Python fallback
      defs/               Back-compat shim that aliases legacy PAOFLOW.defs.* imports and emits FutureWarning
    rust/paoflow_rs/      Optional Rust (PyO3 + rayon) kernels for ACBN0 ERIs and epsilon/JDOS. Parity < 1e-12
    tests/unit/           Fast unit tests, organized by subpackage
    tests/integration/    qe/, transport/, TBmodel/: example-driven runs compared against Reference data
    examples/             qe_examples, transport_examples, acbn0_examples, TBmodel_examples, elphon_example,
                          pyskeaf_examples, qo_analysis_examples, unfold_examples, plot_examples, new_examples, vasp_examples
    BASIS/                Per-element radial orbital tables (e.g. BASIS/Si/3P.dat) for internal projections
    PSEUDOS/              Norm-conserving pseudopotential sets (nc-sr-05, nc-fr-04 PBE standard)
    agentic/              Agentic workflows run on top of the library (not installed). skeaf/: playbook,
                          prompts, templates, git-ignored projects/; has its own AGENTS.md / CLAUDE.md
                          thermo_optics/: same layout, optics-vs-experiment playbook (PAOFLOW.agentic.*)
    docs/                 Sphinx + MyST site (Read the Docs); docs/internals/ holds developer notes
    .github/              CI workflows and asset_generation scripts for integration-test tarballs
    ```

5. **Architecture and design rules**
    - Standard workflow in `main.py` (see `examples/qe_examples/example01/main.py`):
      `PAOFLOW(savedir=...)` → `read_atomic_proj_QE()` → `projectability()` → `pao_hamiltonian()` → `bands()` / `interpolated_hamiltonian()` → `pao_eigh()` → `gradient_and_momenta()` → `adaptive_smearing()` → `dos()` / `transport()` / ... → `finish_execution()`
    - Transport workflow: after building the PAO Hamiltonian, run `Transport(paoflow.data_controller)` → `define_partition` → `build_hamiltonian_blocks` → `configure_*` → `compute_leads_self_energy` → `compute_greens_functions` → `compute_transmission` / `compute_dos` / `compute_current`
    - Orchestrators (`PAOFLOW.py`, `Transport.py`, `SparsePAOFLOW.py`) expose physics-named methods and call procedural backend functions in the subpackages. `main.py` must read as a physics workflow and must not reach into internal arrays
    - Shared state goes through `DataController` (`arrays, attr = self.data_controller.data_dicts()`). Store data there only when a later step reuses it; keep scratch arrays local
    - Do not rename existing `do_*` modules. Do not use the `do_*` prefix for new modules; give them descriptive names
    - Pass inputs as Python arguments. Do not add YAML/JSON config files for runtime configuration
    - Refactors must not change numerical output
    - joblib/loky parallelism (`sk_fitting`, `band_unfold`, `sparse_bands`, `pyskeaf`) must stay within the MPI rank's core budget. Pass every `n_jobs`/`n_workers` through `resolve_n_jobs` (and cap thread pools with `core_budget()`) from `src/PAOFLOW/utils/parallel_resources.py`. Importing `PAOFLOW` exports `LOKY_MAX_CPU_COUNT` under `mpirun`
    - Sparse backend: never materialize dense `(nk, nawf, nawf)` or `(nawf, nawf, nR)` tensors. Unsupported methods must raise loudly, with no silent fallback to dense

6. **Testing**
    - Run all tests with `pytest`, or select with `pytest -m unit`, `pytest -m integration`, `pytest -q tests/unit/<pkg>`, `pytest tests/... -k exampleNN`
    - Integration tests only run PAOFLOW; they do not run QE. The `*.save/` and `Reference/` inputs are downloaded as release-asset tarballs (see `tests/integration/qe/README.md` and `tests/integration/transport/README.md`). The generators live in `.github/assets_generation/`
    - New features ship with tests. Keep them fast by using reduced grids and k-points

7. **Style, CI and branches**
    - Ruff lint and format run through pre-commit: line length 100, single quotes. Run `pre-commit run --all-files` before committing. `BASIS/` and `examples/` are excluded
    - CI (`.github/workflows/ci.yaml`) runs on PRs into `develop`: pre-commit, then the QE and transport integration matrices under MPI
    - `develop` is the integration branch, and PRs into `main` are only accepted from `develop`. Feature work happens on branches such as `develop-marco`
    - Docs: `./docs/serve_docs.sh` gives a live preview, and `--full` builds with source links. Developer notes are in `docs/internals/`
