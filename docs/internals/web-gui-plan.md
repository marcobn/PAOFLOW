# PAOFLOW Studio: a setup GUI for `paoflow-gen` and a results viewer (design, 7 November 2026)

Status: implemented on 7 November 2026 (P1 to P6, uncommitted on `develop-marco`). Section 0 records what was built
and how it differs from the plan; the rest of the page is the design as proposed. User documentation:
[Working in the browser](../tutorials/howtos/browser-gui.md). Sections 4 to 7 are the generator (setup) GUI and sections
8 to 10 the plotting (results) GUI. Both share the server and the security model of section 3.

Decisions (user, 7 November 2026):

- **D1:** one app with Setup, Runs and Results tabs.
- **D2:** runs are local only, on the machine that serves the GUI and generates the files. There is no Slurm
  submission from the GUI, so the Slurm items of section 7 are dropped.
- **D3:** the declarative-schema refactor of the CLI.
- **D4:** smart defaults from the QE save, in the GUI and in the interactive CLI.
- **D5:** Plotly.js from the installed `plotly` package (extra `gui`).
- **D6:** an output manifest `paoflow_outputs.json`.
- **D7:** fix the writer issues 9.2.3 and 9.2.5. Done on 7 November 2026:
  - the Hall entry of the generated `plot.py` now matches `hall_trace*_0.dat` and reads the 3-column layout
    (T, E, R_H);
  - `DataController.write_bxsf(..., skeaf_sheets=True)` gained the flag, and the three Hall writers pass `False`;
  - the writer file names are unchanged, for backward compatibility;
  - the stale Berry-curvature `Fermi_surf_band_{1,2}.bxsf` files in the `Reference/` folders of `qe_examples` 04,
    05, 07, 08 and 14, `new_examples` 04 and 05 and `TBmodel_examples/Kane_mele` were left in place (the
    comparisons read `*.dat` only).

## 0. As built (7 November 2026)

The application is called **PAOFLOW Studio** and the command `paoflow-studio` (renamed from `paoflow-gui`,
user, 7 November 2026). The Python package stays `PAOFLOW.gui` and the extra stays `gui`.

**Modules:**

| layer | modules |
|---|---|
| generator questions | `gen/questions/` (`model`, `prompt`, `resolve`, `common`, `run`, `acbn0`, `phonon`, `elphon`, `sscha`) |
| generator support | `gen/outputs.py` (`workflow_outputs`, `GeneratedFile`, `run_plan`, `RunStep`), `gen/safe_lists.py` (AST list expressions instead of `eval`), `gen/system_summary.py` (QE save, `pw.x` input and UPF headers; the D4 defaults) |
| results layer | `results/` (`model`, `columns`, `catalogue`, `discovery`, `loaders`, `volumes`, `figures`, `palette`, `export`, `report`) |
| output manifest | `writers/output_manifest.py`, called by `PAOFLOW.finish_execution` (D6) |
| server and front end | `gui/` (`cli`, `server`, `security`, `workspace`, `generator_api`, `results_api`, `jobs`) and `gui/static/` (`index.html`, `css/app.css`, `js/{app,api,dom,form,setup,browser,codeview,plots,results,runs}.js`) |
| packaging | console script `paoflow-studio`, extra `gui = ["plotly>=5.18"]`, `paoflow-gen --gui` |

**Tests:** 165 new unit tests in `tests/unit/gen`, `tests/unit/gui`, `tests/unit/results` and
`test_write_bxsf_sheets.py`, among them:

- the frozen parity harness (35 scenarios, unused-rule check);
- the builder-key coverage test;
- a GUI-against-CLI byte comparison;
- security, jobs, results and the manifest.

The full unit suite passed (1669 tests) after P1. Headless-Chrome checks (Selenium) were run by hand, outside the
suite. They covered the Setup and write flow; a real 4-rank PAOFLOW run started from the Runs tab and opened in
Results; light and dark themes; the 3D Fermi surface. No console errors.

**Departures from the plan:**

- **Logs and live mode poll** the server (1.5 s for logs, 5 s for rescans) instead of using Server-Sent Events.
  This is simpler with the standard-library server and needs no streaming code path.
- **Ordered series are capped at five.** The palette validator requires an adjacent ΔL of at least 0.06, which
  allows five steps of one hue. The temperature selectors show at most five temperatures at once.
- **The results readers are new modules.** `GPAO` and `read_pao_output` are unchanged, so there is still a
  matplotlib route and a results route (§8 planned to share them). The results readers accept every format seen
  in the references: dense and sparse band layouts, `#`-header tables, CSV, blank-line blocks, and the old BXSF
  headers that give an absolute E_F.
- **Extra D4 defaults.** `ibrav` also defaults from the save, found in the end-to-end run: the generated `main.py`
  failed with "IBRAV = 0 not permitted" until then. The electron–phonon defaults come from the `pw.x` template.
- **GUI-only advanced questions** (`cli=False`) cover the builder keys that no CLI question set:
  - `coupled_emin` and `coupled_components`;
  - the band-plot window of ACBN0;
  - `tr2_ph`;
  - the SSCHA minimiser settings.

**Second round (7 November 2026)** of the open items:

- **Band-path editor.** `gen/band_paths.py` reads the default path of the lattice
  (`kpnts_interpolation_mesh._getHighSymPoints` from the save's `ibrav` and cell, or the 2D tables). It also
  validates a custom path and builds the preview: the path inside the first Brillouin zone (the
  `fermi_plotter.brillouin_zone` geometry), served by `/api/gen/bandpath`.
  - The questions are GUI-only: `band_path` (basic, `widget='bandpath'`) and `high_sym` (advanced point table).
  - `assemble_run` fills in the side the user left untouched.
  - `build_run_script` writes `BAND_PATH` and `HIGH_SYM`. An unchanged path writes the CLI's `main.py`.
- **Field help from docstrings.** `gen/questions/docs.py` maps 102 questions (`FIELD_DOCS`, or `Question.doc`)
  to the parameter of the PAOFLOW method they feed, or to the `system.py` comment dictionaries. Both docstring
  styles are parsed (NumPy `Parameters` / `Other Parameters`, and `Arguments:`), and reST is reduced to plain text.
  `test_field_docs.py` requires every reference to resolve.
- **Clickable EDC/MDC cuts.** These are built in the browser from the heatmap data, for every map (ARPES,
  spectral functions, Berry slices), with crosshairs on the map.
- **Stacked PDOS.** A *display* option draws the PDOS as stacked areas: `role@alpha` colours, split by a
  surface-coloured line, with the total as a line. The matplotlib export uses `stackplot`.
- **`GPAO` on the shared parser.** `read_dos_PAO`, `read_bands_PAO` and `read_transport_PAO` now parse through
  `results.columns.read_columns`, so `GPAO` and the Results tab read files the same way.
  - The arrays are identical to the frozen copy (`tests/unit/inputs/legacy_read_pao_output.py`) on all 114
    reference files checked.
  - `#` headers are accepted.
  - The Hall and Nernst layouts, and the sparse k-path layout, raise clear errors instead of misreading.
- **Browser tests in the suite.** `tests/integration/gui/test_browser.py` (integration and slow) runs three tests:
  Setup writes the resolved files and the zone preview draws; Results stacked PDOS and map cuts; a run from the
  Runs tab. All check that the page logs no error.
- **JS checks.** The pre-commit hook `js-check` and `test_static_js.py` (`.github/lint/check_js.py`) parse every
  module with `node --check` and refuse HTML-string insertion, `eval` and `new Function`.

- **Biome** (installed by the user, 2.6.6) lints and formats the front end:
  - the configuration is `biome.json`: recommended rules, single quotes, line width 100;
  - the pre-commit hook `biome` pins `@biomejs/biome@2.6.6`;
  - the first pass reformatted the JavaScript and the CSS, and removed: assignments inside conditions (the code-view
    tokenizer is now table driven), `forEach` callbacks that return values, unused bindings, a global `isNaN`, and
    descending-specificity orders in the CSS (the `!important` of `[hidden]` is kept on purpose);
  - the browser tests and the end-to-end flows passed again after the reformat.

Nothing from the open list remains.

**Read-only folders (8 November 2026, user request).** `Sandbox(root, read_roots=...)` splits the path rule in
two:

- `resolve`, for writes, the working folder and runs: inside `--root` only;
- `resolve_readable`, for listings, previews, the QE-input and basis pseudopotentials, results, experiment files,
  and the path answers the server reads (`savedir`, `pw_template`): inside the root or a read-only folder.

The read-only folders are the `--allow-read DIR` options plus PAOFLOW's own `PSEUDOS/` and `BASIS/`. They are
fixed at start-up, so no request can add one, and symbolic links that leave them are refused. Listings report
their `place` (`writable` false), and the folder picker shows them as places. Tests: `test_read_roots.py`, plus a
browser test that picks `PSEUDOS` in the QE-input tool.

## 1. Goals and non-goals

Goals:

1. **G1: setup GUI.** Every option of the `paoflow-gen` CLI in a browser form: the `run`, `acbn0`, `phonon` (with
   Raman), `elphon` and `sscha` workflows. The two companion generators, `paoflow-gen-qe` (QE input from AFLOW or
   C2DB) and `paoflow-genbasis-ps` (pseudo-atom basis tables), are included. For the same answers, the GUI writes
   files that are **byte-identical** to the ones the CLI writes.
2. **G2: results GUI.** One browser viewer for the outputs PAOFLOW writes. It discovers the datasets in a folder,
   plots them interactively, overlays runs and experimental data, and exports PNG, SVG, PDF, CSV, a Python
   snippet, or a self-contained HTML report.
3. **G3: runs where PAOFLOW runs.** The GUIs work on a laptop, a workstation, an HPC login node through an SSH
   tunnel, and offline compute nodes. They need no node.js, no build step, no new core dependency and no internet
   access (except for the AFLOW and C2DB fetch, which needs the network anyway).
4. **G4: one source of truth.** The CLI and the GUI read the same question definitions and call the same script
   builders. The matplotlib route (`GPAO`, `plot.py`) and the browser route read the outputs through the same
   readers. Nothing is reimplemented in JavaScript.

Non-goals:

- a general `pw.x` input editor;
- a multi-user web service or cloud hosting;
- job orchestration across machines (the run manager only drives the machine the server runs on: local processes,
  or its Slurm);
- VASP-driven setup (`paoflow-gen` is QE-only today).

## 2. What exists (findings)

| component | where | what the GUI can use |
|---|---|---|
| prompt helpers | `gen/paoflow_driver.py` `ask`, `ask_int`, `ask_float`, `ask_yes_no`, `ask_choice` | prompt kinds to model |
| prompts → `cfg` | `collect_common`, `collect_run`, `collect_nonlinear`, `collect_phonon`, `collect_acbn0` (`paoflow_driver.py:3034-3363`); `collect_elphon_system` and `_collect_{superconductivity,optics,transport,arpes,thermal}` (`elphon_system.py:511-960`); `collect_sscha` (`sscha_driver.py:445`) | to be expressed as data (section 4) |
| `cfg` → text | `build_run_script`, `build_plot_script`, `build_acbn0_script`, `build_acbn0_plot_script`, `build_phonon_script`, `build_phonon_plot_script`, `build_raman_script`, `build_raman_plot_script`, `build_sscha_script`; `render_system_py`, `render_run_qe`, `qe_inputs`, `workflow_files` (elphon) | **pure functions**: previews come free |
| file writing | `main()` (`paoflow_driver.py:3365`), `write_system_folder` (`elphon_system.py:440`) | split into "file map" plus "writer" |
| next steps | printed by `main_elphon` (`elphon_system.py:963`); docstrings of the others | to become a structured run plan (section 7) |
| artefact detection | `detect_savedir`, `detect_upfs`, `detect_prefix`, `detect_pw_input`, `detect_v_cutoff`, `_read_namelist_value`, `epw_inputs.species_masses` | the Workspace step |
| QE input generator | `gen/qe_input/cli.py` (argparse only) → `source.fetch(...)` → `build_qe_input(record, ...)` | already non-interactive |
| basis generator | `basis_gen/cli.py` (argparse: `--pseudo` or `--pseudo-dir`, `--out`, `--preset`, `--shells`, `--r-box`, `--n-points`) | a job (it computes) |
| plotting | `GPAO.py` (about 30 `plot_*` methods: read file, then call `graphics/plot_functions.py`); `inputs/read_pao_output.py` (5 readers); the generated `plot.py` holds the de facto discovery logic (`_one`, `_many` with the prefix fallback, `_spin_channels`) | readers to share; discovery to move into the library |
| Fermi surface | `gen/fermi_plotter.py`: the geometry (`read_fermi_bxsf`, `brillouin_zone`, `fold_into_bz`, `fermi_velocity_field`, `tile_mesh`, `field_direction`) is separate from the Mayavi drawing | WebGL mesh without Mayavi |
| experiment data | `agentic/thermo_optics/datasets.py` (`read_dataset` with provenance header), `metrics.py` (`curve_deviation`) | experiment overlay in the viewer |
| fixtures | `examples/*/Reference/` (for example `qe_examples/example01/Reference`: bands, DOS, 18 PDOS, σ, S, κ, PF), elphon example folders | discovery and reader tests |
| tests | `tests/unit/gen/test_paoflow_driver.py` (feeds `input` via `monkeypatch`), `test_elphon_system.py`, `test_sscha_driver.py` | the parity harness of section 4.4 |
| environment | the `work` env has `plotly` 6.0.1, which ships `plotly/package_data/plotly.min.js` (4.6 MB) | Plotly.js offline, nothing vendored in the repo |

The prompts use these features, and the question model must cover each one:

- free text, int, float, yes/no and single choice;
- multi-select (the numbered property menu, the elphon workflows);
- comma lists (floats, ints, strings, tensor components) and isotropic grids;
- repeated rows (the ACBN0 Hubbard manifolds);
- validation with a re-prompt (the q-grid divides the k-grid, `nk % nq == 0`, unknown workflow names);
- defaults derived from earlier answers (`NK_DENSE = 8·kg`, a smearing that depends on the adaptive prefactor, a
  photon window that depends on the emissivity choice) and from files (the `detect_*` functions, the species masses
  and `pseudo_dir` of the `pw.x` template);
- forced overrides with a printed note (eACBN0 switches the projector to `projwfc`; the EPW source needs the
  explicit k list);
- conditional sections (non-linear, coupled response, Raman, QHA, Slurm);
- the restricted-`eval` laser list (`parse_laser_list`).

## 3. Architecture

```
browser: vanilla ES modules + Plotly.js, no build step
   │  HTTP on 127.0.0.1 (SSH tunnel when remote), token cookie
   ▼
PAOFLOW.gui.server            stdlib ThreadingHTTPServer: JSON endpoints + Server-Sent Events
   ├── /api/workspace/*   →  PAOFLOW.gui.workspace     browse (sandboxed), detect QE artefacts, system summary
   ├── /api/gen/*         →  PAOFLOW.gen.questions     resolve form → gen builders (preview, write)
   ├── /api/jobs/*        →  PAOFLOW.gui.jobs          local runs, live logs (phase 5)
   └── /api/results/*     →  PAOFLOW.results           discover, read, figure spec, export
CLI (unchanged for the user): paoflow-gen → PAOFLOW.gen.questions prompt runner → the same builders
```

**One server, three tabs: Setup, Runs and Results.** They share the workspace, the security, the styles and the
form renderer. After a run, the Results tab opens directly on the run's `output/`.

Entry points:

- `paoflow-studio [--root DIR] [--port 8765] [--no-browser] [--read-only] [--max-ranks N]` serves all three tabs;
- `paoflow-studio plot DIR` opens the Results tab;
- `paoflow-studio report DIR -o report.html` is headless: no server, a static file;
- `paoflow-gen --gui` is an alias that opens the Setup tab, for discoverability.

### 3.1 Technology choices

| choice | decision | alternatives considered |
|---|---|---|
| server | stdlib `http.server.ThreadingHTTPServer` plus a small router of about 300 lines. SSE for logs and live refresh | starlette or uvicorn (installed in `work`, but a new dependency for little gain); Flask |
| front end | vanilla ES modules, one `index.html`, CSS variables (light and dark), no framework, no npm | React or Vue (needs a toolchain); Panel, Streamlit, Dash or NiceGUI (heavy, framework lock-in, hard to keep byte parity with the CLI) |
| plots | Plotly.js (2D, heatmaps, `mesh3d`, `cone`; typed-array JSON) served from the installed `plotly` package | Bokeh; three.js (3D only); vendoring plotly.min.js (4.6 MB in git) |
| figure logic | **server-side Python** builds Plotly JSON specs as plain dicts. The browser only renders and restyles | JS renderers per kind (duplicate physics logic, untested in pytest) |
| notebooks | the figure specs load into `plotly.graph_objects.Figure(spec)`, so Jupyter users get the same plots | Voila (needs a Jupyter server) |

Dependencies:

- **No new core dependency.**
- New extra `gui = ["plotly>=5.18"]`. It is used for the bundled JS, and for the figure objects in notebooks.
- The 3D Fermi surface needs `scikit-image`, which the `fermisurface` and `pyskeaf` extras already list.
- PDF export uses matplotlib (the `graphics` extra).
- If an optional package is missing, the GUI disables that feature and names the extra that enables it.

### 3.2 Security and deployment

The threat model has five parts:

1. On a multi-user login node, every user can reach `127.0.0.1:8765`.
2. A web page in the user's browser can attempt DNS rebinding or cross-site POSTs.
3. Path traversal outside the project.
4. Command injection through run parameters.
5. Script injection through file names or labels.

The controls, in the same order:

1. Bind to `127.0.0.1` only. A random token (`secrets.token_urlsafe(32)`) is printed in the URL once, then becomes
   an `HttpOnly; SameSite=Strict` cookie. Every request without it is rejected.
2. The `Host` header must be `127.0.0.1:<port>` or `localhost:<port>`, and the `Origin` must match on POST.
3. Every path resolves under `--root` (`Path.resolve().is_relative_to(root)`). Writes happen only through the
   workflow writers into the chosen working directory. `--read-only` turns off writes and runs, for viewing-only
   sessions.
4. Runs come from a whitelist built from the generated files (section 7). They take `argv` lists (never
   `shell=True`) and bounded integer rank counts. User input is never evaluated: an AST parser (numbers, lists,
   `range`, a single comprehension) replaces the `eval` in `parse_laser_list`, and the CLI uses it too.
5. File content and labels are inserted as text (`textContent`), never as HTML. A `Content-Security-Policy:
   default-src 'self'` header is sent (check whether the WebGL traces of Plotly need `'unsafe-eval'`).

For remote use the server prints a hint such as `ssh -N -L 8765:127.0.0.1:8765 <host>` and runs the same way on
Snellius, LS6, aflowlib or gb10. Where the centre forbids long processes on login nodes, run the server on a
compute node (`srun --pty`) and tunnel twice. The documentation will cover this.

## 4. Shared foundation: declarative questions (`PAOFLOW.gen.questions`)

This is the key refactor. The prompts become data that two front ends read: the CLI prompt runner and the web form.
The builders do not change.

### 4.1 Model

```python
@dataclass(frozen=True)
class Question:
    key: str                                   # cfg key, e.g. 'nlo_eta'
    prompt: str                                # CLI text, verbatim from today
    kind: str                                  # text | int | float | bool | choice | multichoice
                                               # | float_list | int_list | str_list | grid | rows | path
    default: object | Callable[[Answers, Context], object] = None
    choices: Sequence[str] | Callable[[Answers, Context], Sequence[str]] | None = None
    when: Callable[[Answers, Context], bool] | None = None          # asked / shown only if True
    validate: Callable[[object, Answers], str | None] | None = None # error text → CLI re-prompts
    to_cfg: Callable[[object, Answers], dict] | None = None         # e.g. (Tmin, Tmax, n) → me_temps
    unit: str | None = None
    help: str | None = None                    # long help; may be pulled from the method docstring
    section: str = ''
    level: str = 'basic'                       # 'advanced' is collapsed in the GUI; the CLI asks everything
    cli: bool = True                           # False: GUI-only refinements


@dataclass(frozen=True)
class Note:                                    # printed or shown when `when` holds
    text: str
    when: Callable[[Answers, Context], bool] | None = None
    effect: Callable[[Answers], dict] | None = None   # forced override, e.g. projector='projwfc'


@dataclass(frozen=True)
class Workflow:
    name: str                                  # run | acbn0 | phonon | elphon | sscha | qe_input | basis
    title: str
    items: Sequence[Question | Note]           # CLI order
    files: Callable[[dict], dict[str, FileSpec]]   # cfg → {relative path: text | template copy}
    run_plan: Callable[[dict], list[RunStep]]      # section 7
```

`Context` holds the working directory and the detections: savedir, UPFs, prefix, `pw.x` input, V cutoff, masses and
`pseudo_dir`. A GUI-only system summary is read from `data-file-schema.xml` and the UPF headers: formula, cell, nat,
nspin, noncollinear or SOC, nbnd, nelec, E_F, occupations, k-grid and pseudopotential types.

### 4.2 Two front ends

- **CLI prompt runner** (`questions/prompt.py`). It walks `items` in order, skips any whose `when` is false, calls
  the existing `ask_*` helpers with the same text and defaults, re-prompts on a `validate` error with the same
  message, and prints the `Note`s. `collect_run(common)` and the others become one-line wrappers, so imports and
  tests keep working.
- **Form resolver** (`questions/resolve.py`). `resolve(workflow, answers, context) -> ResolvedForm` returns, for each
  item, whether it is visible, its effective default, its value, its error and its choices. When the form is
  complete it also returns the `cfg` and the file previews. The browser calls it on every edit (debounced, about
  250 ms). The builders only assemble strings, so a round trip takes milliseconds. All predicates and defaults stay
  in Python: no JS duplicate.

### 4.3 Splitting the writers

Three writers change. File writing is separate from file content everywhere, so every workflow can be previewed
before anything touches the disk.

- `paoflow_driver.main` gets a workflow-to-files table:
  - `run` → `main.py`, `plot.py`
  - `acbn0` → `main.acbn0.py`, `plot.acbn0.py`
  - `phonon` → `main.phonon.py`, `plot.phonon.py`, `main.raman.py`, `plot.raman.py`
  - `sscha` → `main.sscha.py`
- `write_system_folder` becomes `system_folder_files(cfg) -> dict[str, FileSpec]` plus a writer. `FileSpec` holds
  either text or a template copy, and the writer keeps the executable bit on `.sh` files.
- `qe_input/cli.main` becomes `generate_qe_input(identifier, pseudo_dir, ...) -> (record, text)` plus a thin CLI.

### 4.4 Parity harness (the acceptance test of phase 1)

- Freeze today's `collect_*` functions into `tests/unit/gen/legacy_collect.py`.
- For each workflow, write an answer matrix that covers every branch: SOC, 2D, molecule NLO, coupled response, the
  Raman flavours, QHA 3 and 5 volumes, eACBN0 forcing `projwfc`, EPW and DFPT sources, metal and semiconductor
  elphon, and the three SSCHA launchers.
- Feed each answer list through `monkeypatch` to both paths. Assert that the `cfg` dicts are equal and that every
  generated file is byte-identical.
- A coverage test checks that every key a builder reads (`cfg[...]`, `cfg.get(...)`) is produced by the schema, so a
  new builder option without a question fails CI.

### 4.5 Smart defaults (after parity, a separate commit)

The system summary can propose defaults:

- `SPIN_ORBIT` from `lspinorb`;
- "metal" from the occupations;
- `nelec`;
- 2D from the vacuum along c (or `assume_isolated`);
- the `epw` coupling source when a pseudopotential is US or PAW (the native DFPT vertex is norm-conserving only);
- `nbnd` sanity against the extended-basis orbital count.

Because the questions are shared, the interactive CLI gets these defaults too (decision D4). The scripts generated
for the same answers do not change.

## 5. Setup GUI

```
┌ PAOFLOW   Setup · Runs · Results ────────── ~/Work/Si   [Si · fcc · 2 atoms · NC · no SOC · gap 0.6 eV] ┐
│ Steps              │ Properties run                                  │ Preview  main.py │ plot.py     │
│ ✓ Workspace        │ ┌ Properties ─────────────────────────────────┐ │  38 NPOOL = 1                │
│ ✓ Workflow         │ │ [x] Band structure      [x] DOS / PDOS      │ │  39 SMEARING = 'gauss'       │
│ ● Common           │ │ [ ] Boltzmann transport [x] Optical (ext.)  │ │  …                           │
│ ○ Properties       │ │ [ ] Spin Hall  (needs SOC: save has none) ⚠ │ │  52 EMIN, EMAX, NE = -8.0, … │
│ ○ Energy grid      │ └─────────────────────────────────────────────┘ │ diff vs existing: +12 −3     │
│ ○ Review & write   │ EMIN [-8.0] eV  EMAX [4.0] eV  NE [1000]  ▸ Adv. │            [ Write files ]  │
└────────────────────┴─────────────────────────────────────────────────┴──────────────────────────────┘
```

1. **Workspace.** A directory picker sandboxed to `--root` shows what was found: a `*.save`, UPFs, `scf.in`,
   `*.scf.in`, `nscf.in`, a `*.dyn0`, a V-cutoff comment, an existing `main*.py` or `system.py`. A system-summary
   card follows. If no save is found, the GUI offers "QE input from a database" or "electron-phonon from a `pw.x`
   input".
2. **Workflow cards.**
   - Workflows: Properties (`run`), ACBN0/eACBN0, Harmonic phonons (with Raman), Electron–phonon system, SSCHA.
   - Tools: QE input from AFLOW/C2DB, Pseudo-atom basis tables.
3. **Form.** Sections come from the schema, with "advanced" fields collapsed, units next to the inputs, inline
   validation, and notes that explain each implication:
   - emissivity implies the optical run;
   - the optical and NLO runs use the extended basis;
   - spin texture and spin Hall need a SOC save;
   - eACBN0 forces `projwfc`;
   - the EPW source needs the explicit k list;
   - the coupled response is for 3D crystals with internal projections only.

   Special widgets:
   - a property card grid (multi-select with one-line descriptions);
   - a grid widget (isotropic, as the CLI);
   - temperature lists with a "min, max, n" generator and a live preview of the values;
   - a Hubbard-manifold table: species from the detected UPFs, shells read from the UPF, initial U;
   - a band-path editor: the default path for the `ibrav`, the in-plane path for 2D, a custom path with labels;
     phase 4 adds a Brillouin-zone preview from `fermi_plotter.brillouin_zone`;
   - a laser-list builder (a list or a range, no `eval`);
   - a Slurm block (partition, nodes, tasks, wall time, account, environment line, python);
   - field help taken from the NumPy docstrings of the PAOFLOW methods the field feeds, such as
     `dielectric_tensor` and `nonlinear_optics`, so the help stays as current as the API.
4. **Preview and write.**
   - A file tree of every output: for elphon, `system.py`, `qe/phonon/{scf.in, ph.in}`, `qe/nscf/nscf.in`,
     `run_qe.sh`, `kmesh.py` and the driver copies.
   - A code view with syntax highlighting and a diff against existing files.
   - Overwrite only after confirmation, with an optional timestamped `.bak` (this replaces `--force`).
   - A "Python equivalent" snippet (`build_run_script({...})`) for scripted, reproducible generation. No answers
     file is written, consistent with the no-config-file rule.
5. **Next steps.** After writing, the run plan of section 7 is shown with copy buttons, and with Run buttons once
   phase 5 lands.

The **QE input** page takes:

- an identifier (an AFLOW URL or AUID, or a C2DB UID), with the source detected automatically;
- the pseudopotential directory;
- toggles for SOC, metallic (auto, metallic, insulating), magnetic (auto, magnetic, non-magnetic), the k-points,
  `ibrav` (auto or 0), `degauss`, `nbnd`, `symprec` and `assume_isolated_2d`.

It then fetches the entry (with network errors shown), previews `scf.in` and the structure summary (lattice, formula,
suggested V cutoff), and writes the file.

The **basis** page takes the arguments of `paoflow-genbasis-ps` and runs the generation as a job. The Results tab
then plots the radial tables.

## 6. Mapping of the `paoflow-gen` workflows

| workflow | sections in the GUI | files |
|---|---|---|
| `run` | common (save, prefix, UPFs, basis directory, `ibrav`, 2D, projector) · properties · basis configuration · `npool`, smearing, SOC · band path (`NK`) · energy grid (`EMIN`, `EMAX`, `NE`), PDOS · double grid (`NFFT`) · non-linear (molecule, `nk`, photon window, `eta`, scissor, components, responses, dipoles, SHG terms, coupling) · coupled response (`nk`, kernels, window, SHG energies, `eta`) · plot script | `main.py`, `plot.py` |
| `acbn0` | common · eACBN0 V · projection scheme · convergence · `NK` · Gaussian tolerance · Hubbard table · V cutoff and initial V · launch commands (pw.x, python, Hartree, QE and python paths) | `main.acbn0.py`, `plot.acbn0.py` |
| `phonon` | common · supercell, displacement, q-mesh, units · thermal · `pp_dir`, Hubbard injection · Born (dfpt or field) · vibrational ε(ω) and γ, reststrahlen emissivity and T · QHA (volumes, strain, EOS, T grid, pressure) · pw.x launch · plot · Raman (static, resonance or all; δ, nbnd, `npool`, smearing, `nfft`, static window, T, FWHM, lasers, lifetime) | `main.phonon.py`, `plot.phonon.py`, `main.raman.py`, `plot.raman.py` |
| `elphon` | `pw.x` template · prefix, name, `pseudo_dir` · coupling source · coarse k and q (with the divisibility check) · irreducible wedge · `nbnd`, exclude bands · masses, `nelec`, metal · basis, `pthr` · workflows (multi-select, defaults depend on metal) · superconductivity · optics · transport (resistivity or mobility) · ARPES · thermal · optics nscf | the system folder |
| `sscha` | save, prefix, `pp_dir` · temperatures, supercell · harmonic matrices (detected `*.dyn0`), q-grid · k-grid · electrons (parent, ionic or fixed, T) · population sizes, Hessian size · launcher (local, Slurm or manual) with the Slurm block | `main.sscha.py` |
| QE input | identifier, source, pseudo directory, overrides | `<compound>.scf.in` |
| basis | pseudo or pseudo directory, out, preset or shells, `r_box`, `n_points`, overwrite | `BASIS_PS/<El>/*.dat` (job) |

A future workflow could add a generator for Green's-function (Landauer) transport (`Transport.py`); it is not in
`paoflow-gen` today.

## 7. Runs tab (phase 5)

**The run plan.** Each workflow gets `run_plan(cfg) -> list[RunStep]`. A `RunStep` holds a label, `argv`, `cwd`,
whether it runs under MPI, an optional phase argument, the steps it depends on, and the files it expects to produce.

- The CLI prints the plan as "Next steps" (the text `main_elphon` prints today comes from the same data).
- The GUI shows it with copy and Run buttons.

The plans:

- `run`: `mpirun -np N python main.py`, then `python plot.py`.
- `acbn0`: `python main.acbn0.py`.
- `phonon`: `python main.phonon.py generate | run | born | qha | analyse`, then `main.raman.py`.
- `sscha`: `python main.sscha.py harmonic | run | status`.
- `elphon`, in order: `bash qe/run_qe.sh`, then `paoflow-genbasis-ps`, then one driver per workflow
  (`eliashberg.py` → `me.py` → `me_aniso.py` → `plot_me.py`, `absorption.py`, `mobility.py`, `arpes.py`, `zg.py
  --write`, `--spectral`, ...).

**Local execution.**

- Each run is a `subprocess.Popen` with an `argv` list, `start_new_session=True`, `nice` and
  `OMP/OPENBLAS/MKL/VECLIB_NUM_THREADS=1`.
- The rank count is capped by `--max-ranks` (default: `utils.parallel_resources.core_budget()`).
- Logs go to `<workdir>/.paoflow-studio/jobs/<id>/run.log`, next to a job record (`argv`, `cwd`, pid, start, end,
  return code). The record is GUI state only; no driver reads it.
- A server restart reattaches running jobs by pid and start time.
- Cancel sends SIGTERM to the process group.

**Slurm:** out of scope (D2). The generated SSCHA driver keeps its own Slurm launcher, which the GUI can start as a
local process.

**Live view.**

- SSE streams the log (tail `-f`).
- A step checklist is parsed from the PAOFLOW timing lines and the `run_qe.sh` steps.
- An output watcher refreshes the Results tab while a run writes files.

## 8. Results data layer (`PAOFLOW.results`, GUI-independent)

The layer is usable from scripts, tests, notebooks and the agentic adapters.

- **`catalogue.py`.** A registry of `DatasetKind` entries. Each holds:
  - the name, the title and the category (Electronic structure, Transport, Topology and Hall, Optics, Non-linear
    optics, Phonons, Electron–phonon, Green's-function transport, ACBN0, SSCHA, Quantum oscillations, Models and
    unfolding);
  - prefix-tolerant filename patterns;
  - the grouping keys (spin channel, tensor component, temperature, kernel level, label);
  - the reader, the default figure, the option schema (the same `Question` model as section 4) and the matching
    GPAO call.
- **`discovery.py`.** `discover(root, max_depth=4) -> list[DatasetRef]` recognises these layouts:
  - a dense run (`output/`);
  - an elphon system folder (`system.py`, `<workflow>/output/`, `figures/`);
  - a phonon folder;
  - SSCHA (`output/sscha_T<T>/state.json`);
  - Green's-function transport;
  - an example `Reference/`;
  - an agentic project.

  Results are cached on (path, mtime, size). The `_one`, `_many` and `_spin_channels` logic of `plot.py` moves here,
  and the generated `plot.py` keeps working.
- **`readers/`.** Typed containers, with units carried in metadata:
  - `Curves` (x and named y arrays);
  - `BandStructure` (k distance, energies of shape `(nk, nbnd)`, ticks and labels, optional weights);
  - `TensorSpectrum` (energy, components, temperatures);
  - `VolumeGrid` (BXSF);
  - `Map2D` (ARPES, unfolding);
  - `Table` (λ, ω_log, T_c, U and V).

  The readers wrap the existing ones (`read_pao_output`, `read_fermi_bxsf`, the elphon loaders, thermo-optics
  `read_dataset`). **GPAO is refactored to call the same readers**, with identical figures, so matplotlib and the
  browser cannot disagree on parsing.
- **`figures/`.** `figure_spec(kind, data, options) -> dict` returns a Plotly JSON spec as plain dicts. The options
  cover:
  - the energy window and reference (E_F, VBM, raw);
  - spin overlay;
  - the tensor component (3×3 or 3×3×3 picker), or the trace divided by 3;
  - the temperature index;
  - normalisation;
  - log axes;
  - unit conversions: S/m and (Ω·cm)⁻¹; eV, nm and cm⁻¹ for optics; THz, cm⁻¹ and meV for phonons.

  Arrays are sent as typed binary (`{dtype, bdata}`), which plotly.js 2.28 and later accept.
- **`export.py`.** Exports:
  - CSV of the plotted traces;
  - a PDF through the GPAO and `plot_functions` route, with the same look as `plot.py`;
  - a Python snippet that reproduces the view with GPAO;
  - **`report.py`**: a self-contained HTML report (figure JSON plus plotly.min.js inlined, offline, one file) that
    can be mirrored to aflowlib or attached to a `REPORT.md`.

## 9. Results catalogue

This catalogue was compiled from the writers on 7 November 2026, and each row becomes one `DatasetKind`. Shorthand
used in the tables:

- "2col" means the `DataController.write_file_row_col` layout: lines of x and y, with no header.
- "2col+H" is the same layout with one `# units` header line.
- `{s}` is the spin channel and `{ab}` a tensor component.
- Electronic energies are in eV relative to E_F = 0.

### 9.1 Catalogue

**Dense electronic properties** (in `outputdir`):

| pattern | format | reader today | GPAO today |
|---|---|---|---|
| `bands_{s}.dat` + `kpath_points.txt` | k index, then one column per band; `label npts` lines | `read_bands_PAO`, `read_band_path_PAO` | `plot_bands`, `plot_dos_beside_bands` |
| `site-projected-bands_{s}.dat` | k, E, weight | `read_site_projected` | `plot_weighted_bands` |
| `dos[dk]_{s}.dat`, `{El}{n}_{orb}_pdos[dk]_{s}.dat`, `pdos[dk]_sum_{s}.dat` | 2col | `read_dos_PAO` | `plot_dos`, `plot_pdos` |
| `{sigma,Seebeck,kappa,PF}[smear]_{s}.dat` | T, E, then 6 components (σ/τ, S, κ/τ, PF/τ) | `read_transport_PAO` | σ, S (none for κ, PF) |
| `hall_trace[_smear]_{s}.dat` (3 columns), `hall…` (9 components), `nernst…` (27) | text | none (`read_transport_PAO` misreads them) | none |
| `doping_{n,p}{conc}.dat`, `cond_{ij}_{s}.dat` | 2col | `read_dos_PAO` | none |
| `{shc,ohc}Ef_{s}_{ij}.dat`, `ahcEf_{ij}.dat`; AC variants `SCD{r,i}`, `OCD{r,i}`, `MCD{r,i}` | 2col (S/cm, (ħ/e) S/cm) | `read_dos_PAO` | `plot_shc`, `plot_ohc`, `plot_ahc` (DC only) |
| `{Spin_,Orbital_,}Berry_{…}.bxsf` | BXSF | `pyskeaf.io_bxsf.read_bxsf` | none |
| `Omega[j]_{s}_{ij}.dat`, `Omegaj_{Spin,Orbital}_…[_bands_{s}].dat`, `velocity_{i}_{s}.dat` | 2col, or the bands layout | `read_dos_PAO`, `read_bands_PAO` | `plot_berry`, `plot_berry_under_bands` |
| optics `{epsi,epsr,ieps,sigmar,sigmai}_{ab}[_{s}].dat`, `{eels,nref,kref,alpha,refl}_{aa}…`, `jdos[_{s}].dat` | 2col (ε, S/m, 1/m, R) | `read_dos_PAO` | `plot_dielectric`, `plot_optical`, `optical_color` |
| emissivity `refl_th{deg}_…`, `emis_th{deg}_…`, `emish_{aa}…`, `emist_{aa}…` (x = T) | 2col | `read_dos_PAO` | `plot_optical` |
| `FermiSurf_{s}.bxsf`, `Fermi_surf_band_{n}.bxsf`, `Fermi_surf_band_{n}_{s}.npz` | BXSF (E_F = 0), npz | `fermi_plotter.read_fermi_bxsf` | none (Mayavi CLI) |
| `spin-texture-bands.dat` (k, E, S_x, S_y, S_z, blocks per band) or `spin_text_band_{ib}.npz`; same for `orbital-…` | text, npz | none | none |
| `effective_masses_{s}.dat`, `effmass_{ij}_{s}.dat` | text (two formats) | none | none |
| `Z2.dim`, `weyl_points.dat`, Berry-phase file, `ipr.npy` | text, npy | none | none |
| Rashba–Edelstein `[orbital_]kai.dat`, `…current.dat`, `…Ekai_{ij}.dat`, `{spin,orbital}_reeEf_{i}{s}.dat` | text, 2col | 2col files only | none |
| linear response `SHC_{EVEN,ODD}_…`, `REE_…`, `AHC_{ci}…`, `Cond_{ci}…` | 2col+H (units in the header) | none (`read_dos_PAO` fails on `#`) | none |
| NLO `{shift,shiftm,injectc,injectm,shgr,shgi,shga}_{abc}[_{s}].dat`, `shgt{r,i}_{term}_{abc}.dat` | 2col (μA/V², A V⁻² s⁻¹, pm/V; sheet units in 2D) | `nonlinear_plots.component_files` | `plot_nonlinear`, `plot_shg_terms` |
| molecule `beta_T_{r,i}_{abc}.dat`, `dmudt_…`, `beta_{molecular,hrs,depolarization,coupled}.dat` | 2col and text | none | `plot_molecular_beta` |
| coupled response `ceps{r,i}_{tag}_{bb}.dat`, `cshg{r,i,a}_{tag}_{abc}.dat` (tag = ipa, lf, alda, lrc, boot), `coupled_response.dat` | 2col | none | `plot_coupled_response` |

**Phonons** (`#` header, default name `phonon`):

| pattern | GPAO today |
|---|---|
| `{f}_band.dat` + `{f}_band.labels` (tick positions as distances), `{f}_dos.dat` | `plot_phonons` |
| `{f}_thermal.dat` | `plot_phonon_thermal` |
| `{f}_ir_{modes,spectrum}.dat`, `{f}_raman_{modes,spectrum}.dat` | `plot_ir_spectrum`, `plot_raman_spectrum` |
| `qha_{ev,volume,thermal_expansion,bulk_modulus,gibbs,heat_capacity,gruneisen}.dat`, `qha_gruneisen_band.{dat,labels}` | `plot_qha`, `plot_gruneisen_band` |
| `vibdielectric/{epsr,epsi,eels,refl}_{ab}.dat` + emissivity files, `{f}_vibdielectric_static.dat` | `plot_optical(path=…)` |
| elphon `phonons/output`: the same layout plus `<prefix>.fc` | `plot_phonons` |

**Electron–phonon** (`<workflow>/output/`, `#` header text plus npz):

| folder | files | GPAO today |
|---|---|---|
| `superconductivity/output` | `alpha2F.dat` (λ, ω_log and T_c in the header), `alpha2F_tr.dat`, `eliashberg.npz`, `fs_coupling.npz` | `plot_metal_transport` |
| `…/me/` | `<prefix>.{imag,pade,acon,qdos}_iso_<T>`, `gap_vs_T.dat`, `max_eigenvalue.dat`, `migdal_eliashberg.npz` | `plot_migdal_eliashberg` |
| `…/me_aniso/` | `<prefix>.lambda_FS`, `.lambda_k_pairs`, `.imag_aniso…_<T>`, `.pade_aniso…_<T>`, `gap_vs_T_aniso.dat`, `migdal_eliashberg_aniso.npz` | `plot_migdal_eliashberg_aniso` |
| `transport/output` | semiconductors: `mobility.dat` (11 columns), `rates_<T>K.dat`, `scattering_rates.npz`, `carrier_scattering.npz`; metals: `resistivity.dat` | `plot_carrier_mobility`, `plot_metal_transport` |
| `optics/output` | `epsilon2_{indabs[_lorenz],dirabs}_<T>K.dat`, `alpha_<T>K.dat` (cm⁻¹, one column per η), `absorption.npz`; emissivity `eps_`, `emish_`, `emis_th<deg>_<T>K.dat`, `emist.dat`, `emissivity.npz` | `plot_phonon_assisted_absorption`, `plot_thermal_emissivity` |
| `arpes/output` | `arpes_bare.npz`, `arpes_<kind>_<T>K.npz` (kpoints, w, intensity, …), `sigma_{isotropic,anisotropic}_<T>K.dat`, `self_energy_fan_migdal.npz` | `plot_arpes` (takes an object; **no npz loader**) |
| `thermal/output` | `thermal_spectral_T<T>K[_seed<s>].npz`, `thermal_spectral_reference.npz`, `thermal_summary.{json,dat}`, `lambda_summary.dat` | `load_thermal_spectral`, `load_thermal_summary` (no GPAO) |
| `coupling/output` | `g_path.dat`, `quadrupole.fmt` | none |

**Other areas:**

| area | files | reader / GPAO today |
|---|---|---|
| Green's-function transport | `conductance{post}.dat`, `doscond{post}.dat` (E, total, channels; `#` header), k-resolved `cond{post}-NNNN.dat`, `current.dat`, `eigchn_ik…_ie….npz` | none |
| pyskeaf | `qo_EF_<E>_freqvsangle[_<band>].out` (comma-separated: azimuth, polar, F in kT, m*, curvature, type) | `PAOFLOW.agentic.qo_analysis` |
| ACBN0 | `{orb}_UJ.txt`, `{l1}_{l2}_{i1}_{i2}_V.txt` (**written in the working directory, not `outputdir`**), driver `bands_{label}_0.dat` | `plot.acbn0.py` only |
| SSCHA | `sscha_T<T>/state.json`, `pop<n>/summary.json` (F ± error, Kong–Liu, lowest mode), `hessian/summary.json`, `*.fc` | none |
| band unfolding | in memory only (`plot_unfolded`); on disk only through the ZG `thermal_spectral_*.npz` | — |

### 9.2 Gaps the catalogue exposes

1. **No metadata on disk.** No dense output records its prefix, units, DFT Fermi energy, basis or version. Energies
   are pre-shifted to E_F = 0 and the BXSF writes `Fermi Energy: 0.0`. The DFT E_F survives only in some npz files.
   Proposal (decision D6): `finish_execution` writes `outputdir/paoflow_outputs.json`, a list of the files written,
   each with its kind, units, spin, component, E_F (DFT), prefix and version. This is an output manifest, not a
   configuration that anything reads back. Discovery uses it when present and falls back to the patterns otherwise.
2. **Ambiguous names.**
   - `alpha_{aa}.dat` (2col, 1/m) vs elphon `alpha_<T>K.dat` (cm⁻¹, several columns);
   - `emish_`, `emist_` in the dense run vs elphon;
   - `epsr_xx.dat` at the top level, under `vibdielectric/` and under `dielectric_*`;
   - two effective-mass formats;
   - two AHC/SHC families (`ahcEf_` vs `AHC_`).

   Discovery tells them apart by directory and column count.
3. **Inconsistent smearing suffix.** It is joined without an underscore for σ, S, κ and PF (`sigmagauss_0.dat`) but
   with one for `hall_trace_gauss_0.dat` and `nernst_gauss_0.dat` (`boltzmann/do_transport.py:139-146`).
   **Existing bug:** the generated `plot.py` looks for `hall_trace_0.dat`, which never exists when a smearing is set
   (the default in the generated `main.py` is `'gauss'`).
4. **Sparse bands reuse the dense names in other layouts.** `sparse_bands.py:1226` writes `bands_0.dat` with the
   k-distance in column 0 and `kpath_points.txt` as `position label`, which `read_band_path_PAO` cannot read. The
   readers must detect the layout.
5. **BXSF overwrite.** `DataController.write_bxsf` always calls `write2bxsf4skeaf`, so every Berry-curvature BXSF of
   the Hall runs also overwrites `Fermi_surf_band_{1,2}.bxsf` (`DataController.py:739-742`).
6. **Reader gaps.**
   - `read_dos_PAO` fails on any `#` header (linear response, phonons, elphon, Green's transport).
   - `read_transport_PAO` assumes 8 columns, so it fails on the Hall and Nernst files.
   - There are no readers for spin texture, effective masses, Weyl points, Green's-function transport, ARPES npz,
     pyskeaf or the SSCHA JSON.

   Phase 3 adds them in `PAOFLOW.results.readers`, and makes `read_dos_PAO` comment-tolerant (same numbers, more
   files).
7. **ACBN0 writes into the working directory.** For ACBN0, discovery therefore treats the working directory as the
   output folder.

Gaps 1, 3 and 5 touch writers. They are fixes or features in their own commits, outside the GUI, and the GUI never
depends on them.

## 10. Results GUI

```
┌ PAOFLOW   Setup · Runs · Results ──────────── output/   + compare: ../Si_soc/output ┐
│ Datasets               │  Band structure · Si.bands_0.dat              │ Options          │
│ ▾ Electronic (22)      │                                                │ window  [-6, 6]  │
│   Bands  (spin 0)      │        [ Plotly canvas: bands | DOS ]          │ reference  E_F ▾ │
│   DOS / PDOS (18)      │                                                │ spin  overlay ▾  │
│ ▾ Transport (5)        │                                                │ beside: DOS ☑    │
│ ▾ Optics (8)           │                                                │ Export ▾         │
│ ▸ Electron–phonon      │ output/Si.bands_0.dat · 400 k · 16 bands       │ PNG SVG PDF CSV  │
└────────────────────────┴────────────────────────────────────────────────┴──────────────────┘
```

- **Dataset tree.** Grouped by category, with counts, badges (spin, temperatures, components) and a search box.
  Adding folders turns on comparison mode.
- **Canvas.** Plotly, with linked layouts: bands beside the DOS on a shared energy axis, the Berry curvature under
  the bands, and the transport tensor against μ at each T.
- **Options panel.** Generated from the kind's option schema. Restyles that need no server (limits, log scale,
  legend) stay in the browser.
- **Special viewers:**
  - Fermi surface in 3D: `mesh3d` from the `fermi_plotter` geometry, coloured by |v_F|, with the BZ wireframe, band
    toggles, a supercell or centred view, and the pyskeaf field direction and slice plane. The server runs marching
    cubes and decimates the mesh.
  - ARPES and unfolding maps: a heatmap with clickable EDC and MDC cuts.
  - Spin texture: arrows or cones on the k-plane.
  - PDOS: stacked by atom, orbital or ℓ from the site-projected labels.
  - Tensor components: a picker for the transport, Hall and NLO tensors.
  - Migdal–Eliashberg: Δ(T) and the gap distributions.
  - SSCHA: populations, free energy and frequencies against T, from `state.json`.
- **Comparison.** Overlay the same kind across runs (SOC against no SOC, DFT against an SK model, kernel levels, basis
  presets), with automatic labels, an optional difference panel and Fermi-level alignment.
- **Experiment overlay.** Load a CSV or a thermo-optics dataset with provenance, overlay it, and print the
  `curve_deviation` metrics.
- **Live mode.** The viewer polls for new or changed files during a run.
- **Dashboard.** Pin figures, then export the HTML report or a PDF bundle.

Large data: a BXSF at 60³ with many bands, and ARPES maps. The server decimates the meshes and sends typed arrays, and
the maps are streamed as binary. Bands of 500 k-points × 200 bands, and the DOS, are small.

## 11. Package layout

```
src/PAOFLOW/gen/questions/         declarative questions (shared by CLI and GUI)
  model.py            Question, Note, Workflow, Context, FileSpec, RunStep
  prompt.py           CLI prompt runner (reproduces today's prompts)
  resolve.py          form resolution for the GUI
  safe_lists.py       AST parser of list and range specs (replaces eval)
  run.py  acbn0.py  phonon.py  elphon.py  sscha.py  qe_input.py  basis.py
src/PAOFLOW/results/                data layer, no GUI dependency
  catalogue.py  discovery.py  containers.py
  readers/  electronic.py transport.py topology.py optics.py nonlinear.py phonons.py
            elphon.py green_transport.py acbn0.py sscha.py pyskeaf.py
  figures/  (same split)  fermi_mesh.py (fermi_plotter geometry → mesh)
  export.py  report.py
src/PAOFLOW/gui/
  cli.py  server.py  security.py  workspace.py  generator_api.py  results_api.py  jobs.py  slurm.py
  static/ index.html  css/app.css  js/{api,form,codeview,filetree,setup,runs,results,plot,fermi3d,maps}.js
tests/unit/gen/test_questions_parity.py  tests/unit/gui/  tests/unit/results/
docs/gui.md (user page)  docs/internals/web-gui.md (developer page, replaces this plan when done)
```

The naming follows the feature-design rules: descriptive module names, procedural functions, no `do_*` modules, and
nothing added to `DataController`.

## 12. Testing

- **Parity** (section 4.4). Per workflow: the answer matrices, identical `cfg`, byte-identical files, and the
  builder-key coverage test.
- **Form resolution.** Visibility, derived defaults, validation messages, forced overrides (eACBN0 and `projwfc`;
  EPW and the explicit list), and completeness.
- **Server.** Token, Host and Origin checks; path escapes (`..`, symlinks out of the root); `--read-only`; overwrite
  confirmation; SSE. The server starts on an ephemeral port in a thread and is driven with `urllib`.
- **Jobs.** A whitelisted test command (`python -c ...`), cancel and reattach.
- **Results.** Discovery on the `examples/*/Reference/` folders and the elphon examples. Each reader gives the same
  numbers as today's readers. Figure specs: trace counts, axis titles, units, spin overlay. The report renders
  offline (no external URLs).
- **End to end** (`@pytest.mark.slow`, not in CI). Headless Chrome through Selenium, which is installed in `work`:
  load Setup, generate a `run` script, open Results, plot bands, export PNG.
- **CI.** Unit tests only. The `paoflow-gen-qe` fetch is mocked, and there is no network.

## 13. Phases and estimates (agentic execution)

| phase | deliverable | files | estimate |
|---|---|---|---|
| P1 | questions model, CLI prompt runner, the seven workflow schemas, writer splits, parity harness; old `collect_*` become wrappers | ~12 | 3 h |
| P2 | server, security, workspace detection and system summary, Setup tab for every workflow plus QE input and basis, previews and diff, write | ~17 | 4.5 h |
| P3 | results catalogue, discovery, readers and figure specs: dense properties, phonons, NLO, elphon, Green's-function transport, ACBN0, SSCHA, pyskeaf; GPAO on the shared readers | ~22 | 5 h |
| P4 | Results tab: tree, canvas, options, comparison, experiment overlay, exports, `paoflow-studio report`, Fermi-surface 3D, ARPES maps, BZ and path preview | ~12 | 3.5 h |
| P5 | run plans, local job manager, live logs, live results | ~5 | 2 h |
| P6 | user page with screenshots, the internals page, `cli-generators.md` and installation updates | ~4 | 1 h |

The total is about 20 h. Each phase ships on its own:

- after P1, the CLI is unchanged for the user;
- after P2, the setup GUI is usable;
- after P4, the viewer is usable;
- P5 is optional (decision D2).

## 14. Risks

- **Parity drift during the refactor.** The frozen legacy `collect_*` tests catch it, and the old code is deleted only
  after the matrix passes.
- **Login-node policies.** Covered by `--read-only` sessions, running the server on a compute node, and the run
  manager submitting to Slurm instead of running locally.
- **Large 3D data.** Server-side marching cubes, decimation and typed arrays.
- **JS maintenance without lint.** The JS stays small and schema-driven, with no framework. An optional node-free
  formatter (Biome binary) can be added to pre-commit later.
- **Generator growth.** New `paoflow-gen` options appear often (the NLO hooks, SSCHA). With the shared schema, a new
  question appears in both front ends at once, and the coverage test flags a builder key that has no question.

## 15. Decisions for the user

- **D1.** One server with Setup, Runs and Results tabs (recommended), or two separate tools.
- **D2.** Run manager (P5) in scope, or generate-and-view only.
- **D3.** Declarative-schema refactor of the CLI (recommended), or GUI-only form definitions (faster to start, but
  drifts from the CLI).
- **D4.** Smart defaults from the QE save in the interactive CLI too (recommended), or only in the GUI.
- **D5.** Plotly.js from the installed `plotly` package through the extra `gui` (recommended), or vendored in the repo.
- **D6.** An output manifest `paoflow_outputs.json` written by `finish_execution` (recommended; a new output, no
  existing number changes), or discovery from the filename patterns only.
- **D7.** Fix the writer issues 9.2.3 (the Hall smearing suffix, or only the `plot.py` pattern) and 9.2.5 (the BXSF
  overwrite) now, as separate commits, or leave them.

**QE runs, guide and brand (8 November 2026, user request).**

- **From the QE input to a run.** Writing `<compound>.scf.in` with the QE-input tool now returns a run plan
  (`gen/outputs.qe_input_plan`): a `pw.x -in <compound>.scf.in` step (MPI, output to `<compound>.scf.out`), then
  an *action* step that opens the Setup tab, Property run, on that folder, so the new save feeds the next
  workflow. The Setup tab shows the plan and sends it to the Runs tab.
  - The run manager accepts the QE programs (`jobs.QE_PROGRAMS`: `pw.x`, `ph.x`, `projwfc.x`, `pp.x`, `q2r.x`,
    `matdyn.x`, `dos.x`, `bands.x`, `epw.x`, `hp.x`, `dynmat.x`). The arguments are limited to the input flags
    and the parallelisation flags, and the program must be on the PATH of `paoflow-studio`; otherwise the
    launch fails with that message, and the Setup tab warns before the run.
  - A step can name a `stdout` file: the program output goes there with no header, and the job log shows it.
    ANSI escape codes are stripped from the logs.
- **The guide.** `static/js/help.js` holds a 14-section guide: getting started, folders, the working folder,
  workflows, forms, writing, the QE input, the basis, runs, results (two sections), remote use, the
  command line, and troubleshooting. A **Help** button or the `?` key opens it; it is searchable, `Esc` closes
  it, and it opens on the section of the current tab. It opens once by itself on the first visit.
- **Brand.** The name in the header and the favicon use the green brand colour (`--brand` `#00853e`, `#00b833`
  in the dark theme). The base font is 15 px (was 14), with the other sizes and the Plotly fonts (13 px) scaled
  to match. The docs screenshots are quantised to 256 colours by octree, which keeps the green.
- **Switches.** A switch must be a `<label>` around its hidden checkbox, or sit inside one: the switches of
  the QE-input tool (spin-orbit, `assume_isolated`), the plot-script option and the basis tool were bare `<span>`s
  next to a separate name label, so clicks never reached the checkbox and they kept their defaults
  (`assume_isolated` always on). Fixed; a browser test clicks each of them and checks the generator receives the
  values.
- **Tests.** Browser tests for the first-visit guide and for the chain QE input → `pw.x` (a stand-in program on
  the PATH) → Setup; unit tests for the QE plan, the QE argument whitelist and the `stdout` file.

**Existing inputs and stale servers (8 November 2026, user report: no way to run `pw.x`).** The only route to
`pw.x` was the dialog after *Write*. The user's server had started before the plan endpoint existed, so *Write*
returned no plan, and the dialog fell back to the property form's plan or to "Complete the form first".

- `POST /api/gen/qe-input/plan` (`generator_api.qe_input_run_plan`) returns the plan of any `pw.x` input in the
  root (a file with a `&control` namelist; 404 for no file, 400 for no input, 403 outside the root).
- The QE-input tool lists the `pw.x` inputs of the working folder at its top, each with *Run pw.x…*; the list is
  refreshed after *Write*. The Runs tab has *Run a pw.x input…* (a file picker inside the root).
- `showPlan` no longer falls back to the property form's plan for a tool; it asks for a restart instead.
- `GuiApp.started` and `server.code_changed_since`: the session reports `restart_needed` when a module of
  `gui/`, `gen/`, `results/` or `writers/` is newer than the server. The page shows a banner (checked at load
  and when the tab becomes visible). Static files are read from disk, so only the Python needs the restart.
- Found on the way: a resolve that was pending or in flight when the user switched from a workflow to a tool
  posted the tool's name (404, an error toast) and could render into the tool's preview. `selectWorkflow` now
  drops it (`requestId`, `form = null` for tools).
- Tests: the plan endpoint and the restart flag (unit), and a browser test of an existing input run from the
  tool and loaded from the Runs tab.

**Choice of the Quantum ESPRESSO build (8 November 2026, user request: "the pw path points to the 7.4.1, but
there are other QE versions").**

- `gui/qe_installations.py`: `discover_installations(extra, path, search)` lists the builds (one per real
  `pw.x`, newest first), and `default_installation` picks the default (the PATH one, else the newest).
  - Where it looks: the PATH; the siblings of the PATH builds; `--qe-bin`; and, with `search`, `~/Programs`,
    `~/Local/Programs`, `~/software`, `~/opt`, `~/apps`, `/opt`, `/usr/local` (`bin`, `*/bin`, `*/*/bin`), the
    conda bases and environments, `$CONDA_PREFIX`, and Homebrew.
  - The installation folder is `bin/..` for a build tree (`bin/pw.x` links to `PW/src/pw.x`), and the package
    prefix for a package.
  - The version comes from `include/qe_version.h`, then `conda-meta/qe-*.json`, then the folder name. MPI comes
    from the `DFLAGS` of `make.inc` or the `QE_ENABLE_MPI` of `CMakeCache.txt`. Nothing is executed.
  - On the Mac Studio it finds the seven builds of `~/Local/Programs/qe-*` in 0.1 s.
- `GuiApp.qe_installations()` (cached, `rescan`) and `GuiApp.qe_installation(bin)`: the server accepts only a
  `bin` of its list (400 otherwise), so the browser chooses and never names an executable.
  - `build_command(..., qe_bin)`: the QE program comes from that `bin` ("ph.x is not in ..." when missing).
    The `bin` goes first on the PATH of the run, whatever its program. A serial build refuses more than 1 rank.
    QE steps use an `mpirun` shipped in the `bin` (conda) before the PATH one.
  - `GET /api/qe` returns the list and the default (`?rescan=1` searches again); the session's
    `features.pw` means "a build was found".
- CLI: `--qe-bin DIR` (repeatable, checked at start-up); the start-up message prints the default and the
  number of other builds.
- Front end: `static/js/qe.js` (`qePicker`, `chosenQe`): a menu in the Runs plan card and in the QE-input tool,
  remembered per root, the pickers in sync; `RunsView.launch` sends `qe_bin`.
- Tests: `test_qe_installations.py` (stand-in trees: in-tree links, siblings, serial, conda metadata and
  `mpirun`, an extra folder, duplicates); the run-manager tests (the chosen build runs, PATH order, unknown
  folder, serial with 2 ranks, missing program; the no-QE test uses `qe_search=False` so that it never starts
  a real QE); a browser test (choose 7.5 in the tool, the Runs menu follows, the run uses 7.5, the choice
  survives a reload).
- Found on the way: the sibling search used the parent of `PW/` (the linked `pw.x`) instead of the
  installation folder; the common places hid it on the Mac.


**Brillouin-zone preview (8 November 2026, user report: "the preview of the BZ did not work").** Checked on a
copy of the user's `Si2.save` (`ibrav` 2, from the GUI's own `pw.x` run): the zone and path draw. Three
defects were found:

- **Figures drawn in a hidden tab.** When the page opens on the Runs tab, the Setup form draws the zone while
  hidden. Plotly then sizes it at its default 700 × 300 px (or nothing for a 3D scene), and once shown it is
  shifted and cropped, or blank. `renderFigure` now follows its container with a `ResizeObserver` and calls
  `Plotly.Plots.resize` when the size changes (`purge` disconnects it). This holds for every figure, the
  Results tab included. A browser test opens `#runs`, switches to Setup, and checks that the figure fills its
  box; it fails without the observer.
- **Γ hidden under L.** The fixed camera (eye 1.05, 1.05, 0.7) looks along about [111] for fcc. `camera_eye`
  now picks, among 120 directions (elevations 15–75°, azimuths in steps of 15°), the one with the largest
  smallest projected distance between the labelled points. For the fcc path this doubles the smallest gap
  (`test_camera_keeps_the_points_apart`).
- **The message without a save** now says what to do: "the zone needs the cell of a QE save
  (<prefix>.save) in the working folder: run pw.x first, or choose the save above".

**Defaults that follow the folder (8 November 2026, user reports: "how do I choose the save above", "the
suggested values are not applied, one has to rewrite them (the .save, the pseudos)").** In the user's
sequence, Property run was open before `pw.x` wrote `Si2.save`. The form kept `pwscf.save`, `pseudo.UPF`,
`ibrav` 0 and an empty band path, because it re-resolved only after an edit; the folder card stayed stale too.
Retyping the value triggered a resolve, which is why it seemed to "apply" it.

- `SetupView.refreshFolder` detects the working folder again (14 ms, stable output) and re-resolves when the
  detection changed. It runs when the Setup tab is shown (`activate`, now called by `App.show` for every
  tab), when the window gets the focus, and every 4 s while Setup is visible. Edited answers keep their
  values; untouched ones follow the new defaults.
- The `rows` fields (the table of high-symmetry points) now follow their default until edited; before, the
  table kept its first contents.
- `band_paths.save_problem` / `saves_in`: the zone preview and the resolve name the cause. The causes are:
  no save in the folder, a Save directory that does not exist (with the saves found), or a save without
  `data-file-schema.xml` (run not finished). The preview offers *Use <save>* buttons. The *Save directory*
  item gets a non-blocking `warning` and `suggestions` (`generator_api._warn_about_the_save`), shown under
  the field with the same buttons.
- `widget='folder'` on the `savedir` questions (`common.py`, `sscha.py`): a *Browse…* button
  (`FormView(browse=...)`, a picker that also reaches the read-only places).
- Tests: `test_save_problems_name_the_cause`; `test_save_directory_without_a_save_is_flagged`; browser tests
  `test_defaults_follow_the_folder` (the save appears while the form is open: the Save directory, UPF,
  `ibrav`, points and preview follow, with no answer stored) and `test_stale_save_answer_offers_the_found_save`.

