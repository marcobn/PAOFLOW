// The user guide of PAOFLOW Studio: a searchable panel with one section per task.
//
// Content is data (paragraphs, lists, commands) rendered with text nodes; `inline code` in
// backticks becomes <code>. Sections are linked from the tabs (helpFor) and from the header.

import { clear, h } from './dom.js';

const SECTIONS = [
  {
    id: 'start',
    title: 'Getting started',
    body: [
      'PAOFLOW Studio is the browser interface of PAOFLOW. It writes the driver scripts of `paoflow-gen`, runs them on this machine and plots what they produce. It has four tabs:',
      [
        '**Setup**: choose a folder, pick a workflow, answer its questions, preview and write the files.',
        '**Runs**: start the commands of the written workflow and follow their logs.',
        '**Results**: open a folder with PAOFLOW outputs and explore them as interactive figures.',
        '**Plots**: run the generated plot scripts (`plot.py`, …) without a display and see the figures they make.',
      ],
      '**You already have a QE run** (a `<prefix>.save` folder):',
      [
        'Setup → *Change…* to its folder → *Property run* (or ACBN0, phonons, electron–phonon, SSCHA).',
        'Answer the questions; the right panel shows `main.py` and `plot.py` as you type.',
        '*Write files* → *Open in Runs* → *Run* on “PAOFLOW properties”.',
        'When the run finishes, *Results* opens its `output/` folder.',
      ],
      '**You start from a material** (no QE run yet):',
      [
        'Setup → *QE input from a database* → paste an AFLOW or C2DB identifier → *Fetch & preview* → *Write*.',
        'In the plan that appears: *Open in Runs* → *Run* the `pw.x` step (QE must be on the PATH, see Runs).',
        'When `pw.x` has finished, *Open Setup (Property run)* continues with the PAOFLOW drivers.',
      ],
      'Every form gives the same files as the command line (`paoflow-gen`) for the same answers, so the scripts can be edited, versioned and run elsewhere.',
    ],
  },
  {
    id: 'folders',
    title: 'Folders and permissions',
    body: [
      'The server starts on its **root** (`paoflow-studio --root DIR`, shown in the header). Paths in the forms are relative to the working folder unless they are absolute.',
      [
        'The folder picker (*Change…*, *Browse…*) is not limited to the root: its *Places* are **root**, **home** and **computer** (`/`), so it walks the whole directory tree of the machine. Folders your account cannot read are left out of the listings.',
        'Choosing a folder outside the root **opens** it for the session: a working folder (Setup and Runs) is opened for reading and writing, a folder picked to read from (pseudopotentials, results, data) for reading only. The picker says which before you choose.',
        'Opened folders are listed under *Places* and remembered across restarts of the server (`.paoflow-studio/opened.json` in the root).',
        'Browsing alone opens nothing: a folder only joins the session when it is chosen.',
      ],
      'Some folders are **read but never written**: PAOFLOW’s own `PSEUDOS/` and `BASIS/`, and every folder given with `--allow-read DIR`. *Places* shows them with a lock.',
      '`--fixed-root` keeps the session inside the root and the `--allow-read` folders: no *home* or *computer* places and no opening. Use it when the Studio is shared or when the root should be the only place written.',
      '`--read-only` turns the session into a viewer: no writes and no runs (folders chosen outside the root open for reading).',
    ],
  },
  {
    id: 'setup-folder',
    title: 'Setup: the working folder',
    body: [
      '*Change…* picks the working folder. The card shows what PAOFLOW found there: the QE save, the pseudopotentials, the `pw.x` inputs, existing drivers, and a summary read from the save (formula, lattice, electrons, gap or metal, k-grid, pseudopotential types, 2D slab).',
      'The save also gives defaults: `ibrav`, spin–orbit and the 2D flag of the property run; for the electron–phonon system the `pw.x` input gives the metal flag, `nelec`, `nbnd` and the coupling source (EPW for ultrasoft or PAW pseudopotentials). The command line uses the same defaults.',
      'Workflows marked *suggested* match what is in the folder.',
      [
        '**One folder for every tab.** A folder chosen in one tab becomes the folder of the others: *Change…* here moves Runs, Results and Plots to it, *Run a pw.x input…* in Runs moves Setup and Results to the folder of the input, and *Change…* in Results moves Setup and Runs.',
        'A Results folder named `output/` gives Setup and Runs its parent, the run folder; Setup on that run folder leaves Results on its `output/`.',
        'A Results folder that can only be read (an `--allow-read` folder, PAOFLOW’s `PSEUDOS/`, a folder opened for reading) stays in Results: Setup and Runs keep their folder and say so.',
        'Runs shows the last plan written in the folder, or none yet.',
      ],
    ],
  },
  {
    id: 'workflows',
    title: 'Setup: the workflows',
    body: [
      [
        '**Property run** (`main.py`, `plot.py`): bands, DOS and PDOS, Boltzmann transport, Fermi surface, spin texture, spin/orbital/anomalous Hall, topology, optics and emissivity (extended basis), non-linear optics and the coupled response.',
        '**ACBN0 / eACBN0** (`main.acbn0.py`, `plot.acbn0.py`): self-consistent Hubbard U, and the intersite V with the projwfc.x projector. The form has no launch commands: the ACBN0 step of the Runs tab sets the ranks of pw.x and of the Hartree step, the Quantum ESPRESSO build and the pw.x options (`-nk 4`), and the script reads them at run time (`paoflow-gen` in a terminal still asks for them; they are the defaults of the script). The same holds for the phonon and Raman launch settings and for the local SSCHA launcher (the Slurm settings stay in the form).',
        '**Harmonic phonons** (`main.phonon.py`, `plot.phonon.py`, optional `main.raman.py`): finite displacements with phonopy, Born charges, vibrational dielectric function, QHA, static and resonance Raman.',
        '**Electron–phonon system** (a folder): `system.py`, the QE inputs (`scf`, `ph.x`, `nscf`, `run_qe.sh`) and the drivers of the chosen workflows (superconductivity, optics, transport, ARPES, thermal, phonons, coupling).',
        '**Anharmonic phonons, SSCHA** (`main.sscha.py`): populations and minimisation at a list of temperatures, local, Slurm or manual launch.',
        '**Quantum oscillations, pyskeaf** (`main.pyskeaf.py`): de Haas–van Alphen frequencies against the field direction from the `Fermi_surf_band_<n>.bxsf` files of a Property run with the Fermi surface (no QE save needed). The field rotates from (θ, φ) start to end — θ is its azimuth in the xy plane, φ its polar angle from +z — or takes one direction or a reciprocal vector (b1, b2, b3); the bands, the energies (one or a scan) and the interpolation are set in the form, the orbit tolerances under *advanced*. The Results tab plots the frequencies against the angle that varies.',
        '**QE input from a database** and **Pseudo-atom basis tables** are tools: they write a `pw.x` input or the basis tables used by the internal projections.',
      ],
    ],
  },
  {
    id: 'forms',
    title: 'Setup: filling a form',
    body: [
      [
        'Questions appear only when they apply: the non-linear block when *Non-linear optics* is selected, the Raman settings after choosing a Raman flavour, and so on.',
        'A value in grey is the **default**, and it is used as it is: there is nothing to retype. Typing replaces it and pins your value; ↺ puts the default back. Defaults follow earlier answers (dense grids from the coarse ones, the smearing floor from the adaptive prefactor) and the files of the working folder: when a `pw.x` run writes its save, the *Save directory*, the UPF files, `ibrav` and the band path follow within a few seconds (and whenever the Setup tab or the window comes back).',
        '*Save directory* has *Browse…*. A Save directory that holds no save (an answer remembered from another system, a run that has not finished) is flagged under the field, with a *Use <prefix>.save* button for each save of the working folder.',
        'Invalid values are marked under the field (a q-grid that does not divide the k-grid, a number that is not a number). The *Files* panel lists what is missing.',
        '*docs · <method>* under a field shows the documentation of the PAOFLOW parameter it feeds.',
        'The *advanced* switch shows settings the command line leaves at their defaults (SSCHA minimiser, ph.x threshold, coupled-SHG components, band-plot window, high-symmetry points). They are written only when changed.',
        '*Reset* forgets the answers of the form; the browser otherwise keeps them per folder and workflow.',
      ],
      '**Band path.** With *Band structure* selected, the *Band path* section shows the path PAOFLOW follows for the lattice, drawn in the first Brillouin zone (drag to rotate; the view is chosen so that no point hides another). Without a usable save the preview says why and offers the saves of the folder. Edit the path (`gG-X-W-K-gG-L|U-X`: `-` joins points, `|` starts a new segment, `gG` is Γ) or, under *advanced*, the table of labels and crystal coordinates. A changed path is written as `BAND_PATH` and `HIGH_SYM` in `main.py`.',
    ],
  },
  {
    id: 'write',
    title: 'Setup: preview and write',
    body: [
      [
        'The *Files* panel shows every file of the workflow; the dot is green for a new file, amber for one that differs from the copy on disk. *diff* shows the differences.',
        'The *Output* box of every workflow has the driver file name (the system folder for electron–phonon) and *Write the plot scripts*: `plot.py`, `plot.acbn0.py`, `plot.phonon.py` (and `plot.raman.py`), the `plot_*.py` of the electron–phonon workflows, `plot.sscha.py` and `plot.pyskeaf.py` (these two save the figures of the Results tab as PDF files and run from the Runs tab). It starts from what paoflow-gen does by default (SSCHA: off).',
        '*Edit* opens the selected file in an editor before it is written; *Done* closes it. An edited file is marked ✎ and *Write files* writes your text. The edit is kept (per folder and workflow) while the form changes; when the form would now generate a different file, the panel says so and *Take the new file* discards your edit.',
        '*Write files* asks before overwriting: unchecked files are left as they are, and a copy `<file>.bak-<date>` is kept by default.',
        '*Python* shows a script that writes the same files without the GUI.',
        '*Next steps* lists the commands of the workflow in order; *Open in Runs* takes them to the Runs tab.',
      ],
    ],
  },
  {
    id: 'qe-input',
    title: 'Tools: QE input from a database',
    body: [
      'Identifiers: an AFLOW AUID (`aflow:0a66d228d896a855`), an AFLOWDATA or material-page URL, or a C2DB UID (`2C-1`). *Database* auto-detects the source.',
      'The pseudopotential folder needs `<element>.upf` for each element plus `PeriodicTableJSON.json` and `reference.json` (cutoffs). PAOFLOW’s `PSEUDOS/` sets qualify: *Browse…* → *Places*.',
      'Overrides: spin–orbit cards, metallic or insulating, magnetic, the k-point grid, explicit cell (`ibrav` 0), `degauss`, `nbnd`, symmetry tolerance and `assume_isolated = 2D` (written for 2D entries only, such as C2DB slabs).',
      '*Fetch & preview* needs the network. *Write* saves `<compound>.scf.in` in the working folder and shows the plan: the `pw.x` run (output `<compound>.scf.out`), then the PAOFLOW drivers in Setup.',
      'Running `pw.x`: the list *pw.x inputs in this folder*, at the top of the tool, shows every `pw.x` input of the working folder (written now, earlier, by hand or by `paoflow-gen-qe`) and the *Quantum ESPRESSO* build that runs it (the same choice as in the Runs tab). *Run pw.x…* → *Open in Runs* → *Run*. The Runs tab has the same through *Run a pw.x input…*.',
    ],
  },
  {
    id: 'basis',
    title: 'Tools: pseudo-atom basis tables',
    body: [
      'Builds the radial tables of the PAO basis from the UPF files (`paoflow-genbasis-ps`): a folder of pseudopotentials or one UPF, a preset (`minimal` ⊂ `standard` ⊂ `extended`; the optics and non-linear runs use `extended`) or explicit shells, the output folder (default `BASIS_PS`).',
      '*Run* starts it in the Runs tab. The property drivers also create missing tables on their first run.',
    ],
  },
  {
    id: 'runs',
    title: 'Runs',
    body: [
      [
        'The *Run plan* is the list of commands of the last workflow written in the working folder (each folder keeps its own plan; *No run plan in …* when nothing was written there). *Run* starts a step; MPI steps ask for the number of ranks (at most the cap shown, `--max-ranks`). *Run a pw.x input…* loads the plan of a `pw.x` input file instead, and its folder becomes the folder of every tab.',
        'Python drivers run with the Python of the server, one thread per rank, niced. Shell scripts get their rank count through `NP` (`run_qe.sh`).',
        'QE programs (`pw.x`, `ph.x`, `projwfc.x`, ...) run under `mpirun -np N` with their input file; their output goes to the file named in the plan (for example `scf.out`).',
        'Steps that start Quantum ESPRESSO themselves — ACBN0, the run phases of phonons and Raman, SSCHA with the local launcher, `run_qe.sh` and `run_zg.sh` of an electron–phonon system — run one process that launches pw.x (ph.x, the ACBN0 Hartree step): their *ranks*, the Quantum ESPRESSO build and the *options* are those of pw.x, passed as `PAOFLOW_MPI_QE`, `PAOFLOW_QE_BIN` and `PAOFLOW_QE_OPTIONS`. Every command shows what runs (`mpirun -np N` of an MPI step, `NP=`, the variables), ready for a terminal.',
        '*options* under a QE step adds runtime options: the parallelisation flags `-nk` (k-point pools), `-ni` (images), `-nd` (linear algebra), `-nt` (task groups), `-nb` (band groups) and their long names, as `-nk 4 -nd 1`. The command above updates as you type; the options are remembered per program. *Run* stays disabled while they cannot work with the ranks: images × pools must divide the ranks, task and band groups the ranks of a pool. Other options are not accepted (copy the command to a terminal for those).',
        '*Quantum ESPRESSO* chooses the build: every installation found on the machine (the one on the PATH, the folders next to it, `~/Programs`, `~/Local/Programs`, `/opt`, `/usr/local`, conda environments, and `paoflow-studio --qe-bin DIR`), with its version; *serial* marks a build without MPI (1 rank). The choice is remembered; *↻* searches again. Its `bin/` also goes first on the PATH of every run, so `run_qe.sh` and the drivers that call QE use it.',
        'The log follows the run live; scrolling up pauses the follow. *Cancel* stops the whole process group. *Results* opens the folder of the run.',
        'Runs keep going when the browser or the server is closed. A restarted server lists them again (records and logs in `.paoflow-studio/jobs/` under the root).',
        'Plot steps (`plot*.py`) open windows: *Run in Plots* runs them in the Plots tab instead, without a display. Other steps marked *interactive* need a terminal. Steps with an *Open Setup* button continue in the Setup tab.',
      ],
    ],
  },
  {
    id: 'results',
    title: 'Results: finding and plotting',
    body: [
      [
        '*Change…* opens any folder of the machine (see *Folders and permissions*): an `output/` folder, a run folder, an electron–phonon system folder. Subfolders are scanned too. The folder chosen becomes the working folder of Setup and Runs (the run folder for an `output/` folder; not for a read-only folder), and a working folder chosen in Setup or Runs opens here.',
        'The tree groups the datasets by category; tensor components, spin channels and temperatures of one quantity are the series of one figure. *filter* searches titles and file names.',
        '*Options* depend on the dataset: energy window of band plots, tensor component and temperatures of transport (at most five at once), PDOS grouping (element and ℓ, site, shell, orbital) and display (lines or stacked areas), the quantity of tables that mix units, the η column of absorption spectra, the bands, smoothing and zone of a Fermi surface.',
        'Figures zoom by dragging, pan with the toolbar, and reset with a double click; hovering shows values. *Table view* lists the plotted numbers.',
        '**Axes** sets the x and y limits by number (one row per y panel of an overlay): *Enter* applies, a blank end is automatic (the data decide it), *Automatic* clears them. A zoom with the mouse shows its range in the fields, ready to be adjusted. The limits are kept per dataset (also across a reload) and the PDF, PNG, SVG and Python exports use them.',
        '**Maps** (ARPES intensity, spectral functions, Berry-curvature slices): click a point for the energy and momentum distribution curves through it.',
        '**Fermi surfaces** are drawn in 3D, coloured by the Fermi velocity, in the Wigner–Seitz zone (needs scikit-image). `FermiSurf_<s>.bxsf` (all bands, eV) and the pyskeaf files `Fermi_surf_band_<n>.bxsf` (one band each, written in Rydberg with the reciprocal vectors without 2π, as SKEAF wants them) give the same picture: the latter are converted, and their bands are named from the file content (file 1 may hold band 2).',
        '*PAOFLOW runs of this folder* (`paoflow_outputs.json`) records the DFT Fermi energy, spin and basis facts and the files of every run.',
      ],
    ],
  },
  {
    id: 'plots',
    title: 'Plots: the figures of the plot scripts',
    body: [
      'The Plots tab lists the plot scripts of the working folder: the `plot*.py` written by Setup (*Write the plot scripts* in the Output box) and the `plot_*.py` of the workflow folders of an electron–phonon system (`superconductivity/plot_me.py`, ...).',
      [
        '*Run* starts the selected script with no display, through the Runs tab (its log is there too). Each figure the script shows (`plt.show()`) or saves (`savefig`) is kept, once, as PNG (shown here) and PDF.',
        'The figures go to `figures/<script>/` next to the script (`figures/plot/01_Band_structure.png`, ...), listed in `figures.json`; a new run replaces them. They appear one by one while the script runs.',
        'The arguments field is passed to the script: `--all` is filled in when the script has it (every quantity, no menu); add `--emin -6 --emax 4` and the like (the options of the script are listed under the field). Prompts (`input()`) take their default answers.',
        'Click a figure to enlarge it; *PNG* and *PDF* download it. *Log of the last run* shows what the script printed, and opens by itself when it failed (for example `run main.py first`).',
        'The Plots tab follows the working folder of Setup, Runs and Results.',
      ],
      {
        code: 'python -m PAOFLOW.graphics.figure_capture --out figures/plot plot.py --all',
      },
      'The same command runs a plot script on a machine with no display (a cluster login node).',
    ],
  },
  {
    id: 'results-more',
    title: 'Results: compare, experiment, export',
    body: [
      [
        '**Overlay properties**: with a curve open, tick other datasets of the folder that share its x axis (Re ε with Im ε, n with k, DOS with PDOS, σ/τ with S). Properties of one unit share a panel; another unit gets its own panel below, on the same x axis (never a second y axis). The colour is the property (it keeps its colour while others come and go), the line style the series: *Series* picks the tensor components, `xx` by default. Up to three on top of the open one; *Clear the overlay* goes back.',
        '**Compare runs**: open a dataset, *+ this dataset*, open the same kind of dataset in another folder, *+ this dataset* again, then *Overlay* (up to eight; each run keeps its colour).',
        '**Experiment overlay**: *File…* picks a measured curve (columns or CSV), choose its x and y columns and a scale, *Overlay*. The GUI reports the RMS, the bias and the relative RMS against the computed curve.',
        '**Export**: CSV of the plotted data, *Python* (a script that rebuilds the figure, with the `GPAO` call where one exists), PDF / PNG / SVG through matplotlib. The camera icon of the figure saves the current view.',
        '**Report** writes one HTML file with every figure of the folder; it opens offline.',
        '**live** rescans the folder every few seconds while a run writes files; new datasets are marked *new*.',
      ],
    ],
  },
  {
    id: 'remote',
    title: 'Remote machines',
    body: [
      'Start the server on the remote machine without a browser (inside tmux, so that it survives the logout), then tunnel its port from your computer and open the printed URL:',
      {
        code: '# on the remote machine\ntmux new -s studio\nconda activate <env with PAOFLOW>\npaoflow-studio --root ~/runs --no-browser --port 8780\n\n# on your computer: the SAME port on both ends\nssh -N -o ExitOnForwardFailure=yes -L 8780:127.0.0.1:8780 user@remote',
      },
      [
        '**The same port number on both ends.** The Studio only accepts requests addressed to its own port, so `-L 8780:127.0.0.1:8780` works and `-L 8766:127.0.0.1:8765` does not.',
        '**A port that is free on your computer.** If you also run `paoflow-studio` locally, it holds 8765: give the remote one another `--port`. `ExitOnForwardFailure` makes ssh stop with an error instead of tunnelling to nothing; without it the browser reaches your local Studio, which rejects the remote token.',
        '**The whole printed URL**, `http://127.0.0.1:8780/?token=…`; after the first visit the token is kept in a cookie. If the port was busy on the remote machine, the Studio takes the next free one and prints it: tunnel that one.',
        '*Ctrl-b d* leaves tmux with the Studio running; `tmux attach -t studio` returns to it (Ctrl-C stops it). Plots need Plotly in the remote environment (`pip install "plotly>=5.18"`).',
      ],
      'The server listens on 127.0.0.1 only and every request needs the session token of the URL. On a cluster, keep heavy runs off the login nodes: run the server on a compute node, or use the Runs tab only for light steps.',
    ],
  },
  {
    id: 'cli',
    title: 'Command line',
    body: [
      {
        code: 'paoflow-studio [--root DIR] [--port 8765] [--no-browser] [--allow-read DIR]...\n               [--qe-bin DIR]... [--fixed-root] [--read-only] [--max-ranks N] [--verbose]\npaoflow-studio plot DIR                     # open the Results tab on DIR\npaoflow-studio report DIR -o report.html    # static HTML report, no server\npaoflow-gen --gui                           # the paoflow-gen questions in the browser',
      },
      'Without a free port at 8765 the next twenty are tried. The forms, plots and runs have command-line equivalents: `paoflow-gen`, `paoflow-gen-qe`, `paoflow-genbasis-ps`, the generated `plot.py`, and `PAOFLOW.results.ResultsFolder` in scripts and notebooks.',
    ],
  },
  {
    id: 'troubleshooting',
    title: 'Troubleshooting',
    body: [
      [
        '“… is outside the GUI root …”: the path is outside the root and every opened folder (a path typed into a form). Choose its folder once with *Change…* or *Browse…*; with `--fixed-root`, start the server with a wider `--root` or add the folder with `--allow-read`.',
        '“… is read-only …”: an `--allow-read` folder, or one opened for reading only (picked with *Browse…* for pseudopotentials or results). Choose it with *Change…* in Setup to open it for writing.',
        '“this PAOFLOW Studio keeps to its root (--fixed-root)”: the server was started with `--fixed-root`; restart it without that option, or with a wider `--root`.',
        'A system folder that lists fewer entries than `ls`: entries your account cannot read are skipped.',
        '“no Quantum ESPRESSO installation found”, or a build missing from the list: load QE in the shell, or name its folder with `paoflow-studio --qe-bin DIR`, and start the server again (*↻* finds builds installed in the searched places meanwhile).',
        '“existing files differ”: a file changed on disk since the preview; write again and confirm.',
        '“PAOFLOW was updated after this server started” (a yellow banner), or a button that does nothing: the page is newer than the running server (a source checkout after `git pull` or an edit). Stop `paoflow-studio` with Ctrl-C and start it again.',
        '“Plotly is not available”: install it in the environment that runs `paoflow-studio` (on the remote machine for a remote Studio), `pip install "PAOFLOW[gui]"` or `pip install "plotly>=5.18"`, then restart the server. The 3D Fermi surface also needs `pip install "PAOFLOW[fermisurface]"`.',
        'A page that asks for the token: open the full URL printed by `paoflow-studio` (it ends with `?token=…`).',
        "“not this session's token” on a remote URL: the port is taken on your computer (by a local `paoflow-studio`), so the tunnel could not use it and the browser reached the local Studio. Restart the remote Studio with another `--port` and tunnel that port (see *Remote machines*).",
        '“use the same port on both ends”: the tunnel maps two port numbers (`-L 8766:127.0.0.1:8765`); use one number on both sides.',
        'A run shown as *lost*: its process ended while no server was watching (for example after a reboot); its log is still in `.paoflow-studio/jobs/`.',
      ],
    ],
  },
];

// `inline code` and **bold** and *emphasis* as elements (text nodes only).
function inline(text) {
  const parts = [];
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > last) parts.push(text.slice(last, match.index));
    const token = match[0];
    if (token.startsWith('`')) parts.push(h('code', {}, token.slice(1, -1)));
    else if (token.startsWith('**')) parts.push(h('strong', {}, token.slice(2, -2)));
    else parts.push(h('em', {}, token.slice(1, -1)));
    last = match.index + token.length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

function block(item) {
  if (typeof item === 'string') return h('p', {}, inline(item));
  if (Array.isArray(item))
    return h(
      'ul',
      {},
      item.map((entry) => h('li', {}, inline(entry))),
    );
  if (item.code) return h('pre', { class: 'help-code' }, item.code);
  return null;
}

function text(section) {
  return [section.title, ...section.body.flat(2).map((b) => (typeof b === 'string' ? b : b.code))]
    .join(' ')
    .toLowerCase();
}

export class HelpPanel {
  constructor() {
    this.search = h('input', {
      type: 'search',
      placeholder: 'search the guide…',
      oninput: () => this.renderContents(),
    });
    this.contents = h('nav', { class: 'help-contents' });
    this.page = h('article', { class: 'help-page' });
    this.root = h(
      'aside',
      { class: 'help-panel', hidden: true, 'aria-label': 'PAOFLOW Studio guide' },
      h(
        'div',
        { class: 'help-head' },
        h('strong', { class: 'grow' }, 'PAOFLOW Studio guide'),
        h('button', { class: 'icon-btn', title: 'Close (Esc)', onclick: () => this.close() }, '✕'),
      ),
      h(
        'div',
        { class: 'help-body' },
        h('div', { class: 'help-side' }, this.search, this.contents),
        this.page,
      ),
    );
    document.body.append(this.root);
    document.addEventListener('keydown', (event) => {
      const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName);
      if (event.key === 'Escape' && !this.root.hidden) this.close();
      else if (event.key === '?' && !typing) this.toggle();
    });
    this.current = 'start';
    this.renderContents();
    this.show('start');
  }

  renderContents() {
    const query = this.search.value.trim().toLowerCase();
    clear(this.contents);
    for (const section of SECTIONS) {
      if (query && !text(section).includes(query)) continue;
      this.contents.append(
        h(
          'button',
          {
            class: `help-link${section.id === this.current ? ' selected' : ''}`,
            onclick: () => this.show(section.id),
          },
          section.title,
        ),
      );
    }
    if (!this.contents.children.length)
      this.contents.append(h('div', { class: 'small muted' }, 'nothing matches'));
  }

  show(id) {
    const section = SECTIONS.find((s) => s.id === id) || SECTIONS[0];
    this.current = section.id;
    clear(this.page).append(h('h2', {}, section.title), ...section.body.map(block).filter(Boolean));
    this.page.scrollTop = 0;
    this.renderContents();
  }

  open(id) {
    if (id) this.show(id);
    this.root.hidden = false;
    document.body.classList.add('help-open');
  }

  close() {
    this.root.hidden = true;
    document.body.classList.remove('help-open');
  }

  toggle() {
    if (this.root.hidden) this.open();
    else this.close();
  }
}

// The section that explains a tab (the ? buttons of the tabs).
export const TAB_HELP = { setup: 'forms', runs: 'runs', results: 'results', plots: 'plots' };
