# Tutorial template

A tutorial is a Jupyter notebook `docs/learn/tutorials/tutorialNN.ipynb` that takes one material through one complete workflow, from the DFT output to an interpreted physical result. The reader is learning: they leave knowing both the call sequence and what each step computes. Tutorial 01 (electronic structure of Si) is the base that every other page may assume. A tutorial has at most 3000 words of prose (see the table at the top of SKILL.md); a workflow that needs more is split into two tutorials, or its variants move to how-tos.

## Structure

Cells in this order. Headings are numbered so that other pages can link to a step. Items 1 to 5 are always there; the others, and the optional parts of a step (the `**Output:**` list, the expert checks after a figure), are written only when they add information the reader does not already have from the cells above. A section that repeats values or checks already stated next to their output is left out.

1. **Title and opening** (markdown). `# Tutorial NN: <Physical topic>`. One paragraph: what the reader computes, for which material and which physical question, and the PAOFLOW methods involved. The first sentence says what the reader does ("In this tutorial you compute ..."); the relation to Tutorial 01 follows it (see "Openings" in SKILL.md). Then a `**What you will compute**` label followed by a short list of the quantities and concepts the reader will meet. Each item is a physics statement ("How the PAO Hamiltonian $H(\mathbf{R})$ is built by Fourier transform"), not a method name.
2. **Download tip** (markdown). A `:::{tip}` admonition that opens with what the reader gains by running the notebook, with the link in the same sentence: "To run the calculation yourself, change its parameters, or start your own material from this workflow, {download}`download this notebook <tutorialNN.ipynb>`." A bare "Download this notebook" tells the reader neither why nor when to do it (see "Connectors" in SKILL.md: an instruction carries its reason). Then the release tag and the file the setup cell downloads, and the directory it extracts next to the notebook, so that a reader without internet access on the compute node can fetch it by hand. One sentence on the prerequisites, for example "This tutorial assumes [Tutorial 01](tutorial01.ipynb)."
3. **Required input** (markdown). The QE runs that made the `.save` directory, named by their input files, which the release asset holds next to it (`scf.in`, `nscf.in`, `proj.in`), and the files PAOFLOW reads, one line each on what it contains. Then a sentence and a code block for a reader who would rather make the `.save` directory than download it: "If you would like to create the `.save` directory yourself instead of downloading it, ...", followed by the `pw.x` and `projwfc.x` commands in order. Do not list DFT settings here: a partial list does not reproduce the run, and the input files do. The reason for each setting PAOFLOW depends on (k-grid, `nbnd`, occupations, SOC, `nosym`) goes in the step where it takes effect (see "Point of use" in SKILL.md).
4. **Setup cell** (code). `savedir` and `outputdir`, relative to the notebook, then the download and extraction of the release asset, run only when `savedir` does not exist. Extract only the files of this tutorial, with `filter='data'` so that no member of the archive is written outside the notebook's directory. A reader who runs the notebook on their own calculation replaces `savedir` and the download does not run. Pattern:

   ```python
   from pathlib import Path
   from tarfile import open as open_tar
   from urllib.request import urlretrieve

   savedir = 'tutorial01/silicon.save'
   outputdir = 'output'

   # fetch the input of this tutorial from the release named in the tip, unless it is here already
   if not Path(savedir).is_dir():
       archive = Path('tutorial_assets.tar.gz')
       if not archive.is_file():
           urlretrieve(
               'https://github.com/marcobn/PAOFLOW/releases/download/'
               'tutorial-assets-v1/tutorial_assets.tar.gz',
               archive,
           )
       with open_tar(archive, 'r:gz') as tar:
           members = [m for m in tar.getmembers() if m.name.startswith('tutorial01/')]
           tar.extractall(members=members, filter='data')
   ```
5. **One section per step**, each made of:
   - a markdown cell `## N. <What this step does>`. Begin with what is computed and why the workflow needs it here. Give an equation only where the reader needs it to choose a parameter or read the output, as the code evaluates it, with one sentence on what it means for the output; otherwise one sentence saying what the quantity is. Link to the section of the theory page that derives it, name the approximations in one line each, and give the parameters that matter for this material, with the reason for the chosen values (see "Physics depth" in SKILL.md);
   - the code cell(s), with stored output;
   - a markdown cell that tells the reader how to read the output, with numbers from the run. After a step that writes files, add an `**Output:**` label and list each file with one line on its contents and columns.
6. **Figures**: wherever a step's concept or result is easier to see than to read (see "Where a picture carries the explanation" in SKILL.md), a `hide-input` plotting cell after the step that produced the data, followed by a markdown cell interpreting the figure with numbers (gap, band width, peak position, sign of a tensor component). Then say what an expert would check in it: agreement with the DFT bands inside the projectability window, symmetry-required degeneracies, sum rules.
7. **Where to go next** (markdown, optional). Links to the how-tos that vary this workflow, the theory pages behind it and the next tutorial, one clause each. No recap.
8. **Asserts** (code, tag `remove-cell`). The last cell asserts the few numbers the conclusions of the tutorial rest on, at the precision the text quotes, each with a message naming the quantity (see "Running the code" in SKILL.md). The website hides the cell, and the page test fails when an assert does.

A tutorial is long enough to cover its workflow and no longer. When a step is only a variant of something shown earlier (another k-path, another plot), it goes in a `:::{tip}` or a how-to, not in a new section.

## Cell rules

- Every code cell has its output stored. The setup cell and the cell of asserts are the only cells allowed without output.
- Plotting cells carry the `hide-input` tag and work on their own (they import what they need and read the output files).
- A step that the reader may want to change gets a `:::{tip}` with the alternative call inside a fenced code block, not a commented-out code cell.
- `verbose=True` output is kept when it shows something the text discusses (the projectability per band, timings); otherwise trim it with `verbose=False`.
- Setting the tag in JSON: `"metadata": {"tags": ["hide-input"]}` on the cell (`["remove-cell"]` for the asserts).
- Plotting cells save nothing to the repository: the figures of a tutorial are its stored outputs.

## Before and after

From Tutorial 01, section 3.

Before:

> ## 3. Check projectability
>
> The projectability of band $n$ at k-point $\mathbf{k}$ measures how completely the PAO basis captures that band: [equation]. Values near 1 indicate a good representation; values well below 1 correspond to high-energy, plane-wave-like states not captured by localised orbitals.
>
> Bands with $p_{n\mathbf{k}}$ below the threshold (default 0.95) are excluded from Hamiltonian construction. For silicon, the four valence bands and four lowest conduction bands are well within the PAO window.

After:

> ## 3. Projectability: which bands the PAO basis can represent
>
> You keep a Kohn–Sham state in the PAO Hamiltonian only if the basis reproduces it. The measure is the projectability, the norm of the state's projection onto the PAOs:
>
> $$p_{n\mathbf{k}} = \sum_\alpha |A_{n\alpha}(\mathbf{k})|^2 .$$
>
> A state built from Si $3s$ and $3p$ orbitals has $p_{n\mathbf{k}} \approx 1$. A state well above the conduction-band minimum has plane-wave character between the atoms and a much smaller projectability. `projectability()` keeps the states with $p_{n\mathbf{k}} > 0.95$ (`pthr`) and sets the energy above which the PAO bands stop following the DFT bands. For the Si grid of this tutorial that energy is {TODO: energy from run} eV above the valence-band maximum, so all four valence bands and the lowest four conduction bands are inside it. Why a low projectability distorts the bands, and how the PAO shift fills the rest of the space, is explained in [Projectability](../theory/pao-hamiltonian.md#projectability).

What changed. The heading says what the step decides. The first sentence gives the reason for the step. The equation stays because the reader sets `pthr` on it; the physics behind it is one link away. The intuition is tied to Si orbitals. The threshold carries its code name. The claim about which bands survive is a number from the run. The voice is "you" throughout.
