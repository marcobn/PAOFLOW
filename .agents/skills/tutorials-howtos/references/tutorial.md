# Tutorial template

A tutorial is a Jupyter notebook `docs/tutorials/tutorialNN.ipynb` that takes one material through one complete workflow, from the DFT output to an interpreted physical result. The reader leaves knowing both the call sequence and what each step computes. Tutorial 01 (electronic structure of Si) is the base that every other page may assume.

## Structure

Cells in this order. Headings are numbered so that other pages can link to a step. Items 1 to 5 are always there; the others, and the optional parts of a step (the `**Output:**` list, the expert checks after a figure), are written only when they add information the reader does not already have from the cells above. A section that repeats values or checks already stated next to their output is left out.

1. **Title and opening** (markdown). `# Tutorial NN: <Physical topic>`. One paragraph: the material, the physical question, what the reader computes, and the PAOFLOW methods involved. Then a `**What you will compute**` label followed by a short list of the quantities and concepts the reader will meet. Each item is a physics statement ("How the PAO Hamiltonian $H(\mathbf{R})$ is built by Fourier transform"), not a method name.
2. **Download tip** (markdown). A `:::{tip}` admonition: the notebook link (`[tutorialNN.ipynb](tutorialNN.ipynb)`), the release tag and the file the setup cell downloads, and the directory it extracts next to the notebook, so that a reader without internet access on the compute node can fetch it by hand. One sentence on the prerequisites, for example "This tutorial assumes [Tutorial 01](tutorial01.ipynb)."
3. **Required input** (markdown). The files PAOFLOW reads and what each contains, and the DFT settings that matter for this workflow (k-grid, nbnd, SOC, nosym), so that a reader can produce the same input for their own material.
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
   - a markdown cell `## N. <What this step does>`. Begin with what is computed and why the workflow needs it here. Then the defining equation as the code evaluates it, one to three sentences of intuition, and the parameters that matter for this material, with the reason for the chosen values;
   - the code cell(s), with stored output;
   - a markdown cell that tells the reader how to read the output, with numbers from the run. After a step that writes files, add an `**Output:**` label and list each file with one line on its contents and columns.
6. **Figures**: wherever a step's concept or result is easier to see than to read (see "Where a picture carries the explanation" in SKILL.md), a `hide-input` plotting cell after the step that produced the data, followed by a markdown cell interpreting the figure with numbers (gap, band width, peak position, sign of a tensor component). Then say what an expert would check in it: agreement with the DFT bands inside the projectability window, symmetry-required degeneracies, sum rules.
7. **Where to go next** (markdown, optional). Links to the how-tos that vary this workflow and to the next tutorial, one clause each. No recap.

A tutorial is long enough to cover its workflow and no longer. When a step is only a variant of something shown earlier (another k-path, another plot), it goes in a `:::{tip}` or a how-to, not in a new section.

## Cell rules

- Every code cell has its output stored. The setup cell is the only cell allowed without output.
- Plotting cells carry the `hide-input` tag and work on their own (they import what they need and read the output files).
- A step that the reader may want to change gets a `:::{tip}` with the alternative call inside a fenced code block, not a commented-out code cell.
- `verbose=True` output is kept when it shows something the text discusses (the projectability per band, timings); otherwise trim it with `verbose=False`.
- Setting the tag in JSON: `"metadata": {"tags": ["hide-input"]}` on the cell.

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
> A state built from Si $3s$ and $3p$ orbitals has $p_{n\mathbf{k}} \approx 1$. A state well above the conduction-band minimum has plane-wave character between the atoms and a much smaller projectability. `projectability()` keeps the states with $p_{n\mathbf{k}} > 0.95$ (`pthr`) and sets the energy above which the PAO bands stop following the DFT bands. For the Si grid of this tutorial that energy is {TODO: energy from run} eV above the valence-band maximum, so all four valence bands and the lowest four conduction bands are inside it.

What changed. The heading says what the step decides. The first sentence gives the reason for the step. The intuition is tied to Si orbitals. The threshold carries its code name. The claim about which bands survive is a number from the run. The voice is "you" throughout.
