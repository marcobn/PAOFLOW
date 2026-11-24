# How to plot PAOFLOW output with GPAO

On this page you plot the output files of a finished PAOFLOW run with the {py:class}`~PAOFLOW.GPAO.GPAO` class: the band structure, the density of states (DOS) and the projected DOS (PDOS). GPAO reads the text files that each property writes to `outputdir`, so you can plot a run without its `PAOFLOW` object, after a cluster job, for example. GPAO computes nothing, so a figure carries the approximations of the run that wrote the files.

:::{tip}
This page is built on the case [`si_bands_transport`](https://github.com/marcobn/PAOFLOW/tree/develop/examples/cases/si_bands_transport): open it in PAOFLOW Studio with *Examples…* → `si_bands_transport`, or copy it with `paoflow-gen --example si_bands_transport`. Download {download}`plotting.py <../../../examples/cases/si_bands_transport/plotting.py>` to plot your own run: it reads the files in `output/`, the `outputdir` of the case's `main.py`, and saves the figures of this page. Download {download}`figures.py <plotting/figures.py>` to check that a rerun still gives the numbers quoted here. To plot without running the case, extract the files of the figures, `plotting/si/` of `howto_assets.tar.gz` in the release [tutorial-assets-v2](https://github.com/marcobn/PAOFLOW/releases/tag/tutorial-assets-v2), into `output/` with `mkdir output && tar xzf howto_assets.tar.gz -C output --strip-components=2 plotting/si`. `plotting.py` runs in about 10 s on one core.
:::

## Setting up

```{literalinclude} ../../../examples/cases/si_bands_transport/plotting.py
:language: python
:end-before: '# Band structure and DOS'
```

The calls below pass the colors of `palette` through `col` or `cols`, because the GPAO defaults (black, red, green and blue) differ from those of this documentation.

## Band structure and DOS

You plot the bands from `bands_0.dat` and `kpath_points.txt` of `bands()`, with energies relative to $E_F$, and the DOS from `dosdk_0.dat` of `dos()`.

```{literalinclude} ../../../examples/cases/si_bands_transport/plotting.py
:language: python
:start-after: '# Band structure and DOS'
:end-before: '# Projected DOS grouped by orbital'
```

Each GPAO method draws one new figure and calls `matplotlib.pyplot.show()`, which opens a window when you run a script. To write the figure to a file instead, as the script does, run with `MPLBACKEND=Agg python plotting.py` and call `matplotlib.pyplot.savefig('name.png')` after the GPAO call.

The tick marks of the k axis come from `kpath_points.txt`, which `sym_points` reads; a jump in the path gets one tick with both labels, such as `K|U`. `label` (`band_label` in {py:meth}`~PAOFLOW.GPAO.GPAO.plot_dos_beside_bands`) sets the energy-axis label and `y_lim` the energy range. Keep `y_lim` of `plot_dos_beside_bands` inside the energy range of `dos(emin, emax)`, because the DOS panel shares the energy axis and stops where the DOS file stops. To compare two runs, pass a list of files and a list of `labels` to {py:meth}`~PAOFLOW.GPAO.GPAO.plot_bands`, which overlays them.

![Si band structure along the fcc path with the DOS on the same energy axis to its right](../../_static/images/howtos/plotting/si_bands_dos.png)

The figures show a run of this case with the projections of `projwfc.x` (the projector *projwfc* of the case's form), with the calls `bands(ibrav=2, nk=2000)` and `dos(emin=-12.0, emax=2.2, ne=1000)` of the case. The energy zero is $E_F$, here the valence-band maximum (VBM) at $\Gamma$, and the conduction-band minimum (CBM) is at 0.754 eV on the $\Gamma$–X line, so the DOS is zero between 0 and 0.754 eV. The DOS panel ends at `emax`, 2.2 eV.

## Projected DOS grouped by orbital

To see which orbitals make up each part of the DOS, you plot the PDOS ([Projected density of states](../theory/density-of-states.md#projected-density-of-states)) summed over groups of orbitals, here the shells of both Si atoms. `dos()` writes one PDOS file per pseudo-atomic orbital, named by atom and orbital (`Si1_3s_pdosdk_0.dat`, `Si1_3px_pdosdk_0.dat`, ...). {py:meth}`~PAOFLOW.GPAO.GPAO.plot_pdos` draws one curve per file, or, with `psum_inds`, one curve per group of files, where each inner list holds the positions in the file list of the files to add up.

```{literalinclude} ../../../examples/cases/si_bands_transport/plotting.py
:language: python
:start-after: '# Projected DOS grouped by orbital'
```

The script reads the shell of each file from its name, so the groups follow the basis of the run. With the `projwfc.x` projections of the figure, the PDOS covers the $3s$, $3p$ and $3d$ pseudo-orbitals, and the groups hold 2, 6 and 10 files; with the internal projections of the case's `main.py`, the $4s$, $4p$ and $4d$ shells of its `standard` basis add three more groups. With `psum_inds` given, `labels` and `cols` take one entry per group rather than per file.

![Si PDOS summed over both atoms for the 3s, 3p and 3d orbitals](../../_static/images/howtos/plotting/si_pdos.png)

The $3s$ orbitals carry most of the weight at the bottom of the valence band, below $-10$ eV, and the $3p$ orbitals between $-4$ eV and the VBM. Integrated over the valence band, the three curves hold 0.81, 2.27 and 0.92 of the 4.00 states per spin; the [bands and DOS tutorial](../tutorials/tutorial01.ipynb#density-of-states-and-projected-dos) discusses this distribution.

## Other properties

GPAO plots the files of the other properties in the same way, one method per file type, and the case of each property has a `plot.py`, written by Studio or `paoflow-gen`, that calls it: the Berry curvature under the bands in [`fe_anomalous_hall`](https://github.com/marcobn/PAOFLOW/tree/develop/examples/cases/fe_anomalous_hall), the spin Hall conductivity in [`pt_spin_hall`](https://github.com/marcobn/PAOFLOW/tree/develop/examples/cases/pt_spin_hall) and [`snte_spin_texture`](https://github.com/marcobn/PAOFLOW/tree/develop/examples/cases/snte_spin_texture), and the conductivity and Seebeck coefficient in [`al_dos_transport`](https://github.com/marcobn/PAOFLOW/tree/develop/examples/cases/al_dos_transport) and [`alp_hubbard_optics`](https://github.com/marcobn/PAOFLOW/tree/develop/examples/cases/alp_hubbard_optics). The Fermi surface, the file `FermiSurf_0.bxsf` of `fermi_surface()` in [`snte_spin_texture`](https://github.com/marcobn/PAOFLOW/tree/develop/examples/cases/snte_spin_texture), is drawn by the `fermi-plotter` command instead of GPAO ({py:mod}`PAOFLOW.gen.fermi_plotter` lists its options); it needs the `fermisurface` extra, `pip install "PAOFLOW[fermisurface]"`.

## Where to go next

- {py:class}`~PAOFLOW.GPAO.GPAO` lists every plotting method, including those for Berry curvature, spin Hall conductivity, transport, optics, phonons and electron–phonon properties, with their arguments.
- [Band structure and density of states of silicon](../tutorials/tutorial01.ipynb) produces the band and DOS files used here and explains what they contain.
- [Band velocities, density of states and adaptive smearing](../theory/density-of-states.md) derives the DOS and the PDOS in these files.
