# How to plot PAOFLOW output with GPAO

Every PAOFLOW property writes its result as text files in `outputdir`: {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.bands` writes `bands_0.dat` and `kpath_points.txt`, {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.dos` the DOS and one projected-DOS (PDOS) file per orbital, {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.transport` the transport tensors, {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.spin_Hall` the spin Hall conductivity (SHC), {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.fermi_surface` a BXSF file. The {py:class}`~PAOFLOW.GPAO.GPAO` class reads these files and draws them with matplotlib, so you can plot a finished run without its `PAOFLOW` object, for example after a cluster job or on another machine. This page shows the call for each kind of output, the arguments that change what the figure shows, and how to check that a figure matches its run. GPAO computes nothing: a figure carries the approximations of the run that wrote the files.

Base examples: [examples/qe_examples/example01/](https://github.com/marcobn/PAOFLOW/tree/master/examples/qe_examples/example01) (Si), [examples/qe_examples/example02/](https://github.com/marcobn/PAOFLOW/tree/master/examples/qe_examples/example02) (Al), [examples/qe_examples/example04/](https://github.com/marcobn/PAOFLOW/tree/master/examples/qe_examples/example04) (Fe), [examples/qe_examples/example05/](https://github.com/marcobn/PAOFLOW/tree/master/examples/qe_examples/example05) (Pt), [examples/qe_examples/example06/](https://github.com/marcobn/PAOFLOW/tree/master/examples/qe_examples/example06) (AlP), [examples/qe_examples/example08/](https://github.com/marcobn/PAOFLOW/tree/master/examples/qe_examples/example08) (SnTe monolayer), [examples/pyskeaf_examples/Bi/](https://github.com/marcobn/PAOFLOW/tree/master/examples/pyskeaf_examples/Bi) (Bi)

:::{tip}
The output files of these examples are in `howto_assets.tar.gz` of the release [tutorial-assets-v1](https://github.com/marcobn/PAOFLOW/releases/tag/tutorial-assets-v1). Download it and extract it with `tar xzf howto_assets.tar.gz`, which creates one folder per system under `plotting/`. Run the code below, or the whole script {download}`main.py <plotting/main.py>`, from the directory that contains `plotting/`. To plot your own run, replace the paths with files from its `outputdir`.
:::

## Setting up

```{literalinclude} plotting/main.py
:language: python
:start-after: '# [setup]'
:end-before: '# [/setup]'
```

Each GPAO method opens a new figure and calls `matplotlib.pyplot.show()`. In a notebook the figure appears below the cell. A script opens it in a window; to write it to a file instead, run the script with a non-interactive backend, `MPLBACKEND=Agg python main.py`, where `show()` leaves the figure open with a warning, and call `matplotlib.pyplot.savefig('name.png')` after the GPAO call. The script of this page does so for every figure. The default colors of GPAO are black, red, green and blue; the calls below pass the colors of `palette` through `col` or `cols`.

## Band structure and DOS

`bands()` writes `bands_0.dat`, one row per k-point with the point index followed by every band energy in eV relative to $E_F$, and `kpath_points.txt`, the high-symmetry points of the path with the number of points to the next one, followed by the coordinates of every point. `dos()` writes `dosdk_0.dat`, the energy and the DOS per spin.

```{literalinclude} plotting/main.py
:language: python
:start-after: '# [bands]'
:end-before: '# [/bands]'
```

`sym_points` reads the tick positions and labels from `kpath_points.txt`. A segment with 0 points marks a jump in the path, and its tick carries both labels: `K|U` is the end of one piece of the path at K and the start of the next at U. `label` (`band_label` in {py:meth}`~PAOFLOW.GPAO.GPAO.plot_dos_beside_bands`) is the energy-axis label, and `y_lim` the energy range. `plot_dos_beside_bands` draws the DOS on the energy axis of the bands, so set `y_lim` inside the energy range of the DOS file: the DOS stops where `dos(emin, emax)` stopped. {py:meth}`~PAOFLOW.GPAO.GPAO.plot_bands` also takes a list of files and a list of `labels`, and overlays them, for example the bands of two runs.

![Si band structure along the fcc path with the DOS on the same energy axis to its right](../../_static/images/howtos/plotting/si_bands_dos.png)

The figure shows the files of example01, which calls `bands(ibrav=2, nk=2000)` and `dos(emin=-12.0, emax=2.2, ne=1000)`. The energy zero is $E_F$, here the valence-band maximum (VBM), and the DOS panel ends at 2.2 eV, the `emax` of the run. The VBM is at $\Gamma$ and the conduction-band minimum (CBM) at 0.754 eV on the $\Gamma$–X line, so the DOS is zero between 0 and 0.754 eV.

## Projected DOS grouped by orbital

`dos()` writes one PDOS file per pseudo-atomic orbital, named by atom and orbital (`Si1_3s_pdosdk_0.dat`, `Si1_3px_pdosdk_0.dat`, ...), with the energy and the PDOS per spin. {py:meth}`~PAOFLOW.GPAO.GPAO.plot_pdos` draws one curve per file, or, with `psum_inds`, one curve per group of files: each inner list holds the positions in the file list of the files to add up.

```{literalinclude} plotting/main.py
:language: python
:start-after: '# [pdos]'
:end-before: '# [/pdos]'
```

The groups here are built from the file names, so they hold the 2 $3s$, 6 $3p$ and 10 $3d$ files of both atoms. `labels` and `cols` take one entry per curve, that is, per group when `psum_inds` is given.

![Si PDOS summed over both atoms for the 3s, 3p and 3d orbitals](../../_static/images/howtos/plotting/si_pdos.png)

The $3s$ orbitals carry most of the weight at the bottom of the valence band, below $-10$ eV, and the $3p$ orbitals between $-4$ eV and the VBM. Integrated over the valence band, the three curves hold 0.81, 2.27 and 0.92 of the 4.00 states per spin, which [Tutorial 01](../tutorial01.ipynb#7-density-of-states-and-projected-dos) discusses.

## Berry curvature along the path

{py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.topology` with `Berry=True, spol=..., ipol=..., jpol=...` writes the Berry curvature $\Omega_{ij}(\mathbf{k})$ along the path of `bands()`, summed over the occupied bands, to `Omega_{spol}_{ipol}{jpol}.dat`: the point index and $\Omega$ in bohr$^2$. The first label in the name is the spin direction `spol`, which only the spin Berry curvature `Omegaj_*.dat` uses. Example04 (bcc Fe with spin–orbit coupling) calls `topology(spol=2, ipol=1, jpol=2, Berry=True)`, so its file is `Omega_z_yz.dat`.

```{literalinclude} plotting/main.py
:language: python
:start-after: '# [berry]'
:end-before: '# [/berry]'
```

{py:meth}`~PAOFLOW.GPAO.GPAO.plot_berry_under_bands` puts the curvature in a panel under the bands, on the same k axis; `y_lim` applies to the bands, and the curvature panel spans the range of the data. To choose the range of the curvature as well, plot it alone with `plotter.plot_berry(fe + 'Omega_z_yz.dat', sym_points=fe + 'kpath_points.txt', y_lim=(-50.0, 50.0))`.

![Fe band structure with the Berry curvature along the same path in a panel below](../../_static/images/howtos/plotting/fe_berry.png)

The curvature has a peak of $1.47\times10^{3}$ bohr$^2$ on the P–H line, close to H, and is small elsewhere on this scale. At that point two bands lie 1.2 meV above and 31 meV below $E_F$. Because $\Omega$ contains the factor $1/(\varepsilon_{n\mathbf{k}}-\varepsilon_{m\mathbf{k}})^2$ for an occupied band $n$ and an empty band $m$, it peaks where such a pair nearly touches at $E_F$.

## Spin Hall conductivity

`spin_Hall(s_tensor=[[ipol, jpol, spol]])` writes `shcEf_{spol}_{ipol}{jpol}.dat`: the Fermi energy relative to $E_F$ (eV) and the spin Hall conductivity $\sigma^{s}_{ij}$ with the bands kept rigid. {py:meth}`~PAOFLOW.GPAO.GPAO.plot_shc` takes one file, or a list of files drawn together, with a legend entry made of the last four characters of each file name (`z_xy`).

```{literalinclude} plotting/main.py
:language: python
:start-after: '# [shc]'
:end-before: '# [/shc]'
```

The default axis label is $(\Omega\,\mathrm{cm})^{-1}$, the unit of a bulk calculation. A two-dimensional run, `spin_Hall(twoD=True)` as in example08, divides by the area of the cell instead of its volume, so the bulk label does not apply; `label` replaces it.

![Spin Hall conductivity of Pt against the Fermi energy](../../_static/images/howtos/plotting/pt_shc.png)

In Pt (example05) $\sigma^z_{xy}$ is 1961 $(\Omega\,\mathrm{cm})^{-1}$ at $E_F$ and has its maximum, 2025 $(\Omega\,\mathrm{cm})^{-1}$, 0.09 eV above it.

![Spin Hall conductivity of the SnTe monolayer, xy and yx components](../../_static/images/howtos/plotting/snte_shc.png)

In the SnTe monolayer (example08) the two components have opposite signs. Between $-0.30$ and 0.38 eV, the VBM and CBM of the run, both stay below 0.0023, about 1% of their largest value: inside the gap the occupied states do not change as $E_F$ moves.

## Conductivity and Seebeck coefficient

`transport()` writes one file per tensor, with the columns temperature (K), chemical potential $\mu$ relative to $E_F$ (eV), and the components $xx$, $yy$, $zz$, $xy$, $xz$, $yz$. Without scattering channels the relaxation time is constant and equal to 1 s, so `sigmagauss_0.dat` holds $\sigma/\tau$ (see [Tutorial 02](../tutorial02.ipynb)). `t_ele` selects the components: the default is the three diagonal ones, `[(0, 1)]` gives $xy$, and `[]` plots the average of the diagonal. Each temperature in the file gets its own figure. {py:meth}`~PAOFLOW.GPAO.GPAO.plot_seebeck` converts $S$ from V/K to $\mu$V/K.

```{literalinclude} plotting/main.py
:language: python
:start-after: '# [transport]'
:end-before: '# [/transport]'
```

The automatic y range of these plots covers the whole $\mu$ range of the file, not only `x_lim`. For AlP it would reach $-1.88\times10^{4}\,\mu$V/K, the value at $\mu = -8$ eV, so the call gives `y_lim`.

![Conductivity over relaxation time of Al against the chemical potential, three diagonal components](../../_static/images/howtos/plotting/al_sigma.png)

For cubic Al (example02) the three diagonal components coincide, and $\sigma/\tau$ is $3.20\times10^{21}\,(\Omega\,\mathrm{m\,s})^{-1}$ at $E_F$ and 300 K.

![Seebeck coefficient of AlP against the chemical potential, average of the diagonal components](../../_static/images/howtos/plotting/alp_seebeck.png)

For AlP (example06) the VBM and CBM of the run are at $-0.81$ and 1.62 eV. The Seebeck coefficient is positive for $\mu$ below 1.43 eV, where holes carry the current, and negative above, with extremes of $+4495\,\mu$V/K at 1.35 eV and $-2876\,\mu$V/K at 1.51 eV.

## Fermi surface

`fermi_surface()` writes `FermiSurf_0.bxsf`, the energies of the bands that cross $E_F$ on the full k-grid, relative to $E_F$, in the BXSF format of XCrySDen. The `fermi-plotter` command draws the surface where these energies equal the `Fermi Energy` line of the file (0 for PAOFLOW files), colored by $|\nabla_\mathbf{k} E|$ in eV bohr. It needs Mayavi, which the `fermisurface` extra installs (`pip install "PAOFLOW[fermisurface]"`), and for the interactive window also a Qt binding such as PySide6.

```bash
fermi-plotter plotting/bi/FermiSurf_0.bxsf --interp 2 --cmap viridis
fermi-plotter plotting/bi/FermiSurf_0.bxsf --interp 2 --cmap viridis --save bi_fermi_surface.png
```

The first command opens an interactive window. The second writes the PNG instead; on Linux and Windows it renders off-screen and needs no Qt binding. The options that change the surface:

- `--interp N`: Fourier upsampling of the k-grid by a factor `N` before the surface is drawn; the BXSF grid of the run is often too coarse for smooth sheets.
- `--fermi-shift dE`: draws the surface at $E_F + dE$ (eV), the Fermi surface of the rigidly doped crystal.
- `--band`: one band or a list (`--band 15,16`) instead of all bands in the file.
- `--bz ws|cell|none`: folds the sheets into the Wigner–Seitz cell (default), keeps the reciprocal cell of the grid, or draws no outline.
- `--supercell N --center`: repeats the sheets over N cells with $\Gamma$ in the middle, to show pockets that the cell boundary cuts.

![Fermi surface of Bi in the first Brillouin zone, colored by the magnitude of the energy gradient](../../_static/images/howtos/plotting/bi_fermi_surface.png)

The figure shows the file of the Bi example, whose `fermi_surface()` keeps the bands within 1 eV of $E_F$ on the k-grid of the run (33 points per direction in the file), drawn with `--interp 2 --cmap viridis`. Two bands cross $E_F$. Band 16 forms the elongated sheets: its energy rises from a minimum of $-0.42$ eV inside them, so they enclose electrons. Band 15 reaches only 0.02 eV above $E_F$, and its hole pockets are the small fragments, which this grid barely resolves; `--band 15` draws them alone. A pocket of this size needs a denser k-grid in the run, because `--interp` only interpolates between the energies the file holds.

## Where to go next

- {py:class}`~PAOFLOW.GPAO.GPAO` lists every plotting method, including those for optics, phonons and electron–phonon properties, with their arguments.
- [Tutorial 01](../tutorial01.ipynb) and [Tutorial 02](../tutorial02.ipynb) produce the band, DOS and transport files used here and explain what they contain.
- [Comparing pyskeaf orbits with quantum oscillations](qo_analysis.md) uses the BXSF files of `fermi_surface()` for de Haas–van Alphen orbits.
