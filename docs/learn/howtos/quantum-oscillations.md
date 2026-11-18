# How to compare pyskeaf orbits with measured quantum-oscillation frequencies

On this page you test whether the extremal orbits of your calculated Fermi surface explain the frequencies measured in a de Haas–van Alphen, Shubnikov–de Haas or torque rotation. {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.pyskeaf` gives, for each field direction, the frequency $F = \hbar A_\mathrm{ext}/(2\pi e)$ (the Onsager relation) of every extremal cross-section $A_\mathrm{ext}$ of the Fermi surface perpendicular to the field, with its cyclotron mass and carrier type. The class {py:class}`~PAOFLOW.agentic.qo_analysis.analysis_driver.QuantumOscillationAnalysis` reads these tables and the measured peaks, puts both on the same angle axis, and tests the agreement against chance. The comparison inherits the approximations of the calculation: a rigid Fermi surface from the DFT bands at $E_F$, and band masses without electron–phonon or electron–electron renormalisation.

Base example: [examples/qo_analysis_examples/NbSn2/](https://github.com/marcobn/PAOFLOW/tree/master/examples/qo_analysis_examples/NbSn2)

:::{tip}
The inputs of this page are in `howto_assets.tar.gz` of the release [tutorial-assets-v2](https://github.com/marcobn/PAOFLOW/releases/tag/tutorial-assets-v2). Extract it with `tar xzf howto_assets.tar.gz`, which creates `quantum-oscillations/`, and run the code below, or the whole script {download}`main.py <quantum-oscillations/main.py>`, from the directory that contains it; {download}`figures.py <quantum-oscillations/figures.py>` draws the figure of this page from the outputs of `main.py`. The analysis needs the optional dependencies of `pip install "PAOFLOW[qo_analysis]"`. The script runs in about 30 s.
:::

## The inputs

The calculated side is the folder of a pyskeaf rotation, with one file `qo_EF_0_freqvsangle_<band>.out` per band (plain or gzip-compressed). For the NbSn2 example it holds the spin–orbit bands 11 to 18 on a $49^3$ grid, at 90 polar angles $1.011^\circ$ apart, from $0^\circ$ to $90^\circ$, at azimuthal angle $29.95^\circ$. pyskeaf measures the polar angle from $+z$ and the azimuthal angle from $+x$, in the Cartesian frame of the BXSF files.

The measured side is two CSV tables, whose `#` lines record where the numbers come from. The sweep table, `angle_deg,freq_T,rel_mag`, holds the peaks of the rotation, with `rel_mag` the FFT magnitude divided by the largest in the data set. The Lifshitz–Kosevich (LK) table, `direction,exp_angle_deg,day,label,freq_T,freq_err_T,mstar_me,mstar_err_me,observed,source`, holds the labelled frequencies and cyclotron masses at single directions. In this example the sweep peaks were digitised from an image of a torque FFT-vs-angle map and the LK values transcribed from text, so they are not raw data. When your rotation exists only as such an image, {py:meth}`~PAOFLOW.agentic.qo_analysis.analysis_driver.QuantumOscillationAnalysis.digitize_fft_map` reads the peaks from it, given the pixel positions of two ticks on each axis and of the colour bar, and checks them against the LK table.

## Setting up the analysis

```{literalinclude} quantum-oscillations/main.py
:language: python
:end-before: '# Field directions in both conventions'
```

Group the bands into the Fermi surfaces they form with `degenerate_groups`. With spin–orbit coupling in a centrosymmetric non-magnetic crystal such as NbSn2 every band is a Kramers pair, so bands 11 and 12 describe one surface and their orbits count once. `'auto'` detects the pairs from the BXSF files in `bxsf_dir`, and on NbSn2 it finds the same four pairs, but without BXSF files it keeps each band alone; pass `2` when you have only the frequency tables. `sweep_angle` and `fixed_angle` name the angle the rotation varied and the one it held.

pyskeaf finds each orbit in many copies of the reciprocal cell, in the supercell it slices, and writes their number in the last column. Genuine orbits appear in many copies, while artefacts at the supercell walls appear in a few. `fermi_surface_inventory()` writes the histogram of these counts:

```text
      copies  entries
         0-2     1960
         2-4     1028
        4-10     1005
       10-20      638
       20-50      995
       50-80      785
      80-100      425
     100-150     2516
     150-200     3509
```

For this run the histogram is not cleanly bimodal: the number of orbits per unit copy count falls up to 80 to 100 copies and rises above, so no count separates the two kinds of orbit and the cut is a choice. The comparison below keeps orbits with at least 50 copies, the `min_copies` cut, and repeats the test at 100 to show that the conclusion does not depend on it.

## Field directions and the angle convention

A crystal direction becomes a pair of pyskeaf angles only with a convention: a field normal to the $(hkl)$ planes, along $h\mathbf{a}^* + k\mathbf{b}^* + l\mathbf{c}^*$ (`field_convention='reciprocal'`, the usual meaning of the indices of a sample facet), or a field along the row $h\mathbf{a} + k\mathbf{b} + l\mathbf{c}$ (`'real'`). The two agree only for single-index directions of orthogonal axes. {py:meth}`~PAOFLOW.agentic.qo_analysis.analysis_driver.QuantumOscillationAnalysis.field_directions` prints both for each measured direction and checks the selected one against the angles you expect. The label $[11y]$ of the source is the direction measured at $45^\circ$, here given as $[1\,1\,3.902]$:

```{literalinclude} quantum-oscillations/main.py
:language: python
:start-after: '# Field directions in both conventions'
:end-before: '# The measured peaks and the LK table'
```

```text
[11y] hkl=[1, 1, 3.902]: target az=29.95, pol=45.0 -> OK
   reciprocal: azimuthal   29.949, polar   45.003  <== selected
   real      : azimuthal   60.051, polar    8.633
[110] hkl=[1, 1, 0]: target az=29.95, pol=90.0 -> OK
   reciprocal: azimuthal   29.949, polar   90.000  <== selected
   real      : azimuthal   60.051, polar   90.000
```

In orthorhombic NbSn2 the $[110]$ field lies at $29.95^\circ$ from $\mathbf{a}$ in the reciprocal convention and at $60.05^\circ$ in the real one, so the wrong convention compares the measurement with a rotation in another plane. The conventional cell enters only here, as `primitive_cell` and the integer rows of `conventional_from_primitive`. A `day` entry selects the LK rows of one measurement session.

## Reading the measurement

```{literalinclude} quantum-oscillations/main.py
:language: python
:start-after: '# The measured peaks and the LK table'
:end-before: '# Matches against chance'
```

`angle_scale` and `angle_offset` map the measured rotation angle onto the calculated one. Here the sample turned from $[001]$ to $[110]$ in the plane that pyskeaf rotated through, so the two angles coincide; a sample mounted with an offset needs `angle_offset`, and a rotation measured from $[110]$ towards $[001]$ needs `angle_scale=-1.0, angle_offset=90.0`.

## Testing the rotation against chance

{py:meth}`~PAOFLOW.agentic.qo_analysis.analysis_driver.QuantumOscillationAnalysis.sweep_comparison` tests each significant measured peak $(\theta_i, F_i)$, those with `rel_mag` at least `weak_threshold`. Among the calculated orbits $j$ within `angle_window` of $\theta_i$ it takes the nearest in $\ln F$, and counts the peak as matched when

$$
\frac{|F_{j^*} - F_i|}{F_i} \le t, \qquad j^* = \arg\min_{j:\,|\theta_j - \theta_i| \le \Delta\theta} |\ln F_j - \ln F_i| ,
$$

with $t$ the `tolerance` and $\Delta\theta$ the `angle_window`. Choose $\Delta\theta$ from the uncertainty of the sample alignment and the angle step of the calculation, and $t$ from the accuracy you expect of the calculated areas. Both settings let in chance coincidences as well as genuine matches, which is why the rate is always quoted with the baselines below.

```{literalinclude} quantum-oscillations/main.py
:language: python
:start-after: '# Matches against chance'
:end-before: '# Anisotropy and mass enhancement'
```

```text
robust orbits: 7654 (NumOrbCopy >= 50); significant peaks: 160
matched within 5 % (+/-2.6 deg): 44 %
chance: log-uniform 41 +/- 4 %, angle-shuffled 39 +/- 4 %
robust orbits: 6648 (NumOrbCopy >= 100); significant peaks: 160
matched within 5 % (+/-2.6 deg): 38 %
chance: log-uniform 35 +/- 4 %, angle-shuffled 34 +/- 4 %
```

The match rate is meaningful only next to the rate that random data reach with the same orbits. The first baseline replaces the measured frequencies by frequencies drawn log-uniformly between 15 and 8000 T at the measured angles. The second keeps the measured frequencies and moves each to a randomly drawn measured angle, so it rewards only agreement specific to each angle. Both are averaged over 200 draws. On NbSn2, 44 % of the 160 peaks match, against 41 % and 39 % by chance with a spread of 4 %. With 7654 orbits spread over three decades of frequency, a $\pm 5\,\%$ window around any frequency contains an orbit about 40 % of the time, so the pointwise rate is no evidence that the calculation reproduces the measurement. At `min_copies=100` the rate and the baselines both fall by 6 points, so the cut does not change this.

![Calculated extremal-orbit frequencies of NbSn2 against the rotation angle from [001] to [110], coloured by Kramers pair, with the measured torque peaks marked as matched or unmatched within 5 %](../../_static/images/howtos/quantum-oscillations/nbsn2_sweep_overlay.png)

Assign the measured branches from their shape instead: a calculated branch of one group that follows the measured frequency and its slope over several angles. The figure shows the 160 significant peaks over the orbits with at least 50 copies; the weak peaks are not tested and not drawn. The highest measured branch, between 3.7 and 5.0 kT from $5^\circ$ to $35^\circ$, lies along the amber orbits of bands 15,16, the sheet that the single-direction test below also gives for it. Yet at $5^\circ$ the test matches its 3692 T peak to a hole orbit of bands 13,14 at 3669 T, found in only 56 copies, because the 15,16 orbit nearest in frequency, at 3449 T, lies 6.6 % below. A pointwise match names the nearest orbit, not the sheet that makes the branch. `sweep_comparison` also writes `per_angle_listing.txt` to `qo_analysis/sweep/`, the measured and calculated frequencies side by side at each measured angle, which is the table to read while you follow a branch.

## Anisotropy and mass enhancement at single directions

The LK table adds two tests that a coincidence rarely passes. {py:meth}`~PAOFLOW.agentic.qo_analysis.analysis_driver.QuantumOscillationAnalysis.direction_analysis` takes each label measured at two directions, tries every pair of orbits of one group within `anisotropy_window` (35 %) of the two measured frequencies, and ranks the pairs by

$$
s = \left|\frac{F_2^\mathrm{c}}{F_1^\mathrm{c}} - \frac{F_2^\mathrm{e}}{F_1^\mathrm{e}}\right| + \frac{1}{2}\sum_{i=1,2} \frac{|F_i^\mathrm{c} - F_i^\mathrm{e}|}{F_i^\mathrm{e}} + \frac{w}{2}\sum_{i=1,2}\left|\ln\frac{m_i^\mathrm{e}}{m_i^\mathrm{c}}\right| ,
$$

with c for calculated, e for measured and $w = 0.2$. A ratio within 0.03 of the measured one is "reproduced", and deviations above 5 % at both directions that differ by less than 3 points are flagged as a uniform offset.

```{literalinclude} quantum-oscillations/main.py
:language: python
:start-after: '# Anisotropy and mass enhancement'
```

```text
[001] -> [11y]
 label group  F_ref_exp  F_ref_calc  F_cmp_exp  F_cmp_calc  ratio_exp  ratio_calc  1+lambda_ref                                                         verdict
 delta 13,14      977.0    1061.412     1245.0    1333.559      1.274       1.256         1.094 anisotropy reproduced; uniform offset (right shape, wrong size)
    mu 15,16     3672.0    3446.367     5804.0    5455.078      1.581       1.583         2.478 anisotropy reproduced; uniform offset (right shape, wrong size)
sigma1 11,12      478.0     563.460      512.0     552.901      1.071       0.981         3.179                                WRONG SIGN - likely coincidental
sigma2 11,12      586.0     563.460      578.0     552.901      0.986       0.981         2.753                                           anisotropy reproduced
```

Both $\sigma$ branches find the same 563 T sheet of bands 11,12, whose frequency falls by 2 % towards $[11y]$. $\sigma_2$ falls by 1.4 %, but $\sigma_1$ rises by 7 %, so its match at $[001]$ is a coincidence. $\mu$ keeps its ratio, 1.581 against 1.583, with the calculation about 6 % low at both directions: the sheet has the right shape and the wrong size, so the assignment holds and the 6 % measures the error of the calculated area. The last column is the mass enhancement $1 + \lambda = m^*_\mathrm{LK}/m^*_\mathrm{band}$, which should exceed 1 and be similar for branches of comparable character: 2.48 for $\mu$, 2.75 for $\sigma_2$ and 2.85 for $\beta_1$ (from the $[001] \to [110]$ table), whereas $\delta$, whose ratio is reproduced, gives 1.09, almost no enhancement, which makes it the least secure of these assignments.

## When a branch has no calculated partner

Before you report a measured branch as unexplained, check the calculation. {py:meth}`~PAOFLOW.agentic.qo_analysis.analysis_driver.QuantumOscillationAnalysis.compare_runs` compares the orbits with those of a denser or scalar-relativistic run registered with `add_reference_run()`; {py:meth}`~PAOFLOW.agentic.qo_analysis.analysis_driver.QuantumOscillationAnalysis.fermi_level_scan` follows the orbits in runs at $E_F \pm$ a few meV and checks their shift against the rigid-band slope $dF/dE = \pm 8.637\,(m^*/m_e)$ T/meV, positive for electron orbits; and {py:meth}`~PAOFLOW.agentic.qo_analysis.analysis_driver.QuantumOscillationAnalysis.breakdown_gaps` finds near-contacts between sheets where magnetic breakdown can create orbits that pyskeaf does not trace. The base example runs all three. The orbits of the breakdown network itself, from `breakdown_network()`, take minutes per angle and core, so run one angle first.

## Where to go next

- {py:class}`~PAOFLOW.agentic.qo_analysis.analysis_driver.QuantumOscillationAnalysis` lists every method with its parameters and the files it writes.
- {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.pyskeaf` documents the rotation that writes the frequency tables, from the BXSF files of {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.fermi_surface`.
- The folder [agentic/skeaf/](https://github.com/marcobn/PAOFLOW/tree/master/agentic/skeaf) holds the playbook and prompts with which an AI agent runs this analysis in stages, from the geometry check to a report for experimental collaborators, on the same class.
