# How to compare pyskeaf orbits with measured quantum-oscillation frequencies

On this page you test whether the extremal orbits of your calculated Fermi surface explain the frequencies measured in a de Haas–van Alphen, Shubnikov–de Haas or torque rotation. For each field direction, {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.pyskeaf` finds every extremal cross-section $A_\mathrm{ext}$ of the Fermi surface perpendicular to the field. It gives the frequency of each one, $F = \hbar A_\mathrm{ext}/(2\pi e)$ by the [Onsager relation](../theory/quantum-oscillations.md#the-onsager-relation), together with its cyclotron mass and carrier type. The class {py:class}`~PAOFLOW.agentic.qo_analysis.analysis_driver.QuantumOscillationAnalysis` reads these tables and the measured peaks, and puts both on the same angle axis. It then tests whether the agreement is better than chance. The comparison rests on three approximations:

- rigid Fermi surface: the sheets of the DFT bands at the DFT Fermi level;
- band masses: no electron–phonon or electron–electron renormalisation, which the measured masses contain;
- single-band orbits: no magnetic breakdown between sheets.

:::{tip}
This page is built on the case [`nbsn2_qo_analysis`](https://github.com/marcobn/PAOFLOW/tree/develop/examples/cases/nbsn2_qo_analysis): open it in PAOFLOW Studio with *Examples…* → `nbsn2_qo_analysis`, or copy it with `paoflow-gen --example nbsn2_qo_analysis`. It holds the pyskeaf tables and the measured peaks, so no QE or PAOFLOW run is needed. Download {download}`qo_rotation.py <quantum-oscillations/qo_rotation.py>` into the copy, or beside your own data, to run this comparison (about 5 s; it needs `pip install "PAOFLOW[qo_analysis]"`). Run {download}`figures.py <quantum-oscillations/figures.py>` after it to draw the figure and check the numbers of this page.
:::

## Setting up the analysis

The calculated side is the folder of a pyskeaf rotation. It holds one file `qo_EF_0_freqvsangle_<band>.out` per band, plain or gzip-compressed, which `pyskeaf()` writes from the band files of {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.fermi_surface` (see [Finding extremal orbits on the interpolated Fermi surface](../theory/quantum-oscillations.md#finding-extremal-orbits-on-the-interpolated-fermi-surface)). The constructor takes this folder and describes the rotation in it:

```{literalinclude} quantum-oscillations/qo_rotation.py
:language: python
:end-before: '# Field directions in both conventions'
```

| Argument | Meaning | NbSn2 |
| --- | --- | --- |
| `skeaf_dir` | folder of the pyskeaf rotation | `calc/soc/`: spin–orbit bands on a $49^3$ k-mesh |
| `bands` | bands whose orbits are compared | 11 to 18 |
| `degenerate_groups` | size of the groups of bands that each form one Fermi surface | `2`, the Kramers pairs |
| `sweep_angle` | the pyskeaf angle that the rotation varied | `'polar'`: 90 angles $1.011^\circ$ apart, from $0^\circ$ to $90^\circ$ |
| `fixed_angle` | the value of the angle held fixed (deg) | azimuthal angle $29.95^\circ$ |

pyskeaf measures the polar angle from $+z$ and the azimuthal angle from $+x$, in the Cartesian frame of the band files. The remaining arguments turn crystal directions into these angles (see [Field directions and the angle convention](#field-directions-and-the-angle-convention)).

Group the bands so that each group is one Fermi surface. With spin–orbit coupling in a centrosymmetric non-magnetic crystal such as NbSn2, every band is a Kramers pair. Bands 11 and 12 therefore describe one surface, and their orbits count once (see [Spin–orbit coupling and Kramers pairs](../theory/quantum-oscillations.md#spinorbit-coupling-and-kramers-pairs)). The default `'auto'` detects the pairs from the band files in `bxsf_dir`, and without these files it keeps each band alone. Pass `2` when you have only the frequency tables.

pyskeaf slices a supercell that holds several copies of the reciprocal cell, so it finds each orbit several times. It writes the number of copies in the last column of its tables. A genuine orbit on a smooth sheet appears in many copies, and an artefact at the supercell walls in only a few. The copy count therefore measures how far you can trust an orbit (see [Supercell copies](../theory/quantum-oscillations.md#supercell-copies)). `fermi_surface_inventory()` writes the histogram of these counts:

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

The histogram is not bimodal, so it does not separate artefacts from genuine orbits, and the cut is your choice. The comparison below keeps the orbits with at least 50 copies, the `min_copies` cut. It repeats the test at 100 copies to show that the conclusion does not depend on the cut.

## Field directions and the angle convention

A measured field direction is given by crystal indices $[hkl]$, and turning them into a pair of pyskeaf angles needs a convention:

| `field_convention` | Field along | Meaning of $[hkl]$ |
| --- | --- | --- |
| `'reciprocal'` | $h\mathbf{a}^* + k\mathbf{b}^* + l\mathbf{c}^*$ | normal to the $(hkl)$ planes, the usual meaning of the indices of a sample facet |
| `'real'` | $h\mathbf{a} + k\mathbf{b} + l\mathbf{c}$ | along the lattice row |

The two agree only for single-index directions of orthogonal axes (see [Field direction and crystal indices](../theory/quantum-oscillations.md#field-direction-and-crystal-indices)). {py:meth}`~PAOFLOW.agentic.qo_analysis.analysis_driver.QuantumOscillationAnalysis.field_directions` prints both for each measured direction, and checks the selected one against the angles you expect. The source labels one direction $[11y]$; it is the direction measured at $45^\circ$, here given as $[1\,1\,3.902]$:

```{literalinclude} quantum-oscillations/qo_rotation.py
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

In NbSn2 the $[110]$ field lies at $29.95^\circ$ from $\mathbf{a}$ in the reciprocal convention and at $60.05^\circ$ in the real one. The wrong convention would therefore compare the measurement with a rotation in another plane. The conventional cell enters only here, through `primitive_cell` and the integer rows of `conventional_from_primitive`. A `day` entry selects the Lifshitz–Kosevich (LK) rows of one measurement session.

## Reading the measurement

The measured side is two CSV tables. Their `#` lines record where the numbers come from.

| Table | Columns | Content |
| --- | --- | --- |
| sweep table | `angle_deg,freq_T,rel_mag` | the peaks of the rotation; `rel_mag` is the FFT magnitude divided by the largest in the data set |
| LK table | `direction,exp_angle_deg,day,label,freq_T,freq_err_T,mstar_me,mstar_err_me,observed,source` | labelled frequencies at single directions, with the cyclotron masses from the temperature dependence of their amplitudes (see [Temperature damping and the measured mass](../theory/quantum-oscillations.md#temperature-damping-and-the-measured-mass)) |

Here the sweep peaks were digitised from an image of a torque FFT map, and the LK values transcribed from text, so they are not raw data. Three functions help you build such tables from a publication:

| Function | Reads | Gives |
| --- | --- | --- |
| {py:meth}`~PAOFLOW.agentic.qo_analysis.analysis_driver.QuantumOscillationAnalysis.digitize_fft_map` | an image of an FFT map | the peaks, checked against the LK table |
| {py:func}`~PAOFLOW.agentic.qo_analysis.experiment_data.parse_text_peaks` | text | frequencies and masses, for you to check against the source |
| {py:func}`~PAOFLOW.agentic.qo_analysis.experiment_data.extract_slide_deck` | a `.pptx` deck | its images and text, for either of the two above |

```{literalinclude} quantum-oscillations/qo_rotation.py
:language: python
:start-after: '# The measured peaks and the LK table'
:end-before: '# Matches against chance'
```

`angle_scale` and `angle_offset` map the measured rotation angle onto the calculated one. Here the sample turned from $[001]$ to $[110]$ in the plane that pyskeaf rotated through, so the two angles coincide. A sample mounted with an offset needs `angle_offset`. A rotation measured from $[110]$ towards $[001]$ needs `angle_scale=-1.0, angle_offset=90.0`.

## Testing the rotation against chance

{py:meth}`~PAOFLOW.agentic.qo_analysis.analysis_driver.QuantumOscillationAnalysis.sweep_comparison` takes each significant measured peak and finds the calculated orbit nearest in $\ln F$ within an angle window. It counts the peak as matched when the two frequencies differ by less than a relative tolerance (see [Pointwise matches and chance baselines](../theory/quantum-oscillations.md#pointwise-matches-and-chance-baselines)). Three arguments set the test:

| Argument | Meaning | Here | How to choose it |
| --- | --- | --- | --- |
| `angle_window` | half-width of the window around each measured angle (deg) | 2.6 | from the uncertainty of the sample alignment and the angle step of the calculation |
| `tolerance` | largest relative difference of a matched frequency | 0.05 | from the accuracy you expect of the calculated areas, which the [resolution of the k-mesh](../theory/quantum-oscillations.md#resolution-of-the-k-mesh) and the position of the Fermi level limit |
| `weak_threshold` | fraction of the largest FFT magnitude below which a peak is not tested | 0.15 | high enough to leave out harmonics and combinations of stronger peaks, which are not extremal orbits |

Both windows also let in chance coincidences. The method therefore compares the match rate with two baselines. The first draws random frequencies at the measured angles. The second moves each measured frequency to a random measured angle, so it rewards only agreement specific to each angle.

```{literalinclude} quantum-oscillations/qo_rotation.py
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

On NbSn2, 44 % of the 160 peaks match, against $41 \pm 4$ % and $39 \pm 4$ % by chance. This rate is no evidence that the calculation reproduces the measurement. The orbits are too dense: with 7654 orbits over three decades of frequency, a $\pm 5\,\%$ window around any frequency holds an orbit about 40 % of the time. At `min_copies=100` the rate and the log-uniform baseline both fall by 6 points, so the cut does not change this conclusion.

![Calculated extremal-orbit frequencies of NbSn2 against the rotation angle from [001] to [110], coloured by Kramers pair, with the measured torque peaks marked as matched or unmatched within 5 %](../../_static/images/howtos/quantum-oscillations/nbsn2_sweep_overlay.png)

Assign the measured branches from their shape instead. A measured branch belongs to a calculated branch of one group when it follows that branch's frequency and slope over several angles. The figure shows the 160 significant peaks over the orbits with at least 50 copies.

The highest measured branch, between 3.7 and 5.0 kT from $5^\circ$ to $35^\circ$, follows the orbits of bands 15,16. The single-direction test below assigns it to the same sheet. The pointwise test does not: at $5^\circ$ it matches the 3692 T peak to a hole orbit of bands 13,14 at 3669 T, found in only 56 copies. The nearest orbit of bands 15,16, at 3456 T, lies 6.4 % below the peak, outside the tolerance. A pointwise match names the nearest orbit, not the sheet that makes the branch. To follow a branch, read `qo_rotation/sweep/per_angle_listing.txt`, which lists the measured and calculated frequencies side by side at each measured angle.

## Anisotropy and mass enhancement at single directions

The LK table adds two tests that coincidences rarely pass: the anisotropy of a branch and its mass. {py:meth}`~PAOFLOW.agentic.qo_analysis.analysis_driver.QuantumOscillationAnalysis.direction_analysis` takes each label measured at two directions. It ranks every pair of calculated orbits of one group by how well the pair reproduces the ratio of the two measured frequencies, then the frequencies themselves, then the masses (see [Anisotropy ratios](../theory/quantum-oscillations.md#anisotropy-ratios)). The ratio tests the shape of the sheet, because an error that scales the whole sheet cancels in it. `anisotropy_window` (35 %) limits the orbits it tries to those near the two measured frequencies. Widen it when a sheet has the right shape but a large error in size.

```{literalinclude} quantum-oscillations/qo_rotation.py
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
[001] -> [110]
label group  F_ref_exp  F_ref_calc  F_cmp_exp  F_cmp_calc  ratio_exp  ratio_calc  1+lambda_ref                                                         verdict
beta1 13,14      102.0      90.795      312.6     277.936      3.065       3.061         2.854 anisotropy reproduced; uniform offset (right shape, wrong size)
```

Both $\sigma$ branches find the same 563 T sheet of bands 11,12, whose frequency falls by 2 % towards $[11y]$. The measured $\sigma_2$ falls by 1.4 %, so its assignment holds. The measured $\sigma_1$ rises by 7 %, so its match at $[001]$ is a coincidence.

$\mu$ keeps its ratio, 1.581 against 1.583, and the calculation is about 6 % low at both directions. The sheet has the right shape and the wrong size. The assignment holds, and the 6 % measures the error of the calculated area.

The column `1+lambda_ref` is the [mass enhancement](../theory/quantum-oscillations.md#mass-enhancement) $1 + \lambda = m^*_\mathrm{LK}/m^*_\mathrm{band}$ at the first direction. It contains every renormalisation that the DFT bands lack. It should therefore exceed 1, and be similar for branches of comparable character. It is 2.48 for $\mu$, 2.75 for $\sigma_2$ and 2.85 for $\beta_1$. $\delta$ gives 1.09, almost no enhancement, although its ratio is reproduced, which makes it the least secure of these assignments.

## When a branch has no calculated partner

Check the calculation before you report a branch as unexplained. Three methods test the parts of the calculation that can hide an orbit:

| Method | Tests | How |
| --- | --- | --- |
| {py:meth}`~PAOFLOW.agentic.qo_analysis.analysis_driver.QuantumOscillationAnalysis.compare_runs` | the [resolution of the k-mesh](../theory/quantum-oscillations.md#resolution-of-the-k-mesh), or the effect of spin–orbit coupling | compares the orbits, one by one, with those of a denser or a scalar-relativistic run registered with `add_reference_run()` |
| {py:meth}`~PAOFLOW.agentic.qo_analysis.analysis_driver.QuantumOscillationAnalysis.fermi_level_scan` | the position of the Fermi level | follows the orbits in runs at $E_F \pm$ a few meV, and checks their shift against the [rigid-band slope](../theory/quantum-oscillations.md#rigid-shift-of-the-fermi-level) $dF/dE = \pm 8.637\,(m^*/m_e)$ T/meV, positive for electron orbits |
| {py:meth}`~PAOFLOW.agentic.qo_analysis.analysis_driver.QuantumOscillationAnalysis.breakdown_gaps` | [magnetic breakdown](../theory/quantum-oscillations.md#magnetic-breakdown) | finds near-contacts between sheets, where breakdown can create orbits that pyskeaf does not trace |

The case's `main.py` runs all three. It also runs `breakdown_network()`, which follows the orbits across these contacts and takes minutes per angle and core, so run it on one angle first.

The weak peaks that the sweep comparison leaves out can also be explained without a new orbit. {py:meth}`~PAOFLOW.agentic.qo_analysis.analysis_driver.QuantumOscillationAnalysis.weak_peak_analysis` tests, against chance, whether they are harmonics or combinations of significant peaks.

## Where to go next

- [Quantum oscillations and extremal Fermi-surface orbits](../theory/quantum-oscillations.md) derives the frequencies, masses and tests used here.
- {py:class}`~PAOFLOW.agentic.qo_analysis.analysis_driver.QuantumOscillationAnalysis` lists every method with its parameters and the files it writes.
- [agentic/skeaf/](https://github.com/marcobn/PAOFLOW/tree/develop/agentic/skeaf) holds the playbook and prompts for an AI agent that runs this analysis in stages on the same class.
