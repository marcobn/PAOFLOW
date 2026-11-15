# <SYSTEM>: quantum oscillations vs calculated Fermi surface

<!-- Written for experimental collaborators. Fill every <...>; delete sections that do not apply.
     Keep numbers identical to the outputs of plot_figures.py / weak_peak_tests.py. -->

| File | Content |
|---|---|
| `fig1_sweep_overlay.png` | calculated extremal orbits vs measured peaks over the rotation, with branch assignments |
| `fig2_network_orbits.png` | magnetic-breakdown network orbits vs the peaks no ordinary orbit explains (if computed) |
| `plot_figures.py`, `weak_peak_tests.py` | regenerate figures and statistics (Python 3 + numpy + matplotlib) |
| `data/` | all inputs (see the end) |

> **Provenance caveat** - <state exactly where the experimental numbers come from: raw data, published tables,
> transcribed slide text, or DIGITISED images (give file, slide, method and the validation accuracy).
> Say what is reliable (e.g. frequencies to ~1 %) and what is not (amplitudes, weak peaks).>

## 1. Inputs
- Experiment: <technique, rotation plane, angle step, field window / FFT resolution, temperature>.
- Calculation: <code, functional, SOC or not, k-grid, pyskeaf num_interpolation, E_F, bands>.
- Geometry: <field-angle convention, real vs reciprocal [hkl], how the experimental angle maps to the calculation>.

## 2. Figure 1 - how to read it, assignments
- How to read: dots = calculated extremal orbits (colour = band group, size ~ curvature amplitude factor),
  circles = measured peaks (size = FFT magnitude), arrows = assignments, grey italic = unexplained branches.
- Chance level: <x % of peaks lie within 5 % of some orbit vs y +/- z % for random frequencies> ->
  assignments rest on branch shape over the rotation and on 1 + lambda.

| Measured branch | Calculated sheet | Agreement | 1 + lambda | Confidence |
|---|---|---|---|---|
| <label> | <group, carrier, max/min> | <deviation over the rotation> | <m_exp/m_band> | <high/medium/low> |

## 3. What was tested to explain the failures
| Test | Result |
|---|---|
| convergence (grid, num_interpolation) | <...> |
| Fermi-level shift | <...> |
| spin-orbit coupling | <...> |
| magnetic breakdown | <...> |

## 4. Bottom line
1. <what is explained>  2. <what is not>  3. <what is expected but not demonstrated>

## 5. What would help from the experimental side
<raw data; field-window FFTs (breakdown vs mixing); angle calibration; LK masses; second sample>

## 6. Reproducing
`python plot_figures.py` ; `python weak_peak_tests.py` - data files and columns:
`calc_orbits.csv` (angle_deg, band, group, freq_T, mstar_me, carrier, extremum, curv_kTA2, amp_factor,
num_orbit_copies), `exp_sweep_peaks.csv` (angle_deg [calculation angle], freq_T, rel_mag),
`branch_assignments.csv`, `network_orbits.csv`, `meta.json` (thresholds, colours, touching groups).

## 7. How to read calc_orbits.csv (FAQ)
- One row = one extremal orbit (dA/dk_H = 0), not one Fermi surface. F[T] = 10 476 x A[A^-2].
- With SOC + inversion + time reversal, bands come in Kramers pairs (same Fermi surface); count one per pair.
- Repeated, equal frequencies within one band = symmetry-equivalent orbits at different k-points.
- `extremum`: max (curv < 0) or min (curv > 0, e.g. necks); tips of pockets are not extremal orbits.
- `num_orbit_copies` = supercell repetitions of the same orbit (robustness), already averaged into the row.
