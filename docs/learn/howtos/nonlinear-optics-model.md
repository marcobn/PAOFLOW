# How to compute the non-linear optical response of a tight-binding model

On this page you compute the second-order optical response of a tight-binding model with {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.nonlinear_optics`: the shift current, the injection currents and the second-harmonic generation (SHG) susceptibility $\chi^{(2)}$. These responses vanish in a crystal with inversion symmetry. A model is therefore the natural place to test them: you switch inversion on and off with one parameter, and the point group tells you which components must vanish. The example is graphene with a sublattice potential, the simplest crystal without inversion. The [theory page](../theory/nonlinear-optics.md) derives the formulas the code evaluates. The responses rest on these approximations:

- independent electrons in the bands of the model, without excitons or local fields;
- the position operator diagonal on the orbital centres (the tight-binding approximation);
- interband transitions only, so the results apply to insulators and semiconductors;
- one constant broadening for all transitions, and injection currents given as rates, without a carrier relaxation time.

:::{tip}
This page is built on the case [`graphene_nlo_model`](https://github.com/marcobn/PAOFLOW/tree/develop/examples/cases/graphene_nlo_model): open it in PAOFLOW Studio with *Examples…* → `graphene_nlo_model`, or copy it with `paoflow-gen --example graphene_nlo_model`. The model is built in code, so the case needs no input files, and its run takes under a minute on one core. Download {download}`nonlinear_optics_model.py <nonlinear-optics-model/nonlinear_optics_model.py>` to compute the responses of your own model: replace `model` and set `nelec`, `components` and `nk` for it. Download {download}`figures.py <nonlinear-optics-model/figures.py>` to redraw the figures of this page from that run and to check the numbers it quotes, by running it in the same directory after `python nonlinear_optics_model.py > nonlinear_optics_model.log`.
:::

## Computing the responses

The script uses the `graphene` model with $\delta = 1$ eV, which puts the on-site energies $\pm\delta/2$ on the two sublattices. This model has a gap of 1 eV at K and K′, and the point group $D_{3h}$. Its lattice vector $\mathbf{a}_1$ lies along $x$, and one of its mirror planes is perpendicular to $x$. The script computes the responses directly after the constructor:

```{literalinclude} nonlinear-optics-model/nonlinear_optics_model.py
:language: python
:end-before: '# The same model with inversion symmetry'
```

`nonlinear_optics()` interpolates $H_{\alpha\beta}(\mathbf{R})$ at the k-points of its own mesh. On a model, you therefore call it right after the constructor, without `interpolated_hamiltonian()` or `pao_eigh()`. On a DFT Hamiltonian, it must come after `pao_hamiltonian()` and before these two calls, because they release $H_{\alpha\beta}(\mathbf{R})$.

Five arguments depend on the model:

| argument | in the script | what it sets | how to choose it |
| -------- | ------------- | ------------ | ---------------- |
| `components` | `['yyy', 'yxx', 'xxy', 'xxx']` | the tensor components $abc$: $a$ is the direction of the current or polarisation, $b$ and $c$ those of the field | the independent components of your point group, and those that test its relations ([Point groups and inversion](../theory/nonlinear-optics.md#point-groups-and-inversion)) |
| `nk` | `(300, 300, 1)`, 90 000 k-points of the two-dimensional zone | the k-mesh | sample the transition energies near K and K′, where the bands are closest, more finely than `eta`; otherwise the spectra oscillate with the mesh |
| `eta` | `0.05` eV, the default | the width of the Gaussians that replace the $\delta$ functions of the shift and injection currents, and the imaginary part of the resonance denominators of $\chi^{(2)}$ ([Broadening](../theory/nonlinear-optics.md#broadening)) | the resolution of the spectra: features narrower than `eta`, such as the onset at the gap, are broadened by it |
| `nelec` | `2`, which fills the lower band | the number of electrons, which a model does not have of its own | twice the number of occupied bands, because without spin–orbit coupling each band holds two electrons ([Occupations and spin](../theory/nonlinear-optics.md#occupations-and-spin)) |
| `twoD` | `True` | the responses of the sheet: the bulk responses multiplied by the height of the cell, $\lvert\mathbf{a}_3\rvert = 10a$ for this model, in nm µA/V² (shift current), nm A V⁻² s⁻¹ (injection currents) and pm²/V ($\chi^{(2)}$) | `True` for a layer, because without it the responses depend on the vacuum in the cell, which is not a property of the layer ([Bulk and sheet units](../theory/nonlinear-optics.md#bulk-and-sheet-units)) |

The point group fixes which components are independent. In $D_{3h}$ with this orientation, $yyy = -yxx = -xxy$ and $xxx = 0$ for every response. The list holds these four components, so the run checks the relations as well as computing the one independent component. To check the mesh, repeat the run on a denser one: the peaks should not move.

The position matrix elements are measured from the orbital centres. `nonlinear_optics()` takes the centres from the atomic positions of the model. An error in them breaks the symmetry of the responses (see [Checking the result](#checking-the-result)), so for a model of your own, give exact positions, or pass the centres in crystal coordinates as `orbital_positions`.

The run writes one file per response and component, with the photon energy (eV) and the value:

| files | response |
| ----- | -------- |
| `shift_*` | shift current |
| `shiftm_*` | magnetic shift current |
| `injectc_*` | circular injection current |
| `injectm_*` | magnetic injection current |
| `shgr_*`, `shgi_*`, `shga_*` | real part, imaginary part and modulus of $\chi^{(2)}$ |

{py:meth}`~PAOFLOW.GPAO.GPAO.plot_optical` reads the files by these prefixes, and `figures.py` draws the two figures below with it.

![Shift current of gapped graphene against photon energy, for the yyy, yxx, xxy and xxx components](../../_static/images/howtos/nonlinear-optics-model/nlo_shift.png)

The shift current sets in at the 1 eV gap, with an onset broadened by `eta`. It peaks just above the gap, at 14.74 nm µA/V² at 1.07 eV, and decays as the transitions move away from the valleys. Each transition displaces the electron by the [shift vector](../theory/nonlinear-optics.md#shift-current). The current is therefore largest where many transitions with a large displacement fall at one photon energy. The curves also obey the $D_{3h}$ relations: `yxx` and `xxy` coincide at $-\sigma^{yyy}$, and $\sigma^{xxx}$ is zero.

![Real part, imaginary part and modulus of the SHG susceptibility chi^yyy of gapped graphene against photon energy](../../_static/images/howtos/nonlinear-optics-model/nlo_shg.png)

$\chi^{(2)}$ has two resonances, at half the gap and at the gap ([Second-harmonic generation](../theory/nonlinear-optics.md#second-harmonic-generation)). The two-photon resonance, where $2\hbar\omega$ reaches the gap, gives the main peak: $|\chi^{yyy}|$ reaches $8.85 \times 10^{5}$ pm²/V at 0.51 eV. At the one-photon resonance, $|\chi^{yyy}|$ has only a shoulder. The real part has its minimum there, $-3.33 \times 10^{5}$ pm²/V at 0.98 eV. Below both resonances, $\chi^{(2)}$ tends to the static value $|\chi^{yyy}(0)| = 2.07 \times 10^{5}$ pm²/V.

## Checking the result

Symmetry gives the cheapest checks, because a component that the point group forbids must come out at the level of numerical noise. A model carries no symmetry operations, so `nonlinear_optics()` sums over the full mesh, and the relations test the calculation. On a DFT run, in contrast, the default `symmetry=True` imposes them ([Symmetry reduction of the k-mesh](../theory/nonlinear-optics.md#symmetry-reduction-of-the-k-mesh)).

A second run tests that the responses vanish with inversion. It uses the same model with $\delta = 0$, on a coarser mesh, because it only tests that the responses vanish:

```{literalinclude} nonlinear-optics-model/nonlinear_optics_model.py
:language: python
:start-after: '# The same model with inversion symmetry'
:end-before: '# Symmetry checks'
```

The script prints the checks after the two runs:

```{literalinclude} nonlinear-optics-model/nonlinear_optics_model.py
:language: python
:start-after: '# Symmetry checks'
```

The output is:

```text
shift current: peak 14.74 nm muA/V^2 at 1.07 eV
  |yyy + yxx|, |yyy + xxy| <= 7.3e-04, |xxx| <= 3.8e-15
|chi_yyy|: 2.070e+05 pm^2/V at 0, peak 8.849e+05 at 0.51 eV
Re chi_yyy: minimum -3.330e+05 pm^2/V at 0.98 eV
max |shiftm|: yyy 0.0e+00, yxx 0.0e+00, xxy 9.2e-14, xxx 0.0e+00
max |injectm|: yyy 2.5e-04, yxx 4.2e-04, xxy 8.7e-05, xxx 4.6e-04
max |injectc|: yyy 1.3e-06, yxx 1.1e-06, xxy 8.6e+04, xxx 9.7e-07
delta = 0: max |shift| 3.1e-13, max |chi| 7.5e-06
```

Each line tests one reason for a response to vanish:

| check | why it must vanish | largest value in the run |
| ----- | ------------------ | ------------------------ |
| $\sigma^{yyy} + \sigma^{yxx}$, $\sigma^{yyy} + \sigma^{xxy}$ | the $D_{3h}$ relations | $7 \times 10^{-4}$ nm µA/V², $5 \times 10^{-5}$ of the peak |
| $\sigma^{xxx}$ | the mirror perpendicular to $x$ | below $10^{-14}$ |
| `shiftm_*`, `injectm_*` | the magnetic currents are odd under [time reversal](../theory/nonlinear-optics.md#time-reversal), which the model preserves | below $10^{-13}$ and below $5 \times 10^{-4}$, in the units of their files |
| shift current and $\lvert\chi^{(2)}\rvert$ at $\delta = 0$ | inversion symmetry | $3 \times 10^{-13}$ nm µA/V² and $10^{-5}$ pm²/V |

Each of these responses vanishes, to the precision of the last column.

The circular injection current is the exception. It is antisymmetric in its two field indices, so its `yyy`, `yxx` and `xxx` components vanish for any crystal. Only `xxy` tests the point group. This component is allowed only in gyrotropic point groups, and $D_{3h}$ is not one of them ([Permutation of the field indices](../theory/nonlinear-optics.md#permutation-of-the-field-indices)). Yet `injectc_xxy.dat` reaches $-8.6 \times 10^{4}$ nm A V⁻² s⁻¹ at 1.75 eV.

The cause is the orbital centres. The built-in `graphene` model stores its atomic positions to five decimals. This rounding breaks the threefold rotation of the position operator at the $10^{-5}$ level, and it is also the source of the residual of the $D_{3h}$ relations in the table. With the exact centres, `orbital_positions=[[1/3, 1/3, 0], [2/3, 2/3, 0]]`, both drop to the level of the other forbidden components. Do not use `injectc_xxy` of the built-in model as a physical result.

## Applying it to a DFT Hamiltonian

On a Hamiltonian from a DFT run, three further arguments matter:

| argument | what it does | why it matters |
| -------- | ------------ | -------------- |
| `scissor` (eV) | shifts the conduction bands rigidly in every transition energy, and keeps the position matrix elements ([Scissor shift](../theory/nonlinear-optics.md#scissor-shift)) | it corrects the PBE gap |
| `position_dipoles=True` | adds the dipoles between orbitals of one atom to the position operator ([Position dipoles](../theory/nonlinear-optics.md#position-dipoles)) | for transitions such as $s$–$p$ across the gap of a III–V semiconductor |
| `degeneracy_tol` (eV, default 0.01) | groups nearly degenerate bands ([Degenerate bands](../theory/nonlinear-optics.md#degenerate-bands)) | without the grouping, the responses amplify the numerical asymmetries of these bands into forbidden components |

To test the symmetry relations of a DFT run, pass `symmetry=False`, so that the code sums over the full mesh as on a model.

## Where to go next

- [Second-order optical response](../theory/nonlinear-optics.md) derives the shift current, the injection currents and the SHG, with their symmetry constraints and approximations.
- [Non-linear optics](../../internals/nonlinear-optics.md) explains how the code evaluates them and lists the symmetry tests on model Hamiltonians and the GaAs and GeS examples.
- [How to run PAOFLOW on a tight-binding model](tight-binding-model.md) lists the other models and shows how to build one with your own parameters.
- [Berry phases and Chern number of gapped graphene](../tutorials/tutorial06.ipynb) computes the Berry curvature and Chern number of the same gapped graphene.
