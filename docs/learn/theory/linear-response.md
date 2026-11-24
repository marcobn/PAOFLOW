# Kubo linear response and the independent-particle dielectric tensor

This page derives the response tensors that PAOFLOW computes from the Kubo formula: the static response of an observable to an electric field, evaluated by {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.linear_response` (charge conductivity, spin Hall conductivity, Rashba–Edelstein and anomalous Hall tensors), and the frequency-dependent dielectric tensor of independent electrons, evaluated by {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.dielectric_tensor`. Both are sums over the eigenstates $\psi_{n\mathbf{k}}$ and eigenvalues $\varepsilon_{n\mathbf{k}}$ of the interpolated PAO Hamiltonian $\tilde H(\mathbf{k})$ and over matrix elements of the velocity operator, whose construction is explained in [The PAO Hamiltonian](pao-hamiltonian.md). The smearing of the occupations follows [Density of states](density-of-states.md). Each section names the approximation it makes and the argument of the method that controls it.

## The Kubo formula for a static electric field

A static field $\mathbf{E}$ changes the expectation value of an operator $\hat O_i$ by

$$\delta\langle \hat O_i\rangle = \chi_{ij}\,E_j ,$$

and Kubo's theory gives $\chi_{ij}$ as a correlation function of $\hat O_i$ with the current operator that couples to the field [1]. For electrons that do not interact except through a mean field, Bastin et al. [2] wrote this correlation function at zero frequency with the retarded and advanced Green's functions $G^{R,A}(E)$ of the crystal, and Crépieux and Bruno [3] split it into two parts with different physics. The first part, $\chi^{I}$, contains the derivative of the Fermi function, $-\partial f/\partial E$, so only states at the Fermi energy $E_F$ contribute: it is the Fermi-surface term. The second part, $\chi^{II}$, contains $f(E)$ itself and runs over all occupied states: it is the Fermi-sea term.

PAOFLOW evaluates both parts in the constant-broadening model of Freimuth et al. [4], in which disorder enters only through one energy $\Gamma$ that broadens every Bloch state into a Lorentzian spectral function,

$$a_{n\mathbf{k}}(E) = \frac{1}{\pi}\,\frac{\Gamma}{(E-\varepsilon_{n\mathbf{k}})^2+\Gamma^2} .$$

A state with spectral width $\Gamma$ has an amplitude that decays as $e^{-\Gamma t/\hbar}$, so its occupation decays with the lifetime

$$\tau = \frac{\hbar}{2\Gamma} .$$

For example, $\Gamma = 10$ meV corresponds to $\tau \approx 33$ fs. The model has no vertex corrections, so for the Hall-type responses it gives the intrinsic contribution of the band structure and leaves out the extrinsic side-jump and skew-scattering mechanisms of impurities [5].

### Velocity, spin and spin-current operators

All responses pair an operator $\hat O_i$ with the velocity $\hat v_j$, which carries the coupling to the field. Their matrix elements between the Bloch eigenstates at the same $\mathbf{k}$ are written

$$O^i_{nm}(\mathbf{k}) = \langle\psi_{n\mathbf{k}}|\hat O_i|\psi_{m\mathbf{k}}\rangle, \qquad v^j_{nm}(\mathbf{k}) = \frac{1}{\hbar}\langle\psi_{n\mathbf{k}}|\partial_{k_j}\tilde H(\mathbf{k})|\psi_{m\mathbf{k}}\rangle ,$$

where the diagonal element $v^j_{nn}$ is the band velocity $\mathbf{v}_{n\mathbf{k}}$ and the off-diagonal elements carry the interband transitions; [The PAO Hamiltonian](pao-hamiltonian.md) explains how $\partial_{\mathbf{k}}\tilde H$ is built from $H_{\alpha\beta}(\mathbf{R})$. Three operators take the place of $\hat O_i$:

- the velocity $\hat v_i$, for the charge current $-e\hat v_i$ of electrons with charge $-e$;
- the spin $\hat S_s = \frac{\hbar}{2}\hat\sigma_s$, built by {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.spin_operator` in the PAO basis;
- the spin current $\hat J^s_i = \frac{1}{2}\{\hat S_s, \hat v_i\}$, which carries spin $s$ along direction $i$.

The spin current is the conventional symmetrised product. In a crystal with spin–orbit coupling spin is not conserved, and this current differs from the conserved current of Shi et al. [6] by a torque-dipole term, which PAOFLOW does not include. The spin and the spin current are meaningful only when the bands carry spin–orbit coupling, so for `response='shc'` and `'ree'` the DFT run must be fully relativistic.

### Time-reversal parity of the two terms

Which of the two terms survives in a crystal depends on how $\hat O_i$ behaves under time reversal. Under time reversal, a matrix element at $\mathbf{k}$ maps to the complex conjugate of the one at $-\mathbf{k}$, multiplied by the parity $\eta_O = \pm 1$ of the operator. The velocity is odd, so

$$O^i_{nm}(\mathbf{k})\,v^j_{mn}(\mathbf{k}) \;\to\; -\eta_O\left[O^i_{nm}(-\mathbf{k})\,v^j_{mn}(-\mathbf{k})\right]^* .$$

In a crystal with time-reversal symmetry the sum over $\mathbf{k}$ is therefore invariant under this map. The real part of the product survives the sum only if $-\eta_O = 1$, and the imaginary part, which changes sign under conjugation, only if $\eta_O = 1$. The Fermi-surface term, built from real parts (see below), is therefore nonzero in a non-magnetic crystal only for time-odd operators ($\hat v$, $\hat S$), and the Fermi-sea term, built from imaginary parts, only for time-even ones ($\hat J^s$). A response that needs broken time-reversal symmetry, a magnetic order, is called T-odd:

| $\hat O$ | Fermi-surface term $\chi^{I}$ | Fermi-sea term $\chi^{II}$ |
| --- | --- | --- |
| $-e\hat v$ | charge conductivity (T-even) | anomalous Hall conductivity (T-odd) |
| $\hat S$ | Rashba–Edelstein effect (T-even) | magnetic Rashba–Edelstein effect (T-odd) |
| $\hat J^s$ | magnetic spin Hall effect (T-odd) | spin Hall conductivity (T-even) |

The physics of the anomalous and spin Hall conductivities, the Berry curvature behind them, is on [Berry curvature and topology](berry-topology.md).

## The Fermi-surface term

The Fermi-surface term describes the electrons at $E_F$ that the field accelerates and scattering relaxes, so it grows as the broadening shrinks. Its three forms in PAOFLOW are the full broadened sum, equation (1), and its intraband and interband parts at small $\Gamma$, equations (4) and (5).

### Equation (1): Fermi-surface term with broadened states

At zero temperature the Fermi-surface term of the constant-broadening model [4] is the Kubo–Greenwood product of two spectral functions at $E_F$,

$$\chi^{I}_{ij} = -\frac{\pi e\hbar}{V}\sum_{\mathbf{k}}\sum_{nm}\mathrm{Re}\left[O^i_{nm}\,v^j_{mn}\right]a_{n\mathbf{k}}(E_F)\,a_{m\mathbf{k}}(E_F),$$

with $V = N_\mathrm{mesh}\Omega$ the volume of the crystal of $N_\mathrm{mesh}$ cells of volume $\Omega$. Inserting the Lorentzians gives the form that PAOFLOW evaluates,

$$\chi^{I}_{ij}(E) = -\frac{e\hbar}{\pi V}\sum_{\mathbf{k}}\sum_{nm}\frac{\Gamma^2\,\mathrm{Re}\left[O^i_{nm}\,v^j_{mn}\right]}{\left[(E-\varepsilon_{n\mathbf{k}})^2+\Gamma^2\right]\left[(E-\varepsilon_{m\mathbf{k}})^2+\Gamma^2\right]} . \qquad (1)$$

The sum runs over all bands and needs no occupations, because the two Lorentzians place it at the energy $E$. The code evaluates equation (1) on a grid of energies $E$ in place of $E_F$, so one run gives the response as a function of the chemical potential in the rigid-band approximation: the bands stay those of the undoped crystal while $E$ moves through them. For the charge conductivity the operator is the charge current $-e\hat v_i$, which flips the sign, and equation (1) becomes the Kubo–Greenwood conductivity

$$\sigma_{ij}(E) = \frac{\pi e^2\hbar}{V}\sum_{\mathbf{k}}\sum_{nm}\mathrm{Re}\left[v^i_{nm}\,v^j_{mn}\right]a_{n\mathbf{k}}(E)\,a_{m\mathbf{k}}(E),$$

which is positive on the diagonal. Because the trace over each degenerate multiplet does not depend on the basis chosen inside it, equation (1) is free of the gauge freedom of degenerate states.

### Equation (4): intraband limit

When $\Gamma$ is small compared with the separation of the bands, the diagonal terms $n = m$ of equation (1) dominate. Each contains the squared Lorentzian, whose integral over energy is

$$\int \frac{\Gamma^2\,dE}{\left[(E-\varepsilon)^2+\Gamma^2\right]^2} = \frac{\pi}{2\Gamma},$$

so that $\Gamma^2/[(E-\varepsilon)^2+\Gamma^2]^2 \to (\pi/2\Gamma)\,\delta(E-\varepsilon)$ and equation (1) becomes [7]

$$\chi^{I,\mathrm{intra}}_{ij}(E) = -\frac{e\hbar}{2\Gamma V}\sum_{\mathbf{k}n}\mathrm{Re}\left[O^i_{nn}\,v^j_{nn}\right]\delta(E-\varepsilon_{n\mathbf{k}}) . \qquad (4)$$

With $\hbar/2\Gamma = \tau$ and $\hat O = -e\hat v$, equation (4) is the conductivity of Boltzmann theory with a constant relaxation time at zero temperature, $\sigma_{ij} = (e^2\tau/V)\sum v^i_{n\mathbf{k}}v^j_{n\mathbf{k}}\,\delta(E-\varepsilon_{n\mathbf{k}})$: the response scales as $1/\Gamma$, as a dissipative current must. The code replaces the $\delta$ function by a Gaussian of the adaptive width of each state (see [Density of states](density-of-states.md)) and divides by $2\Gamma + 10^{-4}$ eV to avoid a division by zero. Unlike equation (1), the diagonal elements $O^i_{nn}$ depend on the basis inside a degenerate multiplet; PAOFLOW uses the basis in which $\hat O_i$ is diagonal within each multiplet. {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.conductivity` evaluates the same Fermi-surface sum for $\hat O = \hat v$ without the prefactor $e^2\tau/V$.

### Equation (5): interband term at small broadening

The second contribution that Li et al. [7] keep at small $\Gamma$ couples different bands and is of first order in $\Gamma$. PAOFLOW evaluates it as

$$\chi^{I,\mathrm{inter}}_{ij}(E) = \frac{e\hbar}{V}\sum_{\mathbf{k}}\sum_{n\neq m}\left(f_{n\mathbf{k}}-f_{m\mathbf{k}}\right)\frac{\mathrm{Re}\left[O^i_{nm}\,v^j_{mn}\right]\,\Gamma\,(\varepsilon_{m\mathbf{k}}-\varepsilon_{n\mathbf{k}})}{\left[(\varepsilon_{m\mathbf{k}}-\varepsilon_{n\mathbf{k}})^2+\Gamma^2\right]^2+\delta_b^2}, \qquad (5)$$

with $f_{n\mathbf{k}} = f(\varepsilon_{n\mathbf{k}} - E)$ the occupation at the chemical potential $E$, smeared with the adaptive width of each state, and $\delta_b = 10^{-3}$ eV a regulariser of the denominator. The code sums $2f_{n\mathbf{k}}$ over all $m \neq n$, which equals the form above because the summand without the occupation is antisymmetric in $n$ and $m$. For the charge conductivity the sign flips, as in equation (1). Equation (5) is of first order in $\Gamma$, against $1/\Gamma$ for equation (4), so it is a small correction unless bands split by spin–orbit coupling lie within a few $\Gamma$ of each other near $E$.

## The Fermi-sea term

The Fermi-sea term describes how the field mixes the occupied states with the empty ones, which changes the expectation value of $\hat O$ without any scattering. It is a property of the occupied bands, survives as $\Gamma \to 0$, and gives the intrinsic Hall responses.

### Equation (2): Fermi-sea term at finite broadening

In the constant-broadening model the Fermi-sea term is [4, 7]

$$\chi^{II}_{ij}(E) = \frac{2e\hbar}{V}\sum_{\mathbf{k}}\sum_{n}f_{n\mathbf{k}}\sum_{m\neq n}\frac{\mathrm{Im}\left[O^i_{nm}\,v^j_{mn}\right]\left(\Gamma^2-\Delta_{mn}^2\right)}{\left(\Delta_{mn}^2+\Gamma^2\right)^2+\delta_b^2}, \qquad \Delta_{mn} = \varepsilon_{m\mathbf{k}}-\varepsilon_{n\mathbf{k}} . \qquad (2)$$

The sum over $m$ runs over all bands, not only the empty ones: $\mathrm{Im}[O^i_{nm}v^j_{mn}]$ is antisymmetric in $n$ and $m$ for Hermitian operators and the energy factor is symmetric, so the pairs of two occupied states cancel. The occupations are smeared with the adaptive widths and $\delta_b = 10^{-3}$ eV, as in equation (5). For the anomalous Hall conductivity the operator is $-e\hat v_i$ and the code flips the sign of equation (2).

### Equation (3): clean limit

As $\Gamma \to 0$, the factor $(\Gamma^2-\Delta^2)/(\Delta^2+\Gamma^2)^2$ tends to $-1/\Delta^2$, and equation (2) becomes a sum of Berry-curvature-like terms over the occupied states. PAOFLOW evaluates it as

$$\chi^{II}_{ij}(E) = -\frac{2e\hbar}{V}\sum_{\mathbf{k}}\sum_{n}f_{n\mathbf{k}}\sum_{m\neq n}\frac{\mathrm{Im}\left[O^i_{nm}\,v^j_{mn}\right]}{\Delta_{mn}^2+\delta_p^2}, \qquad \delta_p = 0.05\ \mathrm{eV} . \qquad (3)$$

The fixed $\delta_p$ takes the place of $\Gamma$: `gamma` does not enter equation (3), and the contribution of a pair of bands closer than about 50 meV is suppressed, which caps the large terms near avoided crossings that spin–orbit coupling opens. For `response='ahc'` the code applies equation (3) without the sign change that equations (1), (2), (4) and (5) apply to the charge current, so `ahc` with `full_chi2=True` (equation 2) and with `full_chi2=False` (equation 3) have opposite signs in the limit $\Gamma \to 0$. For $\hat O = -e\hat v$ equation (3) is the intrinsic anomalous Hall conductivity, and for $\hat O = \hat J^s$ the intrinsic spin Hall conductivity, both derived from the Berry curvature on [Berry curvature and topology](berry-topology.md).

## Equations, units and parameters of `linear_response()`

The argument `response` chooses the operator, and `t_odd`, `full_chi2`, `intraband` and `interband` choose the equation:

| `response` | $\hat O$ | `t_odd=False` | `t_odd=True` |
| --- | --- | --- | --- |
| `'cond'` | $-e\hat v_i$ | (1); (4) and/or (5) with `intraband`/`interband` | same |
| `'ahc'` | $-e\hat v_i$ | (3); (2) with `full_chi2` | same |
| `'ree'` | $\hat S_s$ | (1); (4) and/or (5) with `intraband`/`interband` | (3); (2) with `full_chi2` |
| `'shc'` | $\hat J^s_i$ | (3); (2) with `full_chi2` | (1); (4) and/or (5) with `intraband`/`interband` |

For `'cond'` and `'ahc'` the time parity is fixed by the operator, so `t_odd` has no effect. The tensor components are chosen with `a_tensor` (pairs $ij$) and `s_tensor` (triples $s\,i\,j$ for the spin Hall conductivity).

The code writes the conductivities in S/cm and the spin Hall conductivity in $(\hbar/e)$ S/cm, the convention in which it compares directly with a charge conductivity. The Rashba–Edelstein tensor is the spin per unit cell induced by the field, in units of $\hbar$ cm/V, so its prefactor contains no $1/V$. The sums over $\mathbf{k}$ run over the $N_\mathrm{mesh}$ points of the mesh on which $\tilde H(\mathbf{k})$ was diagonalised.

The energy $E$ runs over `esize` points from `eminH` to `emaxH`, measured from $E_F$. The upper end is clipped to the PAO shift, the energy above which the PAO Hamiltonian does not reproduce the DFT bands (see [The PAO Hamiltonian](pao-hamiltonian.md)). The broadening $\Gamma$ is the argument `gamma`. Equations (2), (3) and (5) also need occupations, smeared with the adaptive widths of {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.adaptive_smearing`, which `linear_response()` computes with its defaults when it has not been called. All arguments are listed in the API entry of {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.linear_response`.

## Approximations of the Kubo responses

The responses rest on five approximations.

- Constant broadening. One $\Gamma$ for every state stands for all scattering. The Fermi-surface responses scale as $1/\Gamma$, so their absolute values are only as good as the estimate of $\tau = \hbar/2\Gamma$, while the Fermi-sea responses are insensitive to $\Gamma$ once it is small compared with the band splittings near $E_F$. Without vertex corrections the extrinsic Hall mechanisms are absent [5], which fails in dilute alloys and clean metals at low temperature, where skew scattering can dominate the anomalous Hall effect.
- Rigid bands. Changing $E$ moves the chemical potential through fixed bands; doping that changes the band structure, or a magnetisation that changes with filling, is outside the model.
- Zero temperature. The occupations are step functions smeared by numerical widths, not Fermi functions at a temperature.
- Conventional spin current. The torque-dipole term [6] is missing.
- Bands of the PAO Hamiltonian. Only the bands below the PAO shift are faithful, so responses that collect interband terms from high-lying states, such as the Fermi-sea sums, converge with the bands and the energy window the projection keeps.

## Independent-particle dielectric tensor

{py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.dielectric_tensor` gives the response to an oscillating field of frequency $\omega$ in the long-wavelength limit, the complex dielectric tensor $\epsilon_{ij}(\omega) = \epsilon'_{ij}(\omega) + i\epsilon''_{ij}(\omega)$, written with $\epsilon$ to keep it apart from the eigenvalues $\varepsilon_{n\mathbf{k}}$. Its absorptive part $\epsilon''$ comes from transitions between bands at the same $\mathbf{k}$, and in a metal also from the free carriers at $E_F$. PAOFLOW adds the two as an interband and an intraband (Drude) term.

### Interband term

For independent electrons the absorptive part follows from Fermi's golden rule for vertical transitions, or equivalently from the head of the Adler–Wiser dielectric matrix without local fields [8, 9]:

$$\epsilon''_{ij}(\omega) = \frac{\pi e^2}{\epsilon_0 V\omega^2}\sum_{\mathbf{k}}\sum_{\Delta_{mn}>0} \left(f_{n\mathbf{k}}-f_{m\mathbf{k}}\right)\mathrm{Re}\left[v^i_{nm}\,v^j_{mn}\right]\delta(\Delta_{mn}-\hbar\omega),$$

where the occupations $f_{n\mathbf{k}}$ include the spin degeneracy (2 without spin–orbit coupling in a non-magnetic run, 1 otherwise) and $\epsilon_0$ is the vacuum permittivity. The code evaluates instead the damped-oscillator form

$$\epsilon_{ij}(\omega) = \delta_{ij} + \frac{2e^2\hbar^2}{\epsilon_0 V}\sum_{\mathbf{k}}\sum_{n\neq m} f_{n\mathbf{k}}\,\frac{\mathrm{Re}\left[v^i_{nm}\,v^j_{mn}\right]}{\Delta_{mn}\left[\Delta_{mn}^2-(\hbar\omega)^2-i\eta\hbar\omega\right]},$$

which contains both the real and the imaginary part. As $\eta \to 0$, $\mathrm{Im}\,[\Delta^2-(\hbar\omega)^2-i\eta\hbar\omega]^{-1} \to \pi\delta(\Delta-\hbar\omega)/(2\hbar\omega)$ for $\omega > 0$, and the imaginary part reduces to the golden-rule expression above. The sum over both orderings of a pair turns $f_{n\mathbf{k}}$ into $f_{n\mathbf{k}}-f_{m\mathbf{k}}$, so pairs with equal occupation drop out; the code skips them explicitly. In an insulator $f$ is a step at $E_F$; in a metal it is smeared with the width `degauss`, Gaussian or Methfessel–Paxton as chosen by `smearing`.

Each term is the response of a damped harmonic oscillator of frequency $\Delta_{mn}/\hbar$ and damping $\eta/\hbar$, with oscillator strength proportional to $|v_{nm}|^2/\Delta_{mn}$. Since $\hbar v_{nm} = -i\Delta_{mn} r_{nm}$ for $n \neq m$, the strength is $\Delta_{mn}|r_{nm}|^2$, with $r_{nm}$ the interband position matrix element. In the static limit each term is $|r_{nm}|^2/\Delta_{mn}$, so transitions with a large dipole and a small energy dominate the static value,

$$\epsilon'_{ij}(0) = \delta_{ij} + \frac{2e^2\hbar^2}{\epsilon_0 V}\sum_{\mathbf{k}}\sum_{n\neq m} f_{n\mathbf{k}}\,\frac{\mathrm{Re}\left[v^i_{nm}\,v^j_{mn}\right]}{\Delta_{mn}^3},$$

which for an insulator is the electronic (clamped-ion) dielectric constant $\epsilon_\infty$.

The broadening $\eta$ is the argument `delta`. When {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.adaptive_smearing` has been called first, each transition gets its own width $\eta_{nm}(\mathbf{k}) = a\,|\nabla_{\mathbf{k}}(\varepsilon_{n\mathbf{k}}-\varepsilon_{m\mathbf{k}})|\,\Delta k$, with $\Delta k$ the mesh spacing and $a$ a constant [10], floored at the spacing of the frequency grid. A transition between nearly parallel bands then keeps a sharp peak, which is how van Hove singularities appear in $\epsilon''$, while a steep transition, poorly sampled by the mesh, is broadened over the frequency range it sweeps.

### Intraband Drude term

In a metal the bands that cross $E_F$ also absorb at low frequency through intraband transitions, which need scattering to conserve momentum. The diagonal terms of the Kubo current response, with $(f_n-f_m)/(\varepsilon_m-\varepsilon_n) \to -\partial f/\partial\varepsilon$, give the Drude form

$$\epsilon^{\mathrm{intra}}_{ij}(\omega) = -\frac{(\hbar\omega_p)^2_{ij}}{(\hbar\omega)^2+i\eta_D\hbar\omega}, \qquad (\hbar\omega_p)^2_{ij} = \frac{e^2\hbar^2}{\epsilon_0 V}\sum_{\mathbf{k}n}v^i_{n\mathbf{k}}v^j_{n\mathbf{k}}\,\tilde\delta(\varepsilon_{n\mathbf{k}}-E_F),$$

where $\omega_p$ is the plasma frequency of the conduction electrons, $\tilde\delta$ the smeared $-\partial f/\partial\varepsilon$ of width `degauss` times the spin degeneracy, and $\eta_D = \hbar/\tau_D$ the Drude damping, the argument `intrasmear`. In the static limit $\epsilon^{\mathrm{intra}}$ diverges as $-1/\omega^2$ and $\epsilon''$ as $1/\omega$: the metal screens a static field completely and the optical conductivity $\epsilon_0\omega\epsilon''$ tends to the Drude dc conductivity $\epsilon_0\omega_p^2\tau_D$. With `intraband=False` the term is left out, so that it can be replaced by a Drude term whose damping depends on frequency and temperature, for example one built from electron–phonon scattering. The material is treated as a metal or an insulator according to the DFT run.

### Real part and the Kramers–Kronig relation

Causality requires that $\epsilon'$ and $\epsilon''$ are related by the Kramers–Kronig relation

$$\epsilon'_{ij}(\omega) = \delta_{ij} + \frac{2}{\pi}\,\mathcal{P}\!\int_0^\infty \frac{\omega'\,\epsilon''_{ij}(\omega')}{\omega'^2-\omega^2}\,d\omega' .$$

Each oscillator and Drude term above is analytic in the upper half of the complex frequency plane, so their real parts satisfy this relation term by term. PAOFLOW therefore takes $\epsilon'$ from the real part of the oscillator sum rather than from a numerical transform of $\epsilon''$, and $\epsilon'$ includes the transitions above the upper end `emax` of the frequency grid, as long as their bands are among those kept. The same relation continued to imaginary frequencies gives

$$\epsilon_{ij}(i\xi) = \delta_{ij} + \frac{2}{\pi}\int_0^\infty \frac{\omega'\,\epsilon''_{ij}(\omega')}{\omega'^2+\xi^2}\,d\omega' ,$$

the input of van der Waals and Casimir–Lifshitz calculations. This integral is taken numerically over the frequency grid, so it misses the absorption above `emax`, and `emax` must reach beyond the strong interband peaks. The run also prints, for each diagonal component, the frequency $\sqrt{(2/\pi)\int\omega\,\epsilon''\,d\omega}$ over the grid, a measure of the oscillator strength that the bands and the frequency window contain.

### Optical constants from the dielectric tensor

The other optical spectra follow from $\epsilon_{ii}$ along a principal axis by the macroscopic Maxwell equations: the complex refractive index $n + i\kappa = \sqrt{\epsilon_{ii}}$, the absorption coefficient $\alpha = 2\omega\kappa/c$, the reflectivity at normal incidence $R = [(n-1)^2+\kappa^2]/[(n+1)^2+\kappa^2]$, the electron energy-loss function $-\mathrm{Im}\,\epsilon_{ii}^{-1} = \epsilon''/(\epsilon'^2+\epsilon''^2)$, and the optical conductivity $\sigma_{ij}(\omega) = -i\epsilon_0\omega[\epsilon_{ij}(\omega)-\delta_{ij}]$. PAOFLOW writes the first four for diagonal components only, because for a tensor with off-diagonal elements they hold only in the principal frame. With `emissivity=True` it also computes the reflectivity at oblique incidence from the Fresnel equations and the emissivity $1-R$ of an opaque sample.

### Approximations of the dielectric tensor

The dielectric tensor is that of independent Kohn–Sham electrons, and each approximation behind it fails on a known scale.

- Independent particles, no excitons. The electron and the hole of a transition do not attract each other, so bound excitons below the gap and the transfer of oscillator strength to the absorption edge are missing [11]. This fails wherever the exciton binding energy is comparable to the features of the spectrum: wide-gap insulators, molecular crystals and two-dimensional materials, where it reaches tenths of an eV or more.
- No local fields. The field acting on the electrons is the macroscopic field, without the microscopic fields induced on the scale of the unit cell [8, 9]. The error grows with the inhomogeneity of the electron density: localised $d$ and $f$ states, layered and molecular crystals.
- Kohn–Sham eigenvalues. With PBE the gap is underestimated, so the absorption edge and the interband peaks lie too low in energy, and $\epsilon_\infty$, which scales as $1/\Delta^3$, is too large. `dielectric_tensor()` applies no scissor shift.
- Matrix elements. The velocity is that of the PAO Hamiltonian, $\hat{\mathbf v} = (i/\hbar)[\tilde H, \hat{\mathbf r}]$ with the position operator diagonal on the orbital centres [12]. The dipoles between orbitals of the same atom, such as $\langle s|\mathbf r|p\rangle$, are left out unless `gradient_and_momenta(position_dipoles=True)` adds them; the mechanism and its effect on silicon are in [the position-dipole note](../../internals/position-dipoles.md).
- Finite band set. Only the bands that the PAO Hamiltonian reproduces enter the sums, so $\epsilon''$ is reliable only up to photon energies whose final states lie below the PAO shift, and $\epsilon_\infty$ lacks the small positive contribution of higher transitions.
- Long wavelength and clamped ions. The tensor is the $\mathbf{q} \to 0$ limit without spatial dispersion, and contains only the electronic response; the ionic contribution in the infrared is computed in the phonon module.
- Phenomenological damping. The constant widths $\eta$ and $\eta_D$ stand for lifetimes that in reality depend on energy and temperature.

## Used in

- [Linear response](../tutorials/tutorial03.ipynb): the spin Hall conductivity, Rashba–Edelstein tensors, anomalous Hall and charge conductivities of `linear_response()`, equations (1) to (5).

## References

1. R. Kubo, "Statistical-mechanical theory of irreversible processes. I. General theory and simple applications to magnetic and conduction problems", J. Phys. Soc. Jpn. 12, 570 (1957).
2. A. Bastin, C. Lewiner, O. Betbeder-Matibet and P. Nozières, "Quantum oscillations of the Hall effect of a fermion gas with random impurity scattering", J. Phys. Chem. Solids 32, 1811 (1971).
3. A. Crépieux and P. Bruno, "Theory of the anomalous Hall effect from the Kubo formula and the Dirac equation", Phys. Rev. B 64, 014416 (2001).
4. F. Freimuth, S. Blügel and Y. Mokrousov, "Spin-orbit torques in Co/Pt(111) and Mn/W(001) magnetic bilayers from first principles", Phys. Rev. B 90, 174423 (2014).
5. J. Sinova, S. O. Valenzuela, J. Wunderlich, C. H. Back and T. Jungwirth, "Spin Hall effects", Rev. Mod. Phys. 87, 1213 (2015).
6. J. Shi, P. Zhang, D. Xiao and Q. Niu, "Proper definition of spin current in spin-orbit coupled systems", Phys. Rev. Lett. 96, 076604 (2006).
7. H. Li, H. Gao, L. P. Zârbo, K. Výborný, X. Wang, I. Garate, F. Doğan, A. Čejchan, J. Sinova, T. Jungwirth and A. Manchon, "Intraband and interband spin-orbit torques in noncentrosymmetric ferromagnets", Phys. Rev. B 91, 134402 (2015).
8. S. L. Adler, "Quantum theory of the dielectric constant in real solids", Phys. Rev. 126, 413 (1962).
9. N. Wiser, "Dielectric constant with local field effects included", Phys. Rev. 129, 62 (1963).
10. J. R. Yates, X. Wang, D. Vanderbilt and I. Souza, "Spectral and Fermi surface properties from Wannier interpolation", Phys. Rev. B 75, 195121 (2007).
11. G. Onida, L. Reining and A. Rubio, "Electronic excitations: density-functional versus many-body Green's-function approaches", Rev. Mod. Phys. 74, 601 (2002).
12. M. Graf and P. Vogl, "Electromagnetic fields and dielectric response in empirical tight-binding theory", Phys. Rev. B 51, 4940 (1995).
