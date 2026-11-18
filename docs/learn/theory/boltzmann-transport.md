# Boltzmann transport in the relaxation-time approximation

This page derives the electronic transport coefficients that PAOFLOW computes from the band structure: the electrical conductivity $\sigma$, the Seebeck coefficient $S$, the electronic thermal conductivity $\kappa$ and the Hall coefficient $R_H$, together with the chemical potential of a doped crystal and the energy-dependent relaxation times that enter them. The methods are {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.transport`, {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.doping` and the {py:class}`~PAOFLOW.boltzmann.TauModel.TauModel` class. Every quantity on the page is a sum over the eigenvalues $\varepsilon_{n\mathbf{k}}$ and band velocities $\mathbf{v}_{n\mathbf{k}}$ of the interpolated Hamiltonian $\tilde H(\mathbf{k})$ on a mesh of $N_\mathrm{mesh}$ k-points; $\tilde H(\mathbf{k})$ is derived on [The PAO Hamiltonian](pao-hamiltonian.md), and the band velocities and the adaptive smearing that turns the k-sums into smooth functions of energy on [Band velocities, density of states and adaptive smearing](density-of-states.md).

## Linearised Boltzmann equation

The Boltzmann equation gives the occupation $f_{n\mathbf{k}}(\mathbf{r})$ of the Bloch state $n\mathbf{k}$ in a crystal driven out of equilibrium by an electric field $\mathbf{E}$ and a temperature gradient $\nabla T$. Electrons are treated as wave packets that move with the band velocity and are accelerated by the force $-e\mathbf{E}$, with $e > 0$ the elementary charge. In a steady state the change of $f$ from drift and acceleration is balanced by scattering:

$$
\mathbf{v}_{n\mathbf{k}} \cdot \nabla_{\mathbf{r}} f_{n\mathbf{k}} - \frac{e}{\hbar}\,\mathbf{E} \cdot \nabla_{\mathbf{k}} f_{n\mathbf{k}} = \left(\frac{\partial f_{n\mathbf{k}}}{\partial t}\right)_\mathrm{coll} .
$$

The relaxation-time approximation replaces the collision integral by a relaxation towards the local equilibrium $f^0$, the Fermi–Dirac distribution at the local $\mu(\mathbf{r})$ and $T(\mathbf{r})$, at a rate $1/\tau_{n\mathbf{k}}$:

$$
\left(\frac{\partial f_{n\mathbf{k}}}{\partial t}\right)_\mathrm{coll} = -\frac{f_{n\mathbf{k}} - f^0(\varepsilon_{n\mathbf{k}})}{\tau_{n\mathbf{k}}} .
$$

For weak fields, $f$ on the left-hand side is replaced by $f^0$, which is linear response. With $\nabla_{\mathbf{r}} f^0 = (-\partial f^0/\partial\varepsilon)\,[\nabla\mu + (\varepsilon - \mu)\nabla T/T]$ and $\nabla_{\mathbf{k}} f^0 = \hbar\mathbf{v}\,\partial f^0/\partial\varepsilon$, the deviation from equilibrium is

$$
f_{n\mathbf{k}} - f^0 = \tau_{n\mathbf{k}} \left(-\frac{\partial f^0}{\partial\varepsilon}\right)_{\varepsilon_{n\mathbf{k}}} \mathbf{v}_{n\mathbf{k}} \cdot \left[-e\boldsymbol{\mathcal{E}} - \frac{\varepsilon_{n\mathbf{k}} - \mu}{T}\,\nabla T\right] ,
$$

with $\boldsymbol{\mathcal{E}} = \mathbf{E} + \nabla\mu/e$ the field that drives the current, the gradient of the electrochemical potential [1]. The factor $-\partial f^0/\partial\varepsilon$ restricts the response to states within a few $k_BT$ of $\mu$: only they can change their occupation. Each of them is shifted by its velocity times the distance $\tau_{n\mathbf{k}}$ lets it travel between collisions.

The relaxation-time form neglects the electrons scattered into a state from other states. It is exact for elastic scattering that is isotropic, and for anisotropic elastic scattering it holds when $\tau$ is read as the momentum relaxation time, which weighs each scattering event by $1 - \cos\theta$ of its deflection angle [2, 5]. It fails for inelastic scattering on the scale of $k_BT$, such as polar optical phonons below the Debye temperature, where no single $\tau_{n\mathbf{k}}$ describes the relaxation. The semiclassical picture itself holds when the energy uncertainty $\hbar/\tau$ is small against $k_BT$ and against the spacing of the bands: for $\tau = 10$ fs, $\hbar/\tau$ is 66 meV, already above $k_BT$ at room temperature. Band velocities alone carry the current, so interband coherence and the anomalous velocity of the Berry curvature are left out.

## Transport moments

The charge and heat currents follow from the deviation of the occupation. Each state carries the charge current $-e\mathbf{v}_{n\mathbf{k}}$ and the heat current $(\varepsilon_{n\mathbf{k}} - \mu)\mathbf{v}_{n\mathbf{k}}$, and the sum over states per unit volume is $g_s/(\Omega N_\mathrm{mesh}) \sum_{n\mathbf{k}}$, with $\Omega$ the cell volume and $g_s$ the spin degeneracy. Inserting $f - f^0$ gives

$$
\begin{aligned}
\mathbf{j} &= e^2 \mathcal{L}^{(0)} \boldsymbol{\mathcal{E}} + \frac{e}{T}\,\mathcal{L}^{(1)} \nabla T , \\
\mathbf{j}_Q &= -e\,\mathcal{L}^{(1)} \boldsymbol{\mathcal{E}} - \frac{1}{T}\,\mathcal{L}^{(2)} \nabla T ,
\end{aligned}
$$

in terms of the moments

$$
\mathcal{L}^{(\alpha)}_{ij}(\mu, T) = \frac{g_s}{\Omega N_\mathrm{mesh}} \sum_{n\mathbf{k}} \tau_{n\mathbf{k}}\, v^i_{n\mathbf{k}}\, v^j_{n\mathbf{k}} \left(-\frac{\partial f^0}{\partial\varepsilon}\right)_{\varepsilon_{n\mathbf{k}}} (\varepsilon_{n\mathbf{k}} - \mu)^\alpha , \qquad \alpha = 0, 1, 2 ,
$$

which {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.transport` evaluates on a grid of $\mu$ between `emin` and `emax` and for each temperature between `tmin` and `tmax`. The sum runs over the bands that the projectability keeps (see [The PAO Hamiltonian](pao-hamiltonian.md)), so $\mu + 10\,k_BT$ must lie below the energy where these bands stop representing the DFT bands. $\mathcal{L}^{(0)}$ counts the states at the chemical potential weighted by $\tau v^2$. $\mathcal{L}^{(1)}$ measures how asymmetric that weight is about $\mu$, and $\mathcal{L}^{(2)}$ how wide it is.

### Transport distribution and smearing

The moments are integrals of one function of energy, the transport distribution

$$
\Sigma_{ij}(\varepsilon) = \frac{g_s}{\Omega N_\mathrm{mesh}} \sum_{n\mathbf{k}} \tau_{n\mathbf{k}}\, v^i_{n\mathbf{k}}\, v^j_{n\mathbf{k}}\, \delta(\varepsilon - \varepsilon_{n\mathbf{k}}) , \qquad
\mathcal{L}^{(\alpha)}_{ij} = \int d\varepsilon\, \Sigma_{ij}(\varepsilon) \left(-\frac{\partial f^0}{\partial\varepsilon}\right) (\varepsilon - \mu)^\alpha ,
$$

introduced by Mahan and Sofo [3] and used in the BoltzTraP code [4]. $\Sigma(\varepsilon)$ holds all the band structure; temperature and chemical potential enter only through the Fermi window $-\partial f^0/\partial\varepsilon = [4k_BT\cosh^2((\varepsilon - \mu)/2k_BT)]^{-1}$, which falls to 1% of its peak at $|\varepsilon - \mu| \approx 6\,k_BT$.

PAOFLOW evaluates the moments in one of two ways, set by the `smearing` argument of the `PAOFLOW` constructor. Without smearing, the Fermi window is taken at each eigenvalue, as in the sum over states above. With adaptive smearing (`'gauss'` or `'m-p'`), each delta function in $\Sigma(\varepsilon)$ becomes a Gaussian (or Methfessel–Paxton function) of the width $\sigma_{n\mathbf{k}}$ of that state, $\Sigma(\varepsilon)$ is sampled on the $\mu$ grid extended by about $22\,k_BT$ on each side, linearly interpolated at half the grid spacing, and integrated against the Fermi window by Simpson's rule. The smeared form converges faster with the k-mesh, because a state contributes over its width instead of only where the Fermi window happens to sample it. It adds two errors of its own: the band edges acquire tails of width $\sigma_{n\mathbf{k}}$ that persist at $T \to 0$, and the $\mu$ spacing sets the energy resolution of $\Sigma(\varepsilon)$, so the spacing must be smaller than the narrowest widths near $\mu$.

### Conductivity, Seebeck coefficient and thermal conductivity

The three coefficients follow from the currents under the conditions of the measurement. The conductivity is the current per unit field at uniform temperature. The Seebeck coefficient is the field that a temperature gradient builds up in an open circuit, $\boldsymbol{\mathcal{E}} = S\,\nabla T$ at $\mathbf{j} = 0$. The thermal conductivity is the heat current per unit gradient in the same open circuit, $\mathbf{j}_Q = -\kappa\nabla T$ at $\mathbf{j} = 0$. Solving the current equations gives

$$
\begin{aligned}
\sigma &= e^2 \mathcal{L}^{(0)} , \\
S &= -\frac{1}{eT}\,\bigl[\mathcal{L}^{(0)}\bigr]^{-1} \mathcal{L}^{(1)} , \\
\kappa &= \frac{1}{T}\Bigl[\mathcal{L}^{(2)} - \mathcal{L}^{(1)} \bigl[\mathcal{L}^{(0)}\bigr]^{-1} \mathcal{L}^{(1)}\Bigr] ,
\end{aligned}
$$

and the power factor $S\sigma S$, which reduces to $S^2\sigma$ for a scalar $S$ [1]. $S$ and $\kappa$ need the inverse of the full $3\times3$ tensor $\mathcal{L}^{(0)}$, so `transport()` stops on a two-dimensional model, whose $\mathcal{L}^{(0)}_{zz}$ vanishes.

The code evaluates $\sigma$, $S$ and the power factor as written. For the thermal conductivity, the second term enters with a unit factor $10^{-15}$ times smaller than the first, so the file `kappa` holds the first term alone,

$$
\kappa_0 = \frac{1}{T}\,\mathcal{L}^{(2)} = \kappa + T\,S\sigma S ,
$$

the heat conductance at zero field rather than at zero current. For a diagonal tensor (a cubic crystal, or any crystal along its principal axes) you obtain $\kappa$ from the two files as $\kappa/\tau = \kappa_0/\tau - T\,S^2\sigma/\tau$. In a metal the difference is of relative order $(k_BT/E_F)^2$ and can be ignored. In a semiconductor with $\mu$ in or near the gap it is comparable to or larger than $\kappa$, because $S$ is large there.

The coefficients take a transparent form in two limits. In a degenerate system, where $\Sigma(\varepsilon)$ varies slowly over $k_BT$, the Sommerfeld expansion of the moments gives $\mathcal{L}^{(1)} \approx (\pi^2/3)(k_BT)^2\,\Sigma'(\mu)$ and $\mathcal{L}^{(2)} \approx (\pi^2/3)(k_BT)^2\,\Sigma(\mu)$, so

$$
S \approx -\frac{\pi^2}{3}\frac{k_B^2 T}{e}\,\frac{d\ln\Sigma}{d\varepsilon}\bigg|_{\mu} , \qquad
\frac{\kappa}{\sigma T} \approx \frac{\pi^2}{3}\left(\frac{k_B}{e}\right)^2 = 2.44\times10^{-8}\ \mathrm{W\,\Omega\,K^{-2}} ,
$$

the Mott formula and the Wiedemann–Franz law [1]. A metal therefore has a small $S$, whose sign is that of the slope of $\Sigma(\varepsilon)$ at $E_F$, not of the carrier charge. In general $S = -\langle\varepsilon - \mu\rangle/(eT)$, with the average weighted by $\Sigma(\varepsilon)(-\partial f^0/\partial\varepsilon)$: the mean energy that a carrier transports above $\mu$, divided by $eT$. In a non-degenerate semiconductor this average is over the carriers in the Boltzmann tail of a band. Electrons, above $\mu$, give a negative $S$, and holes, below it, a positive one. Each $k_BT$ of distance between $\mu$ and the band edge adds about $k_B/e = 86\ \mu$V/K to $|S|$, and when $\mu$ is near midgap the electron and hole contributions cancel, so $S$ changes sign there.

## Units, spin factor and normalisation

The code evaluates the moments in its own units: energies and $k_BT$ in eV, $\hbar\mathbf{v}_{n\mathbf{k}} = \nabla_{\mathbf{k}}\varepsilon_{n\mathbf{k}}$ in eV bohr, and $\Omega$ in bohr$^3$. The temperature in kelvin is converted to $k_BT$ with $k_B = 1/11604.525$ eV/K. Collecting the constants gives

$$
\frac{\sigma_{ij}}{\tau}\ [\mathrm{S\,m^{-1}\,s^{-1}}] = \frac{e^3}{\hbar^2 a_0}\,\frac{g_s}{\Omega N_\mathrm{mesh}} \sum_{n\mathbf{k}} \frac{\partial\varepsilon_{n\mathbf{k}}}{\partial k_i}\frac{\partial\varepsilon_{n\mathbf{k}}}{\partial k_j} \left(-\frac{\partial f^0}{\partial\varepsilon}\right) , \qquad \frac{e^3}{\hbar^2 a_0} = 6.9884\times10^{21}\ \mathrm{S\,m^{-1}\,s^{-1}} ,
$$

with $a_0$ the Bohr radius and the sum in eV bohr$^{-1}$. The same prefactor, divided by $T$ and multiplied by the energy factors in eV, gives $S$ in V/K and $\kappa_0$ in W m$^{-1}$ K$^{-1}$; the Hall coefficient carries $a_0^3/e = 9.249\times10^{-13}$ m$^3$ C$^{-1}$.

The k-sum is normalised by $N_\mathrm{mesh}$, the number of points of the full interpolated mesh, so each k-point weighs $1/N_\mathrm{mesh}$ and the sum is a Brillouin-zone average. The spin factor $g_s$ is 2 for a calculation without spin polarisation and without spin–orbit coupling. It is 1 with spin–orbit coupling, where each band is a spinor and is counted once, and 1 in a spin-polarised calculation, where each spin channel is written to its own files, `_0` and `_1`. In that case the moments of the two channels add: the total conductivity is $\sigma_\uparrow + \sigma_\downarrow$, and the total Seebeck coefficient is the conductivity-weighted average $(\sigma_\uparrow + \sigma_\downarrow)^{-1}(\sigma_\uparrow S_\uparrow + \sigma_\downarrow S_\downarrow)$, the two-current model.

## Constant relaxation time

The moments leave $\tau_{n\mathbf{k}}$ to be specified. Without scattering channels, `transport()` sets $\tau_{n\mathbf{k}} = 1$ s for every state, the constant relaxation-time approximation. $\tau$ then factors out of every moment, so the files hold $\sigma/\tau$ in S m$^{-1}$ s$^{-1}$, $\kappa_0/\tau$ in W m$^{-1}$ K$^{-1}$ s$^{-1}$ and the power factor $S^2\sigma/\tau$ in W m$^{-1}$ K$^{-2}$ s$^{-1}$. $S$ and $R_H$, ratios in which $\tau$ cancels, are absolute predictions. To compare $\sigma$ with experiment you multiply by a $\tau$ from another source, a measured mobility or resistivity, for example; relaxation times of carriers in metals and doped semiconductors at room temperature are of order 1 to 100 fs.

The approximation assumes that all states near $\mu$ scatter at the same rate. It is reasonable in a metal, where only a shell of width $k_BT$ around $E_F$ conducts and $\tau$ varies little across it. It fails where the conducting states span energies over which the scattering changes, as in a semiconductor, where the rate of acoustic-phonon scattering grows as the square root of the carrier energy above the band edge; then $S$ also changes, because it depends on how $\tau$ weighs states above and below $\mu$ (see [Energy-dependent relaxation times](#energy-dependent-relaxation-times)).

## Rigid-band approximation

`transport()` scans $\mu$ through the bands of the undoped crystal: each value of $\mu$ stands for a crystal with a different number of electrons but the same $\varepsilon_{n\mathbf{k}}$ and $\mathbf{v}_{n\mathbf{k}}$. This rigid-band approximation treats doping as filling or emptying the bands of the host. It neglects the change of the self-consistent potential with the added charge, the impurity states that dopants introduce in or near the gap, and the disorder that broadens the bands. It holds at low doping, where a dopant removes or adds a small fraction of an electron per cell and its levels are shallow, and it fails for heavy doping, alloying, or dopants with deep levels or resonant states near the band edges. The band structure is that of the DFT functional, so with PBE the gap is too small, and every result that depends on thermally excited carriers across the gap (the bipolar reduction of $S$, the intrinsic regime of the next section) appears at too low a temperature.

## Chemical potential at fixed carrier density

A doped sample holds a fixed number of carriers, so its chemical potential moves with temperature. {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.doping` finds, for each temperature, the $\mu$ at which the bands hold the valence electrons of the undoped crystal minus the doping:

$$
2\int_{\varepsilon_\mathrm{min}}^{\varepsilon_\mathrm{max}} g(\varepsilon)\, f^0(\varepsilon; \mu, T)\, d\varepsilon + N_\mathrm{core} = N_\mathrm{el} - n_d\,\Omega ,
$$

with $g(\varepsilon)$ the DOS per cell and per spin of the projectable bands, $N_\mathrm{el}$ the valence electrons per cell of the DFT run, $n_d$ the doping `doping_conc` in cm$^{-3}$ (positive for holes, negative for electrons), $\Omega$ converted to cm$^3$, and $N_\mathrm{core}$ the argument `core_electrons`. The integration range is `emin` to `emax`. The DOS is that of [Density of states on a k-mesh](density-of-states.md#density-of-states-on-a-k-mesh), with the adaptive widths or, without smearing, the fixed width `delta`. The factor 2 is fixed, so the equation counts the electrons correctly only for a calculation without spin polarisation and without spin–orbit coupling.

The equation balances a small number against a large one: the doping adds $|n_d|\Omega$ electrons per cell, $4.4\times10^{-4}$ for $10^{19}$ cm$^{-3}$ in a cell of 44 Å$^3$, to the $N_\mathrm{el}$ of the valence bands, so the integral must be accurate to a small fraction of $|n_d|\Omega$. The range must therefore include every valence state, or `core_electrons` must count exactly the electrons below `emin`, which holds only when no band crosses `emin`. The energy grid must be finer than the narrowest smearing widths (the floor `min_width` of {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.adaptive_smearing`), and the k-mesh must resolve the states within a few $k_BT$ of the band edge that hold the carriers.

The solution has a simple form for a non-degenerate semiconductor with one parabolic conduction band. The electron density $n = N_c\,e^{-(E_c - \mu)/k_BT}$, with $E_c$ the conduction-band minimum and $N_c \propto T^{3/2}$ the effective density of states, gives $\mu = E_c - k_BT\ln(N_c/n)$ [5]. At fixed $n$, $\mu$ therefore falls below the band edge faster than linearly in $T$. When the intrinsic density $n_i \propto T^{3/2} e^{-E_g/2k_BT}$, with $E_g$ the gap, exceeds $|n_d|$, electrons and holes are both present and $\mu$ moves towards midgap. At high doping the electrons become degenerate and $\mu$ enters the band.

The equation assumes that every dopant is ionised at every temperature, because the rigid bands have no donor or acceptor levels. It therefore misses the freeze-out of carriers onto shallow dopants at low temperature, which matters below the doping at which the impurity band merges with the host band.

## Energy-dependent relaxation times

In a semiconductor the carriers occupy an energy range of several $k_BT$ above the band edge, and the scattering rate changes across it. `transport()` therefore accepts a list of scattering channels, each of which gives a relaxation time $\tau_c$ for every state.

### Matthiessen's rule

The channels are combined by adding their rates for each state,

$$
\frac{1}{\tau_{n\mathbf{k}}} = \sum_c \frac{w_c}{\tau_c(\varepsilon_{n\mathbf{k}}, T)} ,
$$

with weights $w_c$ from `scattering_weights` (1 by default). Adding rates is exact when the mechanisms act independently, each with a transition probability that does not depend on the others, so that the total probability is their sum. Because the sum is taken state by state before the k-sum, the resistivities of the channels do not add: two mechanisms with different energy dependences weigh different carriers, and the total resistivity exceeds the sum of the resistivities of each alone. This deviation from Matthiessen's rule at the level of the resistivity is included [2].

### Built-in scattering models

The built-in models, selected by name, are textbook rates for one isotropic parabolic band of effective mass $m^*$, the parameter `ms`, with parameters read from `tau_dict` (units in {py:mod}`PAOFLOW.boltzmann.do_tau_models`). Each receives $k_BT$ in eV and an energy $E$ and returns $\tau$ in seconds. $E$ stands for the kinetic energy of a carrier above its band edge; [Energy reference](#energy-reference-of-the-built-in-models) says which energy the code passes.

Acoustic phonons, `'acoustic'`, scatter through the deformation potential $D_\mathrm{ac}$, the shift of the band edge per unit strain [6]. In the elastic, equipartition limit ($k_BT$ larger than the energy of the phonons involved) the rate is proportional to $T$ and to the density of final states,

$$
\frac{1}{\tau_\mathrm{ac}} = \frac{(2m^*)^{3/2} D_\mathrm{ac}^2\, k_BT \sqrt{E}}{2\pi \rho\, \hbar^4 v_s^2} ,
$$

with $\rho$ the mass density and $v_s$ the sound velocity. It fails at low temperature, where the energy of the phonons that scatter a carrier becomes comparable to $k_BT$, and in multivalley semiconductors such as Si, where intervalley phonons add a separate channel.

Non-polar optical phonons, `'optical'`, of energy $\hbar\omega_0$ (`hwlo`) scatter through the optical deformation potential $D_\mathrm{op}$, an energy per unit length [7]:

$$
\frac{1}{\tau_\mathrm{op}} = \frac{(m^*)^{3/2} D_\mathrm{op}^2}{\sqrt{2}\,\pi \rho\, \hbar^2\, \hbar\omega_0} \Bigl[N_0\sqrt{E + \hbar\omega_0} + (N_0 + 1)\sqrt{E - \hbar\omega_0}\;\Theta(E - \hbar\omega_0)\Bigr] ,
$$

with $N_0$ the Bose–Einstein occupation of the phonon. The first term is absorption, the second emission, which needs $E > \hbar\omega_0$. The code multiplies `D_op` by the electron-volt to joule factor only, so the formula is in SI units when `D_op` is given in eV/m. The scattering is inelastic by $\hbar\omega_0$, which the relaxation-time form treats only approximately.

Ionised impurities, `'impurity'`, scatter through a screened Coulomb potential, the Brooks–Herring model [8]:

$$
\frac{1}{\tau_\mathrm{imp}} = \frac{\pi n_I Z^2 e^4}{\sqrt{2m^*}\,(4\pi\epsilon)^2 E^{3/2}} \left[\ln(1 + \gamma^2) - \frac{\gamma^2}{1 + \gamma^2}\right] , \qquad \gamma^2 = \frac{8m^*E}{\hbar^2 q_0^2} , \qquad q_0^2 = \frac{e^2 n_I}{\epsilon\, k_BT} ,
$$

with $n_I$ (`nI`) impurities of charge $Ze$ (`Zi`). The rate falls as $E^{-3/2}$, so slow carriers scatter most. The textbook form uses the static dielectric constant $\epsilon_s$ and the density of free carriers in the Debye screening length. The code uses $\epsilon = \epsilon_0(\varepsilon_s + \varepsilon_\infty)$, with $\epsilon_0$ the vacuum permittivity and the relative constants `eps_0` and `eps_inf`, and screens with $n_I$, which equals the carrier density only for full ionisation without compensation. For Si, where $\varepsilon_s = \varepsilon_\infty$, this doubles the dielectric constant, which divides the prefactor of the rate by four, partly offset by the weaker screening in the logarithm. The Debye form of $q_0$ assumes non-degenerate carriers, and the Born approximation behind the model fails at high impurity density and low energy.

Piezoelectric (polar acoustic) scattering, `'polar_acoustic'`, couples the strain of an acoustic wave to the electrons through the piezoelectric constant `piezo`. Its rate is proportional to $T\,E^{-1/2}$ with a screening correction [5]. Polar optical (Fröhlich) scattering, `'polar_optical'`, couples the electrons to the field of longitudinal optical phonons of energies `hwlo`, with a strength proportional to $1/\varepsilon_\infty - 1/\varepsilon_s$ [5]. The code includes absorption and emission weighted by ratios of Fermi–Dirac occupations of the final and initial states at the Fermi energy `Ef`. Both use $\epsilon_0(\varepsilon_s + \varepsilon_\infty)$ in place of $\epsilon_0\varepsilon_s$, as the impurity model does, and the polar optical model uses $1/\varepsilon_\infty - 1/(\varepsilon_s + \varepsilon_\infty)$ for the Fröhlich strength. Polar optical scattering is inelastic and strongly anisotropic, the case in which the relaxation-time approximation is least accurate [5].

All five models describe a single valley with an isotropic mass. They miss the anisotropic masses of Si and Ge, the non-parabolicity away from the band edge, intervalley scattering, and the coupling of degenerate valence bands. Their parameters are therefore fitted or taken from the literature for each material, as in reference [9]. For electron–phonon rates computed from first principles, {py:mod}`PAOFLOW.boltzmann.scattering_channels` provides channels that enter the same sum; [Scattering channels](../../internals/scattering-channels.md) describes them.

### Energy reference of the built-in models

The formulas above take the kinetic energy of the carrier. The code passes instead $E = |\varepsilon_{n\mathbf{k}}|$, measured from the [energy zero of the run](density-of-states.md#energy-zero-from-the-dft-run), which for a semiconductor computed with fixed occupations is the valence-band maximum. For holes this is their kinetic energy. For an electron in the conduction band it is $E_g$ plus its kinetic energy, so the acoustic rate is evaluated at an energy that changes little across the conducting states: the energy dependence of the rate is lost, and its value is too large by $\sqrt{(E_g + E)/E}$. The ionised-impurity rate, which falls with energy, becomes too small for the same reason. To use the kinetic energy, write the model as a {py:class}`~PAOFLOW.boltzmann.TauModel.TauModel`, a function of $k_BT$, $|\varepsilon_{n\mathbf{k}}|$ and a parameter dictionary, that subtracts the conduction-band minimum from the conduction states. In a metal, where the zero is the Fermi energy, $|\varepsilon_{n\mathbf{k}}|$ is the distance from $E_F$, not the kinetic energy of a parabolic band.

## Hall coefficient

In a weak magnetic field $\mathbf{B}$ the Lorentz force adds a term of first order in $\mathbf{B}$ to the solution of the Boltzmann equation, which involves the curvature of the bands, the inverse mass tensor $(M^{-1})_{jq} = \hbar^{-2}\,\partial^2\varepsilon_{n\mathbf{k}}/\partial k_j\partial k_q$, and a second power of $\tau$ [4]. With `do_hall=True`, `transport()` evaluates the kernel

$$
\mathcal{L}^{H}_{ijp} = \frac{g_s}{\Omega N_\mathrm{mesh}} \sum_{n\mathbf{k}} \tau^2_{n\mathbf{k}} \sum_{qr} \epsilon_{pqr}\, v^i_{n\mathbf{k}}\, v^r_{n\mathbf{k}}\, (M^{-1})_{jq,n\mathbf{k}} \left(-\frac{\partial f^0}{\partial\varepsilon}\right)
$$

with $\epsilon_{pqr}$ the Levi-Civita symbol, and from it the Hall tensor

$$
R_{ijp} = -\frac{1}{e}\Bigl[\bigl(\mathcal{L}^{(0)}\bigr)^{-1} \mathcal{L}^{H}_{p}\, \bigl(\mathcal{L}^{(0)}\bigr)^{-1}\Bigr]_{ij} .
$$

The file `hall_trace` holds the average $(R_{xyz} + R_{zxy} + R_{yzx})/3$, and the file `hall` the antisymmetric parts $(R_{ijp} - R_{jip})/2$. The first moment of the same kernel, with $(\varepsilon - \mu)$, gives the Nernst tensor. $\tau$ cancels in $R$ for a constant relaxation time, and for a single parabolic band $|R_H| = 1/(ne)$, so the Hall coefficient measures the carrier density $n$. With several bands, or an energy-dependent $\tau$, it is a weighted average that does not equal $1/(ne)$ for any single carrier type. The curvature is computed by {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.gradient_and_momenta` with `band_curvature=True`, which must precede a call with `do_hall=True`.

## Used in

- [Tutorial 02](../tutorials/tutorial02.ipynb): $\sigma/\tau$, $S$ and $\kappa_0/\tau$ of Si as functions of $\mu$ at constant $\tau$, $\mu(T)$ of n-type Si at fixed carrier density, and the resistivity with acoustic-phonon and ionised-impurity relaxation times.
- [Tutorial 04](../tutorials/tutorial04.ipynb): the transport tensors of a 16-atom Si supercell on the sparse backend, at constant $\tau$.
- [How to plot PAOFLOW output with GPAO](../howtos/plotting.md): $\sigma/\tau$ at constant $\tau$ and the rigid-band `--fermi-shift`.
- [How to run PAOFLOW on a tight-binding model](../howtos/tight-binding-model.md): why `transport()` stops on a two-dimensional model.

## References

1. N. W. Ashcroft and N. D. Mermin, *Solid State Physics* (Holt, Rinehart and Winston, New York, 1976), chapters 12 and 13.
2. J. M. Ziman, *Electrons and Phonons* (Oxford University Press, Oxford, 1960).
3. G. D. Mahan and J. O. Sofo, The best thermoelectric, *Proc. Natl. Acad. Sci. USA* 93, 7436 (1996).
4. G. K. H. Madsen and D. J. Singh, BoltzTraP. A code for calculating band-structure dependent quantities, *Comput. Phys. Commun.* 175, 67 (2006).
5. M. Lundstrom, *Fundamentals of Carrier Transport*, 2nd ed. (Cambridge University Press, Cambridge, 2000).
6. J. Bardeen and W. Shockley, Deformation potentials and mobilities in non-polar crystals, *Phys. Rev.* 80, 72 (1950).
7. C. Jacoboni, *Theory of Electron Transport in Semiconductors* (Springer, Berlin, 2010).
8. D. Chattopadhyay and H. J. Queisser, Electron scattering by ionized impurities in semiconductors, *Rev. Mod. Phys.* 53, 745 (1981).
9. A. Jayaraj, I. Siloi, M. Fornari and M. Buongiorno Nardelli, Relaxation time approximations in PAOFLOW 2.0, *Sci. Rep.* 12 (2022), [doi:10.1038/s41598-022-08931-5](https://doi.org/10.1038/s41598-022-08931-5).
