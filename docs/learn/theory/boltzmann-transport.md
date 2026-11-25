# Boltzmann transport in the relaxation-time approximation

This page derives the electronic transport coefficients that PAOFLOW computes from the band structure. They are the electrical conductivity $\sigma$, the Seebeck coefficient $S$, the electronic thermal conductivity $\kappa$ and the Hall coefficient $R_H$. The page also derives the chemical potential of a doped crystal and the energy-dependent relaxation times that enter the coefficients:

| Method or class | Computes |
| --- | --- |
| {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.transport` | $\sigma$, $S$, $\kappa$ and $R_H$ as functions of the chemical potential and the temperature |
| {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.doping` | the chemical potential at a fixed carrier density |
| {py:class}`~PAOFLOW.boltzmann.TauModel.TauModel` | a relaxation time that depends on the energy of the state |

Every quantity on the page is a sum over the eigenvalues $\varepsilon_{n\mathbf{k}}$ and band velocities $\mathbf{v}_{n\mathbf{k}}$ of the interpolated Hamiltonian $\tilde H(\mathbf{k})$ on a mesh of $N_\mathrm{mesh}$ k-points. $\tilde H(\mathbf{k})$ is derived on [The PAO Hamiltonian](pao-hamiltonian.md). The band velocities, and the adaptive smearing that turns the k-sums into smooth functions of energy, are derived on [Band velocities, density of states and adaptive smearing](density-of-states.md).

## Linearised Boltzmann equation

The Boltzmann equation gives the occupation $f_{n\mathbf{k}}(\mathbf{r})$ of the Bloch state $n\mathbf{k}$ in a crystal driven out of equilibrium by an electric field $\mathbf{E}$ and a temperature gradient $\nabla T$. Electrons are treated as wave packets that move with the band velocity and are accelerated by the force $-e\mathbf{E}$, with $e > 0$ the elementary charge. In a steady state, scattering balances the change of $f$ from drift and acceleration:

$$
\mathbf{v}_{n\mathbf{k}} \cdot \nabla_{\mathbf{r}} f_{n\mathbf{k}} - \frac{e}{\hbar}\,\mathbf{E} \cdot \nabla_{\mathbf{k}} f_{n\mathbf{k}} = \left(\frac{\partial f_{n\mathbf{k}}}{\partial t}\right)_\mathrm{coll} .
$$

The relaxation-time approximation replaces the collision integral by a relaxation at the rate $1/\tau_{n\mathbf{k}}$ towards the local equilibrium $f^0$. Here $f^0$ is the Fermi–Dirac distribution at the local $\mu(\mathbf{r})$ and $T(\mathbf{r})$:

$$
\left(\frac{\partial f_{n\mathbf{k}}}{\partial t}\right)_\mathrm{coll} = -\frac{f_{n\mathbf{k}} - f^0(\varepsilon_{n\mathbf{k}})}{\tau_{n\mathbf{k}}} .
$$

For weak fields, linear response replaces $f$ on the left-hand side by $f^0$. With $\nabla_{\mathbf{r}} f^0 = (-\partial f^0/\partial\varepsilon)\,[\nabla\mu + (\varepsilon - \mu)\nabla T/T]$ and $\nabla_{\mathbf{k}} f^0 = \hbar\mathbf{v}\,\partial f^0/\partial\varepsilon$, the deviation from equilibrium is

$$
f_{n\mathbf{k}} - f^0 = \tau_{n\mathbf{k}} \left(-\frac{\partial f^0}{\partial\varepsilon}\right)_{\varepsilon_{n\mathbf{k}}} \mathbf{v}_{n\mathbf{k}} \cdot \left[-e\boldsymbol{\mathcal{E}} - \frac{\varepsilon_{n\mathbf{k}} - \mu}{T}\,\nabla T\right] ,
$$

with $\boldsymbol{\mathcal{E}} = \mathbf{E} + \nabla\mu/e$ the field that drives the current, the gradient of the electrochemical potential [1]. The factor $-\partial f^0/\partial\varepsilon$ restricts the response to states within a few $k_BT$ of $\mu$, because only they can change their occupation. The change of each is proportional to its velocity along the driving force and to $\tau_{n\mathbf{k}}$, the time the force acts on it between collisions.

The relaxation-time form neglects the electrons scattered into a state from other states. How well it holds depends on the scattering [2, 5]:

| Scattering | Relaxation-time form |
| --- | --- |
| Elastic and isotropic | exact |
| Elastic and anisotropic | holds when $\tau$ is read as the momentum relaxation time, which weighs each scattering event by $1 - \cos\theta$ of its deflection angle |
| Inelastic on the scale of $k_BT$, such as polar optical phonons below the Debye temperature | fails: no single $\tau_{n\mathbf{k}}$ describes the relaxation |

The semiclassical picture itself needs the energy uncertainty $\hbar/\tau$ to be small against $k_BT$ and against the spacing of the bands. For $\tau = 10$ fs, $\hbar/\tau$ is 66 meV, already above $k_BT$ at room temperature. Band velocities alone carry the current, so interband coherence and the anomalous velocity of the Berry curvature are left out.

## Transport moments

The charge and heat currents follow from the deviation of the occupation. Each state carries the charge current $-e\mathbf{v}_{n\mathbf{k}}$ and the heat current $(\varepsilon_{n\mathbf{k}} - \mu)\mathbf{v}_{n\mathbf{k}}$. The sum over states per unit volume is $g_s/(\Omega N_\mathrm{mesh}) \sum_{n\mathbf{k}}$, with $\Omega$ the cell volume and $g_s$ the spin degeneracy. Inserting $f - f^0$ gives

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

which {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.transport` evaluates on a grid of $\mu$ between `emin` and `emax`, for each temperature between `tmin` and `tmax`. $\mathcal{L}^{(0)}$ counts the states at the chemical potential, weighted by $\tau v^2$. $\mathcal{L}^{(1)}$ measures how asymmetric that weight is about $\mu$, and $\mathcal{L}^{(2)}$ how wide it is.

The sum runs over the bands that the projectability keeps (see [The PAO Hamiltonian](pao-hamiltonian.md)). Therefore $\mu + 10\,k_BT$ must lie below the energy where these bands stop representing the DFT bands.

### Transport distribution and smearing

The moments are integrals of one function of energy, the transport distribution

$$
\Sigma_{ij}(\varepsilon) = \frac{g_s}{\Omega N_\mathrm{mesh}} \sum_{n\mathbf{k}} \tau_{n\mathbf{k}}\, v^i_{n\mathbf{k}}\, v^j_{n\mathbf{k}}\, \delta(\varepsilon - \varepsilon_{n\mathbf{k}}) , \qquad
\mathcal{L}^{(\alpha)}_{ij} = \int d\varepsilon\, \Sigma_{ij}(\varepsilon) \left(-\frac{\partial f^0}{\partial\varepsilon}\right) (\varepsilon - \mu)^\alpha ,
$$

introduced by Mahan and Sofo [3] and used in the BoltzTraP code [4]. $\Sigma(\varepsilon)$ holds all the band structure. Temperature and chemical potential enter only through the Fermi window $-\partial f^0/\partial\varepsilon = [4k_BT\cosh^2((\varepsilon - \mu)/2k_BT)]^{-1}$, which falls to 1% of its peak at $|\varepsilon - \mu| \approx 6\,k_BT$.

PAOFLOW evaluates the moments in one of two ways, set by the `smearing` argument of the `PAOFLOW` constructor:

| `smearing` | How the moments are evaluated | Errors of its own |
| --- | --- | --- |
| `None` | the Fermi window is taken at each eigenvalue, as in the sum over states above | a state counts only where the Fermi window happens to sample it |
| `'gauss'` or `'m-p'` (adaptive smearing) | each delta function in $\Sigma(\varepsilon)$ becomes a Gaussian (or Methfessel–Paxton function) of the width $\sigma_{n\mathbf{k}}$ of that state; $\Sigma(\varepsilon)$ is sampled on the $\mu$ grid extended by about $22\,k_BT$ on each side, linearly interpolated at half the grid spacing, and integrated against the Fermi window by Simpson's rule | the band edges acquire tails of width $\sigma_{n\mathbf{k}}$ that persist at $T \to 0$; the $\mu$ spacing sets the energy resolution of $\Sigma(\varepsilon)$ |

The smeared form converges faster with the k-mesh, because a state contributes over its width instead of only where the Fermi window happens to sample it. With smearing, choose a $\mu$ spacing smaller than the narrowest widths near $\mu$.

### Conductivity, Seebeck coefficient and thermal conductivity

The three coefficients follow from the currents under the conditions of the measurement:

| Coefficient | Condition | Definition |
| --- | --- | --- |
| Conductivity $\sigma$ | uniform temperature | the current per unit field |
| Seebeck coefficient $S$ | open circuit, $\mathbf{j} = 0$ | the field that a temperature gradient builds up, $\boldsymbol{\mathcal{E}} = S\,\nabla T$ |
| Thermal conductivity $\kappa$ | open circuit, $\mathbf{j} = 0$ | the heat current per unit gradient, $\mathbf{j}_Q = -\kappa\nabla T$ |

Solving the current equations under these conditions gives

$$
\begin{aligned}
\sigma &= e^2 \mathcal{L}^{(0)} , \\
S &= -\frac{1}{eT}\,\bigl[\mathcal{L}^{(0)}\bigr]^{-1} \mathcal{L}^{(1)} , \\
\kappa &= \frac{1}{T}\Bigl[\mathcal{L}^{(2)} - \mathcal{L}^{(1)} \bigl[\mathcal{L}^{(0)}\bigr]^{-1} \mathcal{L}^{(1)}\Bigr] ,
\end{aligned}
$$

and the power factor $S\sigma S$, which reduces to $S^2\sigma$ for a scalar $S$ [1]. $S$ and $\kappa$ need the inverse of the full $3\times3$ tensor $\mathcal{L}^{(0)}$. On a two-dimensional model $\mathcal{L}^{(0)}_{zz}$ vanishes, so `transport()` stops.

The code evaluates $\sigma$, $S$ and the power factor as written. For the thermal conductivity, the second term enters with a unit factor $10^{-15}$ times smaller than that of the first. The file `kappa` therefore holds the first term alone,

$$
\kappa_0 = \frac{1}{T}\,\mathcal{L}^{(2)} = \kappa + T\,S\sigma S ,
$$

the heat conductance at zero field rather than at zero current. For a diagonal tensor (a cubic crystal, or any crystal along its principal axes) you obtain $\kappa$ from the two files as $\kappa/\tau = \kappa_0/\tau - T\,S^2\sigma/\tau$. In a metal the difference is of relative order $(k_BT/E_F)^2$ and can be ignored. In a semiconductor with $\mu$ in or near the gap it is comparable to or larger than $\kappa$, because $S$ is large there.

The coefficients take a transparent form in two limits. The first is a degenerate system, where $\Sigma(\varepsilon)$ varies slowly over $k_BT$. There the Sommerfeld expansion of the moments gives $\mathcal{L}^{(1)} \approx (\pi^2/3)(k_BT)^2\,\Sigma'(\mu)$ and $\mathcal{L}^{(2)} \approx (\pi^2/3)(k_BT)^2\,\Sigma(\mu)$, so

$$
S \approx -\frac{\pi^2}{3}\frac{k_B^2 T}{e}\,\frac{d\ln\Sigma}{d\varepsilon}\bigg|_{\mu} , \qquad
\frac{\kappa}{\sigma T} \approx \frac{\pi^2}{3}\left(\frac{k_B}{e}\right)^2 = 2.44\times10^{-8}\ \mathrm{W\,\Omega\,K^{-2}} ,
$$

the Mott formula and the Wiedemann–Franz law [1]. A metal therefore has a small $S$. Its sign is that of the slope of $\Sigma(\varepsilon)$ at $E_F$, not that of the carrier charge.

The second limit is a non-degenerate semiconductor. In general $S = -\langle\varepsilon - \mu\rangle/(eT)$, with the average weighted by $\Sigma(\varepsilon)(-\partial f^0/\partial\varepsilon)$: $S$ is the mean energy that a carrier transports above $\mu$, divided by $eT$. In a non-degenerate semiconductor this average is over the carriers in the Boltzmann tail of a band. Electrons, above $\mu$, give a negative $S$, and holes, below it, a positive one. Each $k_BT$ of distance between $\mu$ and the band edge adds about $k_B/e = 86\ \mu$V/K to $|S|$. When $\mu$ is near midgap, the electron and hole contributions cancel, so $S$ changes sign there.

## Units, spin factor and normalisation

The code evaluates the moments in its own units: energies and $k_BT$ in eV, $\hbar\mathbf{v}_{n\mathbf{k}} = \nabla_{\mathbf{k}}\varepsilon_{n\mathbf{k}}$ in eV bohr, and $\Omega$ in bohr$^3$. It converts the temperature in kelvin to $k_BT$ with $k_B = 1/11604.525$ eV/K. Collecting the constants gives

$$
\frac{\sigma_{ij}}{\tau}\ [\mathrm{S\,m^{-1}\,s^{-1}}] = \frac{e^3}{\hbar^2 a_0}\,\frac{g_s}{\Omega N_\mathrm{mesh}} \sum_{n\mathbf{k}} \frac{\partial\varepsilon_{n\mathbf{k}}}{\partial k_i}\frac{\partial\varepsilon_{n\mathbf{k}}}{\partial k_j} \left(-\frac{\partial f^0}{\partial\varepsilon}\right) , \qquad \frac{e^3}{\hbar^2 a_0} = 6.9884\times10^{21}\ \mathrm{S\,m^{-1}\,s^{-1}} ,
$$

with $a_0$ the Bohr radius and the sum in eV bohr$^{-1}$. The same prefactor, divided by $T$ and multiplied by the energy factors in eV, gives $S$ in V/K and $\kappa_0$ in W m$^{-1}$ K$^{-1}$. The Hall coefficient carries $a_0^3/e = 9.249\times10^{-13}$ m$^3$ C$^{-1}$.

The k-sum is normalised by $N_\mathrm{mesh}$, the number of points of the full interpolated mesh. Each k-point weighs $1/N_\mathrm{mesh}$, so the sum is a Brillouin-zone average. The spin factor $g_s$ depends on the calculation:

| Calculation | $g_s$ | Reason |
| --- | --- | --- |
| Without spin polarisation and without spin–orbit coupling | 2 | each band holds both spins |
| With spin–orbit coupling | 1 | each band is a spinor and is counted once |
| Spin-polarised | 1 | each spin channel is a separate set of bands, written to its own files, `_0` and `_1` |

In a spin-polarised calculation the moments of the two channels add. The total conductivity is $\sigma_\uparrow + \sigma_\downarrow$. The total Seebeck coefficient is the conductivity-weighted average $(\sigma_\uparrow + \sigma_\downarrow)^{-1}(\sigma_\uparrow S_\uparrow + \sigma_\downarrow S_\downarrow)$, the two-current model.

## Constant relaxation time

The moments leave $\tau_{n\mathbf{k}}$ to be specified. Without scattering channels, `transport()` sets $\tau_{n\mathbf{k}} = 1$ s for every state: the constant relaxation-time approximation. $\tau$ then factors out of every moment, and the files hold:

| File | Quantity | Unit |
| --- | --- | --- |
| `sigma` | $\sigma/\tau$ | S m$^{-1}$ s$^{-1}$ |
| `kappa` | $\kappa_0/\tau$ | W m$^{-1}$ K$^{-1}$ s$^{-1}$ |
| `PF` | the power factor $S^2\sigma/\tau$ | W m$^{-1}$ K$^{-2}$ s$^{-1}$ |
| `Seebeck` | $S$, a ratio in which $\tau$ cancels | V/K |
| `hall_trace`, `hall` | $R_H$, a ratio in which $\tau$ cancels | m$^3$ C$^{-1}$ |

$S$ and $R_H$ are therefore absolute predictions. To compare $\sigma$ with experiment, multiply it by a $\tau$ from another source, for example a measured mobility or resistivity. Relaxation times of carriers in metals and doped semiconductors at room temperature are of order 1 to 100 fs.

The approximation assumes that all states near $\mu$ scatter at the same rate. It is reasonable in a metal, where only a shell of width $k_BT$ around $E_F$ conducts and $\tau$ varies little across it. It fails where the conducting states span energies over which the scattering changes. In a semiconductor, for example, the rate of acoustic-phonon scattering grows as the square root of the carrier energy above the band edge. $S$ then changes too, because it depends on how $\tau$ weighs the states above and below $\mu$ (see [Energy-dependent relaxation times](#energy-dependent-relaxation-times)).

## Rigid-band approximation

`transport()` scans $\mu$ through the bands of the undoped crystal. Each value of $\mu$ stands for a crystal with a different number of electrons but the same $\varepsilon_{n\mathbf{k}}$ and $\mathbf{v}_{n\mathbf{k}}$. This rigid-band approximation treats doping as filling or emptying the bands of the host. It neglects three effects of doping: the change of the self-consistent potential with the added charge, the impurity states that dopants introduce in or near the gap, and the disorder that broadens the bands.

The approximation holds at low doping, where a dopant removes or adds a small fraction of an electron per cell and its levels are shallow. It fails for heavy doping, for alloying, and for dopants with deep levels or resonant states near the band edges.

The bands are also those of the DFT functional, so with PBE the gap is too small. Every result that depends on carriers thermally excited across the gap therefore appears at too low a temperature. Two such results are the bipolar reduction of $S$ and the intrinsic regime of the next section.

## Chemical potential at fixed carrier density

A doped sample holds a fixed number of carriers, so its chemical potential moves with temperature. {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.doping` finds, for each temperature, the $\mu$ at which the bands hold the valence electrons of the undoped crystal minus the doping:

$$
2\int_{\varepsilon_\mathrm{min}}^{\varepsilon_\mathrm{max}} g(\varepsilon)\, f^0(\varepsilon; \mu, T)\, d\varepsilon + N_\mathrm{core} = N_\mathrm{el} - n_d\,\Omega ,
$$

| Symbol | Meaning | Argument |
| --- | --- | --- |
| $g(\varepsilon)$ | the DOS per cell and per spin of the projectable bands, that of [Density of states on a k-mesh](density-of-states.md#density-of-states-on-a-k-mesh), with the adaptive widths or, without smearing, a fixed width | `delta` |
| $\varepsilon_\mathrm{min}$, $\varepsilon_\mathrm{max}$ | the integration range | `emin`, `emax` |
| $N_\mathrm{el}$ | the valence electrons per cell of the DFT run | |
| $n_d$ | the doping in cm$^{-3}$, positive for holes and negative for electrons | `doping_conc` |
| $\Omega$ | the cell volume, converted to cm$^3$ | |
| $N_\mathrm{core}$ | the electrons below the integration range | `core_electrons` |

The factor 2 is fixed, so the equation counts the electrons correctly only for a calculation without spin polarisation and without spin–orbit coupling.

The equation balances a small number against a large one. The doping adds $|n_d|\Omega$ electrons per cell to the $N_\mathrm{el}$ of the valence bands: $4.4\times10^{-4}$ for $10^{19}$ cm$^{-3}$ in a cell of 44 Å$^3$. The integral must therefore be accurate to a small fraction of $|n_d|\Omega$, which sets three requirements:

| Setting | Requirement |
| --- | --- |
| `emin` | the range includes every valence state, or `core_electrons` counts exactly the electrons below `emin`, which holds only when no band crosses `emin` |
| Energy grid | finer than the narrowest smearing widths, the floor `min_width` of {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.adaptive_smearing` |
| k-mesh | resolves the states within a few $k_BT$ of the band edge that hold the carriers |

The solution has a simple form for a non-degenerate semiconductor with one parabolic conduction band. The electron density is $n = N_c\,e^{-(E_c - \mu)/k_BT}$, with $E_c$ the conduction-band minimum and $N_c \propto T^{3/2}$ the effective density of states. Solving for $\mu$ gives $\mu = E_c - k_BT\ln(N_c/n)$ [5]. At fixed $n$, $\mu$ therefore falls below the band edge faster than linearly in $T$. When the intrinsic density $n_i \propto T^{3/2} e^{-E_g/2k_BT}$, with $E_g$ the gap, exceeds $|n_d|$, electrons and holes are both present, and $\mu$ moves towards midgap. At high doping the electrons become degenerate and $\mu$ enters the band.

The equation assumes that every dopant is ionised at every temperature, because the rigid bands have no donor or acceptor levels. It therefore misses the freeze-out of carriers onto shallow dopants at low temperature. Freeze-out matters below the doping at which the impurity band merges with the host band.

## Energy-dependent relaxation times

In a semiconductor the carriers occupy an energy range of several $k_BT$ above the band edge, and the scattering rate changes across it. `transport()` therefore accepts a list of scattering channels. Each channel gives a relaxation time $\tau_c$ for every state.

### Matthiessen's rule

The channels are combined by adding their rates for each state,

$$
\frac{1}{\tau_{n\mathbf{k}}} = \sum_c \frac{w_c}{\tau_c(\varepsilon_{n\mathbf{k}}, T)} ,
$$

with weights $w_c$ from `scattering_weights` (1 by default). Adding rates is exact when the mechanisms act independently. Each mechanism then has a transition probability that does not depend on the others, and the total probability is their sum.

The rates are added state by state, before the k-sum, so the resistivities of the channels do not add. Two mechanisms with different energy dependences weigh different carriers, and the total resistivity exceeds the sum of the resistivities of each alone. The calculation includes this deviation from Matthiessen's rule at the level of the resistivity [2].

### Built-in scattering models

The built-in models are textbook rates for one isotropic parabolic band of effective mass $m^*$. You select each by name, and it reads its parameters from `tau_dict`, with $m^*$ as `ms` (units in {py:mod}`PAOFLOW.boltzmann.do_tau_models`). Each model receives $k_BT$ in eV and an energy $E$, and returns $\tau$ in seconds. In the formulas below, $E$ is the kinetic energy of a carrier above its band edge; [Energy reference](#energy-reference-of-the-built-in-models) says which energy the code passes.

| Name | Mechanism | Parameters besides `ms` |
| --- | --- | --- |
| `'acoustic'` | acoustic phonons, through the deformation potential | `D_ac`, `rho`, `v` |
| `'optical'` | non-polar optical phonons | `D_op`, `hwlo`, `rho` |
| `'impurity'` | ionised impurities, Brooks–Herring model | `nI`, `Zi`, `eps_0`, `eps_inf` |
| `'polar_acoustic'` | piezoelectric coupling to acoustic phonons | `piezo`, `doping_conc`, `eps_0`, `eps_inf`, `rho`, `v` |
| `'polar_optical'` | Fröhlich coupling to longitudinal optical phonons | `hwlo`, `Ef`, `eps_0`, `eps_inf` |

Acoustic phonons (`'acoustic'`) scatter through the deformation potential $D_\mathrm{ac}$, the shift of the band edge per unit strain [6]. In the elastic, equipartition limit, where $k_BT$ is larger than the energy of the phonons involved, the rate is proportional to $T$ and to the density of final states,

$$
\frac{1}{\tau_\mathrm{ac}} = \frac{(2m^*)^{3/2} D_\mathrm{ac}^2\, k_BT \sqrt{E}}{2\pi \rho\, \hbar^4 v_s^2} ,
$$

with $\rho$ the mass density and $v_s$ the sound velocity. The model fails in two cases. At low temperature, the energy of the phonons that scatter a carrier becomes comparable to $k_BT$. In multivalley semiconductors such as Si, intervalley phonons add a separate channel.

Non-polar optical phonons (`'optical'`) of energy $\hbar\omega_0$ (`hwlo`) scatter through the optical deformation potential $D_\mathrm{op}$, an energy per unit length [7]:

$$
\frac{1}{\tau_\mathrm{op}} = \frac{(m^*)^{3/2} D_\mathrm{op}^2}{\sqrt{2}\,\pi \rho\, \hbar^2\, \hbar\omega_0} \Bigl[N_0\sqrt{E + \hbar\omega_0} + (N_0 + 1)\sqrt{E - \hbar\omega_0}\;\Theta(E - \hbar\omega_0)\Bigr] ,
$$

with $N_0$ the Bose–Einstein occupation of the phonon. The first term is the absorption of a phonon. The second is emission, which needs $E > \hbar\omega_0$. The code multiplies `D_op` by the electron-volt to joule factor only, so the formula is in SI units when `D_op` is given in eV/m. The scattering is inelastic by $\hbar\omega_0$, which the relaxation-time form treats only approximately.

Ionised impurities (`'impurity'`) scatter through a screened Coulomb potential, the Brooks–Herring model [8]:

$$
\frac{1}{\tau_\mathrm{imp}} = \frac{\pi n_I Z^2 e^4}{\sqrt{2m^*}\,(4\pi\epsilon)^2 E^{3/2}} \left[\ln(1 + \gamma^2) - \frac{\gamma^2}{1 + \gamma^2}\right] , \qquad \gamma^2 = \frac{8m^*E}{\hbar^2 q_0^2} , \qquad q_0^2 = \frac{e^2 n_I}{\epsilon\, k_BT} ,
$$

with $n_I$ (`nI`) impurities of charge $Ze$ (`Zi`). The rate falls as $E^{-3/2}$, so slow carriers scatter most. The code departs from the textbook form in two places:

| | Textbook form | Code |
| --- | --- | --- |
| Dielectric constant | the static dielectric constant $\epsilon_s$ | $\epsilon = \epsilon_0(\varepsilon_s + \varepsilon_\infty)$, with $\epsilon_0$ the vacuum permittivity and the relative constants `eps_0` and `eps_inf` |
| Density in the Debye screening length | the density of free carriers | $n_I$, which equals the carrier density only for full ionisation without compensation |

For Si, where $\varepsilon_s = \varepsilon_\infty$, the code doubles the dielectric constant. This divides the prefactor of the rate by four, partly offset by the weaker screening in the logarithm. The Debye form of $q_0$ assumes non-degenerate carriers. The Born approximation behind the model fails at high impurity density and low energy.

Piezoelectric, or polar acoustic, scattering (`'polar_acoustic'`) couples the strain of an acoustic wave to the electrons through the piezoelectric constant `piezo`. Its rate is proportional to $T\,E^{-1/2}$, with a screening correction [5].

Polar optical, or Fröhlich, scattering (`'polar_optical'`) couples the electrons to the field of longitudinal optical phonons of energies `hwlo`. Its strength is proportional to $1/\varepsilon_\infty - 1/\varepsilon_s$ [5]. The code includes absorption and emission, weighted by ratios of the Fermi–Dirac occupations of the final and initial states at the Fermi energy `Ef`. Polar optical scattering is inelastic and strongly anisotropic, the case in which the relaxation-time approximation is least accurate [5].

Both polar models use $\epsilon_0(\varepsilon_s + \varepsilon_\infty)$ in place of $\epsilon_0\varepsilon_s$, as the impurity model does. For the Fröhlich strength, the polar optical model uses $1/\varepsilon_\infty - 1/(\varepsilon_s + \varepsilon_\infty)$.

All five models describe a single valley with an isotropic mass. They miss the anisotropic masses of Si and Ge, the non-parabolicity away from the band edge, intervalley scattering, and the coupling of degenerate valence bands. Their parameters are therefore fitted or taken from the literature for each material, as in reference [9]. {py:mod}`PAOFLOW.boltzmann.scattering_channels` provides channels with electron–phonon rates computed from first principles, which enter the same sum. [Scattering channels](../../internals/scattering-channels.md) describes them.

### Energy reference of the built-in models

The formulas above take the kinetic energy of the carrier. The code passes instead $E = |\varepsilon_{n\mathbf{k}}|$, measured from the [energy zero of the run](density-of-states.md#energy-zero-from-the-dft-run). For a semiconductor computed with fixed occupations, that zero is the valence-band maximum. In a metal it is the Fermi energy.

| Carrier | Energy passed to the model | Consequence |
| --- | --- | --- |
| Hole in a semiconductor | its kinetic energy | the formulas hold as written |
| Electron in the conduction band | $E_g$ plus its kinetic energy | the acoustic rate is evaluated at an energy that changes little across the conducting states, so the energy dependence of the rate is lost and its value is too large by $\sqrt{(E_g + E)/E}$; the ionised-impurity rate, which falls with energy, becomes too small |
| Carrier in a metal | the distance from $E_F$ | not the kinetic energy of a parabolic band |

To use the kinetic energy, write the model as a {py:class}`~PAOFLOW.boltzmann.TauModel.TauModel`, a function of $k_BT$, $|\varepsilon_{n\mathbf{k}}|$ and a parameter dictionary. The function subtracts the conduction-band minimum from the energies of the conduction states.

## Hall coefficient

In a weak magnetic field $\mathbf{B}$, the Lorentz force adds a term of first order in $\mathbf{B}$ to the solution of the Boltzmann equation. This term involves the curvature of the bands, through the inverse mass tensor $(M^{-1})_{jq} = \hbar^{-2}\,\partial^2\varepsilon_{n\mathbf{k}}/\partial k_j\partial k_q$, and a second power of $\tau$ [4]. With `do_hall=True`, `transport()` evaluates the kernel

$$
\mathcal{L}^{H}_{ijp} = \frac{g_s}{\Omega N_\mathrm{mesh}} \sum_{n\mathbf{k}} \tau^2_{n\mathbf{k}} \sum_{qr} \epsilon_{pqr}\, v^i_{n\mathbf{k}}\, v^r_{n\mathbf{k}}\, (M^{-1})_{jq,n\mathbf{k}} \left(-\frac{\partial f^0}{\partial\varepsilon}\right)
$$

with $\epsilon_{pqr}$ the Levi-Civita symbol, and from it the Hall tensor

$$
R_{ijp} = -\frac{1}{e}\Bigl[\bigl(\mathcal{L}^{(0)}\bigr)^{-1} \mathcal{L}^{H}_{p}\, \bigl(\mathcal{L}^{(0)}\bigr)^{-1}\Bigr]_{ij} .
$$

| File | Content |
| --- | --- |
| `hall_trace` | the average $(R_{xyz} + R_{zxy} + R_{yzx})/3$ |
| `hall` | the antisymmetric parts $(R_{ijp} - R_{jip})/2$ |
| `nernst` | the Nernst tensor, from the first moment of the same kernel, with $(\varepsilon - \mu)$ |

$\tau$ cancels in $R$ for a constant relaxation time. For a single parabolic band $|R_H| = 1/(ne)$, so the Hall coefficient measures the carrier density $n$. With several bands, or an energy-dependent $\tau$, $R_H$ is a weighted average that does not equal $1/(ne)$ for any single carrier type. The band curvature comes from {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.gradient_and_momenta` with `band_curvature=True`, which must precede a call with `do_hall=True`.

## Used in

- [Band structure and density of states of silicon](../tutorials/tutorial01.ipynb): $\sigma/\tau$, $S$, $\kappa_0/\tau$ and the power factor of Si at 300 K as functions of $\mu$, at constant $\tau$ and with rigid bands.
- [Boltzmann transport in silicon](../tutorials/tutorial02.ipynb): $\sigma/\tau$, $S$ and $\kappa_0/\tau$ of Si as functions of $\mu$ at constant $\tau$, $\mu(T)$ of n-type Si at fixed carrier density, and the resistivity with acoustic-phonon and ionised-impurity relaxation times.
- [Large cells on the sparse backend](../tutorials/tutorial04.ipynb): the transport tensors of a 16-atom Si supercell on the sparse backend, at constant $\tau$.
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
