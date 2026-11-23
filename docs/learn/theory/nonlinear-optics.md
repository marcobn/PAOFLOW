# Second-order optical response: shift current, injection current and second-harmonic generation

This page derives the second-order optical responses that {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.nonlinear_optics` evaluates from the PAO Hamiltonian: the shift current and the injection currents, the direct currents that light drives in a crystal without inversion symmetry (the bulk photovoltaic effect), and the second-harmonic generation (SHG) susceptibility. All three are sums over interband transitions of the band energies, the interband position matrix elements and their k-derivatives, in the length-gauge formalism of Aversa and Sipe [1] and Sipe and Shkrebtii [2], written for tight-binding Hamiltonians by Wang et al. [3] and extended to magnetic crystals by Chen et al. [4]. The Hamiltonian is the real-space $H_{\alpha\beta}(\mathbf{R})$ of [The PAO Hamiltonian](pao-hamiltonian.md), Fourier-interpolated at every point of a uniform mesh; the velocity matrix elements are those of [Band velocities](density-of-states.md#band-velocity-and-velocity-matrix-elements). The linear counterpart of these responses is the [independent-particle dielectric tensor](linear-response.md#independent-particle-dielectric-tensor).

## Second-order response to an optical field

A monochromatic field $\mathbf{E}(t) = \mathbf{E}(\omega)e^{-i\omega t} + \mathbf{E}(-\omega)e^{i\omega t}$, with $\mathbf{E}(-\omega) = \mathbf{E}(\omega)^*$, couples to independent electrons through $e\,\mathbf{E}\cdot\hat{\mathbf{r}}$ in the length gauge [1]. At second order in the field, the current acquires a static part and the polarisation a part at $2\omega$. Each is described by a rank-3 tensor with a current or polarisation index $a$ and two field indices $b$, $c$, summed over repeated indices:

$$
J^a = \sigma^{abc}(\omega)\,E^b(\omega)E^c(-\omega) + \sigma_M^{abc}(\omega)\,\mathrm{Im}\bigl[E^b(\omega)E^c(\omega)^*\bigr] ,
$$

$$
\frac{dJ^a}{dt} = \eta_C^{abc}(\omega)\,\mathrm{Im}\bigl[E^b(\omega)E^c(\omega)^*\bigr] + \eta_M^{abc}(\omega)\,\mathrm{Re}\bigl[E^b(\omega)E^c(\omega)^*\bigr] ,
$$

$$
P^a(2\omega) = \epsilon_0\,\chi^{abc}(-2\omega;\omega,\omega)\,E^b(\omega)E^c(\omega) .
$$

The first line is the shift current $\sigma$ and its magnetic counterpart $\sigma_M$, a steady current that flows as long as the light is on. The second is the injection current, whose rate of growth is fixed by the light: the current itself grows until the carriers relax, so the steady value is $\tau\,dJ^a/dt$ with the carrier relaxation time $\tau$, which the code does not compute. The subscripts $C$ and $M$ label the circular injection, driven by the helicity $\mathrm{Im}[E^bE^{c*}]$ of the light, and the magnetic responses, which are odd under time reversal and vanish in a non-magnetic crystal (see [Time reversal](#time-reversal)). The third line defines $\chi^{abc}$ with the vacuum permittivity $\epsilon_0$.

With this field convention, $\sigma^{abb}$ is twice the shift conductivity of Sipe and Shkrebtii [2], who write the current as $2\sigma^{abb}E^b(\omega)E^b(-\omega)$. Compare spectra from the literature only after converting to one convention.

## Band geometry from the PAO Hamiltonian

The three tensors need four quantities at every k-point: the transition energies, the interband position matrix elements, their k-derivatives and the velocity differences between bands. In this section $\hbar = 1$, so energies and frequencies are the same and velocities are in eV bohr; the response sections restore $\hbar$.

### Velocity and position matrix elements

The velocity operator is $v^a = \partial_a \tilde H + i[\tilde H, \tau_a]$, with $\partial_a = \partial/\partial k_a$ and $\tau_a$ the diagonal matrix of orbital centres, as derived in [The gradient of the Hamiltonian](density-of-states.md#the-gradient-of-the-hamiltonian). Between the eigenstates $|n\rangle$ of $\tilde H(\mathbf{k})$, with $\omega_{nm} = \varepsilon_{n\mathbf{k}} - \varepsilon_{m\mathbf{k}}$, the relation $v = i[\tilde H, r]$ gives the interband position matrix elements

$$
r^a_{nm} = \frac{-i\,v^a_{nm}}{\omega_{nm}} , \qquad n, m \text{ in different degenerate groups,}
$$

and the band velocities enter through their differences $\Delta^a_{nm} = v^a_{nn} - v^a_{mm} = \partial_a\omega_{nm}$. The diagonal of the position operator, the Berry connection $A^a_n = i\langle u_n|\partial_a u_n\rangle$, depends on the phase of each eigenvector; it enters the responses only through the covariant derivative below, so no smooth gauge is needed. The position operator implied by $v^a$ is diagonal on the orbital centres, the tight-binding approximation of [Position dipoles](#position-dipoles).

### Generalized derivative and the sum rule

The shift current and the SHG need the k-derivative of $r^b_{nm}$. The covariant, or generalized, derivative

$$
r^b_{nm;a} = \partial_a r^b_{nm} - i\left(A^a_n - A^a_m\right) r^b_{nm}
$$

changes with the phases of $|n\rangle$ and $|m\rangle$ in the same way as $r^b_{nm}$ itself, so products such as $r^b_{mn}r^c_{nm;a}$ are gauge invariant [1, 2]. PAOFLOW does not take it by finite differences, which would need a smooth gauge, but evaluates it from a sum rule. Write $\tilde H^A$ for the Hamiltonian in the basis whose Bloch phases use the orbital centres, $\tilde H^A_{\alpha\beta}(\mathbf{k}) = e^{i\mathbf{k}\cdot(\boldsymbol{\tau}_\beta - \boldsymbol{\tau}_\alpha)}\tilde H_{\alpha\beta}(\mathbf{k})$. Its first derivative has the band matrix elements $v^a_{nm}$, and its second derivative those of

$$
w^{ab} = \partial_a\partial_b \tilde H + i[\partial_a \tilde H, \tau_b] + i[\partial_b \tilde H, \tau_a] - \bigl[[\tilde H, \tau_a], \tau_b\bigr] .
$$

Differentiating $v^b_{nm} = i\omega_{nm} r^b_{nm}$ covariantly gives $(v^b_{nm})_{;a} = i\Delta^a_{nm} r^b_{nm} + i\omega_{nm} r^b_{nm;a}$. On the other side, $(v^b_{nm})_{;a}$ is $w^{ab}_{nm}$ plus the terms with $|\partial_a n\rangle$ and $|\partial_a m\rangle$, whose components on the other bands are $\langle p|\partial_a m\rangle = v^a_{pm}/(\varepsilon_m - \varepsilon_p)$, as in [Berry curvature from the Kubo formula](berry-topology.md#berry-curvature-from-the-kubo-formula); their components along $|n\rangle$ and $|m\rangle$ are the connections that the covariant derivative removes. Solving for $r^b_{nm;a}$,

$$
r^b_{nm;a} = \frac{i}{\omega_{nm}}\left[\frac{v^b_{nm}\Delta^a_{nm}}{\omega_{nm}} - w^{ab}_{nm} + \sum_p\left(\frac{v^b_{np}v^a_{pm}}{\omega_{pm}} - \frac{v^a_{np}v^b_{pm}}{\omega_{np}}\right)\right] ,
$$

where the sum runs over all $n_\mathrm{awf}$ bands, $p = n$ and $p = m$ included, and a term is left out when its denominator joins two bands of one degenerate group. The terms $p = n$ and $p = m$ add $v^a_{nm}\Delta^b_{nm}/\omega_{nm}$, so the bracket contains both velocity differences, as in the plane-wave sum rule [2].

The term $w^{ab}_{nm}$ distinguishes the tight-binding sum rule from the plane-wave one [3]: with plane waves $\partial_a\partial_b H = \delta_{ab}/m_e$ is diagonal and drops out between different bands, while in a basis of localised orbitals it does not. The sum over $p$ is complete within the PAO basis, so the rule is exact for the model and needs no convergence in the number of bands, unlike its plane-wave form; what it does not include is discussed under [Approximations and their limits](#approximations-and-their-limits).

### Shift vector, Berry curvature and quantum metric

Writing $r^b_{nm} = |r^b_{nm}|\,e^{-i\phi^b_{nm}}$, the definition of the covariant derivative gives

$$
r^b_{mn}\,r^b_{nm;a} = |r^b_{nm}|\,\partial_a|r^b_{nm}| - i\,|r^b_{nm}|^2 R^{a,b}_{nm} , \qquad
R^{a,b}_{nm} = \partial_a\phi^b_{nm} + A^a_n - A^a_m .
$$

$R^{a,b}_{nm}$ is the shift vector [2], a length: the difference between the intracell positions (Berry connections) of the two bands, made gauge invariant by the phase gradient of the transition dipole. It is the displacement of the electron's centre of charge when light polarised along $b$ moves it from band $m$ to band $n$.

The injection current and the SHG also contain the two-band products

$$
\Omega^{bc}_{mn} = -2\,\mathrm{Im}\bigl[r^b_{mn}r^c_{nm}\bigr] , \qquad
g^{bc}_{mn} = \mathrm{Re}\bigl[r^b_{mn}r^c_{nm}\bigr] ,
$$

the contribution of the pair to the Berry curvature, $\Omega^{bc}_m = \sum_{n \ne m}\Omega^{bc}_{mn}$, and to the quantum metric of band $m$. The first sum is the [Kubo form of the Berry curvature](berry-topology.md#berry-curvature-from-the-kubo-formula).

### Degenerate bands

At a degenerate level, $r^a_{nm}$ is undefined between the degenerate states and $v^a_{nn}$ depends on the basis chosen inside the degenerate subspace, as for the [Berry curvature](berry-topology.md#degenerate-bands). PAOFLOW sorts the bands at each k-point into groups, in which consecutive bands are closer than `degeneracy_tol` (0.01 eV by default), excludes the pairs inside a group from $r$ and from every band sum, and replaces the band velocities by the block $v^a_\mathrm{intra}$ of the velocity matrix inside each group: $v^b_{nm}\Delta^a_{nm}$ becomes $[v^a_\mathrm{intra}, v^b]_{nm}$ and $r^b_{nm}\Delta^a_{nm}$ becomes $[v^a_\mathrm{intra}, r^b]_{nm}$. With these replacements every quantity is unchanged by a rotation of the states inside a group, and for groups of one band they reduce to the expressions above. The default tolerance is large compared with the exact degeneracies of a model, because DFT-derived Hamiltonians have nearly degenerate pairs whose meV-level asymmetries the factors $1/\omega_{nm}$ amplify into components that symmetry forbids. A transition between two bands of one group is not counted, so features below `degeneracy_tol` in photon energy are lost.

### Position dipoles

The position operator diagonal on the orbital centres omits the dipoles between different orbitals of one atom, such as $\langle s|\mathbf{r}|p\rangle$, which are of the order of the orbital radius. With `position_dipoles=True` the position operator becomes $\boldsymbol{\tau} + \mathbf{d}(\mathbf{k})$, with $\mathbf{d}(\mathbf{k})$ the dipoles of the Löwdin-orthogonalised PAOs. In the band basis $r$ becomes $r + D$ between groups, with $D = U^\dagger d\,U$, and $r_{;a}$ gains the derivative of $D$ and its products with $r$; the expressions and the evaluation of $\mathbf{d}(\mathbf{k})$ are in the developer notes on [non-linear optics](../../internals/nonlinear-optics.md) and [position dipoles](../../internals/position-dipoles.md). The dipoles change only the interband matrix elements, not the band energies or velocities, and their effect on the responses is the same in kind as on the [dielectric tensor](linear-response.md#approximations-of-the-dielectric-tensor). They need the radial functions of the PAO basis and a scalar-relativistic run, so they are not available for models or with spin–orbit coupling. Wannier-function calculations that keep the full position matrix of their orbitals [7] go beyond the orbital-centre approximation in the same way.

## Shift current

The shift current is the second-order current at zero frequency produced by real transitions. The code evaluates, with $\omega_{nm} = (\varepsilon_{n\mathbf{k}} - \varepsilon_{m\mathbf{k}})/\hbar$, $f_{nm} = f_n - f_m$ and the k-integral $\int[d\mathbf{k}] = \int_\mathrm{BZ} d^3k/(2\pi)^3 \to (N_\mathrm{mesh}V)^{-1}\sum_\mathbf{k}$ over the mesh and the cell volume $V$,

$$
\sigma^{abc}(\omega) = \frac{\pi e^3 g_s}{\hbar^2}\int[d\mathbf{k}]\sum_{nm} f_{nm}\,\mathrm{Im}\bigl[r^b_{mn}r^c_{nm;a} + r^c_{mn}r^b_{nm;a}\bigr]\,\delta(\omega_{nm} - \omega) ,
$$

with the spin factor $g_s$ of [Occupations and spin](#occupations-and-spin). For $b = c$ the identity of the previous section turns it into

$$
\sigma^{abb}(\omega) = \frac{2\pi e^3 g_s}{\hbar^2}\int[d\mathbf{k}]\sum_{m\,\mathrm{occ}}\sum_{n\,\mathrm{empty}} |r^b_{nm}|^2\,R^{a,b}_{nm}\,\delta(\omega_{nm} - \omega)
$$

at zero temperature. The physical picture follows: $|r^b_{nm}|^2\delta(\omega_{nm}-\omega)$ is the rate of transitions from Fermi's golden rule, and each transition displaces the charge by the shift vector, so the current is the rate times the displacement. A shift current therefore needs transitions whose initial and final states have different centres of charge, which inversion symmetry forbids.

The magnetic shift current,

$$
\sigma_M^{abc}(\omega) = -\frac{\pi e^3 g_s}{\hbar^2}\int[d\mathbf{k}]\sum_{nm} f_{nm}\,\mathrm{Re}\bigl[r^b_{mn}r^c_{nm;a} - r^c_{mn}r^b_{nm;a}\bigr]\,\delta(\omega_{nm} - \omega) ,
$$

is driven by circularly polarised light and exists only when time reversal is broken, in a magnetic crystal without inversion symmetry [4].

## Injection current

The injection current comes from the asymmetric population that the transitions create. Each transition adds an electron with velocity $v^a_{nn}$ and removes one with $v^a_{mm}$, so the current grows at the transition rate times the velocity difference. With $f_{mn} = f_m - f_n$ and $\Delta^a_{mn} = v^a_{mm} - v^a_{nn}$, the code evaluates [4]

$$
\eta_C^{abc}(\omega) = \frac{2\pi e^3 g_s}{\hbar^2}\int[d\mathbf{k}]\sum_{nm} f_{mn}\,\Delta^a_{mn}\,\mathrm{Im}\bigl[r^b_{mn}r^c_{nm}\bigr]\,\delta(\omega_{nm} - \omega)
= -\frac{\pi e^3 g_s}{\hbar^2}\int[d\mathbf{k}]\sum_{nm} f_{mn}\,\Delta^a_{mn}\,\Omega^{bc}_{mn}\,\delta(\omega_{nm} - \omega) ,
$$

$$
\eta_M^{abc}(\omega) = -\frac{2\pi e^3 g_s}{\hbar^2}\int[d\mathbf{k}]\sum_{nm} f_{mn}\,\Delta^a_{mn}\,g^{bc}_{mn}\,\delta(\omega_{nm} - \omega) .
$$

In both, a pair $(n, m)$ with $\omega_{nm} > 0$ contributes when $m$ is occupied and $n$ empty. The circular injection is weighted by the Berry curvature of the pair: circularly polarised light excites the states at $\mathbf{k}$ and $-\mathbf{k}$ at different rates when $\Omega^{bc}_{mn}$ is nonzero, as in the valley-selective absorption of [gapped graphene](berry-topology.md#gapped-graphene-and-the-valley-berry-phase). The magnetic injection is weighted by the quantum metric and is driven by linearly polarised light; it needs broken time-reversal symmetry, because otherwise the populations at $\mathbf{k}$ and $-\mathbf{k}$ are equal and carry opposite velocities.

## Second-harmonic generation

The SHG susceptibility has, besides the one-photon resonances at $\hbar\omega = \varepsilon_n - \varepsilon_m$, two-photon resonances at $2\hbar\omega = \varepsilon_n - \varepsilon_m$, and terms that involve three bands. PAOFLOW evaluates the form of Rashkeev et al. [5] and Wang et al. [3], with the time-odd term of Chen et al. [4], in which no term diverges as $\omega \to 0$ in an insulator. It splits $\chi$ into a three-band (interband) part and a two-band part that contains the generalized derivatives,

$$
\chi^{abc} = \frac{e^3 g_s}{2\epsilon_0\hbar^2}\int[d\mathbf{k}]\left(\chi_e^{abc} + \chi_i^{abc}\right) ,
$$

$$
\chi_e^{abc} = \sum_{n,m,l}\frac{r^a_{nm}\left(r^b_{ml}r^c_{ln} + r^c_{ml}r^b_{ln}\right)}{\omega_{ln} - \omega_{ml}}
\left[\frac{2f_{nm}}{\omega_{mn} - 2\tilde\omega} + \frac{f_{ln}}{\omega_{ln} - \tilde\omega} + \frac{f_{ml}}{\omega_{ml} - \tilde\omega}\right] ,
$$

$$
\begin{aligned}
\chi_i^{abc} = i\sum_{nm} f_{nm}\Biggl\{&\frac{1}{\omega_{mn} - 2\tilde\omega}\left[\frac{2r^a_{nm}\left(r^b_{mn;c} + r^c_{mn;b}\right)}{\omega_{mn}} - \frac{4r^a_{nm}\left(r^b_{mn}\Delta^c_{mn} + r^c_{mn}\Delta^b_{mn}\right)}{\omega_{mn}^2}\right] \\
+\;&\frac{1}{\omega_{mn} - \tilde\omega}\left[\frac{r^a_{nm;c}r^b_{mn} + r^a_{nm;b}r^c_{mn}}{\omega_{mn}} + \frac{r^a_{nm}\left(r^b_{mn}\Delta^c_{mn} + r^c_{mn}\Delta^b_{mn}\right)}{\omega_{mn}^2} \right. \\
&\left. -\,\frac{r^b_{mn}r^c_{nm;a} + r^c_{mn}r^b_{nm;a}}{2\omega_{mn}} - \frac{\Delta^a_{mn}\,g^{bc}_{nm}}{2\omega_{mn}^2}\right]\Biggr\} ,
\end{aligned}
$$

where $n$, $m$, $l$ in $\chi_e$ are three different bands, and $\omega_{mn} - c\tilde\omega$ stands for $\omega_{mn} - c\,\omega - i\eta/\hbar$ with $c = 1, 2$ and the broadening $\eta$ of [Broadening](#broadening). The prefactor holds the factor $1/2$ that the symmetrised products $\{r^br^c\} = (r^br^c + r^cr^b)/2$ carry in Refs. [3, 5]. The last term of $\chi_i$, with the quantum metric $g^{bc}_{nm}$, is the time-odd term [4]; it vanishes in a non-magnetic crystal. Terms of $\chi_e$ whose denominator $\omega_{ln} - \omega_{ml}$ is smaller than `degeneracy_tol` are left out.

The one-photon resonances of $\chi$ are those of the linear absorption, while the two-photon resonances bring the same transitions to half the photon energy. A spectrum of $|\chi|$ therefore shows the absorption edge twice, at half the gap and at the gap. Below half the gap $\chi$ is nearly real and tends to the static value $\chi(0)$, at which the tensor becomes symmetric under all permutations of $a$, $b$, $c$ (Kleinman symmetry), a useful check of a computed static tensor.

## Symmetry of the second-order tensors

Symmetry fixes which components of the tensors can be nonzero, and these relations are the first test of a calculation.

### Point groups and inversion

A rank-3 tensor $T^{abc}$ that relates polar vectors, as all five here do, is invariant under every operation $R$ of the point group of the crystal (Neumann's principle):

$$
T^{abc} = \sum_{a'b'c'} R_{aa'}R_{bb'}R_{cc'}\,T^{a'b'c'} .
$$

For inversion, $R = -1$, the right side is $-T^{abc}$, so every second-order response vanishes in a centrosymmetric crystal. In the other point groups the law leaves a few independent components: in the zincblende group $T_d$ only $T^{xyz}$ and its permutations, in $D_{3h}$ with a mirror perpendicular to $x$ the relation $T^{yyy} = -T^{yxx} = -T^{xxy} = -T^{xyx}$ with $T^{xxx} = 0$.

### Time reversal

In a crystal with time-reversal symmetry the band quantities at $\mathbf{k}$ and $-\mathbf{k}$ are related by complex conjugation (with spin–orbit coupling, band $n$ at $-\mathbf{k}$ is the Kramers partner of band $n$ at $\mathbf{k}$): $r^b_{nm}(-\mathbf{k}) = r^b_{nm}(\mathbf{k})^*$, $r^b_{nm;a}(-\mathbf{k}) = -r^b_{nm;a}(\mathbf{k})^*$ and $\Delta^a_{nm}(-\mathbf{k}) = -\Delta^a_{nm}(\mathbf{k})$. Inserting these, the integrands of $\sigma$ and $\eta_C$ are even in $\mathbf{k}$, while those of $\sigma_M$, $\eta_M$ and the last term of $\chi_i$ are odd and cancel between $\mathbf{k}$ and $-\mathbf{k}$. The magnetic responses are therefore zero in a non-magnetic crystal, and a nonzero value there measures the numerical error.

### Permutation of the field indices

The formulas fix the symmetry in the two field indices: $\sigma^{abc}$, $\eta_M^{abc}$ and $\chi^{abc}$ are symmetric in $b, c$, while $\sigma_M^{abc}$ and $\eta_C^{abc}$ are antisymmetric, so $\eta_C^{abb} = 0$. The antisymmetric $\eta_C^{abc}$ is equivalent to a rank-2 tensor that relates the current to the axial vector $\mathbf{E}\times\mathbf{E}^*$ of the light, and such a tensor can be nonzero only in the gyrotropic point groups, those that also allow natural optical activity. $D_{3h}$ and $T_d$ are not gyrotropic, so the circular injection current vanishes in them.

### Symmetry reduction of the k-mesh

With `symmetry=True` (the default) and the point-group operations of the Quantum ESPRESSO run, PAOFLOW sums only over the irreducible wedge of the mesh, with each point weighted by the size of its orbit, and averages the result over the group, $\mathrm{Sym}_G\,T = |G|^{-1}\sum_{R\in G} R\otimes R\otimes R\;T$. The average is linear and each integrand is covariant, so the result equals the sum over the full mesh. The operations combined with time reversal are not used. A tensor computed this way obeys the point-group relations exactly, so the relations test the calculation only with `symmetry=False`, or for models, which carry no symmetry operations and are summed over the full mesh.

## Broadening, occupations and units

### Broadening

On a discrete mesh the $\delta$ functions of the shift and injection currents are replaced by functions of width $\eta$, the argument `eta`: by default the Gaussian $e^{-x^2/\eta^2}/(\sqrt{\pi}\,\eta)$, whose standard deviation is $\eta/\sqrt{2}$ as in [Fixed-width smearing](density-of-states.md#fixed-width-smearing), or with `smearing='lorentz'` the Lorentzian $(\eta/\pi)/(x^2 + \eta^2)$. With `adaptive_smearing=a` each pair gets the width $a\,|\mathbf{v}_n - \mathbf{v}_m|\,\delta k$, the [adaptive width](density-of-states.md#width-from-the-band-velocity) of the transition energy, with $\delta k$ the mesh spacing, floored at $\eta$.

In the SHG, $\eta$ enters as the imaginary part $-i\eta$ of every resonant denominator, one-photon and two-photon alike. The textbook replacement $\omega \to \omega + i\eta$ would give $-2i\eta$ in the two-photon denominators, so the two-photon resonances here are half as broad as in that convention. The weights of the resonances are collected on a grid of transition energies of spacing `shg_bin`, $\eta/5$ by default, before the denominators are applied ([Non-linear optics](../../internals/nonlinear-optics.md)); a bin small compared with $\eta$ leaves the spectrum unchanged. Physically, $\eta$ stands for the dephasing of the excited states, constant over energy and temperature, and it must be larger than the spacing of the transition energies on the mesh, or the spectra oscillate with the mesh.

### Occupations and spin

At zero temperature the occupations $f_n$ are a step: the lowest $N_\mathrm{el}/g_s$ bands are filled at each k-point, with $N_\mathrm{el}$ the electron count `nelec`, or the bands below `fermi_energy` when it is given. At a finite `temperature` they are Fermi–Dirac functions at `fermi_energy`, by default the middle of the gap. The spin factor $g_s$ is 2 for a non-magnetic calculation without spin–orbit coupling and 1 otherwise; with two collinear spin channels the responses of each are computed with $g_s = 1$ and summed.

### Bulk and sheet units

The code works in eV and bohr and converts once at the end: the shift currents to µA/V$^2$, the injection coefficients to A V$^{-2}$ s$^{-1}$ and $\chi$ to pm/V. For a two-dimensional crystal in a cell with vacuum, the factor $1/V$ of $\int[d\mathbf{k}]$ makes these bulk values inversely proportional to the height $L$ of the cell, which is not a property of the layer. With `twoD=True` they are multiplied by $L$ (`slab_height`, by default $|\mathbf{a}_3|$), which gives the sheet responses per unit area, in nm µA/V$^2$, nm A V$^{-2}$ s$^{-1}$ and pm$^2$/V. A bulk value for a layer of thickness $t$ is the sheet value divided by $t$.

## Approximations and their limits

The responses are those of independent electrons in the bands of the PAO Hamiltonian, and each approximation fails on a known scale.

- Independent particles, no excitons. The electron and hole of a transition do not interact, so excitonic resonances below the gap and the transfer of strength to the absorption edge are missing, as for the [dielectric tensor](linear-response.md#approximations-of-the-dielectric-tensor). The error is largest in two-dimensional and wide-gap materials, where exciton binding energies reach tenths of an eV.
- No local fields. The field acting on the electrons is the macroscopic one.
- Kohn–Sham energies. With PBE the transition energies are too low; the [scissor shift](#scissor-shift) corrects them rigidly.
- Position operator on the orbital centres, unless `position_dipoles` adds the intra-atomic dipoles (see [Position dipoles](#position-dipoles)). The error grows with the weight of transitions between orbitals of one atom, such as the $s$–$p$ transitions across the gap of a III–V semiconductor.
- Finite band set. The sums over intermediate bands, in the sum rule and in $\chi_e$, run over the $n_\mathrm{awf}$ PAO bands. Bands above the [PAO shift](pao-hamiltonian.md#the-pao-shift) are placeholders, so transitions into them are unreliable, and virtual transitions through them make the three-band terms approximate.
- Interband terms only. Pairs with equal occupation drop out, so the Fermi-surface contributions of a metal or a doped semiconductor, such as the Berry curvature dipole, are not included; the results apply to insulators and semiconductors.
- Long wavelength, clamped ions. The light carries no momentum, and the ions do not move.
- Constant broadening $\eta$ for all transitions, and no carrier relaxation time in the injection coefficients.

### Scissor shift

The argument `scissor` adds a constant $\Delta_s$ to the energies of the conduction bands, those above the $N_\mathrm{el}/g_s$ occupied ones, in every transition energy: in the arguments of the $\delta$ functions, in the resonant denominators and in the explicit factors $1/\omega_{mn}$ of $\chi$. The position matrix elements $r$, their derivatives and the velocity differences keep the values of the unshifted Hamiltonian. This is the scissor implementation of Nastos et al. [6], in which the position matrix elements do not change under the scissor while the velocity matrix elements are rescaled. The shift and injection spectra then move rigidly by $\Delta_s$, while $\chi$ also changes in magnitude through the factors $1/\omega_{mn}$. The scissor assumes a correction that is the same for all conduction states, so it is defined only when a gap separates the occupied bands from the others.

## Used in

- [How to compute the non-linear optical response of a tight-binding model](../howtos/nonlinear-optics-model.md): the shift current, injection currents and SHG of gapped graphene, and the symmetry relations that check them.

## References

1. C. Aversa and J. E. Sipe, "Nonlinear optical susceptibilities of semiconductors: Results with a length-gauge analysis", Phys. Rev. B 52, 14636 (1995).
2. J. E. Sipe and A. I. Shkrebtii, "Second-order optical response in semiconductors", Phys. Rev. B 61, 5337 (2000).
3. C. Wang, X. Liu, L. Kang, B.-L. Gu, Y. Xu and W. Duan, "First-principles calculation of nonlinear optical responses by Wannier interpolation", Phys. Rev. B 96, 115147 (2017).
4. H. Chen, M. Ye, N. Zou, B.-L. Gu, Y. Xu and W. Duan, "Basic formulation and first-principles implementation of nonlinear magneto-optical effects", Phys. Rev. B 105, 075123 (2022).
5. S. N. Rashkeev, W. R. L. Lambrecht and B. Segall, "Efficient ab initio method for the calculation of frequency-dependent second-order optical response in semiconductors", Phys. Rev. B 57, 3905 (1998).
6. F. Nastos, B. Olejnik, K. Schwarz and J. E. Sipe, "Scissors implementation within length-gauge formulations of the frequency-dependent nonlinear optical response of semiconductors", Phys. Rev. B 72, 045223 (2005).
7. J. Ibañez-Azpiroz, S. S. Tsirkin and I. Souza, "Ab initio calculation of the shift photocurrent by Wannier interpolation", Phys. Rev. B 97, 245143 (2018).
