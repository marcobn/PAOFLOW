# Berry curvature and topological invariants

This page derives the geometric quantities of a band structure that PAOFLOW computes. They are the Berry phase of a set of bands on a closed loop in k-space, the Berry curvature, the Chern number, and the anomalous and spin Hall conductivities that the curvature gives in the clean limit. Three methods evaluate them:

| Method | Quantity | Evaluated from |
| --- | --- | --- |
| {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.berry_phase` | Berry phase on a loop, fluxes through plaquettes | overlaps of Bloch states |
| {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.anomalous_Hall` | Berry curvature, anomalous Hall conductivity | a sum over states, integrated over the Brillouin zone |
| {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.spin_Hall` | spin Berry curvature, spin Hall conductivity | a sum over states, integrated over the Brillouin zone |

All of them start from the Hamiltonian $\tilde H(\mathbf{k})$ of [The PAO Hamiltonian](pao-hamiltonian.md). The Hall conductivities also use the velocity matrices and the adaptive smearing of [Band velocities, density of states and adaptive smearing](density-of-states.md). [Kubo linear response](linear-response.md) derives the same conductivities from the Kubo formula, with a finite broadening of the states. At the end of the page, the gapped graphene, Haldane and Kane–Mele models show each quantity on a case with a known answer.

## Berry connection and Berry phase

The Berry phase is the phase that a Bloch state acquires, beyond its dynamical phase, when $\mathbf{k}$ is carried slowly around a closed loop $\mathcal{C}$ [1]. Write $\psi_{n\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}} u_{n\mathbf{k}}(\mathbf{r})$, with $u_{n\mathbf{k}}$ periodic in the cell. All $u_{n\mathbf{k}}$ live in the same space of cell-periodic functions, so their change with $\mathbf{k}$ is well defined. That change gives the Berry connection $\mathbf{A}_n$, and the circulation of $\mathbf{A}_n$ around the loop is the Berry phase $\gamma_n$:

$$
\mathbf{A}_n(\mathbf{k}) = i \langle u_{n\mathbf{k}} | \nabla_{\mathbf{k}} u_{n\mathbf{k}} \rangle ,
\qquad
\gamma_n(\mathcal{C}) = \oint_{\mathcal{C}} \mathbf{A}_n \cdot d\mathbf{k} .
$$

Each state carries an arbitrary phase. A change of that phase, $u_{n\mathbf{k}} \to e^{i\chi(\mathbf{k})} u_{n\mathbf{k}}$, shifts $\mathbf{A}_n$ by $-\nabla_{\mathbf{k}}\chi$. It shifts $\gamma_n$ only by an integer multiple of $2\pi$, because $e^{i\chi}$ must return to its value at the end of the loop. The phase $\gamma_n$ is therefore a property of the bands, defined modulo $2\pi$, while $\mathbf{A}_n$ is not. By Stokes' theorem, $\gamma_n$ is the flux through the loop of the Berry curvature

$$
\Omega_n(\mathbf{k}) = \nabla_{\mathbf{k}} \times \mathbf{A}_n(\mathbf{k}),
\qquad
\Omega^{xy}_n = \partial_{k_x} A_{n,y} - \partial_{k_y} A_{n,x} ,
$$

which is gauge invariant at every $\mathbf{k}$. In a two-dimensional crystal only $\Omega^{xy}_n$ enters, and this page drops the superscript where the plane is clear. The curvature acts as a magnetic field in k-space and bends the trajectory of a wave packet [3].

Bands that touch on the loop are followed as a set. The connection of $N_b$ bands is an $N_b \times N_b$ matrix, $A_{mn} = i\langle u_{m\mathbf{k}}|\nabla_{\mathbf{k}} u_{n\mathbf{k}}\rangle$, and its gauge freedom is a $\mathbf{k}$-dependent unitary mixing of the set. The Wilson loop is the path-ordered exponential of $A$ around the loop. The phase of its determinant survives the mixing, and it is the Berry phase of the set.

## The discrete Berry phase on a loop

A computed eigenvector has a random phase at every $\mathbf{k}$, so $\nabla_{\mathbf{k}} u$ cannot be taken by finite differences. Instead, the loop is cut into $N$ points, with $\mathbf{k}_{N+1} = \mathbf{k}_1$, and the phase is collected from the overlaps of neighbouring points. For a small step, $\langle u_{\mathbf{k}}|u_{\mathbf{k}+\Delta\mathbf{k}}\rangle \approx 1 + \langle u_{\mathbf{k}}|\nabla u_{\mathbf{k}}\rangle \cdot \Delta\mathbf{k} \approx e^{-i\mathbf{A}\cdot\Delta\mathbf{k}}$, so the product of the overlaps around the loop has the phase $-\gamma$ [2]. For several bands each overlap is a matrix, and the phase is that of the determinant of their product:

$$
\gamma = -\arg \det \prod_{i=1}^{N} M(\mathbf{k}_i, \mathbf{k}_{i+1}) ,
\qquad
M_{mn}(\mathbf{k}_i, \mathbf{k}_{i+1}) = \langle u_{m\mathbf{k}_i} | u_{n\mathbf{k}_{i+1}} \rangle ,
$$

with $m$ and $n$ running over the selected bands: those listed in `sub`, or the occupied bands. Each state appears once as a bra and once as a ket, so its arbitrary phase cancels. A unitary mixing inside the set cancels in the determinant. The only error left is that of replacing the loop by a polygon, and it vanishes as the overlaps of neighbouring points approach unitary matrices.

### Overlaps of the orbital coefficients

`berry_phase()` diagonalises $\tilde H(\mathbf{k}) = \sum_{\mathbf{R}} H(\mathbf{R})\, e^{i\mathbf{k}\cdot\mathbf{R}}$ in the orthonormal PAO basis at each point of the loop. The overlaps are those of the eigenvectors,

$$
M_{mn}(\mathbf{k}_i, \mathbf{k}_{i+1}) = \sum_\alpha c^*_{\alpha m}(\mathbf{k}_i)\, c_{\alpha n}(\mathbf{k}_{i+1}) ,
$$

where $c_{\alpha n}(\mathbf{k})$ is the coefficient of orbital $\alpha$ in band $n$. Each overlap matrix is replaced by its unitary part, obtained from a singular-value decomposition. This replacement leaves the phase of the determinant unchanged.

This overlap differs from the textbook overlap of cell-periodic functions. With orthonormal orbitals at positions $\boldsymbol{\tau}_\alpha$ in the cell, $u_{n\mathbf{k}}$ carries the factor $e^{-i\mathbf{k}\cdot\boldsymbol{\tau}_\alpha}$ on orbital $\alpha$. The textbook overlap therefore has the extra factor $e^{-i(\mathbf{k}_{i+1} - \mathbf{k}_i)\cdot\boldsymbol{\tau}_\alpha}$ in each term. The two connections differ by $\sum_\alpha |c_{\alpha n}(\mathbf{k})|^2 \boldsymbol{\tau}_\alpha$, the orbital position averaged with the weights of band $n$.

This difference is periodic in $\mathbf{k}$, so its curl integrates to zero over the zone, and the Chern numbers of the two forms agree. The phase on a loop inside the zone, and the map of plaquette phases below, differ by the circulation of this term around the loop. That circulation vanishes when all orbitals sit on one site. It also vanishes to leading order on a circle centred on an isotropic valley, where the weights depend only on the distance from the valley.

### Loops, plaquettes and Wilson loops

The argument `kspace_method` selects the loop:

| `kspace_method` | Loop | Result |
| --- | --- | --- |
| `'circle'` | a circle of radius `kradius` around `kcenter` in crystal coordinates, which is an ellipse in Cartesian coordinates on a non-square lattice | the phase of the loop |
| `'square'` | the rectangle `kxlim` × `kylim`, divided into plaquettes | the phase of each plaquette |
| `'track'` | a straight path across the zone for each of several transverse points | the eigenphases on each path, with `eigvals=True` |

The plaquettes of `'square'` avoid the $2\pi$ ambiguity of one large loop. Their sum is the flux through the rectangle, provided that each plaquette encloses a phase well below $\pi$. A plaquette near $\pi$ has an ambiguous branch and can shift the sum by $2\pi$. Over one reciprocal cell the sum is the lattice Chern number of Fukui, Hatsugai and Suzuki, an integer on any mesh that meets this condition [5].

A straight path from $\mathbf{k}$ to $\mathbf{k} + \mathbf{G}$ is also a closed loop. Its phase is the Zak phase [13], which measures the centre of charge of the bands along that direction [14]. Because $u_{n,\mathbf{k}+\mathbf{G}} = e^{-i\mathbf{G}\cdot\mathbf{r}} u_{n\mathbf{k}}$, the link that closes the path must carry that factor. With `method='zak'` it carries $e^{-i\mathbf{G}\cdot\boldsymbol{\tau}_\alpha}$ on each orbital, the textbook form of that link.

With `eigvals=True` the eigenphases of the product matrix replace its determinant. They give the centres of the hybrid Wannier functions. `kspace_method='track'` follows them across the zone, and their net winding is the Chern number of the set [3].

## Berry curvature from the Kubo formula

The flux through a loop gives the curvature only as an average over the loop. At a single $\mathbf{k}$, the curvature follows from first-order perturbation theory. Write $\Omega^{xy}_n = i\left[\langle \partial_x u_n | \partial_y u_n \rangle - \langle \partial_y u_n | \partial_x u_n \rangle\right]$ and insert $\sum_m |u_m\rangle\langle u_m|$. The terms $m = n$ give $A_{n,x}A_{n,y} - A_{n,y}A_{n,x} = 0$. For $m \ne n$, differentiate $\tilde H|u_n\rangle = \varepsilon_n|u_n\rangle$ and project on $\langle u_m|$. This gives $\langle u_m|\partial_i u_n\rangle = v^i_{mn}/(\varepsilon_n - \varepsilon_m)$, with the [velocity matrix](density-of-states.md#band-velocity-and-velocity-matrix-elements) $v^i_{mn}(\mathbf{k}) = \langle u_{m\mathbf{k}}|\partial \tilde H/\partial k_i|u_{n\mathbf{k}}\rangle$ in units with $\hbar = 1$. Inserting it in the sum gives [3]

$$
\Omega^{ij}_n(\mathbf{k}) = i \sum_{m \ne n} \frac{v^i_{nm} v^j_{mn} - v^j_{nm} v^i_{mn}}{(\varepsilon_{n\mathbf{k}} - \varepsilon_{m\mathbf{k}})^2}
= -2 \sum_{m \ne n} \frac{\mathrm{Im}\bigl[v^i_{nm}(\mathbf{k})\, v^j_{mn}(\mathbf{k})\bigr]}{(\varepsilon_{n\mathbf{k}} - \varepsilon_{m\mathbf{k}})^2} ,
$$

which needs no derivative of the states and no choice of gauge. `anomalous_Hall()` adds a broadening $\delta$ to the denominator (argument `delta`, default 0.05 eV):

$$
\Omega^{ij}_n(\mathbf{k}) = -2 \sum_{m \ne n} \frac{\mathrm{Im}\bigl[v^i_{nm}(\mathbf{k})\, v^j_{mn}(\mathbf{k})\bigr]}{(\varepsilon_{n\mathbf{k}} - \varepsilon_{m\mathbf{k}})^2 + \delta^2} ,
$$

with $m$ over all $n_\mathrm{awf}$ bands. Pairs whose denominator is below $10^{-4}$ eV$^2$ are left out, which matters only for $\delta < 0.01$ eV. With $\partial\tilde H/\partial k$ in eV bohr, $\Omega_n$ is in bohr$^2$. The broadening is not part of the textbook form. It keeps the sum finite where two bands nearly touch, and in exchange it reduces the term of a pair separated by $\Delta\varepsilon$ by the factor $\Delta\varepsilon^2/(\Delta\varepsilon^2 + \delta^2)$.

A band acquires large curvature where another band comes close and the two are coupled by both velocity components. This happens at an avoided crossing opened by spin–orbit coupling, or at a Dirac point with a small mass. The peak extends over a region whose size is the splitting divided by the band velocity.

The terms of a pair $(n, m)$ in $\Omega_n$ and $\Omega_m$ are equal and opposite. The curvature summed over all bands therefore vanishes at every $\mathbf{k}$, and only a partly filled set of bands carries a net curvature.

{py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.topology` with `Berry=True` evaluates the band-summed curvature on the band path, with $\delta = 0.05$ eV and a step occupation at $E = 0$. Its sum carries the opposite sign, $+2\,\mathrm{Im}$. Its file for the indices $ij$ therefore holds $\Omega^{ji} = -\Omega^{ij}$ of the equation above.

### Degenerate bands

A degenerate band has no unique eigenvector. An example is one band of a Kramers pair in a crystal with time-reversal and inversion symmetry. Its individual $\Omega_n$ therefore depends on how the degenerate states are mixed, and only the sum over the degenerate subspace is gauge invariant.

PAOFLOW rotates each degenerate subspace to the basis that diagonalises the left operator of the sum: $v^i$, or the spin current below. In that basis the pairs inside the subspace drop out; otherwise they would enter with the denominator $\delta^2$ alone. The conductivity does not depend on this choice when the whole subspace has one occupation. The curvature of one band of a degenerate subspace does.

## Anomalous Hall conductivity

In a crystal without time-reversal symmetry the curvatures at $\mathbf{k}$ and $-\mathbf{k}$ do not cancel. An electric field then drives a Hall current without a magnetic field. The current comes from the semiclassical velocity of a wave packet, which carries the anomalous term $-\dot{\mathbf{k}} \times \Omega_n$ besides the band velocity, with $\hbar\dot{\mathbf{k}} = -e\mathbf{E}$. Summing this term over the occupied states gives [3, 10]

$$
\sigma_{ij} = \frac{e^2}{\hbar} \int_\mathrm{BZ} \frac{d^3k}{(2\pi)^3} \sum_n f_{n\mathbf{k}}\, \Omega^{ij}_n(\mathbf{k}) ,
$$

the static limit of the Kubo formula for the antisymmetric part of $\sigma$. `anomalous_Hall()` replaces the integral by the average over the $N_\mathrm{mesh}$ points of the interpolated mesh. It scans the Fermi energy $E$ at fixed $\tilde H$:

$$
\sigma_{ij}(E) = \frac{e^2}{\hbar} \frac{1}{N_\mathrm{mesh} V} \sum_{\mathbf{k}} \sum_n f(\varepsilon_{n\mathbf{k}}; E, w_{n\mathbf{k}})\, \Omega^{ij}_n(\mathbf{k}) ,
\qquad
f(\varepsilon; E, w) = \tfrac12\left[1 - \mathrm{erf}\!\left(\frac{\varepsilon - E}{w}\right)\right] ,
$$

with $V$ the cell volume and $w_{n\mathbf{k}}$ the [adaptive width](density-of-states.md#adaptive-smearing) of {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.adaptive_smearing`. The occupation becomes a Methfessel–Paxton function with `smearing='m-p'`, and a sharp step without adaptive smearing. The code multiplies by $e^2/\hbar$ in siemens and converts bohr$^{-1}$ to cm$^{-1}$, so $\sigma_{ij}(E)$ is in $(\Omega\,\mathrm{cm})^{-1}$. The scan stops at the [PAO shift](pao-hamiltonian.md#the-pao-shift), above which the bands are not those of the DFT calculation.

| Argument | Sets |
| --- | --- |
| `emin`, `emax`, `ne` | the range and the number of points of the scan in $E$ |
| `a_tensor` | the components $(i, j)$ |
| `delta` | the broadening $\delta$ |
| `fermi_dw`, `fermi_up` | the energy window of the curvature map |

| File | Content |
| --- | --- |
| `ahcEf_ij.dat` | $\sigma_{ij}(E)$ in $(\Omega\,\mathrm{cm})^{-1}$ |
| `Berry_ij.bxsf` | the curvature in the window, $\sum_n [f(\varepsilon_{n\mathbf{k}}; E_\mathrm{up}) - f(\varepsilon_{n\mathbf{k}}; E_\mathrm{dw})]\, \Omega^{ij}_n(\mathbf{k})$, on the mesh in bohr$^2$ |

The window of the map runs from $E_\mathrm{dw}$ = `fermi_dw` to $E_\mathrm{up}$ = `fermi_up`, each taken at the nearest point of the scan below it. With $E_\mathrm{dw}$ below the lowest band the map holds the curvature of the whole Fermi sea. With a narrow window around $E_F$ it shows the hot spots near the Fermi surface.

A nonzero $\sigma_{xy}$ in a magnet needs spin–orbit coupling, which transfers the magnetisation to the orbital motion. `anomalous_Hall()` runs only on Hamiltonians with spin–orbit coupling. For a spinless model with broken time reversal, such as the Haldane model below, compute the Chern number from the plaquette sum of `berry_phase()` instead.

## Chern number and the quantised plateau

Consider a two-dimensional band, or set of bands, separated from all others by a gap everywhere in the zone. The flux of its curvature through the zone is quantised:

$$
C = \frac{1}{2\pi} \int_\mathrm{BZ} \Omega^{xy}(\mathbf{k})\, d^2k \in \mathbb{Z} .
$$

The quantisation follows from the shape of the zone, which is a torus, a closed surface. Cut it into two patches, each with its own smooth gauge. The flux through each patch is the Berry phase around their common boundary, traversed in opposite directions. The two phases agree modulo $2\pi$, so the total flux is $2\pi$ times an integer. That integer counts the obstruction to a single smooth gauge over the zone [3, 4]. Because $C$ is an integer, it can change only where the gap closes.

When the Fermi energy lies in such a gap, every occupied band is full. With $(1/N_\mathrm{mesh} A)\sum_{\mathbf{k}} \to \int d^2k/(2\pi)^2$ for a cell of area $A$, the conductivity becomes

$$
\sigma_{xy} = \frac{e^2}{\hbar} \frac{2\pi}{(2\pi)^2} \sum_{n\,\mathrm{occ}} C_n = \frac{e^2}{h} \sum_{n\,\mathrm{occ}} C_n ,
$$

the result of Thouless, Kohmoto, Nightingale and den Nijs [4]. This result explains the plateau of the Fermi-energy scan. Moving $E$ through the gap changes no occupation, so $\sigma_{xy}(E)$ stays at the quantised value until $E$ reaches a band edge. Below the lowest band $\sigma_{xy}$ vanishes because no state is occupied. Above the highest band it vanishes because the curvatures of all bands add to zero. In a metal the Fermi sea ends inside a band, and the integral is not quantised.

Three effects move the computed plateau away from the integer:

| Effect | Consequence | What to do |
| --- | --- | --- |
| The broadening lowers each term by $\Delta\varepsilon^2/(\Delta\varepsilon^2 + \delta^2)$ | the plateau is low by a relative amount of order $(\delta/E_g)^2$ for a gap $E_g$ | choose $\delta \ll E_g$ |
| The smeared occupation leaks a tail of width $w_{n\mathbf{k}}$ into the gap | the edges of the plateau are rounded; its centre is unchanged when $w \ll E_g$ | nothing, as long as $w \ll E_g$ |
| The curvature peaks have a width of order $E_g/(\hbar v)$, for band velocity $v$ at the gap edge | a mesh that does not resolve them misses part of the flux | use a fine mesh for a small gap |

## Spin Berry curvature and spin Hall conductivity

The spin Hall effect is a current of spin transverse to the field. It flows also in crystals with time-reversal symmetry, where the charge Hall current vanishes. Its Kubo formula has the form of the anomalous Hall formula, with the left velocity replaced by a spin current [11, 15]. `spin_Hall()` uses the conventional spin current, $j^{s}_{i} = \tfrac12\{s, v^i\}$. Here $s = \sigma_s/2$ is the spin along direction $s$ in units of $\hbar$, with $\sigma_s$ its Pauli matrix. The method evaluates

$$
\Omega^{s}_{ij,n}(\mathbf{k}) = -2 \sum_{m \ne n} \frac{\mathrm{Im}\bigl[j^{s}_{i,nm}(\mathbf{k})\, v^j_{mn}(\mathbf{k})\bigr]}{(\varepsilon_{n\mathbf{k}} - \varepsilon_{m\mathbf{k}})^2 + \delta^2} ,
\qquad
\sigma^{s}_{ij}(E) = \frac{e^2}{\hbar} \frac{1}{N_\mathrm{mesh} V} \sum_{\mathbf{k}} \sum_n f(\varepsilon_{n\mathbf{k}}; E, w_{n\mathbf{k}})\, \Omega^{s}_{ij,n}(\mathbf{k}) ,
$$

with the broadening, occupations, scan and degenerate subspaces of the anomalous Hall case. The spin current carries $\hbar s$ in place of the charge, so the conductivity is in the conventional unit $(\hbar/e)\,(\Omega\,\mathrm{cm})^{-1}$. The arguments and files that differ from the anomalous Hall case are:

| Argument or file | Meaning |
| --- | --- |
| `s_tensor` | the triplets $(i, j, s)$ |
| `shc_proj` | restricts the spin current to a set of orbitals $P$, through $\tfrac12\{P, j^s_i\}$ |
| `twoD=True` | divides the sum by the cell area in bohr$^2$, without the factor $e^2/\hbar$ |
| `shcEf_s_ij.dat` | $\sigma^{s}_{ij}(E)$ in $(\hbar/e)\,(\Omega\,\mathrm{cm})^{-1}$, or with `twoD=True` the sheet conductivity in units with $\hbar = e = 1$ |
| `Spin_Berry_s_ij.bxsf` | the map of the spin Berry curvature in the window `fermi_dw`, `fermi_up`, as above |

In the units of `twoD=True`, the quantum $e/2\pi$ is $1/2\pi$.

When the Hamiltonian conserves $s_z$, the bands split into two spin sectors, with Chern numbers $C_\uparrow$ and $C_\downarrow$. With $s_z = \pm\tfrac12$ the plateau is

$$
\sigma^{z}_{xy} = \frac{e}{2\pi} \cdot \frac{C_\uparrow - C_\downarrow}{2} .
$$

When spin–orbit coupling mixes the spin components, $\tfrac12\{s, v\}$ is not the current of a conserved quantity ([spin-current operators](linear-response.md#velocity-spin-and-spin-current-operators)). Its Hall conductivity is then not quantised, even in a gap [12]. `spin_Hall()` gives the conductivity of this conventional current, which is the definition used in first-principles studies of metals such as Pt [15].

## Worked models

The three models are honeycomb lattices with one orbital per site and first-neighbour hopping $t$. Near the valleys $K$ and $K'$, $\tilde H$ reduces to the Dirac form $\hbar v(\tau q_x \sigma_x + q_y \sigma_y)$. Here $\mathbf{q}$ is measured from the valley, $\tau = \pm 1$ for $K$ and $K'$, $\hbar v = \tfrac{\sqrt{3}}{2}|t|a$ for lattice constant $a$, and the Pauli matrices act on the sublattices. Each model adds a mass $m_\tau \sigma_z$, which opens a gap $2|m_\tau|$. The models differ in the sign of the mass at the two valleys:

| Model | Mass at valley $\tau$ | Chern number | PAOFLOW model and arguments |
| --- | --- | --- | --- |
| Gapped graphene | $m_\tau = \Delta/2$, the same at both valleys | $C = 0$ | `graphene`, with $\Delta$ = `delta` |
| Haldane | $m_\tau = M - \tau\, 3\sqrt{3}\, t_2 \sin\phi$ | $\lvert C\rvert = 1$ when the two masses have opposite signs, else $C = 0$ | `haldane_pythtb`, from PythTB, with $M$ = `delta` and the hoppings and phase `t1`, `t2`, `phi` |
| Kane–Mele | for each spin, a Haldane mass with $t_2 = \lambda_{SO}$ and $\phi = \pm\pi/2$ | $C_\uparrow = -C_\downarrow = \pm 1$ while $s_z$ is conserved | `Kane_Mele`, with $\lambda_{SO}$ = `soc_par` and the staggered potential $t$ times `v_par` |

### Gapped graphene and the valley Berry phase

An on-site energy $\pm\Delta/2$ on the two sublattices gives the mass $m_\tau = \Delta/2$ at both valleys and a gap $|\Delta|$. For a two-band Hamiltonian $\mathbf{d}(\mathbf{q})\cdot\boldsymbol{\sigma}$, the Berry phase of each band on a loop is half the solid angle swept by $\hat{\mathbf{d}}$, with opposite signs for the two bands [1]. Here $\mathbf{d} = (\hbar v\tau q_x, \hbar v q_y, \Delta/2)$. On a circle of radius $q$ around a valley, $\hat{\mathbf{d}}$ sweeps a cone of solid angle $2\pi(1 - \cos\theta)$, with $\cos\theta = (\Delta/2)/\sqrt{\hbar^2v^2q^2 + \Delta^2/4}$. The phase and the curvature are therefore

$$
|\gamma(q)| = \pi \left( 1 - \frac{|\Delta|}{\sqrt{\Delta^2 + 4\hbar^2 v^2 q^2}} \right) ,
\qquad
\Omega_c(\mathbf{q}) = \tau\, \frac{2\hbar^2 v^2 \Delta}{(\Delta^2 + 4\hbar^2 v^2 q^2)^{3/2}} ,
$$

with $\Omega_c$ the curvature of the conduction band [9]. The phase $\gamma$ has opposite signs at $K$ and $K'$.

Without the gap, $|\gamma| = \pi$ on every loop around the valley, because the curvature is a delta function at the Dirac point. The gap spreads the curvature over a disc of radius of order $|\Delta|/(2\hbar v)$. A loop much smaller than the disc encloses almost no phase, and only a much larger loop approaches $\pi$; $|\gamma| = \pi/2$ at $q = \sqrt{3}|\Delta|/(2\hbar v)$. The phase around one valley therefore measures the flux through the loop, not an invariant.

The flux of a whole valley is $\pm\pi$, and time reversal makes the two valleys opposite, so $C = 0$. Gapped graphene is a trivial insulator, with a valley Hall effect but no charge Hall effect [9].

### The Haldane model

Haldane's model adds two terms to graphene: a sublattice potential $\pm M$ and a second-neighbour hopping $t_2 e^{\pm i\phi}$. The phase of the hopping circulates in the same sense on both sublattices, so it breaks time reversal with no net magnetic flux through the cell [6]. The second-neighbour term gives masses of opposite sign at the two valleys, $m_\tau = M - \tau\, 3\sqrt{3}\, t_2 \sin\phi$. In the Dirac approximation the lower band carries the flux $-\tau\pi\,\mathrm{sgn}(m_\tau)$ around valley $\tau$. This flux is the integral of $\Omega_c$ above, with $\Delta = 2m_\tau$ and the sign reversed. Adding the two valleys gives

$$
C = -\tfrac12 \sum_{\tau = \pm 1} \tau\, \mathrm{sgn}(m_\tau) .
$$

When $|M| < 3\sqrt{3}\,|t_2 \sin\phi|$, the masses have opposite signs and the valley fluxes add. Then $|C| = 1$, so $\sigma_{xy} = \pm e^2/h$ without Landau levels. At $M = 0$ and $\phi = \pi/2$ the gap is $6\sqrt{3}\,|t_2|$. When $|M| > 3\sqrt{3}\,|t_2\sin\phi|$, both masses have the sign of $M$, and $C = 0$. The transition between the two phases passes through a gap closing at one valley.

### The quantum spin Hall model of Kane and Mele

Kane and Mele add to graphene a spin–orbit coupling $i\lambda_{SO}\nu_{ij} \sigma_z$ between second neighbours [7]. Here $\sigma_z$ is the Pauli matrix of eigenvalues $\pm 1$, and $\nu_{ij} = \pm 1$ for paths that turn left or right. For each eigenvalue of $\sigma_z$ this coupling is a Haldane term with $t_2 = \lambda_{SO}$ and $\phi = \pm\pi/2$, of opposite sign for the two spins. With $\sigma_z$ conserved, the model is therefore two Haldane models related by time reversal, with $C_\uparrow = -C_\downarrow = \pm 1$. Its gap is $6\sqrt{3}\,\lambda_{SO}$. The charge Hall conductivity is zero, and the spin Hall plateau has the magnitude $e/2\pi$. A staggered potential larger than $3\sqrt{3}\,\lambda_{SO}$ closes and reopens the gap into a trivial insulator, as $M$ does in the Haldane model.

A Rashba term (`r_par`) mixes the spin components. $C_\uparrow$ and $C_\downarrow$ then lose their separate meaning, and the plateau departs from $e/2\pi$. The insulator still stays distinct from a trivial one while the gap is open, because time reversal protects a $\mathbb{Z}_2$ invariant [8]. That invariant, not the spin Hall conductivity, characterises the quantum spin Hall state. When the spin mixes, compute the $\mathbb{Z}_2$ index with `topology()`.

## Approximations and their limits

The band structure is that of independent particles: the PAO Hamiltonian of the DFT functional. The curvature weighs each pair of bands by $1/\Delta\varepsilon^2$, so an error in a small splitting near $E_F$ enters quadratically. A spin–orbit gap at an avoided crossing is such a splitting. A gap underestimated by the functional enlarges the curvature near it.

The conductivities are intrinsic, in the clean limit, at zero temperature and with rigid bands. [Approximations of the Kubo responses](linear-response.md#approximations-of-the-kubo-responses) says what each of these leaves out, including the side-jump and skew-scattering contributions of impurities [10]. The broadening $\delta$ is a numerical regulator, not a model of disorder. In an insulator, choose it as in [the plateau section](#chern-number-and-the-quantised-plateau). In a metal, compare two values, because near-degeneracies at $E_F$ make the result depend on $\delta$.

In a metal, splittings of all sizes occur on the Fermi surface, so the conductivity converges slowly with the mesh. Check it on a denser mesh of {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.interpolated_hamiltonian`. For the Berry phase, the loop needs enough points that the overlaps of neighbouring points stay close to unitary.

## Used in

- [Spin Hall conductivity of the Kane–Mele model](../tutorials/tutorial05.ipynb): the spin Hall plateau of the Kane–Mele model and the map of its spin Berry curvature.
- [Berry phases and Chern number of gapped graphene](../tutorials/tutorial06.ipynb): Berry phases on loops around a valley of gapped graphene, and its Chern number from the Berry flux through the zone.
- [How to run PAOFLOW on a tight-binding model](../howtos/tight-binding-model.md): the area normalisation of `spin_Hall()` for two-dimensional models, and the gap of the Kane–Mele model.

## References

1. M. V. Berry, Quantal phase factors accompanying adiabatic changes, Proc. R. Soc. Lond. A 392, 45 (1984).
2. R. Resta, Macroscopic polarization in crystalline dielectrics: the geometric phase approach, Rev. Mod. Phys. 66, 899 (1994).
3. D. Xiao, M.-C. Chang and Q. Niu, Berry phase effects on electronic properties, Rev. Mod. Phys. 82, 1959 (2010).
4. D. J. Thouless, M. Kohmoto, M. P. Nightingale and M. den Nijs, Quantized Hall conductance in a two-dimensional periodic potential, Phys. Rev. Lett. 49, 405 (1982).
5. T. Fukui, Y. Hatsugai and H. Suzuki, Chern numbers in discretized Brillouin zone: efficient method of computing (spin) Hall conductances, J. Phys. Soc. Jpn. 74, 1674 (2005).
6. F. D. M. Haldane, Model for a quantum Hall effect without Landau levels: condensed-matter realization of the "parity anomaly", Phys. Rev. Lett. 61, 2015 (1988).
7. C. L. Kane and E. J. Mele, Quantum spin Hall effect in graphene, Phys. Rev. Lett. 95, 226801 (2005).
8. C. L. Kane and E. J. Mele, Z2 topological order and the quantum spin Hall effect, Phys. Rev. Lett. 95, 146802 (2005).
9. D. Xiao, W. Yao and Q. Niu, Valley-contrasting physics in graphene: magnetic moment and topological transport, Phys. Rev. Lett. 99, 236809 (2007).
10. N. Nagaosa, J. Sinova, S. Onoda, A. H. MacDonald and N. P. Ong, Anomalous Hall effect, Rev. Mod. Phys. 82, 1539 (2010).
11. J. Sinova, S. O. Valenzuela, J. Wunderlich, C. H. Back and T. Jungwirth, Spin Hall effects, Rev. Mod. Phys. 87, 1213 (2015).
12. J. Shi, P. Zhang, D. Xiao and Q. Niu, Proper definition of spin current in spin-orbit coupled systems, Phys. Rev. Lett. 96, 076604 (2006).
13. J. Zak, Berry's phase for energy bands in solids, Phys. Rev. Lett. 62, 2747 (1989).
14. R. D. King-Smith and D. Vanderbilt, Theory of polarization of crystalline solids, Phys. Rev. B 47, 1651 (1993).
15. G. Y. Guo, S. Murakami, T.-W. Chen and N. Nagaosa, Intrinsic spin Hall effect in platinum: first-principles calculations, Phys. Rev. Lett. 100, 096401 (2008).
