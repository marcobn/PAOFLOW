# The PAO Hamiltonian from projected Kohn–Sham states

This page derives the tight-binding Hamiltonian from which PAOFLOW computes every property of a DFT calculation. The Kohn–Sham states are projected onto a basis of pseudo-atomic orbitals (PAOs), and their projectability decides which states the basis can represent. From the states it keeps, PAOFLOW builds the Hamiltonian at the k-points of the DFT run, Fourier transforms it to real space, and interpolates it back to arbitrary k. Each step is evaluated by one method:

| Step | Method |
| --- | --- |
| Projection onto the PAOs | {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.read_atomic_proj_QE` or {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.projections` |
| Projectability, kept bands and PAO shift | {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.projectability` |
| Hamiltonian at the DFT k-points and in real space | {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.pao_hamiltonian` |
| Fourier interpolation on a path or a mesh | {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.bands`, {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.interpolated_hamiltonian` |

The other theory pages start from the interpolated Hamiltonian $\tilde H(\mathbf{k})$ built here.

## Pseudo-atomic orbitals and their overlap

The basis consists of the pseudo-atomic orbitals of the atoms in the cell, $\phi^0_\alpha(\mathbf{r}) = R_{al}(r)\,Y_{lm}(\hat{\mathbf{r}})$. Each is the radial function of shell $l$ of the isolated pseudo-atom $a$, times a real spherical harmonic. The radial functions come from one of two sources:

| Method | Radial functions |
| --- | --- |
| {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.read_atomic_proj_QE` | those of the pseudopotential file, as used by `projwfc.x` |
| {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.projections` | PAOFLOW's basis library (see [Internal basis workflow](../../internals/internal-basis.md)) |

The index $\alpha$ runs over atoms, shells and $m$, and the cell has $n_\mathrm{awf}$ orbitals. With spin–orbit coupling the orbitals are spinors, and $n_\mathrm{awf}$ counts both spin components. In a collinear spin-polarised run each spin channel is treated separately, and the spin index is dropped below.

In the crystal each orbital enters as a Bloch sum over the $N_k$ cells of the Born–von Kármán supercell that the DFT grid defines,

$$
\phi^0_{\alpha\mathbf{k}}(\mathbf{r}) = \frac{1}{\sqrt{N_k}} \sum_{\mathbf{R}} e^{i\mathbf{k}\cdot\mathbf{R}}\, \phi^0_\alpha(\mathbf{r} - \boldsymbol{\tau}_\alpha - \mathbf{R}) ,
$$

where $\boldsymbol{\tau}_\alpha$ is the position of the atom that carries orbital $\alpha$ and $\mathbf{R}$ a lattice vector. The phase contains only the lattice vector, and the atomic position enters through the argument of the orbital. This convention fixes the meaning of $H_{\alpha\beta}(\mathbf{R})$ below.

Orbitals on neighbouring atoms overlap, so the Bloch sums are not orthonormal. Their overlap matrix is

$$
S_{\alpha\beta}(\mathbf{k}) = \langle\phi^0_{\alpha\mathbf{k}}|\phi^0_{\beta\mathbf{k}}\rangle = \sum_{\mathbf{R}} e^{i\mathbf{k}\cdot\mathbf{R}}\, \langle\phi^0_{\alpha,\mathbf{0}}|\phi^0_{\beta,\mathbf{R}}\rangle ,
$$

with $\phi^0_{\beta,\mathbf{R}}$ the orbital $\beta$ in the cell at $\mathbf{R}$. $S(\mathbf{k})$ is Hermitian, and positive definite as long as the orbitals are linearly independent. Diffuse orbitals on close atoms bring it close to singular. With ultrasoft or projector-augmented-wave (PAW) pseudopotentials, the pseudo-wavefunctions are orthonormal in an augmented metric, and every inner product on this page is taken in that metric (see [the augmented metric](../../internals/internal-basis.md#ultrasoft-and-paw-pseudopotentials-the-augmented-metric)).

### Löwdin orthogonalisation

Both routes, `projwfc.x` and {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.projections`, replace the PAOs by the orthonormal combinations

$$
\phi_{\alpha\mathbf{k}} = \sum_\beta \phi^0_{\beta\mathbf{k}}\, \bigl[S^{-1/2}(\mathbf{k})\bigr]_{\beta\alpha} ,
$$

the symmetric orthogonalisation of Löwdin [4]. Their overlap is $S^{-1/2} S\, S^{-1/2} = \mathbb{1}$. Any $S^{-1/2}U$ with $U$ unitary would also give an orthonormal set. The symmetric choice is the one that keeps the symmetry of the orbitals. $S^{-1/2}$ is a function of $S$, so it commutes with every symmetry operation that $S$ commutes with. Each $\phi_\alpha$ then transforms under the point group like the $\phi^0_\alpha$ it comes from, and a $p_x$ orbital remains $p_x$-like. No orbital is favoured either, as one would be in a Gram–Schmidt sequence.

The price is locality. Each $\phi_\alpha$ carries tails on the neighbouring atoms, with weights set by the overlaps. A projected density of states on orbital $\alpha$ therefore also counts weight that belongs to the bonds around its atom.

From here on, $\phi_\alpha$ denotes these orthogonalised orbitals. Every Hamiltonian on this page is a matrix in this orthonormal basis, so its eigenvalue problem is an ordinary one. The matrix in the original orbitals is given in [The Hamiltonian in the non-orthogonal basis](#the-hamiltonian-in-the-non-orthogonal-basis).

## Projection amplitudes

The projection amplitudes are the components of the Kohn–Sham states in the orthogonalised basis,

$$
A_{n\alpha}(\mathbf{k}) = \langle\phi_{\alpha\mathbf{k}}|\psi_{n\mathbf{k}}\rangle .
$$

The operator $P(\mathbf{k}) = \sum_\alpha |\phi_{\alpha\mathbf{k}}\rangle\langle\phi_{\alpha\mathbf{k}}|$ projects onto the PAO space. Applied to a Kohn–Sham state it gives $P|\psi_{n\mathbf{k}}\rangle = \sum_\alpha A_{n\alpha}(\mathbf{k})\,|\phi_{\alpha\mathbf{k}}\rangle$, so the $n_\mathrm{awf}$ amplitudes of band $n$ describe the part of $\psi_{n\mathbf{k}}$ that the basis represents. The amplitudes are computed for all the bands of the DFT run, at its k-points. When the run uses symmetry, these are the k-points of the irreducible wedge of the grid.

## Projectability

The projectability of a state is the squared norm of its projection,

$$
p_{n\mathbf{k}} = \sum_\alpha |A_{n\alpha}(\mathbf{k})|^2 = \langle\psi_{n\mathbf{k}}|P(\mathbf{k})|\psi_{n\mathbf{k}}\rangle ,
$$

so $0 \le p_{n\mathbf{k}} \le 1$, and $1 - p_{n\mathbf{k}}$ is the weight of the state outside the PAO space. Bonding and antibonding combinations of the valence orbitals have $p_{n\mathbf{k}}$ close to 1. Conduction states higher up have weight in the interstitial region, with the character of plane waves, or on atomic shells that the basis does not contain. Their projectability falls.

The projectability obeys a sum rule. The Kohn–Sham states at $\mathbf{k}$ form a complete set, so the sum over all of them is $\sum_n p_{n\mathbf{k}} = \mathrm{Tr}\,P(\mathbf{k}) = n_\mathrm{awf}$. At most $n_\mathrm{awf}$ states per k-point can have $p_{n\mathbf{k}} \approx 1$, and the rest of the PAO weight is spread over higher bands. The DFT run must therefore contain enough bands to show where the projectability drops. When every band of the run passes the threshold `pthr` of the next section, {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.projectability` prints a warning to compute more bands.

### Number of kept bands

{py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.projectability` averages the projectability of each band over the k-points of the DFT run and the $N_s$ spin channels,

$$
\bar p_n = \frac{1}{N_s N_\mathrm{run}} \sum_\sigma \sum_{\mathbf{k}\in\mathrm{run}} p_{n\mathbf{k}\sigma} ,
$$

where $N_\mathrm{run}$ is the number of k-points of the run, each with the same weight. For a run without symmetry reduction, this is the average over the full grid, $N_k^{-1}\sum_{\mathbf{k}} p_{n\mathbf{k}}$. For a run on the irreducible wedge it is not. A high-symmetry point stands for few points of the grid but counts as much as a general point, so the average leans towards the high-symmetry points. The difference matters only for a band whose $\bar p_n$ lies close to the threshold.

The number of kept bands $N_b$ is the number of bands with $\bar p_n$ above `pthr` (default 0.95). The Hamiltonian is built from the $N_b$ lowest bands, so the count decides, not the band index. A band below $N_b$ whose average falls under `pthr` is kept, and a band above $N_b$ that passes is not. A kept state is also dropped at a single k-point where its projectability falls below half of `pthr`, for the reason the bound below gives.

The value of `pthr` limits how far the projected states are from orthonormal. Take two different states $n \neq m$ at the same $\mathbf{k}$. Their orthogonality, $\langle\psi_n|\psi_m\rangle = 0$, gives $\langle\psi_n|P|\psi_m\rangle = -\langle\psi_n|(1-P)|\psi_m\rangle$. The Cauchy–Schwarz inequality for the parts outside the PAO space bounds this by $\sqrt{(1-p_n)(1-p_m)}$. After normalisation to unit length, $\mathbf{a}_{n\mathbf{k}} = A_{n\cdot}(\mathbf{k})/\sqrt{p_{n\mathbf{k}}}$, the overlap of two projected states is therefore

$$
|\mathbf{a}_{n\mathbf{k}}^\dagger \mathbf{a}_{m\mathbf{k}}| \le \sqrt{\frac{(1-p_{n\mathbf{k}})(1-p_{m\mathbf{k}})}{p_{n\mathbf{k}}\,p_{m\mathbf{k}}}} .
$$

At $p = 0.95$ the bound is 0.053. For a state with $p$ near zero it allows any overlap, because normalising the state blows up the small part that the basis represents. This is why a state is left out at a k-point where its projectability falls below half of `pthr`. The overlaps enter the error of the Hamiltonian at the DFT k-points (see [Accuracy at the DFT k-points](#accuracy-at-the-dft-k-points)).

## The PAO Hamiltonian at the DFT k-points

At each k-point of the run, {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.pao_hamiltonian` builds an $n_\mathrm{awf}\times n_\mathrm{awf}$ matrix with two properties. Its eigenstates are the kept projected states, with their DFT eigenvalues. The rest of the PAO space sits at an energy $\eta$ above the bands the matrix represents. The kept set at $\mathbf{k}$ is

$$
K_\mathbf{k} = \bigl\{\, n \le N_b \;:\; \varepsilon_{n\mathbf{k}} \le \eta ,\; p_{n\mathbf{k}} > \mathrm{pthr}/2 \,\bigr\} ,
$$

with the eigenvalues $\varepsilon_{n\mathbf{k}}$ measured from the DFT Fermi energy $E_F$, the energy zero of PAOFLOW. Three matrices enter the construction. $A$ is the $n_\mathrm{awf}\times|K_\mathbf{k}|$ matrix whose columns are the normalised projections $\mathbf{a}_{n\mathbf{k}}$. $E$ is the diagonal matrix of their eigenvalues. $O = A^\dagger A$ is the overlap matrix of the projected states, with $O_{nn} = 1$.

If the projected states were orthonormal, $O = \mathbb{1}$, the operator $AEA^\dagger + \eta(\mathbb{1} - AA^\dagger)$ would have exactly the required spectrum. In general they are not, and then $AA^\dagger$ is not a projector. The orthogonal projector onto the space they span is $P_A = A O^{-1} A^\dagger$, since $P_A^2 = P_A$ and $P_A A = A$. The default `shift_type=1` uses it:

$$
\tilde H(\mathbf{k}) = A E A^\dagger + \eta \bigl[\mathbb{1} - A (A^\dagger A)^{-1} A^\dagger\bigr] .
$$

The argument `shift_type` of {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.pao_hamiltonian` chooses the second term:

| `shift_type` | $\tilde H(\mathbf{k})$ | Directions that no kept state spans |
| --- | --- | --- |
| 0 | $AEA^\dagger + \eta(\mathbb{1} - AA^\dagger)$ | at $\eta$ |
| 1 (default) | the equation above | at $\eta$ |
| 2 | $AEA^\dagger$ alone | at zero, which is $E_F$ |

The construction and the shift are derived in Refs. [1–3].

### The PAO shift

The second term gives the energy $\eta$, called the PAO shift, to the directions of the PAO space that no kept state spans. Both terms map the span of the columns of $A$ onto itself. On its orthogonal complement the first term vanishes and the bracket is the identity. $\tilde H(\mathbf{k})$ therefore has $n_\mathrm{awf} - |K_\mathbf{k}|$ eigenvalues exactly equal to $\eta$, for `shift_type` 0 and 1 alike.

These levels are combinations of orbitals, antibonding or from higher shells, whose Kohn–Sham counterparts have too much weight outside the basis. Without the shift (`shift_type=2`) they sit at zero energy, at $E_F$, where they would appear as flat bands in the gap or at the Fermi surface. The shift moves them above the energies the Hamiltonian represents.

With `shift='auto'`, {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.projectability` sets $\eta$ to the bottom of band $N_b + 1$, the first band that is not kept: its lowest energy over the k-points of the run. When $N_b = n_\mathrm{awf}$, band $N_b$ takes its place. Every DFT state below this energy belongs to the $N_b$ lowest bands, so it is in the kept set unless its own projectability fails. Below $\eta$ the PAO bands therefore follow the DFT bands, and above it they do not.

$\eta$ is thus the top of the energy window in which the Hamiltonian can be trusted. Its role is similar to that of the upper edge of the frozen window in Wannier disentanglement. A kept band that rises above $\eta$ at some k-points is dropped there by the energy condition in $K_\mathbf{k}$, and a level at $\eta$ takes its place.

A value of `shift` you set by hand moves this window, and either direction costs states:

| `shift` | Effect on the window |
| --- | --- |
| above the automatic value | Band $N_b + 1$ lies below $\eta$ at some k-points without being in the Hamiltonian, so the PAO spectrum misses a DFT state inside the window. |
| below the automatic value | The energy condition removes the kept states between your $\eta$ and the bottom of band $N_b + 1$, and the window shrinks. |

{py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.projectability` prints the energy range of band $N_b + 1$, the range in which a manual shift keeps the most states.

### Accuracy at the DFT k-points

The PAO bands reproduce the DFT eigenvalues of the kept states at the k-points of the run exactly only when the projected states are orthogonal. The span of $A$ is invariant under $\tilde H(\mathbf{k})$, so the kept eigenvalues are those of $\tilde H$ restricted to it. For `shift_type=1` they are the eigenvalues of $EO$, since $AEA^\dagger A = A(EO)$, and these equal the eigenvalues of $OE$. For `shift_type=0` they are $\eta$ plus the eigenvalues of $O(E - \eta)$. Write $O = \mathbb{1} + \Delta$, with $\Delta$ off-diagonal. Second-order perturbation theory then gives the shift of eigenvalue $n$ for `shift_type=1`,

$$
\delta\varepsilon_{n\mathbf{k}} \approx \sum_{m\neq n} |\Delta_{nm}|^2\, \frac{\varepsilon_{n\mathbf{k}}\,\varepsilon_{m\mathbf{k}}}{\varepsilon_{n\mathbf{k}} - \varepsilon_{m\mathbf{k}}} ,
$$

and for `shift_type=0` the same sum with every $\varepsilon$ in the numerator replaced by $\varepsilon - \eta$. The first-order term vanishes because $\Delta_{nn} = 0$.

The error is small for states of high projectability. By the bound of [Number of kept bands](#number-of-kept-bands), $|\Delta_{nm}|^2$ is of order $(1-p)^2$, so states with $p_{n\mathbf{k}}$ near 1, such as the valence bands of a covalent crystal, are reproduced closely. The error is largest for a conduction state with projectability near `pthr` that has a close partner of the same symmetry. It vanishes at k-points where symmetry makes the projected states orthogonal, such as states of different irreducible representations at $\Gamma$.

The size of the error also depends on the energy zero, because the numerator holds energies measured from it. For the default `shift_type=1` the error grows with the distance of the pair from $E_F$, and for `shift_type=0` with their distance from $\eta$. With $|\Delta_{nm}| \sim 0.03$, energies a few eV from the zero and a splitting of a few tenths of an eV, it is of the order of tens of meV. This error, not the interpolation, is what you see when a PAO band misses a DFT eigenvalue at a point of the DFT grid.

### The Hamiltonian in the non-orthogonal basis

The same Hamiltonian written in the original PAOs follows from the inverse of the Löwdin transformation, $\phi^0_{\beta\mathbf{k}} = \sum_\alpha \phi_{\alpha\mathbf{k}}\,[S^{1/2}(\mathbf{k})]_{\alpha\beta}$:

$$
\langle\phi^0_{\alpha\mathbf{k}}|\hat H|\phi^0_{\beta\mathbf{k}}\rangle = \bigl[S^{1/2}(\mathbf{k})\,\tilde H(\mathbf{k})\,S^{1/2}(\mathbf{k})\bigr]_{\alpha\beta} .
$$

Its eigenvalue problem is the generalised one, $S^{1/2}\tilde H S^{1/2}\,\mathbf{c} = \varepsilon\, S\,\mathbf{c}$. It has the eigenvalues of $\tilde H$, with eigenvectors $\mathbf{c} = S^{-1/2}\mathbf{u}$, where $\mathbf{u}$ is the eigenvector of $\tilde H$. Only the ACBN0 workflow uses this form. With `acbn0=True` in the {py:class}`~PAOFLOW.PAOFLOW.PAOFLOW` constructor, {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.pao_hamiltonian` writes this matrix and stops (see [ACBN0 module](../../internals/acbn0.md)). Every other property is computed in the orthogonalised basis.

### From the irreducible wedge to the full grid

When the DFT run lists only the irreducible wedge, {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.pao_hamiltonian` builds $\tilde H(\mathbf{k})$ there. It obtains the Hamiltonian on the rest of the grid from the symmetry operations $\mathcal{S}$ of the crystal,

$$
\tilde H(\mathcal{S}\mathbf{k}) = D(\mathcal{S},\mathbf{k})\, \tilde H(\mathbf{k})\, D^\dagger(\mathcal{S},\mathbf{k}) ,
$$

where the unitary matrix $D$ combines three parts: the rotation matrices of the angular parts of the orbitals, the permutation of the atoms that $\mathcal{S}$ maps onto each other, and the Bloch phase of the lattice vector that brings each image atom back into the home cell. Time reversal relates $\tilde H(-\mathbf{k})$ to the complex conjugate of $\tilde H(\mathbf{k})$ in the same way. The full grid therefore carries the symmetry of the crystal. An error of the construction at one wedge point is copied to all its images. The argument `expand_wedge` controls this step. A run without symmetry reduction already holds the full grid.

## The real-space Hamiltonian

The real-space Hamiltonian is the discrete Fourier transform of $\tilde H(\mathbf{k})$ over the $N_k = N_1 N_2 N_3$ points of the $\Gamma$-centred DFT grid:

$$
H_{\alpha\beta}(\mathbf{R}) = \frac{1}{N_k} \sum_{\mathbf{k}} e^{-i\mathbf{k}\cdot\mathbf{R}}\, \tilde H_{\alpha\beta}(\mathbf{k}) ,
$$

for the $N_k$ lattice vectors $\mathbf{R} = n_1\mathbf{a}_1 + n_2\mathbf{a}_2 + n_3\mathbf{a}_3$ with $-\lfloor N_i/2\rfloor \le n_i \le \lceil N_i/2\rceil - 1$. These lattice vectors form a box centred on the origin. With the Bloch sums above, translation invariance gives $\langle\phi_{\alpha\mathbf{k}}|\hat H|\phi_{\beta\mathbf{k}}\rangle = \sum_\mathbf{R} e^{i\mathbf{k}\cdot\mathbf{R}}\langle\phi_{\alpha,\mathbf{0}}|\hat H|\phi_{\beta,\mathbf{R}}\rangle$. $H_{\alpha\beta}(\mathbf{R})$ is therefore the hopping from orbital $\beta$ in the cell at $\mathbf{R}$ to orbital $\alpha$ in the home cell, and $H(\mathbf{R})$ is an *ab initio* tight-binding model. Hermiticity of $\tilde H(\mathbf{k})$ gives $H(-\mathbf{R}) = H(\mathbf{R})^\dagger$.

### Aliasing

The construction above defines $\tilde H(\mathbf{k})$ at every $\mathbf{k}$, but the grid samples it at $N_k$ points only. The transform therefore returns a folded version of its true Fourier coefficients. To see how, write $\tilde H(\mathbf{k}) = \sum_{\mathbf{R}'} e^{i\mathbf{k}\cdot\mathbf{R}'} H^\infty(\mathbf{R}')$, with the sum over the whole lattice. Inserting this into the transform and using $N_k^{-1}\sum_\mathbf{k} e^{i\mathbf{k}\cdot(\mathbf{R}'-\mathbf{R})} = \sum_\mathbf{T} \delta_{\mathbf{R}',\mathbf{R}+\mathbf{T}}$, where $\mathbf{T} = m_1N_1\mathbf{a}_1 + m_2N_2\mathbf{a}_2 + m_3N_3\mathbf{a}_3$ runs over the lattice of the supercell, gives

$$
H(\mathbf{R}) = \sum_{\mathbf{T}} H^\infty(\mathbf{R} + \mathbf{T}) .
$$

Each element is the true hopping plus the hoppings to all periodic images of the Born–von Kármán supercell. $H(\mathbf{R})$ is a faithful tight-binding model when $H^\infty$ is negligible outside the box, that is, when the hoppings have decayed within half the supercell in each direction.

### Decay of the hoppings

How fast $H^\infty(\mathbf{R})$ decays is set by how smooth $\tilde H(\mathbf{k})$ is as a function of $\mathbf{k}$. Suppose $\tilde H(\mathbf{k})$ and its derivatives up to order $m-1$ are continuous and periodic. Integrating the Fourier integral by parts $m$ times then moves $m$ factors of $1/|\mathbf{R}|$ in front, and the hoppings decay at least as $|\mathbf{R}|^{-m}$. If $\tilde H(\mathbf{k})$ is analytic, they decay exponentially.

The physical reason for a fast decay is locality. The Kohn–Sham Hamiltonian is a local operator, and its matrix elements between orbitals centred on distant atoms fall off with their overlap. The PAO construction keeps this locality as long as the kept subspace varies smoothly with $\mathbf{k}$. It breaks it wherever the kept set changes:

- where a kept band crosses $\eta$ and is dropped;
- where the projectability of a kept state crosses half of `pthr`;
- where band $N_b$ is entangled with band $N_b + 1$, so that the lowest $N_b$ bands change character abruptly.

At such points $\tilde H(\mathbf{k})$ has a kink or a jump. The hoppings then acquire a power-law tail, and the tail folds back into the box as aliasing. The bands most affected are those near the change, which the automatic shift places at the top of the window.

## Fourier interpolation

The inverse transform evaluated at an arbitrary $\mathbf{k}$,

$$
\tilde H(\mathbf{k}) = \sum_{\mathbf{R}} e^{i\mathbf{k}\cdot\mathbf{R}}\, H(\mathbf{R}) ,
$$

with the sum over the box of lattice vectors, gives the PAO Hamiltonian between the grid points. Its eigenvalues $\varepsilon_{n\mathbf{k}}$ are the interpolated bands, and each k-point costs one $n_\mathrm{awf}\times n_\mathrm{awf}$ diagonalisation. Of its $n_\mathrm{awf}$ bands, those below $\eta$ follow the DFT bands.

### Exactness on the DFT grid

At a point $\mathbf{k}'$ of the DFT grid, inserting the forward transform gives $\sum_\mathbf{R} e^{i\mathbf{k}'\cdot\mathbf{R}} N_k^{-1}\sum_\mathbf{k} e^{-i\mathbf{k}\cdot\mathbf{R}}\tilde H(\mathbf{k}) = \tilde H(\mathbf{k}')$, because $N_k^{-1}\sum_\mathbf{R} e^{i(\mathbf{k}'-\mathbf{k})\cdot\mathbf{R}} = \delta_{\mathbf{k}\mathbf{k}'}$ on the grid. The interpolation thus returns the Hamiltonian it was built from, to rounding error, whatever the grid size and however slowly the hoppings decay.

Agreement between the PAO bands and the DFT eigenvalues at the grid points therefore tests the construction of the Hamiltonian (see [Accuracy at the DFT k-points](#accuracy-at-the-dft-k-points)), not the interpolation. The interpolation error lives between the grid points, and it is the [aliasing](#aliasing) of the real-space Hamiltonian: the hoppings beyond the box are missing, and those folded into it carry the wrong phase. To measure it, compare with a DFT band calculation at k-points off the grid, or repeat the calculation on a denser DFT grid.

### Paths and meshes

PAOFLOW evaluates the sum in two ways:

| | {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.bands` | {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.interpolated_hamiltonian` |
| --- | --- | --- |
| k-points | the points of a path | a mesh of $M_1\times M_2\times M_3$ points, $N_\mathrm{mesh} = M_1M_2M_3$, with $M_i = 2N_i$ unless you set `nfft1`, `nfft2` and `nfft3` |
| Evaluation | the sum, directly at each point | $H(\mathbf{R})$ placed in a box of $M_1\times M_2\times M_3$ lattice vectors, filled with zeros outside the original one, then a fast Fourier transform |
| A coefficient on a face of the box, for even $N_i$ | kept at $n_i = -N_i/2$ | split equally between $n_i = -N_i/2$ and $n_i = +N_i/2$ |

The zeros of the mesh route add no information: the mesh samples the same interpolation more densely.

The two routes differ only in the last row. A lattice vector on the face $n_i = -N_i/2$ of the box and its image at $n_i = +N_i/2$ differ by the supercell vector $N_i\mathbf{a}_i$. On the DFT grid their phases are equal, so the grid cannot tell which of the two the coefficient belongs to. Splitting it equally keeps $\tilde H(\mathbf{k})$ Hermitian between grid points. Both routes agree on the DFT grid. Between grid points they differ by an amount set by the hoppings on the faces of the box, the same quantity that measures the aliasing. A visible difference between the band structure and the mesh eigenvalues therefore means that the DFT grid is too coarse for the range of the hoppings.

## What limits the PAO Hamiltonian

The PAO bands reproduce the DFT calculation, with its exchange–correlation functional and its band gap, and not the experiment. On top of the Kohn–Sham eigenvalues it starts from, three independent choices set the accuracy of $\tilde H(\mathbf{k})$:

| Choice | What it sets | Price of improving it |
| --- | --- | --- |
| DFT grid | The box of lattice vectors, and with it the aliasing error between grid points | A denser DFT calculation |
| `pthr` | The kept bands and $\eta$, the top of the energy window that the Hamiltonian represents | A lower `pthr` keeps more bands and raises $\eta$, but the projected states overlap more, and the eigenvalue errors at the DFT k-points grow as $(1-p)^2$ |
| Basis | The number of bands, $n_\mathrm{awf}$, and their projectability | Every later step costs more: the memory of the interpolated mesh grows as $N_\mathrm{mesh}\, n_\mathrm{awf}^2$ and each diagonalisation as $n_\mathrm{awf}^3$ |

The error of the DFT grid does not show at the grid points. The hoppings must decay within half the supercell, so a material with long-range hoppings, such as one whose kept bands are entangled with higher ones, needs a denser grid than its DFT self-consistency alone would require. Converge it by comparing results from two DFT grids.

Above $\eta$ the spectrum consists of placeholder levels and of states distorted by the energy condition. A property that involves states near or above $\eta$, such as optical transitions into high conduction bands, is therefore not reliable. To raise $\eta$, lower `pthr` or add shells to the basis.

Adding shells, such as empty $d$ orbitals or the shells of the extended preset of {py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.projections`, raises the projectability of the conduction states and moves $\eta$ up. Diffuse orbitals also bring $S(\mathbf{k})$ close to singular. The Löwdin transformation then mixes the orbitals strongly and spreads each orthogonalised orbital over its neighbours, so the projected density of states becomes harder to read as an orbital character.

## Used in

- [Band structure and density of states of silicon](../tutorials/tutorial01.ipynb): the projections, projectability, PAO shift and $H(\mathbf{R})$ of bulk Si, its bands, and their interpolation onto a $24\times24\times24$ mesh.
- [How to choose the projections onto the PAO basis](../howtos/projections.md): the projectability, kept bands and PAO shift of each basis of Si, from `projwfc.x` to the `extended` preset.
- [How to choose the truncation of the sparse Hamiltonian](../howtos/sparse-truncation.md): how fast the hoppings decay.
- [How to choose which states the sparse backend solves](../howtos/sparse-states.md): the cap the PAO shift puts on the solved states.

## References

1. L. A. Agapito, A. Ferretti, A. Calzolari, S. Curtarolo and M. Buongiorno Nardelli, Effective and accurate representation of extended Bloch states on finite Hilbert spaces, *Phys. Rev. B* 88, 165127 (2013).
2. L. A. Agapito, S. Ismail-Beigi, S. Curtarolo, M. Fornari and M. Buongiorno Nardelli, Accurate tight-binding Hamiltonian matrices from ab initio calculations: Minimal basis sets, *Phys. Rev. B* 93, 035104 (2016).
3. L. A. Agapito, M. Fornari, D. Ceresoli, A. Ferretti, S. Curtarolo and M. Buongiorno Nardelli, Accurate tight-binding Hamiltonians for two-dimensional and layered materials, *Phys. Rev. B* 93, 125137 (2016).
4. P.-O. Löwdin, On the non-orthogonality problem connected with the use of atomic wave functions in the theory of molecules and crystals, *J. Chem. Phys.* 18, 365 (1950).
