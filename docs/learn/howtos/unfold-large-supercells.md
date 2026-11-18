# How to unfold the bands of large supercells with the sparse solver

On this page you unfold the bands of a supercell of hundreds to thousands of atoms with {py:func}`~PAOFLOW.models.band_unfold.unfold_bands_sparse`, which computes at each k-point only the `n_eigs` supercell states closest to an energy `sigma`, together with their spectral weights. You need it when the full diagonalisation of {py:func}`~PAOFLOW.models.band_unfold.unfold_bands`, described in [How to unfold supercell bands onto the primitive Brillouin zone](band-unfolding.md), runs out of memory or time. The weights of the returned states are the same as on the full route; the approximation is that only part of the spectrum is unfolded ([Partial spectra](../theory/band-unfolding.md#partial-spectra)). The route needs a tight-binding model whose hoppings are a function of the bond length.

:::{tip}
The model and the cells are built in code, so the page needs no input files. Download {download}`main.py <unfold-large-supercells/main.py>` to unfold your own large supercell: replace the model, the cell builder and the path with yours. Run it as `OMP_NUM_THREADS=1 python main.py`; it takes about 2 minutes on 4 cores. Run {download}`figures.py <unfold-large-supercells/figures.py>` after it, in the same directory, to redraw the figure of this page from that run.
:::

## A carbon model with distance-dependent hoppings

Give the supercell as an environment-dependent tight-binding (EDTB) model whose hoppings are distance dependent, because the sparse route evaluates each hopping from the bond length, not from one parameter set per neighbour shell as in the silicon model of the dense how-to. Each Slater–Koster channel of the carbon model below decays with the bond length $r$ in the form of L. Goodwin, A. J. Skinner and D. G. Pettifor, Europhys. Lett. 9, 701 (1989): its value `V0` at $r_0 = 2.684$ bohr, the nearest-neighbour distance of graphene, falls off as the power `n` of $r_0/r$, multiplied by an exponential decay whose length and sharpness are `r_c` and `n_c`. Energies are in eV and distances in bohr; the basis is $sp^3d^5$, and the screening by the atoms around each bond is described in [Environment-dependent Slater–Koster models](../../internals/sk-edtb.md).

```{literalinclude} unfold-large-supercells/main.py
:language: python
:end-before: '# The (n, m) supercell of AA bilayer graphene'
```

## Building the supercell

The system is AA-stacked bilayer graphene, 3.35 Å between the layers, in the $(n, m)$ supercell spanned by $n\mathbf{a}_1 + m\mathbf{a}_2$ and its $60^\circ$ rotation, which holds $n^2 + nm + m^2$ primitive cells per layer. The function below builds this cell with the lattice constant of graphene as `alat`. It places every atom on a primitive site shifted by a primitive lattice vector inside the cell, as the unfolding requires ([The atom mapping](../theory/band-unfolding.md#the-atom-mapping)):

```{literalinclude} unfold-large-supercells/main.py
:language: python
:start-after: '# The (n, m) supercell of AA bilayer graphene'
:end-before: '# Unfolding with the sparse solver'
```

## Unfolding with the sparse solver

Call `unfold_bands_sparse` with the same arguments as `unfold_bands`, plus the number of states and the energy at which to compute them:

```{literalinclude} unfold-large-supercells/main.py
:language: python
:start-after: '# Unfolding with the sparse solver'
:end-before: '# The edges of the window'
```

The solver is the shift-invert Lanczos method of SciPy's `eigsh`, an iterative method that converges first to the eigenvalues nearest `sigma` and needs only a sparse factorisation of the Hamiltonian at each k-point. The energy range of the `n_eigs` states it returns at a k-point is called the window below. The $(4, 3)$ cell holds $N = 37$ primitive cells of the bilayer, 148 atoms and 1332 orbitals. The full route would store its $H_{\alpha\beta}(\mathbf{R})$ as $1332^2$ complex numbers on each of 125 lattice vectors, 3.5 GB, a memory that grows as the square of the number of orbitals; the sparse route stores the list of bonds, which grows linearly with the number of atoms.

Start the script with `OMP_NUM_THREADS=1`, because `n_workers` distributes the k-points over processes, each running its own linear algebra, and several threads per process oversubscribe the cores. `n_workers=-1` uses all the cores available to the process (its share of the node under `mpirun`), with one factorisation in memory per worker. Before a larger cell, time one k-point and watch its memory, because the factorisation fills in and its cost grows faster than the number of orbitals ([Band unfolding: implementation](../../internals/band-unfolding.md#the-sparse-route)).

## The coordinates of the K point

Give K as $(\tfrac23, \tfrac13)$ when the primitive vectors are at $60^\circ$, as $\mathbf{a}_1 = a(1, 0)$ and $\mathbf{a}_2 = a(\tfrac12, \tfrac{\sqrt3}{2})$ are here, because the fractional coordinates of K depend on the angle between the primitive vectors. For vectors at $120^\circ$, K is $(\tfrac13, \tfrac13)$; for vectors at $60^\circ$, $(\tfrac13, \tfrac13)$ is the point two thirds of the way from $\Gamma$ to M, so a path through it misses the Dirac points. The module's `HEX_SYM_POINTS` uses $(\tfrac13, \tfrac13)$, so do not pass it for cells built at $60^\circ$.

## Choosing the window

Choose `sigma` inside the energy range you want to resolve, here the Dirac energy of the bilayer, and `n_eigs` large enough that the window covers that range at every k-point. The number of supercell states per eV is $N$ times that of the primitive cell, so the same `n_eigs` covers a narrower window in a larger cell, and the window is narrowest where the bands are densest. A larger `n_eigs` costs more per k-point. To check the window, compare its edges with the energy range you plot:

```{literalinclude} unfold-large-supercells/main.py
:language: python
:start-after: '# The edges of the window'
:end-before: '# Saving the bands, the weights and the supercell model'
```

Then save the results and the supercell model for `figures.py`:

```{literalinclude} unfold-large-supercells/main.py
:language: python
:start-after: '# Saving the bands, the weights and the supercell model'
```

![Unfolded bands of AA bilayer graphene from the (4, 3) supercell along Gamma-M-K-Gamma, with the edges of the energy window of the sparse solver](../../_static/images/howtos/unfold-large-supercells/bilayer_window.png)

The dark-blue lines are the bands of the primitive AA bilayer, the vermillion markers the supercell states with marker area proportional to $w_n(\mathbf{k})$, and the dashed amber lines the edges of the window at each k-point. The energy zero $E_\mathrm{D} = -0.519$ eV is the mean of the two Dirac points at K, which the interlayer hopping of AA stacking shifts to $E_\mathrm{D} \pm 0.094$ eV, one for the bonding and one for the antibonding combination of the layers. At every k-point the window covers at least $E_\mathrm{D} - 1.71$ to $E_\mathrm{D} + 1.75$ eV, and inside it every state with weight lies on a primitive band, as it must in a perfect supercell. Outside the window there is no weight even where the primitive bands run, for example on most of $\Gamma$–M, so a window narrower than the plot leaves blank regions that look like gaps.

Keep the range you interpret inside the window also because a degenerate multiplet at its edge can be cut, so that only some of its states, and part of its summed weight, are returned ([Degenerate multiplets](../theory/band-unfolding.md#degenerate-multiplets)). At $\Gamma$ the lower edge of this run falls on a six-fold multiplet, of which the window holds 4 states. For the same reason, and because the states outside the window are missing, the weights at a k-point do not add up to the number of primitive orbitals, as they do on the full route ([The sum rule](../theory/band-unfolding.md#the-sum-rule)).

## Troubleshooting

| message | meaning | what to do |
| --- | --- | --- |
| `eigsh failed to converge for ... eigenvalues near sigma=...` | The Lanczos iterations did not converge at some k-point of a cell above 6000 orbitals; smaller cells fall back to a full diagonalisation. | Lower `n_eigs`, or pass a larger `maxiter` or `ncv`, which `unfold_bands_sparse` forwards to `eigsh`. |
| `No SC atom found matching PC atom ... Check positions/lattice vectors.` | The supercell atoms are not primitive sites shifted by a lattice vector. | See the troubleshooting table of [How to unfold supercell bands onto the primitive Brillouin zone](band-unfolding.md#troubleshooting). |

## Where to go next

- [Band unfolding: spectral weights of supercell states](../theory/band-unfolding.md) derives the weights and explains what a partial spectrum leaves out.
- [How to compute the band structure of twisted bilayer graphene](twisted-bilayer.md) uses the same carbon model and solver for cells that cannot be unfolded.
- [Environment-dependent Slater–Koster models](../../internals/sk-edtb.md) describes the EDTB models and their fitting.
- [Band unfolding: implementation](../../internals/band-unfolding.md) describes the sparse route, its parallel distribution and its memory.
