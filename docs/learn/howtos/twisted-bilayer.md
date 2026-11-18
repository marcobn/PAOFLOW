# How to compute the band structure of twisted bilayer graphene

On this page you build the commensurate cells of twisted bilayer graphene, compute their bands near the Dirac energy, and follow the Dirac velocity as the twist angle $\theta$ decreases. Small angles need cells of thousands of atoms, too many for the dense diagonalisation of [Tutorial 01](../tutorials/tutorial01.ipynb), so a sparse solver computes only the states near the Dirac energy. The Hamiltonian is a tight-binding model, and its layers are rigid and flat: the relaxation of real samples, which changes the bands at small angles, is not included.

The model is the [environment-dependent tight-binding (EDTB) model of carbon](unfold-large-supercells.md#a-carbon-model-with-distance-dependent-hoppings) of the how-to on unfolding large supercells. The script holds its own copy of it, the `carbon` dictionary, and builds the cells in code, so it needs no input files.

```{literalinclude} twisted-bilayer/main.py
:language: python
:end-before: '# The commensurate cell (n, m)'
```

:::{tip}
Download {download}`main.py <twisted-bilayer/main.py>` to compute the bands and the Dirac velocity of other cells or with your own model, and run it as `OMP_NUM_THREADS=1 python main.py > main.log`; its two cells take 6 minutes on 4 cores. Run {download}`figures.py <twisted-bilayer/figures.py>` after it, in the same directory, to redraw the figures of this page from that run. To redraw them without the calculation, run `figures.py` inside the folder `twisted-bilayer/` of `howto_assets.tar.gz` in the release [tutorial-assets-v2](https://github.com/marcobn/PAOFLOW/releases/tag/tutorial-assets-v2), which holds the outputs of `main.py`.
:::

## Commensurate cells

A twisted bilayer is a crystal only at the angles where a lattice vector of the lower layer is also a lattice vector of the rotated upper layer. Each pair of integers $n > m \ge 0$ gives one such angle, at which $n\mathbf{a}_1 + m\mathbf{a}_2$ of the lower layer coincides with $m\mathbf{a}_1 + n\mathbf{a}_2$ of the upper one, and a cell of $n^2 + nm + m^2$ primitive cells per layer ([Commensurate twist angles](../theory/band-unfolding.md#commensurate-twist-angles)). Each atom carries nine $sp^3d^5$ orbitals. Use the cells $(n, n - 1)$, because for them the cell is one period $L$ of the moiré pattern, the smallest cell for its angle ([The moiré period](../theory/band-unfolding.md#the-moiré-period)):

| $(n, m)$ | $\theta$ (deg) | atoms | orbitals | $L$ (Å) |
| --- | --- | --- | --- | --- |
| (2, 1) | 21.79 | 28 | 252 | 6.5 |
| (4, 3) | 9.43 | 148 | 1332 | 15.0 |
| (6, 5) | 6.01 | 364 | 3276 | 23.5 |
| (10, 9) | 3.48 | 1084 | 9756 | 40.5 |
| (15, 14) | 2.28 | 2524 | 22716 | 61.8 |
| (17, 16) | 2.00 | 3268 | 29412 | 70.3 |
| (20, 19) | 1.70 | 4564 | 41076 | 83.1 |
| (31, 30) | 1.08 | 11164 | 100476 | 129.9 |

The function below builds the $(n, m)$ cell with the lattice constant of graphene, $a = 2.46$ Å, as `alat`, the lower layer at $z = 0$ and the upper layer rotated about the origin, a site of both layers:

```{literalinclude} twisted-bilayer/main.py
:language: python
:start-after: '# The commensurate cell (n, m)'
:end-before: '# Dirac points of the AA bilayer at K, split by the interlayer hopping'
```

![Top view of the (6, 5) twisted bilayer with its moiré cell, AA stacking at the corners](../../_static/images/howtos/twisted-bilayer/moire_6_5.png)

The figure shows `twisted_bilayer(6, 5)` repeated in the plane, with one cell outlined. At its corners the atoms of the two layers lie on top of each other (AA stacking); between them the stacking passes through AB and BA regions, where half of the upper atoms sit over the centres of the lower hexagons. The pattern repeats every 23.5 Å, the $L$ of the table.

## The interlayer distance

Set the interlayer distance `d` by the interlayer coupling it gives in your model, because the coupling decides at which angles the bands flatten: the angle at which the Dirac velocity vanishes is proportional to the interlayer tunnelling ([Dirac velocity in the continuum model](../theory/band-unfolding.md#dirac-velocity-in-the-continuum-model)). The default `d=2.68` Å suits this carbon model, whose hoppings between the layers are weak at the spacing of real bilayers, 3.35 Å. In the AA-stacked bilayer the interlayer hopping shifts the Dirac cones of the two layers apart, and its two Dirac points at K lie 0.188 eV apart at 3.35 Å and 0.684 eV apart at 2.68 Å. The shorter distance therefore moves the flat bands to larger angles, whose cells are smaller. Before you compare angles with experiment, check this coupling against the AA splitting from DFT. This block computes both splittings:

```{literalinclude} twisted-bilayer/main.py
:language: python
:start-after: '# Dirac points of the AA bilayer at K, split by the interlayer hopping'
:end-before: '# Bands of the (4, 3) and (6, 5) cells'
```

## Bands with the sparse solver

Compute the bands with {py:class}`~PAOFLOW.spectrum.sparse_bands.SparseEDTB`, which stores the Hamiltonian of the cell as a sparse matrix of bonds and computes, at each k-point, only the `n_eigs` eigenvalues closest to the energy `sigma` (eV), with the shift-invert Lanczos method of SciPy's `eigsh`. The energy range of these states is called the window below. The script runs it for the cells $(4, 3)$ and $(6, 5)$:

```{literalinclude} twisted-bilayer/main.py
:language: python
:start-after: '# Bands of the (4, 3) and (6, 5) cells'
:end-before: '# A short segment from the moire K point towards G'
```

The path is in fractional coordinates of the reciprocal lattice of the cell, so G, M and K are points of the moiré Brillouin zone; because the two cell vectors are at $60^\circ$, K is $(\tfrac23, \tfrac13)$ (see [the coordinates of the K point](unfold-large-supercells.md#the-coordinates-of-the-k-point)). `nk` counts the k-points of the whole path. Set `sigma` to the Dirac energy of the model: `sigma=-0.52` is the mean of the two Dirac points of the AA bilayer at 2.68 Å ($-0.521$ eV) printed by the block above.

Choose `n_eigs` so that the window covers your plot at every k-point. The states per eV grow with the number of atoms, so 40 states span at least $E_\mathrm{D} - 1.623$ to $E_\mathrm{D} + 1.678$ eV at every k-point of the $(4, 3)$ path, but only $E_\mathrm{D} - 1.084$ to $E_\mathrm{D} + 1.151$ eV for $(6, 5)$ ($E_\mathrm{D}$ is defined below the figure). To check a run, compare these edges with your plot.

`n_workers=-1` distributes the k-points over the cores of the process (its share of the node under `mpirun`), with at most four workers above 5000 orbitals (see [Smaller angles](#smaller-angles)). Start the script with `OMP_NUM_THREADS=1`, because several threads per worker oversubscribe the cores and slow the run down. `compute_bands` writes the bands to `bands_0.dat` in `outputdir` and the high-symmetry points to `kpath_points.txt`.

![Bands of twisted bilayer graphene at 9.43 and 6.01 degrees along K-Gamma-M-K, with the Dirac cone of one layer](../../_static/images/howtos/twisted-bilayer/bands.png)

Plot the eigenvalues as markers, because the window moves with k, so the $j$-th eigenvalue need not follow one band. The energy zero $E_\mathrm{D}$ is the mean of the four states at the moiré K point, onto which one Dirac point of each layer folds ([Why a twisted cell does not unfold](../theory/band-unfolding.md#why-a-twisted-cell-does-not-unfold)): $-0.521$ eV for $(4, 3)$ and $-0.519$ eV for $(6, 5)$. The dashed lines are the Dirac cone of one layer of the same model, with the velocity $v_0$ fitted below, drawn from each K.

In the figure, the cones of the two layers meet at M, where the interlayer coupling turns their crossing into a saddle point, and the bands flatten as the angle decreases. At M, the bands above and below $E_\mathrm{D}$ lie at 0.401 and $-0.373$ eV at $9.43^\circ$, but at 0.216 and $-0.191$ eV at $6.01^\circ$: their separation shrinks by a factor of 1.90, faster than the moiré zone (1.57), so the Dirac velocity drops as well.

## The velocity at the Dirac point

To measure that drop, fit the renormalised velocity $v^*$ at the Dirac point: $v^*/v_0 = 1$ for decoupled layers, and $v^*$ vanishes at the magic angle. Compute the bands on a segment from K towards $\Gamma$, 4 % of its length,

```{literalinclude} twisted-bilayer/main.py
:language: python
:start-after: '# A short segment from the moire K point towards G'
:end-before: '# Slope of the Dirac cone'
:dedent: 4
```

and fit the slope of the lower of the two bands that leave the Dirac point upwards:

```{literalinclude} twisted-bilayer/main.py
:language: python
:start-after: '# Slope of the Dirac cone'
:end-before: '# Continuum model of Bistritzer and MacDonald'
```

The k-distances are in radians per `alat`, so the slope is $\hbar v^*$ in eV $\cdot$ `alat`. The fit leaves out the first point after K, because the four states at K are split by up to 1.18 meV, which distorts the slope closest to K. On one layer of the model, on a segment 0.4 % of K–$\Gamma$ long so that $|\mathbf{k} - \mathbf{K}|$ is as small as in the moiré cells, the same fit gives $\hbar v_0 = 1.603$ eV $\cdot$ `alat` $= 3.94$ eV Å, or $v_0 = 5.99 \times 10^5$ m/s. The twisted cells give $\hbar v^* = 1.376$ eV $\cdot$ `alat` at $9.43^\circ$ and 1.177 at $6.01^\circ$, so $v^*/v_0 = 0.858$ and 0.734.

The continuum model of R. Bistritzer and A. H. MacDonald, Proc. Natl. Acad. Sci. USA 108, 12233 (2011), predicts how $v^*$ continues to smaller angles. In it, $v^*/v_0$ depends only on the ratio $\alpha$ of the interlayer tunnelling to the kinetic energy across the moiré zone, which grows as $1/\theta$ at small angles, and vanishes at $\alpha = 1/\sqrt3$ ([Dirac velocity in the continuum model](../theory/band-unfolding.md#dirac-velocity-in-the-continuum-model)). The last block fits $\alpha$ to the $9.43^\circ$ cell, which gives $\alpha = 0.132$, and scales it as $1/\theta$:

```{literalinclude} twisted-bilayer/main.py
:language: python
:start-after: '# Continuum model of Bistritzer and MacDonald'
```

The model then gives $v^*/v_0 = 0.693$ at $6.01^\circ$, 0.041 below the computed value: the tight-binding velocity drops more slowly than the model predicts. The velocity vanishes at $\theta = 2.16^\circ$, or $1.97^\circ$ when $\alpha$ is fitted to the $6.01^\circ$ cell instead; the difference is the uncertainty of the estimate, and the $(17, 16)$ cell at $2.00^\circ$ lies within it. With the real interlayer distance the magic angle would be smaller, because the coupling is weaker.

![Dirac velocity against twist angle: two tight-binding cells and the continuum model, which vanishes at 2.16 degrees](../../_static/images/howtos/twisted-bilayer/velocity.png)

The markers are the two cells, the line the continuum model fitted at $9.43^\circ$, and the dashed line the angle where it vanishes. A cell such as $(10, 9)$ at $3.48^\circ$ would test the extrapolation.

## Smaller angles

Plan a run at a smaller angle from the time of one k-point, because the cost per k-point grows faster than the number of orbitals: the shift-invert method factorises the Hamiltonian at each k-point, and the factors of a sparse matrix fill in. Cells such as $(10, 9)$, or $(17, 16)$ near the magic angle of this model, therefore belong on a cluster node. Time one k-point first, with `bands.eigvals_at_k([2 / 3, 1 / 3, 0], n_eigs=40, sigma=-0.52)`, and watch its memory, then choose `nk` and the number of cores.

The memory of the factors also limits the parallelism: above 5000 orbitals `n_workers=-1` starts at most four workers, and above 20000 orbitals it runs the k-points one after the other. Pass a larger `n_workers` only when the memory of one k-point times the number of workers fits in the node. To lower the cost per k-point, reduce `n_eigs` instead: 8 states are enough for the velocity, 40 for the bands within a few hundred meV of $E_\mathrm{D}$ at these angles.

These bands cannot be unfolded onto the Brillouin zone of graphene with {py:func}`~PAOFLOW.models.band_unfold.unfold_bands`, because the atoms of the rotated layer are not primitive sites shifted by a lattice vector, as the unfolding requires ([Why a twisted cell does not unfold](../theory/band-unfolding.md#why-a-twisted-cell-does-not-unfold)).

## Where to go next

- [Band unfolding: spectral weights of supercell states](../theory/band-unfolding.md#twisted-bilayers) derives the commensurate angles, the moiré period and the continuum-model velocity.
- [How to unfold the bands of large supercells with the sparse solver](unfold-large-supercells.md) introduces the carbon model and the sparse solver.
- [Environment-dependent Slater–Koster models](../../internals/sk-edtb.md) describes the EDTB models and their fitting.
