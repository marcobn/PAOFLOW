# How to choose the truncation of the sparse Hamiltonian

On this page you choose which hoppings of the PAO Hamiltonian $H_{\alpha\beta}(\mathbf{R})$ the sparse backend keeps. The backend stores $H_{\alpha\beta}(\mathbf{R})$ as a list of hoppings between pairs of pseudo-atomic orbitals (PAOs) and drops the small ones once, at the end of `pao_hamiltonian()`, so the truncation you give in `sparse_config` fixes the error of every band and property of the run. Truncate by bond length, with `bond_order` or `rcut`, because a bond-length cutoff keeps the symmetry of the crystal, and converge the cutoff on the bands near the energies your properties use. A cut by the size of the hoppings, `hopping_threshold`, is for quick tests. [Tutorial 04](../tutorials/tutorial04.ipynb) shows the workflow in which the truncation sits; the examples below use its Si calculation.

:::{tip}
The input is `tutorial01/silicon.save` in `tutorial_assets.tar.gz` of the release [tutorial-assets-v1](https://github.com/marcobn/PAOFLOW/releases/tag/tutorial-assets-v1). Extract it with `tar xzf tutorial_assets.tar.gz` and run {download}`main.py <sparse-truncation/main.py>` from the same directory. It computes the Si bands with the full Hamiltonian and 14 truncations, in about a minute on 4 cores. Run {download}`figures.py <sparse-truncation/figures.py>` after it, in the same directory, to draw the figures of this page.
:::

## The change

The truncation is an entry of `sparse_config` in the constructor. The script computes the Si bands of each run with this function, in which only `sparse_config` changes:

```{literalinclude} sparse-truncation/main.py
:language: python
:end-before: '# Truncate by neighbour shells'
```

```{literalinclude} sparse-truncation/main.py
:language: python
:start-after: '# Truncate by neighbour shells'
:end-before: '# Keep every hopping'
```

Because the hoppings are dropped inside `pao_hamiltonian()`, the truncation cannot change later in the run: to compare two truncations, start two runs.

## Which hoppings are small

Plot the hoppings against bond length to see which ones a truncation can drop. `save_sparse_hamiltonian()` in the function above writes the kept hoppings to `hamiltonian.npz` in the output directory, and a threshold of zero keeps all of them:

```{literalinclude} sparse-truncation/main.py
:language: python
:start-after: '# Keep every hopping'
:end-before: '# One shell beyond the R grid'
```

$H(\mathbf{R})$ built from the projections is the exact discrete Fourier transform of $H(\mathbf{k})$ on the DFT k-grid, not a maximally localised model, so it has small but nonzero hoppings out to the edge of the $\mathbf{R}$ grid. The figure shows every element $|H_{\alpha\beta}(\mathbf{R})|$ of the full Si Hamiltonian on its $12\times12\times12$ DFT grid against the length of the bond it connects, $|a\,\mathbf{R} + \boldsymbol{\tau}_\alpha - \boldsymbol{\tau}_\beta|$, where $a$ is the lattice parameter and $\boldsymbol{\tau}_\alpha$, $\boldsymbol{\tau}_\beta$ are the positions of the atoms that carry orbitals $\alpha$ and $\beta$. The vertical lines are the cutoffs of `bond_order=10` and `bond_order=36` and the longest cutoff the grid allows, and the horizontal lines the thresholds of $10^{-3}$ and $10^{-4}$ eV.

![Magnitude of every Si PAO hopping against bond length on a log scale, with the bond_order cutoffs and the longest cutoff of the grid as vertical lines and two hopping thresholds as horizontal lines](../../_static/images/howtos/sparse-truncation/hoppings.png)

The hoppings fall in vertical columns, one per neighbour shell, the set of bonds of one length, marked by the ticks at the top. They decay with the length of the bond, but slowly: the nearest-neighbour shell at 4.59 Bohr has hoppings up to 1.27 eV, and hoppings of $3.7\times10^{-3}$ eV remain beyond 30 Bohr. Within one shell the elements spread over several orders of magnitude, because each couples a different pair of orbitals. A truncation is therefore either a vertical line in this figure, a cut by bond length, or a horizontal line, a cut by size, and the two keep different hoppings.

## Truncation by bond length

Use a cutoff on the bond length, because it keeps the symmetry of the crystal. A symmetry operation maps a bond only onto bonds of the same length, so a cutoff keeps or drops whole shells, and the degeneracies the symmetry protects stay intact. This matters for every property that depends on them: band crossings, topological invariants, and tensor components that vanish by symmetry.

![Two close-ups of the doubly degenerate Si conduction-band pair along Gamma to L: with bond_order=36 the pair stays degenerate but moves up, with a hopping threshold of 1e-3 eV it splits near L](../../_static/images/howtos/sparse-truncation/degeneracy.png)

The figure shows the doubly degenerate conduction-band pair of Si on the second half of the $\Gamma$–L line, 3.390 eV above the valence-band maximum (VBM) at L, with the full Hamiltonian (grey) and two truncations. With `bond_order=36` the pair stays degenerate, 0.02 meV apart at L, but moves up by 46 meV. With a threshold of $10^{-3}$ eV (see [below](#hopping_threshold-a-quick-truncation-for-tests)) it stays closer to the full bands but splits by 11.9 meV at L.

The price of the symmetry is accuracy, which depends on how fast the hoppings decay with distance. In a covalent semiconductor such as Si the decay is slow: the cutoff at the 36th shell (32.1 Bohr) still drops 1956 hoppings above $10^{-3}$ eV, and it moves the bands within 2 eV of the VBM by up to 93 meV.

The cutoff also has an upper limit. It reaches only as far as the $\mathbf{R}$ grid represents each bond by a single lattice vector, 36.7 Bohr for the $12\times12\times12$ grid of Si (the dotted line in the first figure). Beyond that radius one grid value of $H(\mathbf{R})$ mixes bonds of different lengths, so a length has no meaning there, and a longer cutoff stops the run with an error that gives the limit. On the Si grid the longest allowed cutoff still moves the bands near the gap by up to 70 meV. To reach further, use a denser DFT k-grid, which extends the $\mathbf{R}$ grid. You give the cutoff in one of two ways.

### `bond_order`: a number of neighbour shells

`bond_order=n` keeps the bonds up to the $n$-th distinct interatomic distance: 1 is the nearest neighbours (4.59 Bohr in Si), 2 the second neighbours. Because the shells move with the lattice, the same `bond_order` describes the same physical truncation along a series of strained or relaxed structures, which makes it the choice for comparing them. On the $12\times12\times12$ grid of Si the largest allowed value is 47, and one shell more stops the run with an error that gives the limit:

```{literalinclude} sparse-truncation/main.py
:language: python
:start-after: '# One shell beyond the R grid'
:end-before: '# Truncate by radius'
```

### `rcut`: a radius

```{literalinclude} sparse-truncation/main.py
:language: python
:start-after: '# Truncate by radius'
:end-before: '# Converge the cutoff against the full Hamiltonian'
```

`rcut` fixes the range in Bohr, the natural choice to compare materials with different shell structures. The backend moves the radius down into the gap above the outermost shell it reaches, so that it never keeps part of a shell, and `sparse.log` in the output directory records both values:

```text
  cutoff radius (Bohr)   30.000 requested, 29.981 applied (moved into the gap above the last shell)
```

## Choosing the cutoff

Raise `bond_order` or `rcut` until the bands near the energies your property uses stop moving, within the limit of the $\mathbf{R}$ grid. Judge the bands rather than the DOS, because the integration over the Brillouin zone averages band shifts of opposite sign, so a DOS can look converged while single bands have moved. For Si, the script repeats the run for a series of `bond_order` values:

```{literalinclude} sparse-truncation/main.py
:language: python
:start-after: '# Converge the cutoff against the full Hamiltonian'
:end-before: '# Truncate by hopping size'
```

![Largest band shift within 2 eV of the Si VBM against bond_order on a log scale, falling from 564 meV to 70 meV, with the two hopping thresholds as horizontal lines](../../_static/images/howtos/sparse-truncation/convergence.png)

The figure shows this test for Si: the largest shift of the bands within 2 eV of the VBM along the fcc path, against the full Hamiltonian, as only `bond_order` changes. The shift falls from 564 meV at `bond_order=5` to 70 meV at 44 and 47, but not steadily: `bond_order=33` gives 81 meV, while 36 and 40 give 93 and 97 meV. Therefore compare several cutoffs before you accept one, rather than stopping at the first that looks converged. The dashed lines mark the two hopping thresholds: on this grid $10^{-4}$ eV gives smaller shifts than any cutoff, at the price of split degeneracies.

`sparse.log` also prints a bound on the largest eigenvalue shift the truncation can cause at any k-point:

```text
  max eigenvalue shift from truncation (eV)  4.11
```

The bound follows from the Gershgorin theorem: the dropped part of $H(\mathbf{k})$ is Hermitian, so its spectral norm is at most its largest absolute row sum. It is rigorous but loose: for `bond_order=36` on Si it is 4.11 eV, where the largest shift of the lowest eight bands is 0.26 eV. A small bound therefore guarantees a small error, while a large one means that you have to measure the error as above.

Always give `bond_order` or `rcut`, because without either the backend applies the threshold of $10^{-3}$ eV described next.

## `hopping_threshold`: a quick truncation for tests

```{literalinclude} sparse-truncation/main.py
:language: python
:start-after: '# Truncate by hopping size'
```

`hopping_threshold` drops every element with $|H_{\alpha\beta}(\mathbf{R})|$ below the threshold. For a quick test of a workflow it is easier to set than a cutoff, because it needs no knowledge of the shells or of the limit of the $\mathbf{R}$ grid. On Si, $10^{-4}$ eV moves the bands within 2 eV of the VBM by up to 11 meV, and the default $10^{-3}$ eV by up to 91 meV.

However, a threshold cuts through the shells and so breaks the crystal symmetry. Two bonds related by a symmetry operation connect orbitals that are rotated into each other, so the same coupling appears as elements of different size on the two bonds, and the threshold keeps a different set of elements on each. As a result, bands that the symmetry makes degenerate split, by 11.9 meV at L in the figure above. Keep the threshold for tests and exploratory runs. For the same reason it cannot be combined with `bond_order` or `rcut`.

## Troubleshooting

| message | meaning | what to do |
| --- | --- | --- |
| `rcut = ... Bohr exceeds the aliasing-safe radius ...` or `bond_order=... exceeds the ... neighbour shells representable` | the cutoff is longer than the $\mathbf{R}$ grid represents | use a smaller cutoff or a denser DFT k-grid |
| `hopping_threshold cannot be combined with rcut or bond_order` | a threshold and a bond-length cutoff were both given | give one of them |

## Where to go next

- [How to choose which states the sparse backend solves](sparse-states.md), the second approximation of the backend.
- {py:class}`~PAOFLOW.sparse.config.SparseConfig` lists every option of `sparse_config` with its default.
