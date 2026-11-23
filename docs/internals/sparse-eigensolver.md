# The eigensolver of the sparse backend

The sparse backend (`PAOFLOW(..., sparse=True)`) never stores $\tilde H(\mathbf{k})$ on the k-mesh. At each k-point it assembles $\tilde H(\mathbf{k})$ from the hopping list, solves for the states the properties of the pass need, and frees the matrix and its eigenvectors before the next k-point. This page explains how that one eigenproblem is solved, which choice the backend makes for you, and where the backend stops. The user-facing pages are [Tutorial 04](../learn/tutorials/tutorial04.ipynb), which runs the backend, and [How to choose which states the sparse backend solves](../learn/howtos/sparse-states.md), which sets the number of states that this choice depends on.

## The two solvers

```python
sparse_config={'hk_solver': 'auto'}   # 'auto' | 'sparse' | 'dense'
```

`hk_solver` chooses how the eigenproblem at one k-point is solved, and nothing else: with either solver $H(\mathbf{R})$ stays a hopping list and no array indexed by k is formed.

- `'dense'` builds the $n_\mathrm{awf}\times n_\mathrm{awf}$ matrix $\tilde H(\mathbf{k})$ of the current k-point and diagonalises it with the LAPACK driver `zheevr`, which returns only the requested lowest states. Its time grows as $n_\mathrm{awf}^3$ and its memory as $n_\mathrm{awf}^2$, whatever the number of states needed.
- `'sparse'` uses the iterative ARPACK solver in shift-invert mode, with the shift placed below the spectrum by the Gershgorin theorem. It applies the factorised $\tilde H(\mathbf{k})$ to vectors and never forms the dense matrix. Its cost grows with the number of states requested, and once that is a sizeable part of the spectrum it is slower than a full diagonalisation and needs as much memory.
- `'auto'`, the default, takes ARPACK when the requested states, plus four guard states, are less than `dense_ratio` of the spectrum (default 1/8), and LAPACK otherwise. The rule is given in {py:func}`~PAOFLOW.sparse.solver.select_hk_solver`.

The choice depends only on $n_\mathrm{awf}$ and the number of states, never on the matrix elements, so it is made once per pass and printed in `sparse.log`:

```text
  H(k) solver            dense (LAPACK, one 144 x 144 matrix per k-point)
  per-k scratch (MB)     0.3
  solved fraction nev/n  0.43
```

## Why ARPACK runs only for a small part of the spectrum

A window that starts at the bottom of the spectrum must reach the Fermi level, so it holds a fixed fraction of all states (62 of 144 for the 16-atom Si cell of Tutorial 04), and that fraction does not shrink when the cell grows: folding a cell $N$ times multiplies both the number of states below a fixed energy and $n_\mathrm{awf}$ by $N$. In that regime ARPACK gives no saving. For $k$ requested states, scipy chooses a Krylov space of dimension $\max(2k+1, 20)$, which reaches $n_\mathrm{awf}$ itself once $k$ is above about half the spectrum, so ARPACK then allocates a dense $n_\mathrm{awf}\times n_\mathrm{awf}$ basis and is 13 to 66 times slower than `zheevr` (measured for $n_\mathrm{awf}$ = 144, 576 and 1152). Only an interior window, set with `interior_window` (see [How to choose which states the sparse backend solves](../learn/howtos/sparse-states.md)), makes the solved fraction small enough for ARPACK to pay.

A second reason is accuracy at degeneracies. Doubling a cell $N$ times folds $N$ copies of each band onto the same k-point, so exact multiplets of 8 to 64 states are common. ARPACK with the default Krylov dimension can return fewer copies of a multiplet than its multiplicity without raising an error (15 of 32, 7 of 16 and 6 of 8 copies in tests), and the copies it returns are converged eigenvectors that are not orthogonal to each other. The backend therefore starts ARPACK with a Krylov dimension of $\max(4k+1, 40)$, which returned the full multiplicity in every case tested, and orthonormalises each degenerate group by a QR decomposition, because projected DOS weights and the velocities of degenerate states assume an orthonormal basis. `zheevr` returns an orthonormal basis of every cluster by construction.

## The limit `dense_n_max`

A full diagonalisation is allowed up to `dense_n_max` orbitals (default 4096, about 270 MB per matrix). Past that size, a solve that needs more than `dense_ratio` of the spectrum stops with the error

```text
solve_lowest: nev+guard = ... of n = ... is past the iterative regime (> 12.5%), but n exceeds dense_n_max = 4096 ...
```

The two ways out are fewer states per k-point (a narrower property range, `nev` or `interior_window`), or a distributed eigensolver, which the backend does not provide. The same limit applies to the properties that sum over every state (the Hall conductivities, the Berry curvature, linear response), because they need the full spectrum at every k-point, and to the 16 sample diagonalisations that size the solve window. Raise `dense_n_max` if the memory per rank allows a larger matrix.

## Comparing the two solvers

Leave `hk_solver` on `'auto'`, because it makes the choice for each pass. Set `'dense'` or `'sparse'` only to compare the two solvers on a small system with everything else equal: away from degeneracies the eigenvalues agree to the convergence tolerance of ARPACK, and quantities summed over a degenerate group (DOS, transport) agree as well, while quantities of single states inside a degenerate group depend on the basis chosen in the group and differ between the solvers and between runs.
