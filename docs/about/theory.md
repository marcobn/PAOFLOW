# Theory

PAOFLOW is built around one central idea: keep the predictive power of first-principles Density Functional Theory (DFT), but move to a compact representation that is fast enough for dense Brillouin-zone sampling and large property workflows.

Instead of fitting empirical tight-binding parameters, PAOFLOW projects _ab initio_ Bloch states onto an atomic-orbital basis and constructs Hamiltonians directly from the DFT wavefunctions. This gives a model that remains physically grounded while enabling efficient calculations of transport, optical, and topological observables.

## 1. From Bloch States to a PAO Hamiltonian

Starting from a converged plane-wave DFT calculation, PAOFLOW uses Kohn-Sham eigenvalues $E_n(\mathbf{k})$ and eigenstates $\psi_{n\mathbf{k}}$.

The projection step computes overlap amplitudes between Bloch states and atomic-like basis functions $\phi_{m\mathbf{k}}$:

$$
A_{mn}(\mathbf{k}) = \langle \phi_{m\mathbf{k}} \mid \psi_{n\mathbf{k}} \rangle.
$$

This projection defines a finite Hilbert-space representation where the Hamiltonian can be expressed in the PAO subspace. In practice, PAOFLOW combines this with orthogonalization and projectability filtering to remove poorly represented states and keep an accurate low-cost model.

## 2. Finite Hilbert Space and Projectability

The formal basis of the method is developed in the finite-Hilbert-space representation papers. Two practical ideas are essential:

1. **Projectability as a quality metric.** Each DFT band has a projection weight onto the chosen PAO space. Low projectability indicates that a band is not well represented by the basis.
2. **Basis optimization/minimality.** The PAO set is chosen to keep the Hamiltonian compact while preserving spectral accuracy in the energy window of interest.

A simplified projectability indicator is:

$$
p_n(\mathbf{k}) = \sum_m \left|A_{mn}(\mathbf{k})\right|^2.
$$

Bands with insufficient $p_n(\mathbf{k})$ are excluded or treated carefully to avoid artifacts in interpolated properties.

## 3. Real-Space Form and k-Space Interpolation

Once $H(\mathbf{k})$ is available on a coarse mesh, PAOFLOW maps to real-space hopping matrices and back to arbitrarily dense k-grids:

$$
H(\mathbf{R}) = \frac{1}{N_k}\sum_{\mathbf{k}} e^{-i\mathbf{k}\cdot\mathbf{R}} H(\mathbf{k}),
$$

$$
H(\mathbf{k}') = \sum_{\mathbf{R}} e^{i\mathbf{k}'\cdot\mathbf{R}} H(\mathbf{R}).
$$

This is the core reason PAOFLOW is efficient: expensive first-principles information is compressed once, then reused for dense sampling and many observables.

## 4. Physics You Can Compute from the Same Hamiltonian

From the interpolated PAO Hamiltonian, PAOFLOW evaluates derivatives and matrix elements needed for broad classes of observables:

1. **Electronic structure:** band structures, DOS/PDOS, Fermiology.
2. **Topological quantities:** Berry curvature and Hall-type responses (including spin Hall workflows).
3. **Optical response:** dielectric and optical tensors from first-principles-derived matrix elements.
4. **Transport coefficients:** conductivity and thermoelectric quantities via Boltzmann-type post-processing.

Because all quantities come from a common Hamiltonian, cross-property comparisons are consistent by construction.

## 5. Why This Matters for 2D and Layered Materials

For low-dimensional systems, accurate interpolation can be challenging because fine features near band crossings, valleys, and anisotropic dispersions strongly affect measurable quantities. The PAOFLOW formalism addresses this by combining:

1. Controlled basis construction in the physically relevant orbital manifold.
2. Robust interpolation from coarse DFT meshes to dense k-space.
3. Direct access to velocity-related and Berry-phase-related quantities needed for transport and topology.

This makes the approach practical for high-throughput and targeted studies of layered materials where both accuracy and speed are required.

## References

1. Cerasoli, F. _et al._ (2021). Advanced modeling of materials with PAOFLOW 2.0: New features and software design. _Computational Materials Science_, 200, 110828. https://doi.org/10.1016/j.commatsci.2021.110828
2. Agapito, L. A., Curtarolo, S., & Buongiorno Nardelli, M. (2018). PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the projections of electronic wavefunctions on atomic orbital bases. _Computational Materials Science_, 143, 462-466. https://doi.org/10.1016/j.commatsci.2017.11.029
3. Agapito, L. A. _et al._ (2013). Effective and accurate representation of extended Bloch states on finite Hilbert spaces. _Physical Review B_, 88, 165127. https://doi.org/10.1103/PhysRevB.88.165127
4. Agapito, L. A. _et al._ (2016). Accurate tight-binding Hamiltonian matrices from ab-initio calculations: Minimal basis sets. _Physical Review B_, 93, 035104. https://doi.org/10.1103/PhysRevB.93.035104
5. Agapito, L. A. _et al._ (2016). Accurate tight-binding Hamiltonians for 2D and layered materials. _Physical Review B_, 93, 125137. https://doi.org/10.1103/PhysRevB.93.125137
6. Supka, A. R. _et al._ (2016). Accurate ab initio tight-binding Hamiltonians: Effective tools for electronic transport and optical spectroscopy from first principles. _Physical Review B_, 94, 165166. https://doi.org/10.1103/PhysRevB.94.165166
