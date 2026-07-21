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

| Domain | What PAOFLOW computes |
|---|---|
| **Electronic structure** | Band structures, density of states (total & projected), Fermi surfaces |
| **Optical & dielectric response** | Complex dielectric tensor ε(ω), optical conductivity, joint density of states; non-local velocity correction for norm-conserving pseudopotentials |
| **Transport** | Electrical conductivity, Seebeck coefficient, electronic thermal conductivity (Boltzmann transport) |
| **Lattice dynamics & phonons** | Phonon dispersions, DOS and thermal properties ([phonopy](https://phonopy.github.io/phonopy/) finite-displacement); Born effective charges, ε∞ and LO–TO splitting; infrared (IR), non-resonant (Placzek) and resonant (Albrecht) Raman spectra; vibrational (ionic) dielectric ε(ω) and reststrahlen emissivity; quasi-harmonic approximation (thermal expansion, V(T), bulk modulus, C_p, thermodynamic and mode Grüneisen dispersion) |
| **Topology** | Berry curvature, anomalous Hall conductivity, Z₂ invariants, topological surface states |
| **Spin & magnetism** | Spin Hall conductivity, spin texture, non-collinear and fully-relativistic (SOC) Hamiltonians |
| **Model Hamiltonians** | Slater–Koster tight-binding models, Kane–Mele, custom lattice models |
| **ACBN0** | Self-consistent Hubbard U and U+V via the extended ACBN0 functional |
| **pyskeaf** | Fermi surface extremal orbit analysis (de Haas–van Alphen, Shubnikov–de Haas) |
| **Landauer transport** | Quantum transport via Green's function/Landauer–Büttiker formalism |
| **Interoperability** | Quantum ESPRESSO and VASP DFT code integration - other codes are in the development pipline (we welcome contributions from developers!)|

Because all quantities come from a common Hamiltonian, cross-property comparisons are consistent by construction.

## 5. Why This Matters for 2D and Layered Materials

For low-dimensional systems, accurate interpolation can be challenging because fine features near band crossings, valleys, and anisotropic dispersions strongly affect measurable quantities. The PAOFLOW formalism addresses this by combining:

1. Controlled basis construction in the physically relevant orbital manifold.
2. Robust interpolation from coarse DFT meshes to dense k-space.
3. Direct access to velocity-related and Berry-phase-related quantities needed for transport and topology.

This makes the approach practical for high-throughput and targeted studies of layered materials where both accuracy and speed are required.

**References**

> F.T. Cerasoli, A.R. Supka, A. Jayaraj, I. Siloi, M. Costa, J. Slawinska, S. Curtarolo, M. Fornari, D. Ceresoli, and M. Buongiorno Nardelli,
> *Advanced modeling of materials with PAOFLOW 2.0: New features and software design*, Comp. Mat. Sci. **200**, 110828 (2021).

> M. Buongiorno Nardelli, F.T. Cerasoli, M. Costa, S. Curtarolo, R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
> *PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on Atomic Orbital bases, including characterization of topological materials*, Comp. Mat. Sci. **143**, 462 (2018).

> L.A. Agapito, A. Ferretti, A. Calzolari, S. Curtarolo and M. Buongiorno Nardelli,
> *Effective and accurate representation of extended Bloch states on finite Hilbert spaces*, Phys. Rev. B **88**, 165127 (2013).

> L.A. Agapito, S. Ismail-Beigi, S. Curtarolo, M. Fornari and M. Buongiorno Nardelli,
> *Accurate Tight-Binding Hamiltonian Matrices from Ab-Initio Calculations: Minimal Basis Sets*, Phys. Rev. B **93**, 035104 (2016).

> L.A. Agapito, M. Fornari, D. Ceresoli, A. Ferretti, S. Curtarolo and M. Buongiorno Nardelli,
> *Accurate Tight-Binding Hamiltonians for 2D and Layered Materials*, Phys. Rev. B **93**, 125137 (2016).

> P. D'Amico, L. Agapito, A. Catellani, A. Ruini, S. Curtarolo, M. Fornari, M. Buongiorno Nardelli and A. Calzolari,
> *Accurate ab initio tight-binding Hamiltonians: Effective tools for electronic transport and optical spectroscopy from first principles*, Phys. Rev. B **94**, 165166 (2016).