# PAOFLOW EDTB API Reference

API documentation for the environment-dependent tight-binding (EDTB) modules in PAOFLOW.

---

## PAOFLOW.defs.sk_fitting

### class `MultiGeomEDTB`

Multi-geometry environment-dependent tight-binding fitter.

Fits a **single shared** set of parameters
:math:`(\varepsilon, V_\lambda, \gamma, \eta)` to DFT band structures
from **multiple atomic configurations** simultaneously, which is
essential for learning physically meaningful screening strengths
:math:`\gamma` that capture the environment dependence of hopping
integrals.

Each geometry is represented internally by an independent
:class:`SKFitterEDTB` instance (with its own screening sums
:math:`S_{ij}`, design tensors, and reference eigenvalues).  The
combined objective is the (optionally weighted) concatenation of
eigenvalue residuals over all geometries.

.. note::

    All geometries must share the **same species, orbital basis, and
    shell structure** so that the hopping parameter vector is
    identical.  This is automatically satisfied when the training set
    consists of the same material at different lattice parameters,
    strains, surfaces, or defect configurations (with the same
    pseudopotential and projection basis).

Typical training sets
---------------------
* **Volume scan**: equilibrium ± 2%, ± 5% isotropic expansion
  (easiest to generate, same symmetry).
* **Tetragonal distortion**: c/a ≠ 1 strains that break cubic
  symmetry.
* **Surface slab**: 5–7 layer slab with vacuum; atoms at the
  surface have reduced coordination → very different
  :math:`S_{ij}`.

Parameters
----------
geometries : list of (arryp, attrp) tuples
    PAOFLOW data-dict pairs, one per configuration.
n_shells : int
    Number of neighbor shells (default 2).
nkfit : int
    k-mesh subdivision (default 6).
r_cut : float
    Screening cutoff radius **in Bohr**.
gamma_mode : {'global', 'per_lpair', 'per_channel'}
    Screening parameter granularity.
fit_onsite_shift : bool
    Whether to fit η on-site shift parameters (default False).
nkfit : int or list of int/tuple
    Subdivisions along each reciprocal axis for the fitting k-mesh.
    If an int, applies uniformly to all geometries.  If a list,
    must have one entry per geometry; each entry can be an int
    (uniform) or a 3-tuple ``(n1, n2, n3)`` for an anisotropic grid.
    Use ``'auto'`` (default) to automatically detect slab geometries
    and reduce the k-grid to 1 along the vacuum direction.
weights : list of float, optional
    Per-geometry weights for the loss function.  Default: uniform
    (all 1.0).  Increase weight on geometries that should be
    reproduced more accurately (e.g. equilibrium bulk).
verbose : bool
    Print progress information.

Usage
-----
>>> from sk_fitting import SKFitter, MultiGeomEDTB
>>> # Pre-fit SK on equilibrium geometry
>>> fitter_sk = SKFitter(arry_eq, attr_eq, n_shells=3)
>>> p_sk = fitter_sk.fit(n_trials=20)['p_opt']
>>> # Multi-geometry EDTB
>>> geoms = [(arry_eq, attr_eq), (arry_p5, attr_p5), (arry_m5, attr_m5)]
>>> mg = MultiGeomEDTB(geoms, n_shells=3, r_cut=8.0, gamma_mode='per_lpair')
>>> result = mg.fit(p0_sk=p_sk, n_trials=10, n_jobs=-1)
>>> model  = mg.build_model_dict(result['p_opt'])

#### `__init__(self, geometries: 'list[tuple[dict, dict]]', *, n_shells: 'int' = 2, nkfit: 'int | str | list' = 'auto', r_cut: 'float', gamma_mode: 'str' = 'global', fit_onsite_shift: 'bool' = False, weights: 'list[float] | None' = None, verbose: 'bool' = True)`

Initialize self.  See help(type(self)) for accurate signature.

#### `build_model_dict(self, p: 'np.ndarray', geom_idx: 'int' = 0) -> 'dict'`

Convert fitted parameters to a PAOFLOW ``SK_EDTB`` model dict.

Parameters
----------
p : np.ndarray
    Full parameter vector (length ``n_params``).
geom_idx : int
    Which geometry to use for lattice vectors and atom positions
    (default 0 = first / reference geometry).

Returns
-------
dict
    Model dict with ``label='SK_EDTB'``.

#### `eigenvalues(self, p: 'np.ndarray', geom_idx: 'int' = 0) -> 'np.ndarray'`

Compute eigenvalues on the fitting k-mesh for geometry *geom_idx*.

Parameters
----------
p : np.ndarray
    Full parameter vector.
geom_idx : int
    Geometry index (default 0).

Returns
-------
np.ndarray
    Shape ``(Nk, nawf)`` eigenvalues in eV.

#### `extract_onsite_from_HR0(self) -> 'np.ndarray'`

Extract initial on-site energies from the reference geometry.

#### `fit(self, *, p0_sk: 'np.ndarray | None' = None, n_trials: 'int' = 10, seed: 'int | None' = 123, max_nfev: 'int' = 2000, ftol: 'float' = 1e-12, xtol: 'float' = 1e-12, gtol: 'float' = 1e-12, alpha: 'float' = 0.0, n_jobs: 'int' = 1) -> 'dict'`

Multi-start least-squares fit across all geometries.

Parameters
----------
p0_sk : np.ndarray, optional
    Initial SK parameter vector (length ``n_sk``).  Typically from
    a single-geometry :meth:`SKFitter.fit` on the equilibrium
    configuration.
n_trials : int
    Number of random restarts.
seed : int or None
    Random seed for reproducibility.
max_nfev : int
    Max function evaluations per trial (default 2000, larger than
    single-geometry because the combined landscape is harder).
ftol, xtol, gtol : float
    Tolerances for ``scipy.optimize.least_squares``.
alpha : float
    Tikhonov regularization strength.
n_jobs : int
    Parallel workers for multi-start trials (``-1`` = all cores).

Returns
-------
dict
    ``p_opt`` : best parameter vector (length ``n_params``).

    ``rmse`` : best combined RMSE (eV).

    ``per_geom_rmse`` : list of per-geometry RMSE values (eV).

    ``max_err`` : max absolute eigenvalue error (eV).

    ``all_results`` : sorted list of ``(rmse, p, OptimizeResult)``.

    ``param_labels`` : parameter names.

### class `SKFitter`

Slater-Koster eigenvalue-based fitter.

Constructs design tensors from a PAO Hamiltonian and fits SK parameters
by minimising the eigenvalue RMSE on a uniform k-mesh.

Parameters
----------
arryp : dict
    PAOFLOW ``arrays`` dict (needs ``a_vectors``, ``b_vectors``, ``tau``,
    ``atoms``, ``shells``, ``HRs``; optionally ``configuration``).
attrp : dict
    PAOFLOW ``attributes`` dict (needs ``alat``, ``natoms``).
n_shells : int
    Number of neighbor shells to include (default 2 → NN + NNN).
nkfit : int
    Subdivisions along each reciprocal axis for the fitting k-mesh
    (total k-points = ``nkfit**3``).
verbose : bool
    Print progress information.

#### `__init__(self, arryp: 'dict', attrp: 'dict', *, n_shells: 'int' = 2, nkfit: 'int' = 6, verbose: 'bool' = True)`

Initialize self.  See help(type(self)) for accurate signature.

#### `build_model_dict(self, p: 'np.ndarray') -> 'dict'`

Convert a fitted parameter vector into a model dict.

The output uses species-pair-keyed hoppings with explicit
shell reference distances (Bohr), following the ``edtb_params``
schema::

    "hoppings": {
      "Pt-Pt": [
        {"r_ref": 5.247, "params": {"sss": ..., "sps": ...}},
        ...
      ]
    }

Parameters
----------
p : np.ndarray
    Parameter vector (length ``n_params``).

Returns
-------
dict
    Model dict with species-pair-keyed hoppings.

#### `eigenvalues(self, p: 'np.ndarray') -> 'np.ndarray'`

Compute SK eigenvalues for parameter vector *p* on the fitting k-mesh.

Returns
-------
np.ndarray
    Shape ``(Nk, nawf)`` eigenvalues in eV.

#### `extract_onsite_from_HR0(self) -> 'np.ndarray'`

Extract initial on-site energies from H(R=0) diagonal blocks.

Uses QE orbital ordering (m = 0, +1, -1, …) for t2g/eg splitting.

Returns
-------
np.ndarray
    On-site parameter vector of length ``n_onsite``.

#### `fit(self, n_trials: 'int' = 10, seed: 'int | None' = 123, max_nfev: 'int' = 1000, ftol: 'float' = 1e-12, xtol: 'float' = 1e-12, gtol: 'float' = 1e-12, alpha: 'float' = 0.0, n_jobs: 'int' = 1) -> 'dict'`

Run multi-start least-squares optimisation.

Parameters
----------
n_trials : int
    Number of random restarts.
seed : int or None
    Random seed for reproducibility.
max_nfev : int
    Max function evaluations per trial.
ftol, xtol, gtol : float
    Tolerances for ``scipy.optimize.least_squares``.
alpha : float
    Tikhonov regularization strength (default 0 = no penalty).
    Adds ``alpha * w_i * p_i`` penalty rows to the residual, where
    ``w_i`` is proportional to the neighbor-shell distance (farther
    shells are penalized more).  On-site parameters are not penalized.
    Typical values: 0.01–1.0 (start small, increase if far-neighbor
    hoppings blow up).
n_jobs : int
    Number of parallel workers for multi-start trials
    (default 1 = sequential).  Use ``-1`` for all available cores.
    Requires ``joblib`` when ``n_jobs != 1``.

Returns
-------
dict
    ``p_opt`` : best parameter vector.
    ``rmse`` : best RMSE (eV) (data-only, excluding penalty).
    ``max_err`` : max absolute error (eV).
    ``all_results`` : list of ``(rmse, p, OptimizeResult)`` sorted by RMSE.
    ``param_labels`` : parameter names.

### class `SKFitterEDTB`

Environment-dependent tight-binding (EDTB) extension of SKFitter.

Augments the two-center SK hopping integrals with an
environment-dependent screening factor:

.. math::

    V_\lambda^{\text{eff}}(i,j) =
        V_\lambda^{(2c)} \exp\!\bigl(-\gamma_\lambda\,S_{ij}\bigr)

where :math:`S_{ij} = \sum_{k \neq i,j} f_c(d_{ik})\,f_c(d_{jk})` is
a bond screening sum and :math:`f_c` is a smooth cosine cutoff
that tapers to zero between :math:`0.8\,r_\text{cut}` and
:math:`r_\text{cut}`.

Optionally fits environment-dependent on-site shifts:

.. math::

    \varepsilon_\alpha \;\to\;
        \varepsilon_\alpha + \eta_\alpha \sum_k f_c(d_{ik})

The screening strengths :math:`\gamma` can be parametrised at three
granularity levels (``gamma_mode``):

* ``'global'`` — one :math:`\gamma` for all channels (1 parameter).
* ``'per_lpair'`` — one per angular-momentum pair
  (ss, sp, pp, …; up to 6).
* ``'per_channel'`` — one per SK integral
  (ssσ, spσ, ppσ, ppπ, …; up to 10).

Parameters
----------
arryp, attrp : dict
    PAOFLOW data dicts (same as :class:`SKFitter`).
n_shells : int
    Number of neighbor shells (default 2).
nkfit : int
    k-mesh subdivision (default 6).
r_cut : float
    Screening cutoff radius **in Bohr**.
gamma_mode : {'global', 'per_lpair', 'per_channel'}
    Granularity of screening parameters (default ``'global'``).
fit_onsite_shift : bool
    Whether to fit :math:`\eta` on-site shift parameters (default False).
verbose : bool
    Print progress information.

Notes
-----
For a single crystal structure the screening parameters are partially
redundant with the shell-dependent hoppings.  Meaningful :math:`\gamma`
values typically require multi-structure training data or external
constraints on the two-center integrals (e.g. supply ``p0_sk`` to
:meth:`fit` so that only :math:`\gamma` and :math:`\eta` are free to
adjust).

Usage
-----
>>> fitter = SKFitterEDTB(arry, attr, n_shells=2, nkfit=6, r_cut=8.0)
>>> result = fitter.fit(n_trials=10, seed=42)
>>> model  = fitter.build_model_dict(result['p_opt'])

Staged fitting (recommended):

>>> fitter_sk = SKFitter(arry, attr, n_shells=2)
>>> p_sk = fitter_sk.fit(n_trials=20)['p_opt']
>>> fitter_edtb = SKFitterEDTB(arry, attr, n_shells=2, r_cut=8.0)
>>> result = fitter_edtb.fit(p0_sk=p_sk, n_trials=10)

#### `__init__(self, arryp: 'dict', attrp: 'dict', *, n_shells: 'int' = 2, nkfit: 'int' = 6, r_cut: 'float', gamma_mode: 'str' = 'global', fit_onsite_shift: 'bool' = False, verbose: 'bool' = True)`

Initialize self.  See help(type(self)) for accurate signature.

#### `build_model_dict(self, p: 'np.ndarray') -> 'dict'`

Convert fitted parameters to an ``SK_EDTB`` model dict.

The screening ``gamma`` is wrapped in a species-pair key,
consistent with the species-pair-keyed hoppings from the
base class.

Parameters
----------
p : np.ndarray
    Full parameter vector (length ``n_params``).

Returns
-------
dict
    Model dict with ``label='SK_EDTB'`` and ``screening`` block.

#### `eigenvalues(self, p: 'np.ndarray') -> 'np.ndarray'`

Compute SK eigenvalues for parameter vector *p* on the fitting k-mesh.

Returns
-------
np.ndarray
    Shape ``(Nk, nawf)`` eigenvalues in eV.

#### `extract_onsite_from_HR0(self) -> 'np.ndarray'`

Extract initial on-site energies from H(R=0) diagonal blocks.

Uses QE orbital ordering (m = 0, +1, -1, …) for t2g/eg splitting.

Returns
-------
np.ndarray
    On-site parameter vector of length ``n_onsite``.

#### `fit(self, *, p0_sk: 'np.ndarray | None' = None, n_trials: 'int' = 10, seed: 'int | None' = 123, max_nfev: 'int' = 1000, ftol: 'float' = 1e-12, xtol: 'float' = 1e-12, gtol: 'float' = 1e-12, alpha: 'float' = 0.0, n_jobs: 'int' = 1) -> 'dict'`

Multi-start least-squares fit including screening parameters.

Parameters
----------
p0_sk : np.ndarray, optional
    Initial SK parameter vector (length ``n_sk``).  If provided the
    first trial uses these values directly; subsequent trials add a
    small random perturbation to the hopping part.  If *None*,
    on-site energies are extracted from H(R=0) and hoppings are
    randomised (same behaviour as :class:`SKFitter`).
n_trials, seed, max_nfev, ftol, xtol, gtol, alpha
    Same as :meth:`SKFitter.fit`.
n_jobs : int
    Number of parallel workers for multi-start trials
    (default 1 = sequential).  Use ``-1`` for all available cores.
    Requires ``joblib`` when ``n_jobs != 1``.

Returns
-------
dict
    ``p_opt`` : best parameter vector (length ``n_params``).
    ``rmse`` : best RMSE (eV, data-only).
    ``max_err`` : max absolute error (eV).
    ``all_results`` : sorted list of ``(rmse, p, OptimizeResult)``.
    ``param_labels`` : parameter names.

### `sk_design_row(orb_a: 'str', orb_b: 'str', lx: 'float', ly: 'float', lz: 'float') -> 'np.ndarray'`

Coefficient of each SK parameter for matrix element H(orb_a, orb_b).

Returns
-------
np.ndarray
    Length-10 array: entry *k* is the coefficient of ``SK_PARAM_NAMES[k]``.

### `sk_element(orb_a: 'str', orb_b: 'str', lx: 'float', ly: 'float', lz: 'float', sh: 'dict') -> 'float'`

Slater-Koster two-center hopping matrix element H(orb_a, orb_b).

Parameters
----------
orb_a, orb_b : str
    Orbital names from ``ORBITAL_NAMES``.
lx, ly, lz : float
    Direction cosines of the bond vector.
sh : dict
    SK parameter dict with keys from ``SK_PARAM_NAMES``.

Returns
-------
float
    The matrix element value.

---

## PAOFLOW.defs.models

### `Kane_Mele(data_controller, params)`

### `SK_EDTB(data_controller, params)`

Build an environment-dependent Slater-Koster tight-binding model (up to 3NN).

Extends the two-center ``Slater_Koster`` with environment-dependent
screening corrections in the style of Porezag & Frauenheim.

For each bond (i, j) the two-center SK integrals are modulated by a
scalar factor that depends on the local atomic environment:

    V_λ^eff(i,j) = V_λ^(2c) · exp( -γ_λ · S_ij )

where S_ij = Σ_{k ≠ i,j} f_c(d_ik) · f_c(d_jk) is a screening sum
over nearby mediating atoms *k*, and f_c is a smooth cutoff function.

The γ_λ screening strengths can be specified:
  - per SK channel  ('sss', 'pps', …)
  - per l-pair       ('ss', 'sp', 'pp', 'sd', 'pd', 'dd')
  - as a single global value

Additional input (on top of ``Slater_Koster`` requirements):

params['model']['screening'] : dict
    r_cut : float
        Cutoff radius (same units as a_vectors) for mediating atoms.
    gamma : float or dict
        If float → single global screening strength.
        If dict  → per-channel {'sss': γ1, 'pps': γ2, …}
                or per-l-pair {'ss': γ1, 'sp': γ2, …}.
    onsite_shift : dict, optional
        Per-orbital environment-dependent on-site correction strength.
        Keys are orbital labels ('s', 'p', 'd'); values are floats.
        Each on-site energy receives Δε = η · Σ_k f_c(d_ik).

### `Slater_Koster(data_controller, params)`

Build a generalized Slater-Koster tight-binding model (two-center, up to 3NN).

This routine populates the PAOFLOW data containers with a real-space
tight-binding Hamiltonian constructed in the two-center approximation,
up to third nearest neighbors. It uses a simple orbital basis per
atom (s,p and d) and derives hopping matrix elements from the standard
Slater-Koster direction-cosine expressions.

High-level workflow:
- Read lattice vectors and atomic positions from params and set up basic
    attributes (number of atoms, orbitals, hoppings).
- Compute reciprocal lattice vectors and the cell volume.
- Build a 3x3x3 supercell of atomic positions to identify neighbors and
    determine a first-neighbor cutoff.
- Construct the Dnm matrix (orbital-position differences) for gradient
    calculations.
- Allocate the real-space Hamiltonian array HRs and fill on-site energies
    and hopping terms using direction cosines and SK parameters.

Data structure expectations (params):
- params['model']['a_vectors']: 3x3 lattice vectors.
- params['model']['atoms']: dict keyed by string indices ("0", "1", ...),
    each containing:
        - 'name': atomic species label
        - 'tau': fractional/cartesian position in lattice units
        - 'orbitals': list of orbital labels used to map on-site terms
        - on-site energies keyed by orbital label
- params['model']['hoppings']: either a flat dict of SK parameters with keys
    'sss', 'sps', 'pps', 'ppp', etc., or a shell dict with 'nn' and optional
    'nnn'/'nnnn' blocks containing those keys.

Output side-effects (data_controller):
- arry['a_vectors'], arry['b_vectors'], arry['tau'], arry['atoms'],
    arry['shells'], arry['norbitals'], arry['sctau'], arry['Dnm'],
    arry['HRs']
- attr['alat'], attr['omega'], attr['natoms'], attr['nawf'], attr['bnd'],
    attr['nbnds'], attr['nk1'], attr['nk2'], attr['nk3'], attr['nkpnts'],
    attr['nspin'], attr['dftSO'], attr['shift'], attr['cutoff']

Notes:
- The neighbor search uses a 3x3x3 supercell for nn only, 5x5x5 when nnn
    is enabled, and 7x7x7 when nnnn is enabled. Shells are determined from
    distinct neighbor distances and mid-point cutoffs.
- Only the unpolarized case is supported; spin-orbit is disabled here.

### `build_TB_model(data_controller, parameters)`

### `cubium(data_controller, params)`

### `cubium2(data_controller, params)`

### `graphene(data_controller, params)`

### `graphene2(data_controller, params)`

---

## PAOFLOW.defs.edtb_params

### class `EDTBModel`

User-facing interface for environment-dependent tight-binding models.

Holds validated parameter and geometry data and provides a clean API
for saving, loading, converting, and computing band structures.

Construction
------------
Direct::

    model = EDTBModel(params_dict, geometry_dict)

From files::

    model = EDTBModel.from_files("params.json", "geometry.json")

From a PAOFLOW model dict (old *or* new format)::

    model = EDTBModel.from_model_dict(model_dict)

From a fitted ``SKFitter`` / ``SKFitterEDTB``::

    model = EDTBModel.from_fitter(fitter, p_opt)

Serialisation
-------------
::

    model.save("params.json", "geometry.json")
    md = model.to_model_dict()  # for PAOFLOW.PAOFLOW(model=md)

Band computation
----------------
::

    result = model.compute_bands(ibrav=2, nk=500)
    print(result["bands_file"])   # path to bands_0.dat

Transferability
---------------
::

    new_model = model.with_geometry(new_geometry)  # same params, new cell

#### `__init__(self, params: 'dict', geometry: 'dict', *, validate: 'bool' = True)`

Create an EDTB model from parameter and geometry dicts.

Parameters
----------
params : dict
    EDTB parameter dict conforming to the ``edtb_params`` schema.
geometry : dict
    Geometry dict with keys ``alat``, ``a_vectors``, ``atoms``.
validate : bool
    If True (default), validate both dicts on construction.

Raises
------
ValueError
    If validation fails.

#### `compute_bands(self, *, ibrav: 'int' = 0, nk: 'int' = 500, outputdir: 'Optional[str]' = None, band_path: 'Optional[str]' = None, high_sym_points: 'Optional[dict]' = None, smearing: 'str' = 'gauss', verbose: 'bool' = False) -> 'dict'`

Compute the band structure using PAOFLOW.

Parameters
----------
ibrav : int
    Bravais lattice type (default 0).
nk : int
    Number of k-points along the path (default 500).
outputdir : str, optional
    Directory for output files.  If None, a temporary directory
    is used (based on the model label and alat).
band_path : str, optional
    Custom band path (e.g. ``"L-G-X"``).
high_sym_points : dict, optional
    Custom high-symmetry point coordinates.
smearing : str
    Smearing type (default ``"gauss"``).
verbose : bool
    Print PAOFLOW output (default False).

Returns
-------
dict
    ``bands_file`` : str — path to ``bands_0.dat``
    ``sym_file`` : str — path to ``kpath_points.txt``
    ``paoflow`` : PAOFLOW object (for further analysis)

#### `save(self, params_path: 'Union[str, Path]', geometry_path: 'Optional[Union[str, Path]]' = None, *, validate: 'bool' = True) -> 'None'`

Write parameter (and optionally geometry) files.

Parameters
----------
params_path : str or Path
    Output path for the parameter file.
geometry_path : str or Path, optional
    Output path for the geometry file.  If None, only the
    parameter file is written.
validate : bool
    Validate before writing (default True).

#### `summary(self) -> 'str'`

Return a human-readable summary of the model.

#### `to_model_dict(self) -> 'dict'`

Convert to a PAOFLOW model dict.

The returned dict can be passed directly to
``PAOFLOW.PAOFLOW(model=...)``.

Returns
-------
dict
    Model dict with species-pair-keyed hoppings.

#### `with_geometry(self, geometry: 'dict') -> "'EDTBModel'"`

Return a new EDTBModel with the same parameters but a different geometry.

This is the main transferability mechanism: train once,
apply to arbitrary cells / strains / defects with the same
species.

Parameters
----------
geometry : dict
    New geometry dict (``alat``, ``a_vectors``, ``atoms``).

Returns
-------
EDTBModel
    New model sharing the same parameters.

Raises
------
ValueError
    If the new geometry contains species not in the model.

### `active_gamma_labels(l_channels_a: 'List[str]', l_channels_b: 'List[str]') -> 'List[str]'`

Return active γ labels for a species pair.

### `active_sk_names_for_basis(l_channels_a: 'List[str]', l_channels_b: 'List[str]') -> 'List[str]'`

Return the SK parameter names active for a species pair.

### `compute_pair_shell_distances(a_vectors, atoms, sp1, sp2, n_shells=3, r_max=20.0, tol=0.01)`

Compute neighbor-shell distances for a specific species pair.

Unlike ``compute_shell_distances`` (which pools all atom pairs),
this considers only bonds from *sp1* atoms to *sp2* atoms.

Parameters
----------
a_vectors : array-like, shape (3, 3)
    Lattice vectors in Bohr (already scaled by alat).
atoms : list of dict
    Atom dicts with ``"species"`` and ``"tau"`` keys (positions in Bohr).
sp1, sp2 : str
    Species names.
n_shells : int
    Number of distinct shells to return.
r_max : float
    Cutoff for neighbor search (Bohr).
tol : float
    Distance tolerance for grouping into shells (Bohr).

Returns
-------
list of float
    Sorted shell distances for the *sp1*--*sp2* pair (length <= *n_shells*).

### `compute_shell_distances(a_vectors, tau_list, n_shells=3, r_max=20.0, tol=0.01)`

Compute neighbor-shell distances from lattice vectors and atomic basis.

Parameters
----------
a_vectors : array-like, shape (3, 3)
    Lattice vectors in Bohr (already scaled by alat).
tau_list : list of array-like, shape (natoms, 3)
    Atomic positions in Bohr.
n_shells : int
    Number of distinct shells to return.
r_max : float
    Cutoff for neighbor search (Bohr).
tol : float
    Distance tolerance for grouping into shells (Bohr).

Returns
-------
list of float
    Sorted shell distances (length ≤ n_shells).

### `from_model_dict(model_dict: 'dict', shell_distances: 'Optional[Dict[str, float]]' = None) -> 'Tuple[dict, dict]'`

Convert a PAOFLOW model dict to the new schema.

Accepts **both** the old shell-tag-keyed format::

    "hoppings": {"nn": {"sss": ...}, "nnn": {...}}

and the new species-pair-keyed format::

    "hoppings": {"Pt-Pt": [{"r_ref": 5.247, "params": {"sss": ...}}, ...]}

Format detection is automatic.

Parameters
----------
model_dict : dict
    Model dict with keys
    ``label``, ``alat``, ``model.{a_vectors, atoms, hoppings, screening}``.
shell_distances : dict, optional
    *(Old format only)* Shell tag → reference distance (Bohr), e.g.
    ``{"nn": 5.247, "nnn": 7.420, "nnnn": 9.090}``.
    If None, distances are computed automatically from the lattice.

Returns
-------
params : dict
    New-format EDTB parameter dict.
geometry : dict
    New-format geometry dict.

### `read_geometry(filepath: 'Union[str, Path]', *, validate: 'bool' = True) -> 'dict'`

Read a geometry file (JSON).

Returns
-------
dict
    Geometry dict with keys ``alat``, ``a_vectors``, ``atoms``.

### `read_params(filepath: 'Union[str, Path]', *, validate: 'bool' = True) -> 'dict'`

Read EDTB parameters from a JSON file.

Parameters
----------
filepath : str or Path
    Input path.
validate : bool
    If True, validate after reading.

Returns
-------
dict
    EDTB parameter dict.

### `species_pair_key(sp1: 'str', sp2: 'str') -> 'str'`

Canonical species-pair key (alphabetically sorted, hyphen-separated).

Examples
--------
>>> species_pair_key("Ge", "Si")
'Ge-Si'
>>> species_pair_key("Pt", "Pt")
'Pt-Pt'

### `summarize_params(params: 'dict') -> 'str'`

Return a human-readable summary of an EDTB parameter dict.

### `to_model_dict(params: 'dict', geometry: 'dict') -> 'dict'`

Convert new-format params + geometry to a model dict.

Emits the species-pair-keyed hopping format, consistent with
``sk_fitting.build_model_dict``.

Parameters
----------
params : dict
    EDTB parameter dict (new format).
geometry : dict
    Geometry dict (new format).

Returns
-------
dict
    Model dict with species-pair-keyed hoppings.

### `validate_geometry(geometry: 'dict') -> 'List[str]'`

Validate a geometry dict against the schema.

Returns
-------
list of str
    Error messages (empty list = valid).

### `validate_params(params: 'dict') -> 'List[str]'`

Validate an EDTB parameter dict against the schema.

Returns
-------
list of str
    Error messages (empty list = valid).

### `write_geometry(filepath: 'Union[str, Path]', geometry: 'dict', *, validate: 'bool' = True) -> 'None'`

Write a geometry file (JSON).

Parameters
----------
filepath : str or Path
    Output path.
geometry : dict
    Geometry dict with keys ``alat``, ``a_vectors``, ``atoms``.
validate : bool
    Validate before writing.

### `write_params(filepath: 'Union[str, Path]', params: 'dict', *, validate: 'bool' = True) -> 'None'`

Write EDTB parameters to a JSON file.

Parameters
----------
filepath : str or Path
    Output path.
params : dict
    EDTB parameter dict conforming to the schema.
validate : bool
    If True, validate before writing.

Raises
------
ValueError
    If *validate* is True and the dict has schema violations.

---

## PAOFLOW.defs.surface_project

surface_project.py — Projected bulk band structure onto arbitrary surface planes.

Given a bulk tight-binding model (via PAOFLOW model dictionary) and a
surface normal direction, this module:

  1. Identifies the shortest reciprocal-lattice vector  G₀  parallel to
     the surface normal — this sets the k⊥ periodicity.
  2. Finds the two shortest in-plane real-space lattice vectors, defining
     the 2D surface lattice and its reciprocal (surface Brillouin zone).
  3. Auto-detects the surface-BZ type (square, hexagonal, rectangular,
     oblique) and provides default high-symmetry paths.
  4. For each  k∥  along the surface-BZ path, sweeps  k⊥  over one full
     period and collects all bulk eigenvalues.
  5. Computes per-band envelopes (min / max over k⊥) and identifies
     projected gaps and lens-shaped features.
  6. Produces a shaded figure of the projected band structure.

Public API
----------
    project_bulk_bands      — main computation
    project_from_model      — convenience wrapper for EDTBModel objects
    find_absolute_gaps      — locate energy gaps persistent at all k∥
    plot_projected          — shaded band-projection figure
    SurfaceProjectionResult — dataclass container

Pre-defined constants
---------------------
    SURFACE_001, SURFACE_110, SURFACE_111  — common surface normals
    SURFACE_SYM_SQUARE, SURFACE_SYM_HEX, SURFACE_SYM_RECT — surface BZ points

References
----------
  F. J. Himpsel and D. E. Eastman, J. Vac. Sci. Technol. 16, 1297 (1979).
  M. C. Desjonquères and D. Spanjaard, *Concepts in Surface Physics*
  (Springer, 1996).

### class `SurfaceProjectionResult`

Container for projected bulk band structure results.

Attributes
----------
kdist         : (nk_par,) cumulative k∥ distance along the surface path.
sym_ticks     : list of (distance, label) for surface-BZ symmetry ticks.
band_min      : (nk_par, nawf) minimum eigenvalue of band *n* over k⊥.
band_max      : (nk_par, nawf) maximum eigenvalue of band *n* over k⊥.
all_evals     : (nk_par, nk_perp, nawf) full eigenvalue array.
nawf          : number of Wannier functions (bands).
nk_perp       : number of k⊥ sample points.
surface_normal: (3,) unit surface-normal direction (Cartesian / alat).
G_perp        : (3,) shortest reciprocal vector along normal (1/alat).
surf_vectors  : (2, 3) primitive surface real-space lattice vectors.
surf_recip    : (2, 3) surface reciprocal-lattice vectors.
a_bulk        : (3, 3) bulk lattice vectors (rows).
lattice_2d    : str — detected 2D lattice type.

#### `__init__(self, kdist: 'np.ndarray', sym_ticks: 'list', band_min: 'np.ndarray', band_max: 'np.ndarray', all_evals: 'np.ndarray', nawf: 'int', nk_perp: 'int', surface_normal: 'np.ndarray', G_perp: 'np.ndarray', surf_vectors: 'np.ndarray', surf_recip: 'np.ndarray', a_bulk: 'np.ndarray', lattice_2d: 'str' = '') -> None`

Initialize self.  See help(type(self)) for accurate signature.

### `find_absolute_gaps(result: 'SurfaceProjectionResult', y_lim: 'tuple' = (-12, 6), tol: 'float' = 0.01) -> 'List[Tuple[float, float]]'`

Find energy gaps that persist at every k∥ point.

A gap is a contiguous energy interval in which **no** bulk state
exists at **any** k∥.  These are the "absolute" or "true" projected
band gaps where surface states may reside.

Parameters
----------
result : SurfaceProjectionResult from :func:`project_bulk_bands`.
y_lim  : energy window to consider, (E_min, E_max) in eV.
tol    : merge tolerance for near-touching intervals (eV).

Returns
-------
gaps : list of ``(E_lo, E_hi)`` for each absolute gap, sorted by energy.

### `find_band_lenses(result: 'SurfaceProjectionResult', y_lim: 'tuple' = (-12, 6), tol: 'float' = 0.01) -> 'List[dict]'`

Identify "lens" features where a gap opens and closes along k∥.

A band lens occurs when two adjacent projected bands have a gap at
some k∥ values but overlap at others, producing a lens-shaped gap
region in the (k∥, E) plane.

Parameters
----------
result : SurfaceProjectionResult
y_lim  : energy window.
tol    : energy tolerance for gap detection (eV).

Returns
-------
lenses : list of dicts, each with keys:
    ``'gap_center'``  — (float) mean energy of the lens midpoint,
    ``'max_gap'``     — (float) maximum gap width (eV),
    ``'k_range'``     — (float, float) k-distance range where gap exists,
    ``'k_indices'``   — (int, int) index range in kdist.

### `find_gaps_at_kpar(result: 'SurfaceProjectionResult', y_lim: 'tuple' = (-12, 6), tol: 'float' = 0.01) -> 'List[List[Tuple[float, float]]]'`

Find projected band gaps at each individual k∥ point.

Returns
-------
gaps_per_k : list of length nk_par.  Each element is a list of
    ``(E_lo, E_hi)`` gap intervals at that k∥.

### `plot_projected(result: 'SurfaceProjectionResult', *, y_lim: 'tuple' = (-12, 6), color: 'str' = 'steelblue', alpha: 'float' = 0.45, gap_color: 'str' = 'lightyellow', show_gaps: 'bool' = True, mode: 'str' = 'fill', ne: 'int' = 500, surface_bands: 'Optional[np.ndarray]' = None, surface_kdist: 'Optional[np.ndarray]' = None, figsize: 'tuple' = (10, 6), title: 'Optional[str]' = None, show: 'bool' = True, ax=None)`

Plot the projected bulk band structure as a shaded E-vs-k∥ figure.

Parameters
----------
result  : SurfaceProjectionResult from :func:`project_bulk_bands`.
y_lim   : energy window (eV).
color   : fill colour for bulk-band regions.
alpha   : opacity (used in ``'fill'`` mode).
gap_color : background colour for gap regions (``'image'`` mode).
show_gaps : if True, highlight absolute gaps.
mode    : ``'fill'`` — ``fill_between`` per band (overlapping bands
    appear darker, giving a qualitative DOS indication);
    ``'image'`` — binary contour-fill (uniform colour inside bulk
    regions, clean edges).
ne      : energy resolution for ``'image'`` mode.
surface_bands : (nk_par, n_surf_bands) array of surface-state
    eigenvalues to overlay (optional).
surface_kdist : (nk_par,) k-distances for surface bands (optional;
    defaults to ``result.kdist``).
figsize : figure size for a new figure.
title   : plot title.
show    : call ``plt.show()``.
ax      : existing matplotlib Axes (optional).

Returns
-------
fig, ax

### `project_bulk_bands(model_dict: 'dict', surface_normal, *, surface_sym_points: 'Optional[dict]' = None, surface_path: 'Optional[str]' = None, nk_par: 'int' = 100, nk_perp: 'int' = 100, verbose: 'bool' = True) -> 'SurfaceProjectionResult'`

Compute the projected bulk band structure onto a surface plane.

For each  k∥  along a high-symmetry path in the 2D surface Brillouin
zone, the function sweeps  k⊥  over one full period (the 1D BZ
perpendicular to the surface) and collects all bulk eigenvalues.  The
per-band envelopes (min / max over k⊥) define the projected band
structure.

Parameters
----------
model_dict : PAOFLOW-compatible model dictionary for the **bulk**
    crystal.  Must contain ``'model'`` key with ``'a_vectors'`` and
    ``'atoms'``.
surface_normal : (3,) array-like — Cartesian direction perpendicular
    to the surface.  Common choices for cubic crystals:
    ``[0,0,1]`` for (001),  ``[1,1,0]`` for (110),
    ``[1,1,1]`` for (111).  Need not be a unit vector.
surface_sym_points : dict, optional — high-symmetry points in the
    surface BZ as ``{label: [f1, f2]}`` fractional coordinates in
    the surface reciprocal basis.  If *None*, auto-detected from
    the 2D lattice type.
surface_path : str, optional — path string, e.g. ``'Γ-X-M-Γ'``.
    If *None*, auto-detected.
nk_par  : int — number of k∥ points per path segment.
nk_perp : int — number of k⊥ sample points.
verbose : bool — print progress information.

Returns
-------
SurfaceProjectionResult  dataclass with band envelopes, eigenvalues,
and surface geometry metadata.

### `project_from_model(model, surface_normal, **kwargs) -> 'SurfaceProjectionResult'`

Project bulk bands using an EDTBModel object directly.

Parameters
----------
model : EDTBModel for the bulk crystal.
surface_normal : (3,) Cartesian surface-normal direction.
**kwargs : forwarded to :func:`project_bulk_bands`.

Returns
-------
SurfaceProjectionResult

---

## PAOFLOW.defs.dual_params

### class `DualParamModel`

Dual-parameter tight-binding model with site-labeled atoms.

Uses two independently fitted parameter sets (P_bulk and P_surf)
with atoms labeled as "bulk" or "surface".  Interface bonds are
handled with configurable mixing rules.

Construction
------------
From files::

    model = DualParamModel.from_files(
        params_bulk="Si_SK_params.json",
        params_surf="Si_surface_EDTB_params.json",
        geometry="Surface_Si/Si_111_slab_geom.json",
    )

From EDTBModel objects::

    model = DualParamModel(
        params_bulk=bulk_params_dict,
        params_surf=surf_params_dict,
        geometry=geometry_dict,
    )

Labeling is applied automatically on construction (default:
coordination-based).  Override with ``labeling='geometric'``,
``labeling='manual'``, or pass explicit ``labels=[...]``.

Hamiltonian
-----------
Build H(R) and get the PAOFLOW model dict::

    HRs, meta = model.build_hamiltonian()
    model_dict = model.to_model_dict()

Then feed into PAOFLOW and replace HRs::

    pao = PAOFLOW(model=model_dict)
    arry, attr = pao.data_controller.data_dicts()
    arry['HRs'] = HRs
    pao.bands(ibrav=0, band_path='G-M-K-G', nk=500)

#### `__init__(self, params_bulk: 'dict', params_surf: 'dict', geometry: 'dict', *, labels: 'Optional[List[str]]' = None, mixing: 'str' = 'geometric', labeling: 'str' = 'coordination', label_kwargs: 'Optional[dict]' = None)`

Create a dual-parameter model.

Parameters
----------
params_bulk : dict
    EDTB parameter dict for bulk-like atoms.
params_surf : dict
    EDTB parameter dict for surface atoms.
geometry : dict
    Geometry dict (alat, a_vectors, atoms).
labels : list of str, optional
    Explicit 'bulk'/'surface' labels per atom. If None,
    computed from ``labeling`` method.
mixing : str
    Mixing rule: 'geometric' (default), 'arithmetic', or 'bulk'.
labeling : str
    Labeling method if ``labels`` is None:
    'coordination' (default), 'geometric', or 'manual'.
label_kwargs : dict, optional
    Extra kwargs passed to the labeling function.

#### `build_hamiltonian(self, verbose: 'bool' = True) -> 'Tuple[np.ndarray, dict]'`

Build the real-space Hamiltonian H(R).

Returns
-------
HRs : ndarray, complex, shape (nawf, nawf, nk1, nk2, nk3, 1)
    Real-space Hamiltonian on the supercell R-grid.
meta : dict
    Metadata needed for Fourier interpolation:

    - **nawf** : int — number of Wannier functions (= total orbitals)
    - **natoms** : int
    - **tau** : (natoms, 3) — atomic positions in alat units
    - **a_vectors** : (3, 3) — lattice vectors in alat units
    - **b_vectors** : (3, 3) — reciprocal lattice (2π/alat units)
    - **norbitals** : (natoms,) int — orbitals per atom
    - **orbitals_per_atom** : list of list of str — orbital names
    - **atom_block_start** : (natoms,) int — orbital offset per atom
    - **nk1, nk2, nk3** : int — R-grid dimensions
    - **R** : (nrtot, 3) — R-vectors in alat units (FFT order)
    - **Rfft** : (nk1, nk2, nk3, 3) — same, 3-D indexed
    - **Ridx** : (nk1, nk2, nk3) int — 3-D → flat index map
    - **alat** : float — lattice parameter in Bohr
    - **alat_ang** : float — lattice parameter in Å
    - **volume** : float — unit cell volume (alat³ units)
    - **species** : list of str — species name per atom
    - **cutoffs** : list of float — shell cutoffs in alat units
    - **n_bonds** : int — number of bonds processed

#### `label_summary(self) -> 'str'`

Return a summary of atom labels with coordination numbers.

#### `to_model_dict(self) -> 'dict'`

Return a PAOFLOW-compatible model dict.

Uses the **bulk** parameters and the slab geometry.
The returned dict can be passed directly to
``PAOFLOW.PAOFLOW(model=...)``, which will build its own
H(R) internally and set up all metadata.  You then replace
``arry['HRs']`` with the dual-parameter H(R) from
:meth:`build_hamiltonian` before calling ``pao.bands()``.

Returns
-------
dict
    Model dict accepted by PAOFLOW.

### `label_atoms_coordination(geometry: 'dict', r_cut_bohr: 'float' = 8.0, threshold: 'Optional[float]' = None, r_taper_frac: 'float' = 0.8) -> 'Tuple[List[str], np.ndarray]'`

Label atoms as 'bulk' or 'surface' based on coordination number.

The smooth coordination number is defined as:

    Z_i = Σ_{k ≠ i} f_c(d_{ik})

where f_c is a cosine-tapered cutoff function.  Atoms with
Z_i ≥ threshold are labeled 'bulk'; others 'surface'.

Parameters
----------
geometry : dict
    Geometry dict (alat, a_vectors, atoms).
r_cut_bohr : float
    Cutoff radius in Bohr for the smooth cutoff.
threshold : float, optional
    Coordination threshold. If None, uses (max + min) / 2.
r_taper_frac : float
    Fraction of r_cut at which tapering begins (default 0.8).

Returns
-------
labels : list of str
    'bulk' or 'surface' for each atom.
coord : ndarray
    Smooth coordination numbers.

### `label_atoms_geometric(geometry: 'dict', n_surface_layers: 'int' = 2, surface_normal: 'Optional[np.ndarray]' = None) -> 'Tuple[List[str], np.ndarray]'`

Label atoms based on distance from the vacuum surface.

For slab geometries, atoms near the top/bottom surfaces are
labeled 'surface'.  The surface normal is identified from the
longest lattice vector (slab direction).

Parameters
----------
geometry : dict
    Geometry dict.
n_surface_layers : int
    Number of atomic layers from each surface to label 'surface'.
surface_normal : ndarray, optional
    Explicit surface normal. If None, uses the longest a_vector.

Returns
-------
labels : list of str
projections : ndarray
    Signed distances along the surface normal.

### `label_atoms_manual(n_atoms: 'int', surface_indices: 'List[int]') -> 'List[str]'`

Label atoms from an explicit list of surface atom indices.

Parameters
----------
n_atoms : int
    Total number of atoms.
surface_indices : list of int
    Indices of atoms to label as 'surface'.

Returns
-------
labels : list of str

---

## PAOFLOW.defs.band_unfold

band_unfold.py — General band-unfolding from supercell to primitive cell.

Works with any crystal symmetry.  Given the primitive-cell (PC) and
supercell (SC) lattice vectors and atomic positions, the module:

  1. Finds the integer transformation matrix  M  such that  A_SC = M · A_PC.
  2. Enumerates the  N = |det M|  primitive-lattice translations inside
     the supercell.
  3. Builds the atom mapping  I(α, ℓ)  (SC atom ← PC atom α + translation ℓ).
  4. Extracts the real-space Hamiltonian from a PAOFLOW DataController,
     Fourier-transforms along an arbitrary k-path in the PC Brillouin zone,
     and computes the spectral weight  w_n(k)  for every SC eigenstate.

Public API
----------
    unfold_bands(pc_model_dict, sc_model_dict, kpath_frac, *,
                 nk_per_seg=80, verbose=True)

    UnfoldResult  — dataclass returned by unfold_bands()
    plot_unfolded — convenience plotting function

References
----------
  V. Popescu and A. Zunger, Phys. Rev. B 85, 085201 (2012).
  P. B. Allen et al., Phys. Rev. B 87, 085322 (2013).

### class `UnfoldResult`

Container for band-unfolding results.

Attributes
----------
kpath_cart : (nk, 3) Cartesian k-points (units of 1/alat).
kdist      : (nk,)   cumulative k-distance along the path.
sym_ticks  : list of (distance, label) for symmetry-point ticks.
E_pc       : (nk, nawf_pc)  PC reference eigenvalues.
E_sc       : (nk, nawf_sc)  SC eigenvalues.
W          : (nk, nawf_sc)  spectral weights w_n(k).
nawf_pc    : int   — number of orbitals in the PC.
nawf_sc    : int   — number of orbitals in the SC.
N          : int   — volume ratio (= |det M|).
R_translations : (N, 3)  PC lattice translations inside the SC.
atom_map   : (n_at_pc, N)  SC atom index for each (α, ℓ).

#### `__init__(self, kpath_cart: 'np.ndarray', kdist: 'np.ndarray', sym_ticks: 'list', E_pc: 'np.ndarray', E_sc: 'np.ndarray', W: 'np.ndarray', nawf_pc: 'int', nawf_sc: 'int', N: 'int', R_translations: 'np.ndarray', atom_map: 'np.ndarray', a_pc: 'np.ndarray', a_sc: 'np.ndarray', M: 'np.ndarray') -> None`

Initialize self.  See help(type(self)) for accurate signature.

#### `sum_rule_check(self) -> 'np.ndarray'`

Return Σ_n W_n(k) per k-point (should ≈ nawf_pc).

### `make_kpath(sym_points: 'dict', path_str: 'str', a_pc: 'np.ndarray', nk_per_seg: 'int' = 80) -> 'tuple'`

Generate a k-path from a string specification.

Parameters
----------
sym_points : dict mapping label → (3,) fractional coords in PC basis.
path_str   : e.g. 'Γ-X-W-K-Γ-L|U-X'.  Use '-' to connect, '|' for breaks.
a_pc       : (3, 3) PC lattice vectors (rows), Cartesian/alat.
nk_per_seg : number of k-points per linear segment.

Returns
-------
kpath_cart : (nk, 3) Cartesian k-points.
kdist      : (nk,) cumulative distance.
sym_ticks  : list of (distance, label).

### `plot_unfolded(result: 'UnfoldResult', *, y_lim: 'tuple' = (-12, 6), w_thresh: 'float' = 0.02, cmap: 'str' = 'Reds', figsize: 'tuple' = (10, 6), title: 'Optional[str]' = None, show: 'bool' = True, ax=None)`

Plot unfolded band structure.

Parameters
----------
result   : UnfoldResult from unfold_bands().
y_lim    : energy window.
w_thresh : minimum spectral weight to display.
cmap     : matplotlib colormap for scatter points.
figsize  : figure size if creating a new figure.
title    : plot title.
show     : call plt.show().
ax       : existing matplotlib Axes (optional).

Returns
-------
fig, ax

### `unfold_bands(pc_model_dict: 'dict', sc_model_dict: 'dict', sym_points: 'dict', path_str: 'str', *, nk_per_seg: 'int' = 80, verbose: 'bool' = True) -> 'UnfoldResult'`

Unfold supercell bands onto the primitive-cell Brillouin zone.

Parameters
----------
pc_model_dict : PAOFLOW model dict for the **primitive cell**.
sc_model_dict : PAOFLOW model dict for the **supercell**.
sym_points    : dict of high-symmetry point labels → fractional coords
                in the **PC reciprocal basis**. Example:
                ``{'Γ': [0,0,0], 'X': [0.5, 0, 0.5], ...}``
path_str      : band-path string, e.g. ``'Γ-X-W-K-Γ-L|U-X'``.
nk_per_seg    : k-points per linear sub-segment.
verbose       : print progress information.

Returns
-------
UnfoldResult  dataclass with eigenvalues, spectral weights, and metadata.

Notes
-----
Both model dicts must use the **same** alat.  The PC and SC lattice
vectors (``a_vectors``) must be given in Cartesian units of alat (the
PAOFLOW / EDTB convention).

### `unfold_from_models(pc_model, sc_model, sym_points: 'dict', path_str: 'str', *, nk_per_seg: 'int' = 80, verbose: 'bool' = True) -> 'UnfoldResult'`

Unfold bands using EDTBModel objects directly.

Parameters
----------
pc_model   : EDTBModel for the primitive cell.
sc_model   : EDTBModel for the supercell.
sym_points : high-symmetry points in PC fractional coords.
path_str   : band-path string.
nk_per_seg : k-points per segment.
verbose    : print progress.

Returns
-------
UnfoldResult

---
