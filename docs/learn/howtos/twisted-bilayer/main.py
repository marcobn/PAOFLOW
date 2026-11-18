import numpy as np
from PAOFLOW.models.edtb_params import EDTBModel, write_geometry
from PAOFLOW.spectrum.sparse_bands import SparseEDTB

# EDTB model of carbon with distance-dependent hoppings, as on the page on large supercells
spd = ['s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dzx', 'dx2-y2', 'dz2']
carbon = {
    'edtb_version': '1.0',
    'basis': {'C': {'orbitals': spd, 'l_channels': ['s', 'p', 'd']}},
    'onsite': {'C': {'s': -4.52191, 'p': -0.84835, 't2g': 6.33815, 'eg': 6.34247}},
    'hoppings': {'C-C': {
        'type': 'distance_dependent', 'r_0': 2.684, 'r_c': 10.0, 'n_c': 2.59873,
        'channels': {
            'sss': {'V0': -4.28565, 'n': 4.25117}, 'sps': {'V0': 5.03970, 'n': 3.50549},
            'sds': {'V0': 2.96708, 'n': 1.15475}, 'pps': {'V0': 4.50301, 'n': 2.56064},
            'ppp': {'V0': -2.08356, 'n': 4.38254}, 'pds': {'V0': 1.93938, 'n': -0.31188},
            'pdp': {'V0': -6.42159, 'n': 1.33501}, 'dds': {'V0': -0.00009, 'n': -3.40900},
            'ddp': {'V0': 0.00040, 'n': 1.78124}, 'ddd': {'V0': -0.00166, 'n': -1.17807}}}},
    'screening': {'r_cut': 8.0, 'gamma': {'C-C': {
        'ss': 0.00450, 'sp': 0.00743, 'sd': 0.60988, 'pp': 0.00670, 'pd': 1.70099, 'dd': 1.53698}}},
}  # fmt: skip


# The commensurate cell (n, m)
def twisted_bilayer(n, m, d=2.68, vacuum=15.0, a_cc=1.42):
    """Commensurate twisted bilayer graphene (n, m); lengths in angstrom, alat = graphene a."""
    a = a_cc * np.sqrt(3)
    a1, a2 = np.array([1.0, 0]), np.array([0.5, np.sqrt(3) / 2])
    theta = np.arccos((n * n + m * m + 4 * n * m) / (2 * (n * n + n * m + m * m)))
    c, s = np.cos(theta), np.sin(theta)
    cell = np.array([n * a1 + m * a2, -m * a1 + (n + m) * a2])
    span = range(-2 * (n + m), 2 * (n + m) + 1)
    atoms = []
    for z, rot in ((0.0, np.eye(2)), (d / a, np.array([[c, s], [-s, c]]))):
        sites = {}
        for i in span:
            for j in span:
                for t in (0, 1 / 3):  # sublattices A and B
                    f = np.linalg.solve(cell.T, rot @ (i * a1 + j * a2 + t * (a1 + a2)))
                    f -= np.floor(f + 1e-8)
                    sites[tuple(np.round(f, 6))] = f @ cell
        atoms += [{'species': 'C', 'tau': [x, y, z]} for x, y in sites.values()]
    a_vectors = [[*cell[0], 0], [*cell[1], 0], [0, 0, (d + vacuum) / a]]
    return {'alat': a / 0.529177249, 'a_vectors': a_vectors, 'atoms': atoms}, np.degrees(theta)


# Dirac points of the AA bilayer at K, split by the interlayer hopping
a = 1.42 * np.sqrt(3)  # graphene lattice constant (angstrom)
for d in (3.35, 2.68):
    geom, _ = twisted_bilayer(1, 0, d=d)
    lower = [atom for atom in geom['atoms'] if atom['tau'][2] == 0]
    geom['atoms'] = lower + [
        {'species': 'C', 'tau': [x, y, d / a]} for x, y, _ in (atom['tau'] for atom in lower)
    ]
    aa_bands = SparseEDTB(EDTBModel(carbon, geom).to_model_dict(), verbose=False)
    e = aa_bands.eigvals_at_k([2 / 3, 1 / 3, 0], n_eigs=4, sigma=-0.52)
    print(f'AA bilayer, d = {d} A: Dirac points {e[2:].mean() - e[:2].mean():.3f} eV apart, '
          f'mean {e.mean():.4f} eV')  # fmt: skip

# Bands of the (4, 3) and (6, 5) cells
for n, m in ((4, 3), (6, 5)):
    geom, _ = twisted_bilayer(n, m)
    write_geometry(f'tbg_{n}_{m}/geometry.json', geom)
    bands = SparseEDTB(EDTBModel(carbon, geom).to_model_dict(), verbose=False)
    moire = {'G': [0, 0, 0], 'M': [0.5, 0, 0], 'K': [2 / 3, 1 / 3, 0]}
    bands.compute_bands(
        'K-G-M-K',
        moire,
        nk=60,
        n_eigs=40,
        sigma=-0.52,
        outputdir=f'tbg_{n}_{m}',
        n_workers=-1,
    )
    # A short segment from the moire K point towards G
    moire['P'] = [0.64, 0.32, 0]  # 4 % of the way from K to G
    bands.compute_bands(
        'K-P',
        moire,
        nk=8,
        n_eigs=8,
        sigma=-0.52,
        outputdir=f'near_K_{n}_{m}',
        n_workers=-1,
    )


# Slope of the Dirac cone
def dirac_slope(k, E, n_dirac=4):
    """Slope (eV alat) of the lower of the bands that leave the n_dirac states at K upwards."""
    near = np.array([np.sort(e[np.argsort(np.abs(e + 0.52))[:n_dirac]]) for e in E])
    return np.polyfit(k[2:6], near[2:6, n_dirac // 2], 1)[0]  # skips the first point after K


# one layer, on a segment 0.4 % of K-G long, so that |k - K| is as small as in the moire cells
geom, _ = twisted_bilayer(1, 0)
geom['atoms'] = [atom for atom in geom['atoms'] if atom['tau'][2] == 0]
layer = SparseEDTB(EDTBModel(carbon, geom).to_model_dict(), verbose=False)
points = {'K': [2 / 3, 1 / 3, 0], 'P': [0.996 * 2 / 3, 0.996 / 3, 0]}
near_K = layer.compute_bands('K-P', points, nk=8, n_eigs=8, sigma=-0.52)
v0 = dirac_slope(near_K['k_dist'], near_K['eigenvalues'], n_dirac=2)
hbar = 6.582119569e-16  # eV s
print(
    f'one layer: hbar v0 = {v0:.4f} eV alat = {v0 * a:.3f} eV A, v0 = {v0 * a * 1e-10 / hbar:.3g} m/s'
)
v = {}
for n, m in ((4, 3), (6, 5)):
    data = np.loadtxt(f'near_K_{n}_{m}/bands_0.dat')
    v[n, m] = dirac_slope(data[:, 0], data[:, 1:])
    print(f'({n}, {m}): hbar v* = {v[n, m]:.3f} eV alat, v*/v0 = {v[n, m] / v0:.4f}')

# Continuum model of Bistritzer and MacDonald, alpha fitted to the (4, 3) cell, alpha ~ 1/theta
angles = np.array([twisted_bilayer(4, 3)[1], twisted_bilayer(6, 5)[1]])
ratio = np.array([v[4, 3], v[6, 5]]) / v0
alpha = np.sqrt((1 - ratio[0]) / (3 + 6 * ratio[0]))
x = alpha * angles[0] / angles[1]
predicted = (1 - 3 * x**2) / (1 + 6 * x**2)
magic = angles[0] * alpha * np.sqrt(3)
magic_65 = angles[1] * np.sqrt(3 * (1 - ratio[1]) / (3 + 6 * ratio[1]))  # fitted at 6.01 instead
print(f'alpha({angles[0]:.2f} deg) = {alpha:.4f}, predicted v*/v0 at {angles[1]:.2f} deg = '
      f'{predicted:.4f}, magic angle {magic:.3f} deg ({magic_65:.2f} deg if fitted at '
      f'{angles[1]:.2f} deg)')  # fmt: skip
