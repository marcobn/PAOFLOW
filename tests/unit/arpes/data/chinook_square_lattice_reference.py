"""chinook reference for the PAOFLOW parity fixture: square lattice, two Sb atoms with 5p orbitals,
Slater-Koster nearest + next-nearest neighbours, no spin, fixed radial integrals."""
import numpy as np
import chinook.build_lib as build_lib
from chinook.ARPES_lib import experiment, pol_2_sph

a, c = 5.0, 5.0
avec = np.array([[np.sqrt(0.5) * a, np.sqrt(0.5) * a, 0], [np.sqrt(0.5) * a, -np.sqrt(0.5) * a, 0], [0, 0, c]])
Sb1 = np.array([0.0, 0.0, 0.0])
Sb2 = np.array([np.sqrt(0.5) * a, 0, 0])
spin = {'bool': False}
basis = {'atoms': [0, 0], 'Z': {0: 51}, 'orbs': [['51x', '51y', '51z'], ['51x', '51y', '51z']], 'pos': [Sb1, Sb2], 'spin': spin}
basis_object = build_lib.gen_basis(basis)
Ep, Vpps, Vppp = 0.7, 0.25, -1.0
V1 = {'051': Ep, '005511S': Vpps, '005511P': Vppp}
V2 = {'005511S': Vpps / a, '005511P': Vppp / a}
hamiltonian = {'type': 'SK', 'V': [V1, V2], 'avec': avec, 'cutoff': [0.8 * a, 1.1 * a], 'renorm': 1.0, 'offset': 0.0, 'tol': 1e-4, 'spin': spin}
TB = build_lib.gen_TB(basis_object, hamiltonian)
hops = []
for me in TB.mat_els:
    for h in me.H:
        hops.append([me.i, me.j, h[0], h[1], h[2], np.real(h[3]), np.imag(h[3])])
hops = np.array(hops)
pos = np.array([o.pos for o in TB.basis])
labels = [o.label for o in TB.basis]
print('basis:', labels, 'positions', pos.tolist(), 'hoppings', len(hops))

fixed = {'0-5-1-0': 0.6 - 0.2j, '0-5-1-2': 1.0 + 0.4j}
cube = {'X': [0.13, 1.1, 6], 'Y': [0.21, 0.9, 5], 'E': [-3.0, 3.0, 20], 'kz': 0.0}
ARPES_dict = {'cube': cube, 'hv': 21.2, 'W': 4.0, 'pol': np.array([1.0, 0.0, 0.0]), 'T': -1,
              'SE': ['constant', 0.05], 'resolution': {'E': 0.05, 'k': 0.02}, 'rad_type': 'fixed', 'rad_args': fixed}
exp = experiment(TB, ARPES_dict)
exp.datacube()
pks = exp.pks  # (npks, 4): state index, ikx, iky, E
Mk = exp.Mk  # (npks, 2, 3)
pols = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1j, 0.0] / np.sqrt(2), [1.0, 0.0, 1.0] / np.sqrt(2)])
weights = np.array([np.sum(np.abs(np.einsum('ijk,k->ij', Mk, pol_2_sph(p))) ** 2, axis=1) for p in pols])  # (npol, npks)
kx = np.linspace(*cube['X'])
ky = np.linspace(*cube['Y'])
nstates = len(TB.basis)
state = pks[:, 0].astype(int)
k_index = state // nstates
band = state % nstates
ik_x = (k_index // len(ky)).astype(int)  # chinook: X,Y = meshgrid(kx, ky) -> rows ky, cols kx; flattened row-major
ik_y = (k_index % len(ky)).astype(int)
# verify with pks columns: pks[:,1] = floor(k_index / shape(X)[1]) = row index (ky), pks[:,2] = col (kx)
row, col = pks[:, 1].astype(int), pks[:, 2].astype(int)
kxs, kys = kx[col], ky[row]
np.savez('fixture.npz', hops=hops, pos=pos, labels=np.array(labels), avec=avec, kx=kx, ky=ky,
         state_kx=kxs, state_ky=kys, state_band=band, state_energy=pks[:, 3], Mk=Mk, pols=pols, weights=weights,
         eigenvalues=exp.Eb, hv=21.2, W=4.0, fixed_l0=fixed['0-5-1-0'], fixed_l2=fixed['0-5-1-2'],
         th=exp.th)
print('states', pks.shape, 'weights', weights.shape, 'th>=0 fraction', np.mean(exp.th >= 0))
