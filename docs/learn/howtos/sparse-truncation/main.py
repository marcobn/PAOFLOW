import numpy as np

from PAOFLOW.PAOFLOW import PAOFLOW

savedir = 'tutorial01/silicon.save'


def si_bands(outputdir: str, sparse_config: dict) -> np.ndarray:
    """Si bands along the fcc path, one row per k-point, with the truncation ``sparse_config``."""
    paoflow = PAOFLOW(
        savedir=savedir,
        outputdir=outputdir,
        smearing='gauss',
        sparse=True,
        sparse_config=sparse_config,
    )
    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()  # H(R) is truncated here, once, with sparse_config
    paoflow.bands(ibrav=2, nk=2000)
    paoflow.save_sparse_hamiltonian('hamiltonian.npz')  # the kept hoppings, in outputdir
    paoflow.finish_execution()
    return np.loadtxt(f'{outputdir}/bands_0.dat')[:, 1:]


# Truncate by neighbour shells
sparse_config = {'bond_order': 36}
si_bands('output_bo36', sparse_config)

# Keep every hopping
full = si_bands('output_full', {'hopping_threshold': 0.0})

# One shell beyond the R grid
try:
    si_bands('output_bo48', {'bond_order': 48})
except ValueError as error:
    print('bond_order=48:', error)

# Truncate by radius
sparse_config = {'rcut': 30.0}  # Bohr
si_bands('output_rcut30', sparse_config)

# Converge the cutoff against the full Hamiltonian
near_gap = (full > -2.0) & (full < 2.0)  # states within 2 eV of the valence-band maximum
for n in [5, 10, 15, 20, 25, 30, 33, 36, 40, 44, 47]:
    bands = si_bands(f'output_bo{n}', {'bond_order': n})
    shift = np.nanmax(np.abs(bands - full)[near_gap])
    print(f'bond_order={n:2d}: largest band shift within 2 eV of the VBM {1000 * shift:5.1f} meV')

# Truncate by hopping size
sparse_config = {'hopping_threshold': 1.0e-4}  # eV
si_bands('output_t1e-04', sparse_config)
si_bands('output_t1e-03', {'hopping_threshold': 1.0e-3})
