"""Born effective charges and high-frequency dielectric tensor (Stage 2b).

The macroscopic Born effective charge tensor and the clamped-ion (electronic)
dielectric tensor are obtained from one of two routes:

* ``method='dfpt'`` (default): a single Gamma-point density-functional
  perturbation theory run (Quantum ESPRESSO ``ph.x`` with ``epsil=.true.,
  trans=.false.``) yields both tensors directly.  Fast and accurate, but not
  available for DFT+U / hybrid functionals.

* ``method='field'``: finite electric-field (``lelfield``) calculations on the
  primitive cell, with central differences of the per-atom forces (Born
  charges) and the macroscopic polarization (dielectric tensor).  Slower, but
  works whenever ``ph.x`` cannot be used.

The acoustic sum rule (``sum_k Z*_k = 0``) is imposed and both tensors are
symmetrized before the phonopy ``BORN`` file is written.
"""

import numpy as np


def compute_born_and_epsilon(
    data_controller,
    method='dfpt',
    phonon_dir='phonon',
    enforce_sum_rule=True,
    symmetrize=True,
    write_born=True,
):
    """Assemble Born charges and epsilon_inf from a DFPT or finite-field run.

    Returns
    -------
    dict
        ``{'born', 'dielectric'}`` with the Born charges of shape
        ``(natom_prim, 3, 3)`` (units of the elementary charge) and the
        ``(3, 3)`` high-frequency dielectric tensor.
    """
    from .io import harvest_field_results, harvest_ph_results, write_born_file

    arry, attr = data_controller.data_dicts()

    method = str(method).lower()
    if method == 'dfpt':
        res = harvest_ph_results(data_controller, phonon_dir=phonon_dir)
    elif method == 'field':
        res = harvest_field_results(data_controller, phonon_dir=phonon_dir)
    else:
        raise ValueError("method must be 'dfpt' or 'field', got %r" % method)

    born = np.asarray(res['born'], dtype=float)
    epsilon = np.asarray(res['dielectric'], dtype=float)
    born_raw = born.copy()

    if enforce_sum_rule:
        # Acoustic sum rule: the Born charges of all atoms in the cell must sum
        # to zero (charge neutrality of the macroscopic dipole response).
        drift = born.sum(axis=0) / born.shape[0]
        born = born - drift[None, :, :]

    if symmetrize:
        born = 0.5 * (born + born.transpose(0, 2, 1))
        epsilon = 0.5 * (epsilon + epsilon.T)

    arry['born_charges'] = born
    arry['dielectric_tensor'] = epsilon

    if attr.get('verbose', False):
        np.set_printoptions(precision=4, suppress=True)
        print('High-frequency dielectric tensor (epsilon_inf):')
        print(epsilon)
        print('Born effective charges (units of e):')
        for k in range(born.shape[0]):
            print('  atom %d:' % (k + 1))
            print(born[k])
        print('  acoustic sum rule residual (sum_k Z*_k):')
        print(born_raw.sum(axis=0))

    if write_born:
        write_born_file(data_controller, born, epsilon, phonon_dir=phonon_dir)

    return {'born': born, 'dielectric': epsilon}
