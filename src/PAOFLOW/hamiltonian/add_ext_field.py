def add_ext_field(data_controller):
    """Add external electric field, magnetic field, and Hubbard U corrections to the real-space Hamiltonian.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``HRs`` (shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``),
        ``Efield`` (shape ``(3,)``), ``Bfield`` (shape ``(3,)``),
        ``Sj`` (shape ``(3, nawf, nawf)``), ``HubbardU`` (shape ``(nawf,)``),
        ``tau`` (shape ``(natoms, 3)`` in Bohr radii).
        Required attribute: ``natoms``.

    Returns
    -------
    None
        Modifies ``data_controller.data_arrays['HRs']`` in place.

    Notes
    -----
    This function applies up to three perturbations to the on-site elements of
    the real-space Hamiltonian at :math:`\\mathbf{R} = 0`.  Each correction is
    applied only when the corresponding field array is non-zero.

    - **Electric field**: subtracts :math:`\\mathbf{E} \\cdot \\boldsymbol{\\tau}_n`
      from each diagonal entry :math:`H^R_{nn}(0)`, where
      :math:`\\boldsymbol{\\tau}_n` is the orbital position in Ångström:

      .. math::

          H^R_{nn}(0) \\leftarrow H^R_{nn}(0) - \\mathbf{E} \\cdot \\boldsymbol{\\tau}_n

    - **Magnetic field**: subtracts the Zeeman coupling from every
      :math:`\\mathbf{R} = 0` matrix element:

      .. math::

          H^R_{nm}(0) \\leftarrow H^R_{nm}(0)
              - \\sum_{\\alpha} B_{\\alpha} S^{\\alpha}_{nm}

    - **Hubbard U**: subtracts :math:`U_n / 2` from each diagonal entry:

      .. math::

          H^R_{nn}(0) \\leftarrow H^R_{nn}(0) - U_n / 2

    Atomic positions are assumed to be uniform across orbitals within the same
    atom.  The array ``HRs`` is temporarily reshaped to
    ``(nawf, nawf, nk1*nk2*nk3, nspin)`` during processing and restored to its
    original shape on exit.
    """
    import numpy as np

    from .constants import ANGSTROM_AU

    arrays = data_controller.data_arrays
    attributes = data_controller.data_attributes

    nawf, _, nk1, nk2, nk3, nspin = arrays['HRs'].shape
    arrays['HRs'] = np.reshape(arrays['HRs'], (nawf, nawf, nk1 * nk2 * nk3, nspin), order='C')

    l = 0
    natoms = attributes['natoms']
    nwf = nawf // natoms
    tau_wf = np.zeros((nawf, 3), dtype=float)
    for n in range(attributes['natoms']):
        for i in range(nwf):
            tau_wf[l, :] = arrays['tau'][n, :]
            l += 1

    tau_wf /= ANGSTROM_AU

    if arrays['Efield'].any() != 0.0:
        for n in range(nawf):
            arrays['HRs'][n, n, 0, :] -= arrays['Efield'].dot(tau_wf[n, :])

    if arrays['Bfield'].any() != 0.0:
        field = (
            arrays['Bfield'][0] * arrays['Sj'][0]
            + arrays['Bfield'][1] * arrays['Sj'][1]
            + arrays['Bfield'][2] * arrays['Sj'][2]
        )
        for n in range(nawf):
            for m in range(nawf):
                arrays['HRs'][n, m, 0, :] -= field[n, m]

    if arrays['HubbardU'].any() != 0:
        for n in range(nawf):
            arrays['HRs'][n, n, 0, :] -= arrays['HubbardU'][n] / 2.0

    arrays['HRs'] = np.reshape(arrays['HRs'], (nawf, nawf, nk1, nk2, nk3, nspin), order='C')
