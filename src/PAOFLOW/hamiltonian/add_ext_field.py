def add_ext_field(data_controller):
    """Add external electric field, magnetic field, and Hubbard U corrections to the real-space Hamiltonian.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``HRs`` (shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``),
        ``Efield`` (shape ``(3,)``), ``Bfield`` (shape ``(3,)``),
        ``Sj`` (shape ``(3, nawf, nawf)``), ``HubbardU`` (shape ``(nawf,)``),
        and the orbital centres: ``basis`` (projections) or ``tau`` (bohr for DFT runs, alat
        units for models, ``tau_units = 'alat'``), see
        :func:`PAOFLOW.hamiltonian.orbital_centres.orbital_centres_bohr`.
        Required attribute: ``alat``.

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

    Each orbital takes the position of its own centre. Until 16 Nov 2026 the function gave
    ``nawf // natoms`` consecutive orbitals to each atom and read ``tau`` as bohr also for the
    models, which store it in alat units: there the field was too small by the factor ``alat``.
    The array ``HRs`` is temporarily reshaped to
    ``(nawf, nawf, nk1*nk2*nk3, nspin)`` during processing and restored to its
    original shape on exit.
    """
    import numpy as np

    from ..utils.constants import ANGSTROM_AU
    from .orbital_centres import orbital_centres_bohr

    arrays = data_controller.data_arrays
    attributes = data_controller.data_attributes

    nawf, _, nk1, nk2, nk3, nspin = arrays['HRs'].shape
    arrays['HRs'] = np.reshape(arrays['HRs'], (nawf, nawf, nk1 * nk2 * nk3, nspin), order='C')

    if arrays['Efield'].any() != 0.0:
        # orbital centres in angstrom (bohr for DFT runs, alat units for models)
        tau_wf = orbital_centres_bohr(arrays, attributes) / ANGSTROM_AU
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
