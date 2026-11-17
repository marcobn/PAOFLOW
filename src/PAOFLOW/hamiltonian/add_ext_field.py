def add_ext_field(data_controller, efield=None, bfield=None, hubbard_u=None) -> None:
    """Add external electric field, magnetic field, and Hubbard U corrections to the real-space Hamiltonian.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
    efield, bfield, hubbard_u : array_like, optional
        The fields of this call: electric field (V/A, shape ``(3,)``), Zeeman energy (eV, shape
        ``(3,)``) and Hubbard corrections (eV, shape ``(nawf,)``). ``None`` takes the stored
        ``Efield``, ``Bfield``, ``HubbardU`` arrays (the former behaviour); ``add_external_fields``
        passes the fields of each call, so a later call never re-applies an earlier field.
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

      ``B`` is an energy (eV) that lowers the spin parallel to it: an exchange field, or a
      physical field :math:`\\mathbf B_{\\rm T}` as :math:`-g_s\\mu_B\\mathbf B_{\\rm T}`
      (``PAOFLOW.magnetic_field`` takes tesla and adds the orbital term). A collinear run
      (``nspin = 2``) takes only :math:`B_z`: :math:`\\mp B_z/2` on the two channels.

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

    efield = np.asarray(arrays['Efield'] if efield is None else efield, dtype=float)
    bfield = np.asarray(arrays['Bfield'] if bfield is None else bfield, dtype=float)
    hubbard_u = np.asarray(arrays['HubbardU'] if hubbard_u is None else hubbard_u, dtype=float)

    if efield.any() != 0.0:
        # orbital centres in angstrom (bohr for DFT runs, alat units for models)
        tau_wf = orbital_centres_bohr(arrays, attributes) / ANGSTROM_AU
        for n in range(nawf):
            arrays['HRs'][n, n, 0, :] -= efield.dot(tau_wf[n, :])

    if bfield.any() != 0.0:
        if nspin == 2:
            # collinear: -B.S with S_z = +-1/2 on the two channels; no transverse component
            if np.any(bfield[:2] != 0.0):
                arrays['HRs'] = np.reshape(
                    arrays['HRs'], (nawf, nawf, nk1, nk2, nk3, nspin), order='C'
                )
                raise ValueError(
                    'a collinear run (nspin = 2) takes only Bfield along z: call spinor_basis() '
                    'first for a transverse field'
                )
            for n in range(nawf):
                arrays['HRs'][n, n, 0, 0] -= 0.5 * bfield[2]
                arrays['HRs'][n, n, 0, 1] += 0.5 * bfield[2]
        else:
            field = (
                bfield[0] * arrays['Sj'][0]
                + bfield[1] * arrays['Sj'][1]
                + bfield[2] * arrays['Sj'][2]
            )
            for n in range(nawf):
                for m in range(nawf):
                    arrays['HRs'][n, m, 0, :] -= field[n, m]

    if hubbard_u.any() != 0:
        for n in range(nawf):
            arrays['HRs'][n, n, 0, :] -= hubbard_u[n] / 2.0

    arrays['HRs'] = np.reshape(arrays['HRs'], (nawf, nawf, nk1, nk2, nk3, nspin), order='C')
