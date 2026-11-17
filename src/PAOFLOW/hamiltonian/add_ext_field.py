from __future__ import annotations

import numpy as np


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


def hubbard_shifts_from_labels(arrays: dict, shifts: dict) -> np.ndarray:
    """Per-orbital U of the internal projection basis from manifold labels.

    Parameters
    ----------
    arrays : dict
        Data-controller arrays with ``atomic_basis`` (the records of the internal projections:
        species ``atom`` and shell ``label`` of every orbital).
    shifts : dict
        ``{'Al-3p': 0.1, 'P-3p': 2.31}``: species and shell (as ACBN0 names a manifold, any case)
        and U (eV).

    Returns
    -------
    ndarray, shape (nawf,)
        U of every orbital of the basis (0 for the others).

    Raises
    ------
    ValueError
        Without an internal basis (``projwfc.x`` projections have no shell labels: give one U per
        orbital), or for a label that names no orbital of the basis (the message lists the
        manifolds).

    Notes
    -----
    The labels are resolved against the basis of the run, so the same dictionary shifts the same
    manifolds in a run on the ``standard`` basis and in one on the ``extended`` basis, where the
    orbital indices differ.
    """
    basis = arrays.get('atomic_basis')
    if basis is None or len(basis) == 0:
        raise ValueError(
            'Hubbard shifts by manifold label need the internal projections (projections()); '
            'with projwfc.x projections give one U per orbital'
        )
    names = ['%s-%s' % (str(r['atom']).strip(), str(r['label']).strip()) for r in basis]
    wanted = {str(key).strip().lower(): float(value) for key, value in shifts.items()}
    values = np.zeros(len(basis))
    for index, name in enumerate(names):
        values[index] = wanted.get(name.lower(), 0.0)
    known = {name.lower() for name in names}
    missing = sorted(key for key in wanted if key not in known)
    if missing:
        raise ValueError(
            'no orbital of the basis is in %s; the manifolds of this basis are %s'
            % (', '.join(missing), ', '.join(sorted(set(names), key=names.index)))
        )
    return values
