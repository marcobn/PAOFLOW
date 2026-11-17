import numpy as np
import scipy.linalg as la


def doubling_HRs(data_controller):
    """Double the simulation cell along one or more lattice directions.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``HRs`` (shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``),
        ``tau`` (shape ``(natoms, 3)``), ``a_vectors`` (shape ``(3, 3)``).
        Required attributes: ``nx``, ``ny``, ``nz`` (number of doublings along
        each axis), ``nk1``, ``nk2``, ``nk3``, ``nspin``, ``nawf``, ``alat``.

    Returns
    -------
    None
        Modifies ``data_controller.data_arrays`` and
        ``data_controller.data_attributes`` in place.  After each doubling step
        along a given axis, :func:`doubling_attr_arry` is called to update all
        derived quantities.  Final modified entries include:

        - ``HRs`` : np.ndarray, shape
          ``(nawf*2^(nx+ny+nz), nawf*2^(nx+ny+nz), nk1, nk2, nk3, nspin)``
          — the real-space Hamiltonian of the enlarged cell.
        - ``tau`` : np.ndarray — atomic positions extended with the new replica.
        - ``a_vectors`` : np.ndarray — lattice vectors scaled by the doubling
          factor along each axis.

    Notes
    -----
    The doubling is performed axis by axis (x, then y, then z), repeating
    ``nx``, ``ny``, and ``nz`` times respectively.  At each step the
    real-space Hamiltonian is assembled in a block-matrix form:

    .. math::

        H^{\\rm doubled}(\\mathbf{R}) = \\begin{pmatrix}
            H(2\\mathbf{R}) & H(2\\mathbf{R}+\\mathbf{a}_i) \\\\
            H(2\\mathbf{R}-\\mathbf{a}_i) & H(2\\mathbf{R})
        \\end{pmatrix}

    where :math:`\\mathbf{a}_i` is the lattice vector being doubled.
    """
    arry, attr = data_controller.data_dicts()

    nx = attr['nx']
    ny = attr['ny']
    nz = attr['nz']
    nk1 = attr['nk1']
    nk2 = attr['nk2']
    nk3 = attr['nk3']
    nspin = attr['nspin']

    nkpts = nk1 * nk2 * nk3

    cell_index = np.zeros((nk1, nk2, nk3, 3), dtype=int)
    new_index = np.zeros((3, nkpts), dtype=int)

    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                n = k + j * nk3 + i * nk2 * nk3
                Rx = float(i) / float(nk1)
                Ry = float(j) / float(nk2)
                Rz = float(k) / float(nk3)
                if Rx >= 0.5:
                    Rx = Rx - 1.0
                if Ry >= 0.5:
                    Ry = Ry - 1.0
                if Rz >= 0.5:
                    Rz = Rz - 1.0
                Rx -= int(Rx)
                Ry -= int(Ry)
                Rz -= int(Rz)
                # the minus sign in Rx*nk1 is due to the Fourier transformation (Ri-Rj)
                ix = int(-round(Rx * nk1, 0))
                iy = int(-round(Ry * nk2, 0))
                iz = int(-round(Rz * nk3, 0))

                cell_index[ix, iy, iz, 0] = i
                cell_index[ix, iy, iz, 1] = j
                cell_index[ix, iy, iz, 2] = k

                new_index[0, n] = ix
                new_index[1, n] = iy
                new_index[2, n] = iz

    # This construction is doubling along the X direction nx times
    for dx in range(nx):
        HR_double = np.zeros(
            (2 * attr['nawf'], 2 * attr['nawf'], nk1, nk2, nk3, nspin), dtype=complex
        )
        for ix in range(min(new_index[0, :]), max(new_index[0, :]) + 1):
            for iy in range(min(new_index[1, :]), max(new_index[1, :]) + 1):
                for iz in range(min(new_index[2, :]), max(new_index[2, :]) + 1):
                    i, j, k = cell_index[ix, iy, iz, :]  # doubled cell index

                    if ((2 * ix) >= min(new_index[0, :])) and ((2 * ix) <= (max(new_index[0, :]))):
                        i, j, k = cell_index[ix, iy, iz, :]
                        m, n, l = cell_index[2 * ix, iy, iz, :]
                        # Upper left HR_double block
                        HR_double[0 : attr['nawf'], 0 : attr['nawf'], i, j, k, :] = arry['HRs'][
                            :, :, m, n, l, :
                        ]
                        # Lower right HR_double block
                        HR_double[
                            attr['nawf'] : 2 * attr['nawf'],
                            attr['nawf'] : 2 * attr['nawf'],
                            i,
                            j,
                            k,
                            :,
                        ] = arry['HRs'][:, :, m, n, l, :]

                    if ((2 * ix + 1) >= min(new_index[0, :])) and (
                        (2 * ix + 1) <= (max(new_index[0, :]))
                    ):
                        i, j, k = cell_index[ix, iy, iz, :]
                        m, n, l = cell_index[2 * ix + 1, iy, iz, :]
                        # Upper right HR_double block
                        HR_double[0 : attr['nawf'], attr['nawf'] : 2 * attr['nawf'], i, j, k, :] = (
                            arry['HRs'][:, :, m, n, l, :]
                        )

                    if ((2 * ix - 1) >= min(new_index[0, :])) and (
                        (2 * ix - 1) <= (max(new_index[0, :]))
                    ):
                        i, j, k = cell_index[ix, iy, iz, :]
                        m, n, l = cell_index[2 * ix - 1, iy, iz, :]
                        # Lower left HR_double block
                        HR_double[attr['nawf'] : 2 * attr['nawf'], 0 : attr['nawf'], i, j, k, :] = (
                            arry['HRs'][:, :, m, n, l, :]
                        )

        arry['HRs'] = HR_double
        HR_double = None
        shift_bohr = arry['a_vectors'][0, :] * attr['alat']
        centres = _current_orbital_centres(data_controller)
        tau_scale = 1.0 if attr.get('tau_units', 'bohr') == 'alat' else attr['alat']
        arry['tau'] = np.append(
            arry['tau'], arry['tau'][:, :] + arry['a_vectors'][0, :] * tau_scale, axis=0
        )
        arry['a_vectors'][0, :] = 2 * arry['a_vectors'][0, :]
        doubling_attr_arry(data_controller, shift_bohr=shift_bohr, centres_bohr=centres)

    # This construction is doubling along the Y direction ny times
    for dy in range(ny):
        HR_double = np.zeros(
            (2 * attr['nawf'], 2 * attr['nawf'], nk1, nk2, nk3, nspin), dtype=complex
        )

        for ix in range(min(new_index[0, :]), max(new_index[0, :]) + 1):
            for iy in range(min(new_index[1, :]), max(new_index[1, :]) + 1):
                for iz in range(min(new_index[2, :]), max(new_index[2, :]) + 1):
                    i, j, k = cell_index[ix, iy, iz, :]  # doubled cell index

                    if ((2 * iy) >= min(new_index[1, :])) and ((2 * iy) <= (max(new_index[1, :]))):
                        i, j, k = cell_index[ix, iy, iz, :]
                        m, n, l = cell_index[ix, 2 * iy, iz, :]
                        # Upper left HR_double block
                        HR_double[0 : attr['nawf'], 0 : attr['nawf'], i, j, k, :] = arry['HRs'][
                            :, :, m, n, l, :
                        ]
                        # Lower right HR_double block
                        HR_double[
                            attr['nawf'] : 2 * attr['nawf'],
                            attr['nawf'] : 2 * attr['nawf'],
                            i,
                            j,
                            k,
                            :,
                        ] = arry['HRs'][:, :, m, n, l, :]

                    if ((2 * iy + 1) >= min(new_index[1, :])) and (
                        (2 * iy + 1) <= (max(new_index[1, :]))
                    ):
                        i, j, k = cell_index[ix, iy, iz, :]
                        m, n, l = cell_index[ix, 2 * iy + 1, iz, :]
                        # Upper right HR_double block
                        HR_double[0 : attr['nawf'], attr['nawf'] : 2 * attr['nawf'], i, j, k, :] = (
                            arry['HRs'][:, :, m, n, l, :]
                        )

                    if ((2 * iy - 1) >= min(new_index[1, :])) and (
                        (2 * iy - 1) <= (max(new_index[1, :]))
                    ):
                        i, j, k = cell_index[ix, iy, iz, :]
                        m, n, l = cell_index[ix, 2 * iy - 1, iz, :]
                        # Lower left HR_double block
                        HR_double[attr['nawf'] : 2 * attr['nawf'], 0 : attr['nawf'], i, j, k, :] = (
                            arry['HRs'][:, :, m, n, l, :]
                        )

        arry['HRs'] = HR_double
        HR_double = None
        shift_bohr = arry['a_vectors'][1, :] * attr['alat']
        centres = _current_orbital_centres(data_controller)
        tau_scale = 1.0 if attr.get('tau_units', 'bohr') == 'alat' else attr['alat']
        arry['tau'] = np.append(
            arry['tau'], arry['tau'][:, :] + arry['a_vectors'][1, :] * tau_scale, axis=0
        )
        arry['a_vectors'][1, :] = 2 * arry['a_vectors'][1, :]
        doubling_attr_arry(data_controller, shift_bohr=shift_bohr, centres_bohr=centres)

    # This construction is doubling along the Z direction nz times
    for dz in range(nz):
        HR_double = np.zeros(
            (2 * attr['nawf'], 2 * attr['nawf'], nk1, nk2, nk3, nspin), dtype=complex
        )

        for ix in range(min(new_index[0, :]), max(new_index[0, :]) + 1):
            for iy in range(min(new_index[1, :]), max(new_index[1, :]) + 1):
                for iz in range(min(new_index[2, :]), max(new_index[2, :]) + 1):
                    i, j, k = cell_index[ix, iy, iz, :]  # doubled cell index

                    if ((2 * iz) >= min(new_index[2, :])) and ((2 * iz) <= (max(new_index[2, :]))):
                        i, j, k = cell_index[ix, iy, iz, :]
                        m, n, l = cell_index[ix, iy, 2 * iz, :]
                        # Upper left HR_double block
                        HR_double[0 : attr['nawf'], 0 : attr['nawf'], i, j, k, :] = arry['HRs'][
                            :, :, m, n, l, :
                        ]
                        # Lower right HR_double block
                        HR_double[
                            attr['nawf'] : 2 * attr['nawf'],
                            attr['nawf'] : 2 * attr['nawf'],
                            i,
                            j,
                            k,
                            :,
                        ] = arry['HRs'][:, :, m, n, l, :]

                    if ((2 * iz + 1) >= min(new_index[2, :])) and (
                        (2 * iz + 1) <= (max(new_index[2, :]))
                    ):
                        i, j, k = cell_index[ix, iy, iz, :]
                        m, n, l = cell_index[ix, iy, 2 * iz + 1, :]
                        # Upper right HR_double block
                        HR_double[0 : attr['nawf'], attr['nawf'] : 2 * attr['nawf'], i, j, k, :] = (
                            arry['HRs'][:, :, m, n, l, :]
                        )

                    if ((2 * iz - 1) >= min(new_index[2, :])) and (
                        (2 * iz - 1) <= (max(new_index[2, :]))
                    ):
                        i, j, k = cell_index[ix, iy, iz, :]
                        m, n, l = cell_index[ix, iy, 2 * iz - 1, :]
                        # Lower left HR_double block
                        HR_double[attr['nawf'] : 2 * attr['nawf'], 0 : attr['nawf'], i, j, k, :] = (
                            arry['HRs'][:, :, m, n, l, :]
                        )

        arry['HRs'] = HR_double
        HR_double = None
        shift_bohr = arry['a_vectors'][2, :] * attr['alat']
        centres = _current_orbital_centres(data_controller)
        tau_scale = 1.0 if attr.get('tau_units', 'bohr') == 'alat' else attr['alat']
        arry['tau'] = np.append(
            arry['tau'], arry['tau'][:, :] + arry['a_vectors'][2, :] * tau_scale, axis=0
        )
        arry['a_vectors'][2, :] = 2 * arry['a_vectors'][2, :]
        doubling_attr_arry(data_controller, shift_bohr=shift_bohr, centres_bohr=centres)


def _current_orbital_centres(data_controller) -> np.ndarray | None:
    """Orbital centres (bohr) before a doubling step, or None when they cannot be assigned."""
    from .orbital_centres import orbital_centres_bohr

    arry, attr = data_controller.data_dicts()
    try:
        return orbital_centres_bohr(arry, attr)
    except (KeyError, ValueError):
        return None


def doubling_attr_arry(data_controller, shift_bohr=None, centres_bohr=None):
    """Update all scalar and array metadata after a cell-doubling step.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Attributes updated: ``omega``, ``nawf``, ``natoms`` (if present),
        ``nelec`` (if present), ``nbnds`` (if present), ``bnd`` (if present).
        Arrays updated: ``b_vectors``, ``naw``, ``sh``, ``nl``, ``atoms`` (if present),
        ``Sj`` (if present), ``Dnm`` (if present).

    Returns
    -------
    None
        All modifications are made in place.

    Notes
    -----
    Called automatically by :func:`doubling_HRs` after each doubling step, with the Cartesian
    shift of the replica, ``shift_bohr`` (the lattice vector before the step, bohr), and the
    orbital centres before the step, ``centres_bohr``.
    The cell volume ``omega`` and the reciprocal vectors ``b_vectors`` are recomputed from the
    updated ``a_vectors`` and ``alat``.  The spin operator ``Sj`` is extended block-diagonally.
    ``Dnm`` (:math:`\\tau_n - \\tau_m`, bohr) becomes ``[[D, D - shift], [D + shift, D]]``, the
    orbital centres ``orbital_centres`` ``[c, c + shift]``. Until 16 Nov 2026 ``Dnm`` was extended
    block-diagonally (zero between the replicas), which gave the band velocities of doubled cells
    a spurious term, and the models' ``tau`` (alat units) was shifted in bohr.
    """
    arry, attr = data_controller.data_dicts()

    # the reciprocal vectors of the doubled cell (2 pi / alat units, a_i . b_j = delta_ij); they
    # were left at those of the original cell, which put every later k-point off its grid
    if 'b_vectors' in arry:
        arry['b_vectors'] = np.linalg.inv(np.asarray(arry['a_vectors'], dtype=float)).T

    # recalculating volume
    attr['omega'] = attr['alat'] ** 3 * arry['a_vectors'][0, :].dot(
        np.cross(arry['a_vectors'][1, :], arry['a_vectors'][2, :])
    )
    # Increassing nawf/natoms
    attr['nawf'] = 2 * attr['nawf']
    if 'natoms' in attr:
        attr['natoms'] = 2 * attr['natoms']
    if 'nelec' in attr:
        attr['nelec'] = 2 * attr['nelec']
    if 'nbnds' in attr:
        attr['nbnds'] = 2 * attr['nbnds']
    if 'bnd' in attr:
        attr['bnd'] = 2 * attr['bnd']
    # doubling the atom number of orbitals / orbital character / multiplicity
    if 'naw' in arry:
        arry['naw'] = np.append(arry['naw'], arry['naw'])
    if 'sh' in arry:
        arry['sh'] = np.append(arry['sh'], arry['sh'])
    if 'nl' in arry:
        arry['nl'] = np.append(arry['nl'], arry['nl'])
    if 'atoms' in arry:
        arry['atoms'] = np.append(arry['atoms'], arry['atoms'])
    # if Sj is already computed, then double it
    if 'Sj' in arry:
        Sj_double = np.zeros((3, attr['nawf'], attr['nawf']), dtype=complex)
        for spol in range(3):
            Sj = arry['Sj'][spol]
            Sj_double[spol] = la.block_diag(*[Sj, Sj])

        arry['Sj'] = Sj_double
        Sj_double = None

    if 'Dnm' in arry:
        Dnm = np.asarray(arry['Dnm'], dtype=float)
        shift = np.zeros(3) if shift_bohr is None else np.asarray(shift_bohr, dtype=float)
        # Dnm[n, m] = tau_n - tau_m: the replica (second block) sits at tau + shift
        arry['Dnm'] = np.concatenate(
            [
                np.concatenate([Dnm, Dnm - shift], axis=1),
                np.concatenate([Dnm + shift, Dnm], axis=1),
            ],
            axis=0,
        )

    if centres_bohr is not None and shift_bohr is not None:
        centres = np.asarray(centres_bohr, dtype=float)
        arry['orbital_centres'] = np.vstack(
            [centres, centres + np.asarray(shift_bohr, dtype=float)]
        )

    # If the SOC is included pertubative
    if 'adhoc_SO' in attr and (attr['adhoc_SO']):
        if 'lambda_p' in arry:
            arry['lambda_p'] = np.append(arry['lambda_p'], arry['lambda_p'])
        if 'lambda_d' in arry:
            arry['lambda_d'] = np.append(arry['lambda_d'], arry['lambda_d'])
        if 'orb_pseudo' in arry:
            arry['orb_pseudo'] = np.append(arry['orb_pseudo'], arry['orb_pseudo'])
