# this version works only for non-magnetic or non-collienar calculations
def wave_function_site_projection(data_controller):
    """Write real-space site-projected wavefunction weights to a data file.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``tau`` (shape ``(natoms, 3)`` in Bohr radii),
        ``naw`` (number of orbitals per atom), ``v_k``
        (shape ``(nkpnts, nawf, bnd, nspin)``), ``bands_proj``
        (list of band indices to project).
        Required attributes: ``nawf``, ``do_spin_orbit``, ``dimension``,
        ``k_proj`` (k-point index for the projection), ``opath``.

    Returns
    -------
    None
        Writes one text file per band index in ``bands_proj`` to
        ``{opath}/site-projected-wave-function-{bnd_idx}.dat``.
        Each line contains the atomic position coordinates followed by
        the total site weight plus a small offset (0.0001) for plotting.
        The number of coordinate columns depends on ``dimension``:
        3 columns for 3-D, 2 columns for 2-D and 1-D systems.

    Notes
    -----
    For each atom :math:`n` and selected band index :math:`b`, the
    site-projected wavefunction weight at a fixed k-point is

    .. math::

        w_n = \\sum_{\\mu \\in n} |v^\\mu_{bk}|^2

    where the sum runs over all orbital indices :math:`\\mu` belonging to
    atom :math:`n`.  Atomic positions are converted from Bohr radii to
    Ångström using ``ANGSTROM_AU`` before writing.

    When spin–orbit coupling is active (``do_spin_orbit``), the wavefunction
    basis is doubled and both the spin-up sector
    :math:`[\\text{idx}, \\text{fdx})` and spin-down sector
    :math:`[\\text{idx}+s, \\text{fdx}+s)` (with :math:`s = N_{\\text{wf}}/2`)
    are included in the weight sum.

    This function is valid only for non-magnetic or non-collinear calculations
    (spin channel 0 is used exclusively).
    """
    import numpy as np
    from os.path import join

    from .constants import ANGSTROM_AU

    arry, attr = data_controller.data_dicts()

    tau = arry['tau'] / ANGSTROM_AU
    naw, v_k = arry['naw'], arry['v_k']
    bands, k_index = arry['bands_proj'], attr['k_proj']

    do_spin_orbit = attr['do_spin_orbit']
    nawf, dim = attr['nawf'], attr['dimension']

    for idb in range(len(bands)):
        bnd_idx = bands[idb]  # index of the band to be projected
        # open file
        f = open(join(attr['opath'], 'site-projected-wave-function-' + str(bnd_idx) + '.dat'), 'w')
        for n in range(tau.shape[0]):
            # Do to the doubling of the Hamiltonian when SOC is included in the PAO Hamiltonian.
            if do_spin_orbit:
                # creating masks to consirer only the n site.
                # seting up the nonzero parts of the mask and the wave-function
                idx = int(np.sum(naw[0:n]))  # initial
                fdx = int(idx + naw[n])  # final
                s = int(nawf / 2)

                usector_idx = np.arange(idx, fdx, dtype=int)
                dsector_idx = np.arange(idx + s, fdx + s, dtype=int)
                idx_list = list(np.append(usector_idx, dsector_idx))

                total = 0
                total += np.sum(
                    np.absolute(np.square(v_k[k_index : k_index + 1, idx_list, bnd_idx, 0]))
                )

            else:  # no SOC or SOC form QE.
                # creating masks to consirer only the n site.
                # seting up the nonzero parts of the mask and the wave-function
                idx = int(np.sum(naw[0:n]))  # initial
                fdx = int(idx + naw[n])  # final

                total = 0
                total += np.sum(
                    np.absolute(np.square(v_k[k_index : k_index + 1, idx:fdx, bnd_idx, 0]))
                )

            if dim == 3:  # ploting for 3D system
                # we sum a very small part 0.0001 for ploting purpose.
                f.write(
                    ('%5.4f %5.4f %5.4f %5.4f \n')
                    % (tau[n, 0], tau[n, 1], tau[n, 2], total + 0.0001)
                )
            if dim == 2:  # ploting for 2D system
                # we sum a very small part 0.0001 for ploting purpose.
                f.write(('%5.4f %5.4f %5.4f  \n') % (tau[n, 0], tau[n, 1], total + 0.0001))
            if dim == 1:  # ploting for 1D system
                # we sum a very small part 0.0001 for ploting purpose.
                f.write(('%5.4f %5.4f  \n') % (tau[n, 2], total + 0.0001))

        f.close()
