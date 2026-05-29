import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def do_pdos(data_controller, emin, emax, ne, delta):
    """Compute the projected density of states with fixed Gaussian smearing.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``E_k`` (shape ``(nkpnts, nawf, nspin)``),
        ``v_k`` (shape ``(nkpnts, nawf, nawf, nspin)``).
        Required attributes: ``nawf``, ``nspin``, ``nkpnts``, ``shift``.
    emin : float
        Lower energy bound of the PDOS grid (eV).
    emax : float
        Upper energy bound; clipped to ``min(shift, emax)`` (eV).
    ne : int
        Number of energy grid points.
    delta : float
        Gaussian smearing width (eV).

    Returns
    -------
    None
        Writes the following files to the output directory via
        :meth:`DataController.write_file_row_col`:

        - ``{m}_pdos_{ispin}.dat`` for each orbital ``m`` and spin channel.
        - ``pdos_sum_{ispin}.dat`` — the total projected DOS.

    Notes
    -----
    The PDOS for orbital ``m`` at energy ``E`` is

    .. math::

        P_m(E) = \\frac{1}{N_k \\sqrt{\\pi}\\,\\delta}
            \\sum_{\\mathbf{k},n} |\\langle m | n, \\mathbf{k} \\rangle|^2
            \\exp\\!\\left(-\\left(\\frac{E - \\varepsilon_{n\\mathbf{k}}}{\\delta}\\right)^2\\right)

    MPI reduction (:func:`MPI.Reduce`) is used to accumulate the per-rank
    partial sums on rank 0.
    """
    arrays, attributes = data_controller.data_dicts()

    nawf = attributes['nawf']
    nspin = attributes['nspin']
    nktot = attributes['nkpnts']

    # PDOS calculation with gaussian smearing

    emax = np.amin(np.array([attributes['shift'], emax]))
    ene = np.linspace(emin, emax, ne)

    for ispin in range(nspin):
        pdosaux = np.zeros((nawf, ne), dtype=float)
        v_kaux = np.real(np.abs(arrays['v_k'][:, :, :, ispin]) ** 2)

        E_k = arrays['E_k'][:, :, ispin]

        for n in range(ne):
            taux = np.exp(-(((ene[n] - E_k) / delta) ** 2)) / np.sqrt(np.pi)
            for m in range(nawf):
                pdosaux[m, n] += np.sum(taux * v_kaux[:, m, :])

        pdos = np.zeros((nawf, ne), dtype=float) if rank == 0 else None

        comm.Reduce(pdosaux, pdos, op=MPI.SUM)
        pdosaux = None

        if rank == 0:
            pdos /= float(nktot) * np.sqrt(np.pi) * delta

        pdos_sum = np.zeros(ne, dtype=float) if rank == 0 else None

        for m in range(nawf):
            if rank == 0:
                pdos_sum += pdos[m]
            fpdos = '%d_pdos_%d.dat' % (m, ispin)
            data_controller.write_file_row_col(fpdos, ene, (pdos[m] if rank == 0 else None))

        fpdos = 'pdos_sum_%d.dat' % ispin
        data_controller.write_file_row_col(fpdos, ene, pdos_sum)


def do_pdos_adaptive(data_controller, emin, emax, ne):
    """Compute the projected density of states with adaptive smearing.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``E_k`` (shape ``(nkpnts, nawf, nspin)``),
        ``v_k`` (shape ``(nkpnts, nawf, nawf, nspin)``),
        ``deltakp`` (shape ``(nkpnts, nawf, nspin)`` or similar —
        per-k adaptive smearing widths).
        Required attributes: ``nawf``, ``nspin``, ``nkpnts``, ``shift``,
        ``smearing`` (``'gauss'`` or ``'m-p'``).
    emin : float
        Lower energy bound (eV).
    emax : float
        Upper energy bound; clipped to ``min(shift, emax)`` (eV).
    ne : int
        Number of energy grid points.

    Returns
    -------
    None
        Writes the following files to the output directory:

        - ``{m}_pdosdk_{ispin}.dat`` for each orbital ``m`` and spin channel.
        - ``pdosdk_sum_{ispin}.dat`` — the total adaptive PDOS.

    Notes
    -----
    At each energy point the per-k smearing width ``deltakp`` is used to
    evaluate either a Gaussian or a Methfessel-Paxton broadening function
    via :func:`smearing.gaussian` or :func:`smearing.metpax`.  This
    approach follows Yates *et al.*, Phys. Rev. B **75**, 195121 (2007).
    """
    from ..utils.smearing import metpax, gaussian

    arrays = data_controller.data_arrays
    attributes = data_controller.data_attributes

    # PDoS Calculation with Gaussian Smearing
    emax = np.amin(np.array([attributes['shift'], emax]))
    ene = np.linspace(emin, emax, ne)

    nawf = attributes['nawf']

    for ispin in range(attributes['nspin']):
        E_k = np.real(arrays['E_k'][:, :, ispin])

        pdosaux = np.zeros((nawf, ne), dtype=float)

        v_kaux = np.real(np.abs(arrays['v_k'][:, :, :, ispin]) ** 2)

        taux = np.zeros((arrays['deltakp'].shape[0], nawf), dtype=float)

        for n in range(ne):
            if attributes['smearing'] == 'gauss':
                taux = gaussian(ene[n], E_k, arrays['deltakp'][:, :, ispin])
            elif attributes['smearing'] == 'm-p':
                taux = metpax(ene[n], E_k, arrays['deltakp'][:, :, ispin])
            for i in range(nawf):
                # Adaptive Gaussian Smearing
                pdosaux[i, n] += np.sum(taux * v_kaux[:, i, :])

        pdos = np.zeros((nawf, ne), dtype=float) if rank == 0 else None

        comm.Reduce(pdosaux, pdos, op=MPI.SUM)
        pdosaux = None

        if rank == 0:
            pdos /= float(attributes['nkpnts'])

        pdos_sum = np.zeros(ne, dtype=float) if rank == 0 else None

        for m in range(nawf):
            if rank == 0:
                pdos_sum += pdos[m]
            fpdos = '%d_pdosdk_%d.dat' % (m, ispin)
            data_controller.write_file_row_col(fpdos, ene, (pdos[m] if rank == 0 else None))

        fpdos = 'pdosdk_sum_%d.dat' % ispin
        data_controller.write_file_row_col(fpdos, ene, pdos_sum)
