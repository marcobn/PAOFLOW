import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def do_dos(data_controller, emin, emax, ne, delta):
    """Compute the total electronic density of states with fixed Gaussian smearing.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``E_k`` (shape ``(nkpnts, nawf, nspin)``).
        Required attributes: ``bnd``, ``nkpnts``, ``nspin``, ``shift``,
        ``verbose``.
    emin : float
        Lower energy bound of the DOS grid (eV).
    emax : float
        Upper energy bound; clipped to ``min(shift, emax)`` (eV).
    ne : int
        Number of energy grid points.
    delta : float
        Gaussian smearing width (eV).

    Returns
    -------
    None
        Adds or updates the following entries in ``data_controller.data_arrays``:

        - ``dos`` : np.ndarray, shape ``(ne,)`` — the DOS on the energy grid
          (broadcast to all ranks).

        Writes ``dos_{ispin}.dat`` via
        :meth:`DataController.write_file_row_col`.

    Notes
    -----
    The DOS is computed as

    .. math::

        g(E) = \\frac{N_{\\rm bnd}}{N_{\\rm tot} \\sqrt{\\pi}\\,\\delta}
            \\sum_{\\mathbf{k},n} \\exp\\!\\left(
            -\\left(\\frac{E - \\varepsilon_{n\\mathbf{k}}}{\\delta}\\right)^2\\right)

    where :math:`N_{\\rm tot} = N_k \\times N_{\\rm bnd}`.  MPI reduction is
    used to sum partial contributions from each rank.
    """
    arry, attr = data_controller.data_dicts()
    bnd = attr['bnd']
    netot = attr['nkpnts'] * bnd
    emax = np.amin(np.array([attr['shift'], emax]))
    arry['dos'] = np.empty((ne,), dtype=float)
    # DOS calculation with gaussian smearing
    ene = np.linspace(emin, emax, ne)

    if rank == 0 and attr['verbose']:
        print('Writing DoS Files')

    for ispin in range(attr['nspin']):
        dosaux = np.zeros((ne), order='C')

        E_k = arry['E_k'][:, :bnd, ispin]

        for n in range(ne):
            dosaux[n] = np.sum(np.exp(-(((ene[n] - E_k) / delta) ** 2)))

        dos = np.zeros((ne), dtype=float) if rank == 0 else None

        comm.Reduce(dosaux, dos, op=MPI.SUM)
        dosaux = None

        if rank == 0:
            dos *= float(bnd) / (float(netot) * np.sqrt(np.pi) * delta)
            arry['dos'] = dos
        fdos = 'dos_%s.dat' % str(ispin)
        data_controller.write_file_row_col(fdos, ene, dos)
        data_controller.broadcast_single_array('dos', dtype=float)
        # return dos if rank==0 else None


#: Adaptive widths below this (eV) are numerically zero: the velocity vanishes by symmetry on the grid
#: (band extrema and saddle points of a model, a decoupled band), and a Gaussian of zero width is 0/0.
ZERO_WIDTH = 1.0e-6


def resolvable_widths(widths: np.ndarray, spacing: float) -> np.ndarray:
    """Adaptive widths with the numerically zero ones replaced by the energy-grid spacing.

    Parameters
    ----------
    widths : ndarray
        Yates widths ``deltakp`` (eV), any shape.
    spacing : float
        Spacing of the energy grid of the DOS (eV).

    Returns
    -------
    ndarray
        ``widths`` with every entry below :data:`ZERO_WIDTH` set to ``max(spacing, ZERO_WIDTH)``;
        finite widths are unchanged.

    Notes
    -----
    The width :math:`a\\,|\\nabla_{\\mathbf k}\\varepsilon|\\,\\delta k` vanishes where the velocity does,
    exactly on the grid points of high symmetry of a tight-binding model and on flat bands. Those
    states would otherwise give ``NaN`` (zero width) or a spike of height :math:`\\sim 1/\\eta` at one
    energy; with the grid spacing each one counts once. Widths that are small but finite keep the
    warning of :func:`do_dos_adaptive` and the ``min_width`` option of ``adaptive_smearing``.
    """
    return np.where(widths < ZERO_WIDTH, max(float(spacing), ZERO_WIDTH), widths)


def do_dos_adaptive(data_controller, emin, emax, ne):
    """Compute the total density of states with adaptive smearing.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``E_k`` (shape ``(nkpnts, nawf, nspin)``),
        ``deltakp`` (per-k adaptive smearing widths).
        Required attributes: ``bnd``, ``nkpnts``, ``nspin``, ``smearing``
        (``'gauss'`` or ``'m-p'``), ``verbose``.
    emin : float
        Lower energy bound (eV).
    emax : float
        Upper energy bound (eV).
    ne : int
        Number of energy grid points.

    Returns
    -------
    None
        Adds or updates the following entries in ``data_controller.data_arrays``:

        - ``dosdk`` : np.ndarray, shape ``(ne,)`` — the adaptive DOS
          (broadcast to all ranks).

        Writes ``dosdk_{ispin}.dat`` via
        :meth:`DataController.write_file_row_col`.

    Notes
    -----
    At each energy point the per-k adaptive smearing width from
    ``deltakp`` is passed to either :func:`smearing.gaussian` (Gaussian
    broadening) or :func:`smearing.metpax` (Methfessel-Paxton broadening).
    This approach follows Yates *et al.*, Phys. Rev. B **75**, 195121 (2007).
    """
    from ..utils.smearing import gaussian, metpax

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    arry, attr = data_controller.data_dicts()

    # DOS calculation with adaptive smearing
    ene = np.linspace(emin, emax, ne)
    arry['dosdk'] = np.empty((ne,), dtype=float)

    bnd = attr['bnd']
    netot = attr['nkpnts'] * bnd

    if rank == 0 and attr['verbose']:
        print('Writing Adaptive DoS Files')

    spacing = ene[1] - ene[0] if ne > 1 else 0.0
    for ispin in range(attr['nspin']):
        E_k = arry['E_k'][:, :bnd, ispin].reshape(arry['E_k'].shape[0] * bnd)
        delta = resolvable_widths(np.ravel(arry['deltakp'][:, :bnd, ispin], order='C'), spacing)
        narrow = int(np.count_nonzero(delta < spacing))
        if narrow and rank == 0:
            print(
                'WARNING: %d of %d states have an adaptive width below the energy grid spacing '
                '(%.2g eV) and are under-counted in the DOS: raise min_width in '
                'adaptive_smearing() or ne' % (narrow, delta.size, spacing)
            )

        dosaux = np.zeros((ne), dtype=float)

        for n in range(ne):
            if attr['smearing'] == 'gauss':
                # adaptive Gaussian smearing
                dosaux[n] = np.sum(gaussian(ene[n], E_k, delta))

            elif attr['smearing'] == 'm-p':
                # adaptive Methfessel and Paxton smearing
                dosaux[n] = np.sum(metpax(ene[n], E_k, delta))

        dos = np.zeros((ne), dtype=float) if rank == 0 else None
        comm.Reduce(dosaux, dos, op=MPI.SUM)
        dosaux = None

        if rank == 0:
            dos *= float(bnd) / netot
            arry['dosdk'] = dos
        fdosdk = 'dosdk_%s.dat' % str(ispin)
        data_controller.write_file_row_col(fdosdk, ene, dos)
        data_controller.broadcast_single_array('dosdk', dtype=float)
