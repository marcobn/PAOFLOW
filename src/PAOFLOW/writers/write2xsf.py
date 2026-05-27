import numpy as np
from mendeleev import element


def write2xsf(data_controller, filename, data=None):
    """Write crystal structure and optional 3-D volumetric data in XSF format.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``atoms`` (list of element symbols), ``a_vectors``
        (shape ``(3, 3)`` in units of ``alat``), ``tau``
        (shape ``(natoms, 3)`` in Bohr radii).
        Required attribute: ``alat`` (lattice constant in Bohr radii).
    filename : str
        Absolute or relative path of the XSF output file.
    data : Optional[np.ndarray]
        Volumetric scalar field on a real-space grid of shape
        ``(nr1, nr2, nr2)``.  If ``None``, only the crystal structure
        block is written.  Complex arrays are converted to
        :math:`|\\psi|^2` before writing.

    Returns
    -------
    None
        Creates (or overwrites) the file at ``filename``.

    Notes
    -----
    The file is written in the XCrySDen Structure Format (XSF), which is
    supported by XCrySDen and VESTA for structure and charge-density
    visualisation.  The structure block includes primitive cell vectors
    (in Bohr radii) and atomic positions with atomic numbers obtained from
    the ``mendeleev`` library.

    When ``data`` is provided, a ``BEGIN_BLOCK_DATAGRID_3D`` section is
    appended.  The grid is extended by one point in each direction (periodic
    boundary) to satisfy XSF convention, i.e. the grid size written is
    ``(nr1+1, nr2+1, nr3+1)``.
    """
    arry, attr = data_controller.data_dicts()

    fileobj = open(filename, 'w')
    atoms = arry['atoms']

    fileobj.write('CRYSTAL\n')

    fileobj.write('PRIMVEC\n')
    for i in range(3):
        fileobj.write(' %.14f %.14f %.14f\n' % tuple(arry['a_vectors'][i] * attr['alat']))

    fileobj.write('PRIMCOORD\n')
    fileobj.write(str(len(atoms)) + ' 1\n')
    for na in range(len(atoms)):
        atom = element(arry['atoms'][na])
        fileobj.write(' %2d' % atom.atomic_number)
        fileobj.write(' %20.14f %20.14f %20.14f' % tuple(arry['tau'][na]))
        fileobj.write('\n')

    if data is None:
        fileobj.close()
        return

    fileobj.write('BEGIN_BLOCK_DATAGRID_3D\n')
    fileobj.write(' data\n')
    fileobj.write(' BEGIN_DATAGRID_3Dgrid#1\n')

    data = np.asarray(data)
    if data.dtype == np.complex128:
        data = np.abs(data) ** 2

    shape = data.shape
    fileobj.write('  %d %d %d\n' % (shape[0] + 1, shape[1] + 1, shape[2] + 1))

    origin = np.zeros(3)
    fileobj.write('  %f %f %f\n' % tuple(origin))
    # These are the actual coordinates for the data
    for i in range(3):
        fileobj.write('  %f %f %f\n' % tuple(np.array(arry['a_vectors'][i]) * attr['alat']))

    for k in range(shape[2] + 1):
        for j in range(shape[1] + 1):
            fileobj.write('   ')
            for i in range(shape[0] + 1):
                fileobj.write('%12.8e ' % (data[i % shape[0], j % shape[1], k % shape[2]]))
            fileobj.write('\n')

    fileobj.write(' END_DATAGRID_3D\n')
    fileobj.write('END_BLOCK_DATAGRID_3D\n')
    fileobj.close()
