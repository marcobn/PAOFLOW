import numpy as np

import PAOFLOW.transport.io.log_module as log
from PAOFLOW.DataController import DataController
from PAOFLOW.transport.data import ConductorData
from PAOFLOW.transport.grid.rgrid import get_rgrid
from PAOFLOW.transport.io.write_data import populate_real_space_hamiltonian
from PAOFLOW.transport.io.write_header import headered_function
from PAOFLOW.transport.utils.timing import timed_function


@timed_function('atmproj_to_internal')
@headered_function('Conductor Initialization')
def parse_atomic_proj(
    data: ConductorData, data_controller: DataController
) -> dict[str, np.ndarray]:
    opts = data.atomic_proj

    arry, attr = data_controller.data_dicts()

    alat = float(attr['alat'])
    shape = mesh_shape(arry, attr)
    grid_ordered = uses_full_grid_order(arry, attr)
    if grid_ordered:
        # H(k) on the full uniform mesh in FFT order (expanded wedge, models): the k-points are
        # the mesh itself, k = (i/nk1, j/nk2, l/nk3), uniformly weighted.
        k_crystal = mesh_crystal_coordinates(shape)
        arry['wk'] = np.full(k_crystal.shape[0], 1.0 / k_crystal.shape[0])
        reciprocal = np.linalg.inv(np.asarray(arry['a_vectors'], dtype=float)).T  # 2 pi / alat
        arry['vkpts_cartesian'] = (k_crystal @ reciprocal).T * (2.0 * np.pi / alat)  # bohr^-1
    else:
        arry['wk'] = arry['kpnts_wght'] / np.sum(arry['kpnts_wght'])
        arry['vkpts_cartesian'] = arry['kpnts'].T * (2.0 * np.pi / alat)  # bohr^-1

    log.log_proj_summary(data_controller, data)

    # On the grid, H(k) is rebuilt from HRs, the authoritative Hamiltonian: identical to Hks after
    # pao_hamiltonian, and the only one after doubling_Hamiltonian or for a model.
    hk_data = reshape_pao_hamiltonian(data_controller, from_real_space=grid_ordered)

    # The real-space grid is the one conjugate to the mesh of the PAO Hamiltonian: the transverse
    # lattice vectors carry the in-plane hoppings, the ones along the transport direction the
    # couplings between layers (it was hard-coded to (1, 1, 4), right only for wires in vacuum).
    nk = np.asarray(shape, dtype=int)
    nr = nk
    ivr, wr = get_rgrid(nr)
    hk_data.update({'ivr': ivr, 'wr': wr, 'nk': nk, 'nr': nr})
    arry.update(hk_data)

    populate_real_space_hamiltonian(data_controller, hk_data, opts.do_overlap_transformation)

    return hk_data


def mesh_shape(arry: dict, attr: dict) -> tuple[int, int, int]:
    """Shape ``(nk1, nk2, nk3)`` of the mesh of the PAO Hamiltonian (that of ``HRs``).

    Parameters
    ----------
    arry, attr : dict
        Data-controller arrays (``HRs``) and attributes (``nk1``, ``nk2``, ``nk3``).

    Returns
    -------
    tuple of int
    """
    hrs = arry.get('HRs')
    if hrs is not None and np.ndim(hrs) == 6:
        return tuple(int(n) for n in np.shape(hrs)[2:5])
    return int(attr['nk1']), int(attr['nk2']), int(attr['nk3'])


def mesh_crystal_coordinates(shape: tuple[int, int, int]) -> np.ndarray:
    """k-points ``(i/nk1, j/nk2, l/nk3)`` of a uniform mesh in the C order of ``Hks``/``HRs``.

    Parameters
    ----------
    shape : tuple of int

    Returns
    -------
    ndarray, shape ``(nk1 nk2 nk3, 3)``
    """
    axes = [np.arange(n) / n for n in shape]
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)


def uses_full_grid_order(arry: dict, attr: dict) -> bool:
    """True when ``H(k)`` is on the full mesh in FFT order rather than in the order of ``kpnts``.

    Parameters
    ----------
    arry, attr : dict
        Data-controller arrays and attributes.

    Returns
    -------
    bool
        True after :meth:`PAOFLOW.pao_hamiltonian` with ``expand_wedge=True`` (the default: the
        irreducible wedge is unfolded onto the grid) and for model Hamiltonians, which carry only
        ``HRs``; False with ``expand_wedge=False``, where the Hamiltonian follows the k-points of
        the DFT run (``kpnts``, ``kpnts_wght``).
    """
    if 'kpnts_wght' not in arry or 'kpnts' not in arry:
        return True
    return bool(attr.get('expand_wedge', False))


def reshape_pao_hamiltonian(
    data_controller: DataController, from_real_space: bool = False
) -> dict[str, np.ndarray]:
    """``H(k)``, ``H(R)`` and the overlaps in the layout of the transport code.

    Parameters
    ----------
    data_controller : DataController
        Provides ``Hks`` (or ``HRs``), ``nawf``, ``nspin`` and, optionally, ``Sks``/``SRs``.
    from_real_space : bool, optional
        Build ``H(k)`` on the mesh as the FFT of ``HRs`` (models have no ``Hks``, and
        :meth:`PAOFLOW.doubling_Hamiltonian` changes ``HRs`` only),
        :math:`H(\\mathbf k_j) = \\sum_{\\mathbf n} e^{-2\\pi i\\,\\mathbf k_j\\cdot\\mathbf n}
        \\mathtt{HRs}[\\mathbf n]`.

    Returns
    -------
    dict
        ``Hk`` and ``HR`` with shape ``(nspin, nk, nawf, nawf)``; ``Sk``/``SR`` or None.
    """
    arry, attr = data_controller.data_dicts()
    if from_real_space:
        Hks_raw = np.fft.fftn(np.asarray(arry['HRs']), axes=(2, 3, 4))
    else:
        Hks_raw = data_controller.full_hamiltonian_k()  # (nawf, nawf, nk1, nk2, nk3, nspin)
    HRs_raw = arry['HRs']  # shape: (nawf, nawf, nk1, nk2, nk3, nspin)
    nspin = attr['nspin']
    nawf = attr['nawf']
    nkpnts = int(np.prod(np.shape(HRs_raw)[2:5]))

    # reshape to (nawf, nawf, nkpnts, nspin)
    Hks_reshaped = Hks_raw.reshape((nawf, nawf, nkpnts, nspin))
    # transpose to (nspin, nkpnts, nawf, nawf)
    Hk = np.transpose(Hks_reshaped, (3, 2, 1, 0))

    HRs_reshaped = HRs_raw.reshape((nawf, nawf, nkpnts, nspin))
    HR = np.transpose(HRs_reshaped, (3, 2, 1, 0))

    Sks_raw = (
        arry['Sks'] if 'Sks' in arry else None
    )  # TODO Paoflow uses the acbn0 flag to transform Sk in case of non-orthogonality. Check if we need Sks computed with or without the acbn0 flag to get the right answer. In the current implementation, Sks is without the acbn0 flag.
    Sk = Sks_raw[:, :nawf, :] if Sks_raw is not None else None
    Sk = np.transpose(Sk, (2, 1, 0)) if Sk is not None else None

    SRs_raw = arry['SRs'] if 'SRs' in arry else None
    SR = SRs_raw[:, :nawf, :] if SRs_raw is not None else None
    SR = np.transpose(SR, (2, 1, 0)) if SR is not None else None
    return {'Hk': Hk, 'Sk': Sk, 'HR': HR, 'SR': SR}
