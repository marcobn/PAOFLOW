from __future__ import annotations

from typing import Any, Mapping

from mpi4py import MPI

from PAOFLOW.transport.conductor_pipeline import compute_conductor_results
from PAOFLOW.transport.data import ConductorData
from PAOFLOW.transport.results import TransportResults


def apply_onsite_shifts(data: ConductorData, config: Mapping[str, float] | None) -> None:
    """Apply configured rigid on-site energy shifts onto a conductor model.

    Parameters
    ----------
    data : ConductorData
        Conductor input model to update in place.
    config : Mapping[str, float] or None
        Mapping with ``shift_L``, ``shift_C``, ``shift_R``, and ``shift_corr``.
        When ``None``, the model is left unchanged.

    Returns
    -------
    None
        Updates ``data.shift_L``, ``data.shift_C``, ``data.shift_R``, and
        ``data.shift_corr`` in place when ``config`` is provided.
    """
    if config is None:
        return
    data.shift_L = config['shift_L']
    data.shift_C = config['shift_C']
    data.shift_R = config['shift_R']
    data.shift_corr = config['shift_corr']


def apply_lead_convergence(data: ConductorData, config: Mapping[str, Any] | None) -> None:
    """Apply configured lead transfer-matrix iteration settings onto a model.

    Parameters
    ----------
    data : ConductorData
        Conductor input model to update in place.
    config : Mapping[str, Any] or None
        Mapping with ``niterx``, ``transfer_thr``, ``nprint``, ``nfailx``, and
        ``surface``. When ``None``, the model is left unchanged.

    Returns
    -------
    None
        Updates ``data.iteration.*`` and ``data.advanced.surface`` in place when
        ``config`` is provided.
    """
    if config is None:
        return
    data.iteration.niterx = config['niterx']
    data.iteration.transfer_thr = config['transfer_thr']
    data.iteration.nprint = config['nprint']
    data.iteration.nfailx = config['nfailx']
    data.advanced.surface = config['surface']


def apply_eigenchannels(data: ConductorData, config: Mapping[str, Any] | None) -> None:
    """Apply configured eigenchannel decomposition settings onto a model.

    Parameters
    ----------
    data : ConductorData
        Conductor input model to update in place.
    config : Mapping[str, Any] or None
        Mapping with ``do_eigenchannels``, ``neigchnx``, ``do_eigplot``,
        ``ie_eigplot``, and ``ik_eigplot``. When ``None``, the model is left
        unchanged.

    Returns
    -------
    None
        Updates ``data.symmetry.*`` eigenchannel fields in place when ``config``
        is provided.
    """
    if config is None:
        return
    data.symmetry.do_eigenchannels = config['do_eigenchannels']
    data.symmetry.neigchnx = config['neigchnx']
    data.symmetry.do_eigplot = config['do_eigplot']
    data.symmetry.ie_eigplot = config['ie_eigplot']
    data.symmetry.ik_eigplot = config['ik_eigplot']


def require_hamiltonian_blocks(
    conductor_data: ConductorData | None,
    blc_blocks: Mapping[str, Any] | None,
) -> None:
    """Ensure Hamiltonian blocks were built before transport computations.

    Parameters
    ----------
    conductor_data : ConductorData or None
        Validated conductor input model, or ``None`` if not yet built.
    blc_blocks : Mapping[str, Any] or None
        Hamiltonian block operators, or ``None`` if not yet built.

    Raises
    ------
    RuntimeError
        If either ``conductor_data`` or ``blc_blocks`` is ``None``.
    """
    if conductor_data is None or blc_blocks is None:
        raise RuntimeError('Call build_hamiltonian_blocks(...) before transport computations.')


def require_grid_config(energy_grid_config: Mapping[str, Any] | None) -> None:
    """Ensure the energy grid was configured before full-grid calculations.

    Parameters
    ----------
    energy_grid_config : Mapping[str, Any] or None
        Stored energy-grid configuration, or ``None`` if not yet configured.

    Raises
    ------
    RuntimeError
        If ``energy_grid_config`` is ``None``.
    """
    if energy_grid_config is None:
        raise RuntimeError(
            'Call configure_energy_grid(...) before full-grid transport calculations.'
        )


def compute_full_grid_results(
    *,
    conductor_data: ConductorData | None,
    blc_blocks: Mapping[str, Any] | None,
    energy_grid_config: Mapping[str, Any] | None,
    cached_results: TransportResults | None,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> TransportResults:
    """Return cached full-grid transport results or compute them if absent.

    Parameters
    ----------
    conductor_data : ConductorData or None
        Validated conductor input model.
    blc_blocks : Mapping[str, Any] or None
        Hamiltonian block operators.
    energy_grid_config : Mapping[str, Any] or None
        Stored energy-grid configuration used to gate full-grid work.
    cached_results : TransportResults or None
        Previously computed results to reuse, or ``None`` to force a fresh
        computation.
    comm : MPI.Comm, optional
        Communicator used for work distribution and reductions.
        Default is ``MPI.COMM_WORLD``.

    Returns
    -------
    TransportResults
        The reused ``cached_results`` when non-``None``, otherwise the freshly
        computed full-grid observables.

    Raises
    ------
    RuntimeError
        If Hamiltonian blocks or the energy-grid configuration are unavailable.
    """
    require_hamiltonian_blocks(conductor_data, blc_blocks)
    require_grid_config(energy_grid_config)
    if cached_results is not None:
        return cached_results
    return compute_conductor_results(
        data=conductor_data,
        blc_blocks=blc_blocks,
        comm=comm,
    )
