from __future__ import annotations

from typing import Any

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

import PAOFLOW.transport.io.log_module as log
from PAOFLOW.DataController import DataController
from PAOFLOW.transport.conductor_pipeline import run_conductor
from PAOFLOW.transport.current_pipeline import run_current_from_file
from PAOFLOW.transport.io.input_parameters import ConductorData
from PAOFLOW.transport.utils.memusage import MemoryTracker
from PAOFLOW.transport.utils.timing import global_timing
from PAOFLOW.transport.workspace.prepare_data import (
    prepare_conductor_data,
    prepare_conductor_runtime,
)


class Transport:
    """User-facing transport orchestrator with direct-argument APIs.

    Parameters
    ----------
    data_controller : DataController
        Shared PAOFLOW ``DataController`` used by transport preparation stages.
    """

    def __init__(self, data_controller: DataController):
        self.data_controller = data_controller

    def conductor(
        self,
        *,
        datafile_C: str,
        dimC: int,
        dimL: int,
        dimR: int,
        datafile_L: str,
        datafile_R: str,
        emin: float,
        emax: float,
        ne: int,
        delta: float,
        nk: list[int] | tuple[int, int] = (0, 0),
        formula: str = 'landauer',
        transport_direction: int = 1,
        carriers: str = 'electrons',
        work_dir: str = './',
        output_dir: str = './',
        postfix: str = '',
        **kwargs: Any,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.complex128] | None,
        NDArray[np.complex128] | None,
        NDArray[np.complex128] | None,
        NDArray[np.float64],
    ]:
        """Run conductor transport from direct Python arguments.

        Parameters
        ----------
        datafile_C : str
            Path to the conductor Hamiltonian/projection input.
        dimC : int
            Conductor block dimension.
        dimL : int
            Left lead block dimension.
        dimR : int
            Right lead block dimension.
        datafile_L : str
            Path to the left-lead input.
        datafile_R : str
            Path to the right-lead input.
        emin : float
            Minimum energy in eV.
        emax : float
            Maximum energy in eV.
        ne : int
            Number of energy points.
        delta : float
            Broadening parameter.
        nk : list[int] or tuple[int, int], optional
            2D k-grid parameters for ``kpoint_grid.nk``.
        formula : str, optional
            Conductance formula. Default is ``'landauer'``.
        transport_direction : int, optional
            Transport direction index in ``{1, 2, 3}``.
        carriers : str, optional
            Carrier type (``'electrons'`` or ``'phonons'``).
        work_dir : str, optional
            Working directory for transport assets.
        output_dir : str, optional
            Output directory for generated transport files.
        postfix : str, optional
            Output postfix appended to default transport file names.
        **kwargs : Any
            Additional optional ``ConductorData`` fields (for example
            ``write_kdata``, ``write_gf``, ``niterx``, ``transfer_thr``, or
            self-energy file paths for generalized formulas).

        Returns
        -------
        tuple
            ``(conduct, dos, conduct_k, dos_k, gf_out, rsgmL_out, rsgmR_out, egrid)``
            returned by ``run_conductor``.
        """
        input_values: dict[str, Any] = {
            'datafile_C': datafile_C,
            'datafile_L': datafile_L,
            'datafile_R': datafile_R,
            'dimC': dimC,
            'dimL': dimL,
            'dimR': dimR,
            'emin': emin,
            'emax': emax,
            'ne': ne,
            'delta': delta,
            'nk': list(nk),
            'conduct_formula': formula,
            'transport_direction': transport_direction,
            'carriers': carriers,
            'work_dir': work_dir,
            'output_dir': output_dir,
            'postfix': postfix,
        }
        input_values.update(kwargs)

        data = ConductorData(filename='<direct-arguments>', validate=True, **input_values)
        log.initialize_logger(
            self.data_controller,
            log_file_name=f'transport_conductor{data.file_names.postfix}.log',
        )
        memory_tracker = MemoryTracker()

        prepare_conductor_data(data, self.data_controller)
        ham_sys = prepare_conductor_runtime(data, self.data_controller, memory_tracker)

        results = run_conductor(data=data, blc_blocks=ham_sys.blocks, comm=MPI.COMM_WORLD)

        if MPI.COMM_WORLD.Get_rank() == 0:
            global_timing.report()
            memory_tracker.report(include_real_memory=True)

        return results

    def current(
        self,
        *,
        filein: str,
        fileout: str,
        bias_min: float,
        bias_max: float,
        nbias: int,
        sigma: float,
        mu_L: float,
        mu_R: float,
    ) -> NDArray[np.float64]:
        """Run current-vs-bias calculation from direct Python arguments.

        Parameters
        ----------
        filein : str
            Input transmission file path.
        fileout : str
            Output path for current-vs-bias data.
        bias_min : float
            Minimum bias in volts.
        bias_max : float
            Maximum bias in volts.
        nbias : int
            Number of bias samples.
        sigma : float
            Smearing parameter used in Fermi occupations.
        mu_L : float
            Left chemical potential.
        mu_R : float
            Right chemical potential.

        Returns
        -------
        NDArray[np.float64]
            Current values aligned with the generated bias grid.
        """
        log.initialize_logger(self.data_controller, log_file_name='transport_current.log')

        current_data = {
            'fileout': fileout,
            'mu_L': mu_L,
            'mu_R': mu_R,
            'sigma': sigma,
        }
        return run_current_from_file(
            data=current_data,
            filein=filein,
            bias_min=bias_min,
            bias_max=bias_max,
            nbias=nbias,
        )
