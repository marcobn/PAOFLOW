from PAOFLOW.DataController import DataController
from mpi4py import MPI
import datetime
import logging
from pathlib import Path

from PAOFLOW.transport import __version__
from PAOFLOW.transport.io.input_parameters import AtomicProjData, ConductorData

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

_logger = None
_logger_path = None

def initialize_logger(data_controller: DataController, log_file_name: str = "transport.log"):
    global _logger, _logger_path

    if rank != 0 or _logger is not None:
        return

    _, attr = data_controller.data_dicts()
    output_dir = attr["outputdir"]
    if output_dir is None:
        raise RuntimeError("Logger initialization failed: 'outputdir' not set in data_controller.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(output_dir) / log_file_name
    _logger_path = str(log_file)

    _logger = logging.getLogger("rank0_logger")
    _logger.setLevel(logging.INFO)
    _logger.propagate = False

    file_handler = logging.FileHandler(log_file, mode="w")
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    _logger.addHandler(file_handler)


def log_rank0(message: str):
    if rank == 0 and _logger is not None:
        _logger.info(message)


def log_parallelization_info(chunks: int, items: str):
    if rank == 0:
        log_rank0(
            f"Parallelization information: Each rank processes approximately {chunks} {items}."
        )


def log_section_start(name: str):
    log_rank0(f"Started {name}")


def log_section_end(name: str):
    log_rank0(f"Finished {name}")


def log_proj_data(
    proj_data: AtomicProjData,
    data: ConductorData,
) -> list[str]:
    lines = []
    lines.append("  Dimensions found in atomic_proj.{dat,xml}:")
    lines.append(f"    nbnds     : {proj_data.nbnds:>5}")
    lines.append(f"    nkpnts    : {proj_data.nkpnts:>5}")
    lines.append(f"    nspin    : {proj_data.nspin:>5}")
    lines.append(f"    nawf : {proj_data.nawf:>5}")
    lines.append(f"    nelec    : {proj_data.nelec:>12.6f}")
    lines.append(f"    efermi   : {proj_data.efermi_raw:>12.6f}")
    lines.append(f"    energy_units :  {proj_data.energy_units}   ")
    lines.append("")
    lines.append("  ATMPROJ conversion to be done using:")
    lines.append(
        f"    atmproj_nbnd : {proj_data.nbnds if not data.atomic_proj.atmproj_nbnd else data.atomic_proj.atmproj_nbnd:>5}"
    )
    lines.append(f"    atmproj_thr  : {data.atomic_proj.atmproj_thr:>12.6f}")
    lines.append(f"    atmproj_sh   : {data.atomic_proj.atmproj_sh:>12.6f}")
    lines.append(f"    atmproj_do_norm:  {data.atomic_proj.atmproj_do_norm}")
    if not data.atomic_proj.acbn0:
        lines.append("Using an orthogonal basis. acbn0=.false.")
    return lines
