from mpi4py import MPI
import datetime

from PAOFLOW.transport import __version__
from PAOFLOW.transport.io.input_parameters import AtomicProjData, ConductorData

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def log_rank0(message: str):
    if rank == 0:
        print(message)


def log_parallelization_info(chunks: int, items: str):
    if rank == 0:
        print(
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
    if data.atomic_proj.acbn0:
        lines.append("Using an orthogonal basis. acbn0=.true.")
    return lines
