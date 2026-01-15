import logging
from pathlib import Path

import numpy as np
from mpi4py import MPI

from PAOFLOW.DataController import DataController
from PAOFLOW.transport.io.input_parameters import AtomicProjData, ConductorData
from PAOFLOW.transport.utils.constants import amconv, rydcm1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

_logger = None
_logger_path = None


def initialize_logger(
    data_controller: DataController, log_file_name: str = "transport.log"
):
    global _logger, _logger_path

    if rank != 0 or _logger is not None:
        return

    _, attr = data_controller.data_dicts()
    output_dir = attr["outputdir"]
    if output_dir is None:
        raise RuntimeError(
            "Logger initialization failed: 'outputdir' not set in data_controller."
        )

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
    log_rank0(
        f"Parallelization information: Each rank processes approximately {chunks} {items}."
    )


def log_section_start(name: str):
    log_rank0(f"Started {name}")


def log_section_end(name: str):
    log_rank0(f"Finished {name}")


def log_summary(data: ConductorData) -> None:
    """
    Print the summary of transport parameters.

    Parameters
    ----------
    data : ConductorData
        The input data model with validated parameters.
    """

    def format_float(val):
        return f"{val:10.5f}"

    log_rank0("")
    log_rank0(
        "  ======================================================================"
    )
    log_rank0(
        "  =  INPUT Summary                                                     ="
    )
    log_rank0(
        "  ======================================================================"
    )
    log_rank0("")
    log_rank0("  <INPUT>")
    log_rank0(f"          Calculation Type :     {data.calculation_type}")
    log_rank0(f"                    prefix :     {data.file_names.prefix}")
    log_rank0(f"                   postfix :     {data.file_names.postfix}")
    log_rank0(f"                  work_dir :     {data.file_names.work_dir}")
    log_rank0(f"               L-lead dim. :{data.dimL:>10}")
    log_rank0(f"            conductor dim. :{data.dimC:>10}")
    log_rank0(f"               R-lead dim. :{data.dimR:>10}")
    log_rank0(f"       Conductance Formula :     {data.conduct_formula}")
    log_rank0(f"                  Carriers :     {data.carriers}")
    log_rank0(f"       Transport Direction :{data.transport_direction:>10}")
    log_rank0(f"          Have Correlation :     {data.advanced.lhave_corr}")
    log_rank0(f"              Write k-data :     {data.symmetry.write_kdata}")
    log_rank0(f"            Write sgm lead :     {data.symmetry.write_lead_sgm}")
    log_rank0(f"                Write gf C :     {data.symmetry.write_gf}")
    log_rank0(f"           Max iter number :{data.iteration.niterx:>10}")
    log_rank0(f"                    nprint :{data.iteration.nprint:>10}")
    log_rank0("")
    if data.advanced.lhave_corr:
        log_rank0(f"            L-Sgm datafile :     {data.file_names.datafile_L_sgm}")
        log_rank0(f"            C-Sgm datafile :     {data.file_names.datafile_C_sgm}")
        log_rank0(f"            R-Sgm datafile :     {data.file_names.datafile_R_sgm}")
    log_rank0(f"         leads are identical :     {data.advanced.leads_are_identical}")
    log_rank0(f"           ovp orthogonaliz. :     {data.atomic_proj.acbn0}")
    log_rank0("  </INPUT>")
    log_rank0("")

    # ENERGY GRID
    log_rank0("  <ENERGY_GRID>")
    log_rank0(f"                 Dimension :{data.energy.ne:>10}")
    log_rank0(f"                 Buffering :{data.energy.ne_buffer:>10}")

    if data.carriers.strip().lower() == "phonons":
        scale = (rydcm1 / np.sqrt(amconv)) ** 2
        log_rank0(
            f"            Min Frequency :{format_float(data.energy.emin * scale)}"
        )
        log_rank0(
            f"            Max Frequency :{format_float(data.energy.emax * scale)}"
        )
        log_rank0(
            f"              Energy Step :{format_float(data.energy.energy_step * scale)}"
        )
    else:
        log_rank0(f"               Min Energy :{format_float(data.energy.emin)}")
        log_rank0(f"               Max Energy :{format_float(data.energy.emax)}")
        log_rank0(f"              Energy Step :{format_float(data.energy.energy_step)}")

    log_rank0(f"                     Delta :{format_float(data.energy.delta)}")
    log_rank0(f"             Smearing Type :     {data.energy.smearing_type}")
    log_rank0(f"             Smearing grid :{data.energy.nx_smear:>10}")
    log_rank0(f"             Smearing gmax :{format_float(data.energy.xmax)}")
    log_rank0(f"                   Shift_L :{format_float(data.shift_L)}")
    log_rank0(f"                   Shift_C :{format_float(data.shift_C)}")
    log_rank0(f"                   Shift_R :{format_float(data.shift_R)}")
    log_rank0(f"                Shift_corr :{format_float(data.shift_corr)}")
    log_rank0("  </ENERGY_GRID>")
    log_rank0("")

    # K-POINTS
    runtime = data.get_runtime_data()
    if runtime is not None:
        log_rank0("  <K-POINTS>")
        log_rank0(f"       nkpts_par = {runtime.nkpts_par}")
        log_rank0(f"       nrtot_par = {runtime.nrtot_par}")
        log_rank0(f"        use_sym = {data.symmetry.use_sym}")

        nk_par3d = runtime.nk_par3d
        s_par3d = runtime.s_par3d
        log_rank0(
            f"\n       Parallel kpoints grid:        nk = ( {nk_par3d[0]:3} {nk_par3d[1]:3}  {nk_par3d[2]:3} )   s = ( {s_par3d[0]:3} {s_par3d[1]:3}  {s_par3d[2]:3} )"
        )

        for i, (vkpt, weight) in enumerate(zip(runtime.vkpt_par3D, runtime.wk_par), 1):
            log_rank0(
                f"       k ({i:3}) =    ( {vkpt[0]:9.5f} {vkpt[1]:9.5f} {vkpt[2]:9.5f} ),   weight = {weight:8.4f}"
            )

        nr_par3d = runtime.nr_par3d
        log_rank0(
            f"\n       Parallel R vector grid:       nr = ( {nr_par3d[0]:3} {nr_par3d[1]:3}  {nr_par3d[2]:3} )"
        )

        for i, (ivr, weight) in enumerate(zip(runtime.ivr_par3D, runtime.wr_par), 1):
            log_rank0(
                f"       R ({i:3}) =    ( {ivr[0]:9.5f} {ivr[1]:9.5f} {ivr[2]:9.5f} ),   weight = {weight:8.4f}"
            )

        log_rank0("  </K-POINTS>")
        log_rank0("")

    # PARALLELISM
    log_rank0("  <PARALLELISM>")
    log_rank0("       Paralellization over frequencies")
    log_rank0(f"       # of processes: {runtime.nproc if runtime else 'N/A':>5}")
    log_rank0("  </PARALLELISM>")
    log_rank0("")


def log_proj_data(
    proj_data: AtomicProjData,
    data: ConductorData,
) -> list[str]:
    lines = []
    lines.append("  Dimensions found in atomic_proj.{dat,xml}:")
    lines.append(f"    nbnds        : {proj_data.nbnds:>5}")
    lines.append(f"    nkpnts       : {proj_data.nkpnts:>5}")
    lines.append(f"    nspin        : {proj_data.nspin:>5}")
    lines.append(f"    nawf         : {proj_data.nawf:>5}")
    lines.append(f"    nelec        : {proj_data.nelec:>12.6f}")
    lines.append(f"    efermi       : {proj_data.efermi:>12.6f}")
    lines.append(f"    energy_units :  {proj_data.energy_units}   ")
    lines.append("")
    if not data.atomic_proj.acbn0:
        lines.append("Using an orthogonal basis. acbn0=.false.")
    return lines


def log_proj_summary(proj_data: AtomicProjData, data: ConductorData) -> None:
    for line in log_proj_data(proj_data, data):
        log_rank0(line)
