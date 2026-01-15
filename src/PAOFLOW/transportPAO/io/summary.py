import numpy as np

from PAOFLOW.transportPAO.io.log_module import log_rank0
from PAOFLOW.transportPAO.utils.constants import rydcm1, amconv
from PAOFLOW.transportPAO.io.get_input_params import ConductorData


def print_summary(data: ConductorData) -> None:
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
    log_rank0(f"        Conductor datafile :     {data.file_names.datafile_C}")
    if data.calculation_type.strip().lower() == "conductor":
        log_rank0(f"           L-lead datafile :     {data.file_names.datafile_L}")
        log_rank0(f"           R-lead datafile :     {data.file_names.datafile_R}")
    if data.advanced.lhave_corr:
        log_rank0(f"            L-Sgm datafile :     {data.file_names.datafile_L_sgm}")
        log_rank0(f"            C-Sgm datafile :     {data.file_names.datafile_C_sgm}")
        log_rank0(f"            R-Sgm datafile :     {data.file_names.datafile_R_sgm}")
    log_rank0(f"         leads are identical :     {data.advanced.leads_are_identical}")
    log_rank0(f"           ovp orthogonaliz. :     {data.atomic_proj.do_orthoovp}")
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
