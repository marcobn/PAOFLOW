from __future__ import annotations

import os

from mpi4py import MPI

from PAOFLOW.transport.grid.kpoints import (
    KpointsData,
    compute_fourier_phase_table,
    initialize_kpoints,
    initialize_meshsize,
    initialize_r_vectors,
    kpoints_mask,
)
from PAOFLOW.transport.hamiltonian.hamiltonian import HamiltonianSystem
from PAOFLOW.transport.hamiltonian.hamiltonian_init import (
    check_leads_are_identical,
    initialize_hamiltonian_blocks,
)
from PAOFLOW.transport.io.get_input_params import (
    load_conductor_data_from_yaml,
    load_current_data_from_yaml,
)
from PAOFLOW.transport.io.input_parameters import ConductorData, RuntimeData
from PAOFLOW.transport.io.log_module import log_startup
from PAOFLOW.transport.io.summary import print_summary
from PAOFLOW.transport.parsers.atmproj_tools import parse_atomic_proj
from PAOFLOW.transport.parsers.parser_base import read_nr_from_ham
from PAOFLOW.transport.smearing.smearing_T import SmearingData
from PAOFLOW.transport.smearing.smearing_base import smearing_func
from PAOFLOW.transport.utils.memusage import MemoryTracker
from PAOFLOW.transport.workspace.workspace import Workspace


def prepare_conductor(yaml_file: str) -> ConductorData:
    """
    Load input parameters from a YAML file and prepare core conductor data.

    Parameters
    ----------
    yaml_file : str
        Path to the YAML input file.

    Returns
    -------
    data : ConductorData
        Parsed conductor data object with runtime values initialized.
    """
    data = load_conductor_data_from_yaml(yaml_file)

    prefix = os.path.basename(data.file_names.datafile_C)
    work_dir = data.file_names.work_dir
    nproc = MPI.COMM_WORLD.Get_size()

    log_startup("conductor.py")
    if data.carriers == "electrons":
        hk_data = parse_atomic_proj(data)
        nr_full = hk_data["nr"]
    elif data.carriers == "phonons":
        nr_full = read_nr_from_ham(data.file_names.datafile_C)

    nk_par, nr_par = initialize_meshsize(
        nr_full=nr_full, transport_direction=data.transport_direction
    )

    s_par = data.kpoint_grid.s[:2]
    nk_par3d = kpoints_mask(nk_par, 1, data.transport_direction)
    s_par3d = kpoints_mask(s_par, 0, data.transport_direction)
    nr_par3d = kpoints_mask(nr_par, 1, data.transport_direction)

    vkpt_par3D, wk_par = initialize_kpoints(
        nk_par,
        s_par=s_par,
        transport_direction=data.transport_direction,
        use_sym=data.symmetry.use_sym,
    )
    ivr_par3D, wr_par = initialize_r_vectors(nr_par, data.transport_direction)

    data.set_runtime_data(
        runtime=RuntimeData(
            nproc=nproc,
            prefix=prefix,
            work_dir=work_dir,
            nk_par=nk_par,
            s_par=s_par,
            nk_par3d=nk_par3d,
            s_par3d=s_par3d,
            nr_par3d=nr_par3d,
            vkpt_par3D=vkpt_par3D,
            wk_par=wk_par,
            ivr_par3D=ivr_par3D,
            wr_par=wr_par,
            nkpts_par=vkpt_par3D.shape[0],
            nrtot_par=ivr_par3D.shape[0],
        )
    )
    print_summary(data)

    return data


def prepare_smearing(
    data: ConductorData,
    memory_tracker: MemoryTracker,
) -> SmearingData:
    """
    Initialize smearing data and register it with the memory tracker.

    Parameters
    ----------
    memory_tracker : MemoryTracker
        Tracker to record memory usage.

    Returns
    -------
    smearing_data : SmearingData
        Initialized smearing data object.
    """
    smearing_data = SmearingData(smearing_func=smearing_func)
    smearing_data.initialize(
        smearing_type=data.energy.smearing_type,
        delta=data.energy.delta,
        delta_ratio=data.energy.delta_ratio,
        xmax=data.energy.xmax,
    )
    memory_tracker.register_section(
        "smearing", smearing_data.memory_usage, is_allocated=True
    )
    return smearing_data


def prepare_kpoints(data: ConductorData, memory_tracker: MemoryTracker) -> KpointsData:
    """
    Prepare k-point data structure and register with the memory tracker.

    Parameters
    ----------
    data : ConductorData
        Conductor input data object with runtime values set.
    memory_tracker : MemoryTracker
        Tracker to record memory usage.

    Returns
    -------
    kpoints_data : KpointsData
        Prepared k-points data object.
    """
    rt = data.get_runtime_data()

    kpoints_data = KpointsData()
    kpoints_data.vkpt_par3D = rt.vkpt_par3D
    kpoints_data.wk_par = rt.wk_par
    kpoints_data.ivr_par3D = rt.ivr_par3D
    kpoints_data.wr_par = rt.wr_par

    memory_tracker.register_section(
        "kpoints", kpoints_data.memory_usage, is_allocated=True
    )
    return kpoints_data


def prepare_hamiltonian_system(
    data: ConductorData, memory_tracker: MemoryTracker
) -> HamiltonianSystem:
    """
    Prepare Hamiltonian system data and register memory usage.

    Parameters
    ----------
    data : ConductorData
        Conductor input data object with dimensions.
    memory_tracker : MemoryTracker
        Tracker to record memory usage.

    Returns
    -------
    ham_sys : HamiltonianSystem
        Prepared Hamiltonian system object.
    """
    dimL, dimC, dimR = data.dimL, data.dimC, data.dimR
    nkpts_par = data.get_runtime_data().nkpts_par

    ham_sys = HamiltonianSystem(dimL, dimC, dimR, nkpts_par)
    memory_tracker.register_section(
        "hamiltonian data",
        lambda: ham_sys.memusage("ham"),
        is_allocated=ham_sys.allocated,
    )
    memory_tracker.register_section(
        "correlation data",
        lambda: ham_sys.memusage("corr"),
        is_allocated=ham_sys.allocated,
    )
    return ham_sys


def prepare_hamiltonian_blocks_and_leads(
    data: ConductorData, ham_sys: HamiltonianSystem
) -> None:
    """
    Initialize Hamiltonian blocks and check if leads are identical.

    Parameters
    ----------
    data : ConductorData
        Conductor input data object.
    ham_sys : HamiltonianSystem
        Prepared Hamiltonian system object.
    vkpt_par3D : ndarray
        K-point vectors.
    ivr_par3D : ndarray
        R-vectors (integer indices).
    wr_par : ndarray
        R-vector weights.

    Returns
    -------
    None
    """
    table_par = compute_fourier_phase_table(
        vkpts=data._runtime.vkpt_par3D, ivr_par=data._runtime.ivr_par3D
    )

    initialize_hamiltonian_blocks(
        output_dir=data.file_names.output_dir,
        ham_system=ham_sys,
        ivr_par3D=data._runtime.ivr_par3D.T,
        wr_par=data._runtime.wr_par,
        table_par=table_par,
        datafile_C=data.file_names.datafile_C,
        datafile_L=data.file_names.datafile_L,
        datafile_R=data.file_names.datafile_R,
        ispin=data.advanced.ispin,
        transport_direction=data.transport_direction,
        calculation_type=data.calculation_type,
        conductor_data=data,
    )

    data.advanced.leads_are_identical = check_leads_are_identical(
        ham_system=ham_sys,
        datafile_L=data.file_names.datafile_L,
        datafile_R=data.file_names.datafile_R,
        datafile_L_sgm=data.file_names.datafile_L_sgm,
        datafile_R_sgm=data.file_names.datafile_R_sgm,
    )


def prepare_workspace(data: ConductorData, memory_tracker: MemoryTracker) -> Workspace:
    """
    Allocate workspace arrays and register memory usage.

    Parameters
    ----------
    data : ConductorData
        Conductor input data object with dimensions.
    memory_tracker : MemoryTracker
        Tracker to record memory usage.

    Returns
    -------
    workspace : Workspace
        Allocated workspace object.
    """
    dimL, dimC, dimR = data.dimL, data.dimC, data.dimR
    nkpts_par = data.get_runtime_data().nkpts_par
    nrtot_par = data.get_runtime_data().nrtot_par

    workspace = Workspace()
    workspace.allocate(
        dimL=dimL,
        dimC=dimC,
        dimR=dimR,
        dimx_lead=max(dimL, dimR),
        nkpts_par=nkpts_par,
        nrtot_par=nrtot_par,
        write_lead_sgm=data.symmetry.write_lead_sgm,
        write_gf=data.symmetry.write_gf,
    )
    memory_tracker.register_section(
        "workspace", workspace.memusage, is_allocated=workspace.allocated
    )
    return workspace


def prepare_current(yaml_file: str) -> dict | None:
    """
    Load current calculation input parameters from YAML.

    Parameters
    ----------
    `yaml_file` : str
        Path to the YAML input file (e.g. current.yaml).

    Returns
    -------
    `data` : dict or None
        Parsed input parameters, or None if loading fails.
    """
    return load_current_data_from_yaml(yaml_file)
