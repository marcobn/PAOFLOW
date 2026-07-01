"""Driver entry points for the electron-phonon workflow.

* **Generate** (``forces=None``): write one static-SCF QE input per reference-cell
  ``+/-`` displacement (unique prefix, PAO-sized ``nbnd``).
* **Analyse** (``forces='qe'``): rebuild the PAO Hamiltonian of every displaced
  supercell and central-difference to ``dV = dH/du`` (:mod:`dvscf_fd`).  The fold
  to the primitive cell and the ``g_mn^v(k, q)`` assembly / derived properties
  land in P2.
"""

from ..phonon.do_phonopy import init_phonopy
from .displacements import generate_eph_displacements
from .dvscf_fd import compute_dV
from .io import write_eph_displaced_supercells


def generate_eph_inputs(
    data_controller,
    supercell_matrix=None,
    displacement_distance=0.06,
    elphon_dir='elphon',
    pp_dir=None,
    prefix=None,
    kgrid=None,
    hubbard_card=None,
    configuration='standard',
    nbnd=None,
    is_plusminus='auto',
    displacement_mode='symmetry',
):
    """Build the reference-cell displacements and write their QE SCF inputs."""
    arry, attr = data_controller.data_dicts()
    if supercell_matrix is not None:
        attr['phonon_supercell_matrix'] = supercell_matrix
    attr['elphon_displacement_distance'] = displacement_distance

    init_phonopy(data_controller)
    cells, meta = generate_eph_displacements(
        arry['phonopy'],
        displacement_distance,
        is_plusminus=is_plusminus,
        displacement_mode=displacement_mode,
    )
    arry['elphon_displacements'] = meta

    return write_eph_displaced_supercells(
        data_controller,
        cells,
        meta,
        elphon_dir=elphon_dir,
        pp_dir=pp_dir,
        prefix=prefix,
        kgrid=kgrid,
        hubbard_card=hubbard_card,
        configuration=configuration,
        nbnd=nbnd,
        is_plusminus=is_plusminus,
    )


def run_eph(
    data_controller,
    elphon_dir='elphon',
    configuration=None,
    basispath=None,
    pthr=0.95,
    shift_type=1,
    **kwargs,
):
    """Analyse phase (P1): rebuild ``HRs`` per displaced cell -> ``dV``.

    The ``g_mn^v(k, q)`` assembly and the derived properties (Eliashberg
    ``alpha^2 F`` / ``lambda``, phonon-limited transport) land in P2.
    """
    return compute_dV(
        data_controller,
        elphon_dir=elphon_dir,
        configuration=configuration,
        basispath=basispath,
        pthr=pthr,
        shift_type=shift_type,
    )
