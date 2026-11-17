"""``main.landauer.py``: Landauer transport of a crystal or a supercell junction from a QE save.

The script builds the PAO Hamiltonian (internal projections on the pseudo-atomic basis of the run's
pseudopotentials, or the ``projwfc.x`` projections), partitions it along the transport direction and calls
:class:`PAOFLOW.Transport.Transport`: transmission and conductor DOS, the current from the zero-bias transmission
and, for a conductor, the self-consistent finite-bias scan. Every option is an editable constant at the top. The
questions are :mod:`PAOFLOW.gen.questions.landauer`.
"""

from __future__ import annotations

KB_EV = 8.617333262e-5


def build_landauer_script(cfg: dict) -> str:
    """Text of ``main.landauer.py``.

    Parameters
    ----------
    cfg : dict
        Configuration of :func:`PAOFLOW.gen.questions.landauer.assemble_landauer`.

    Returns
    -------
    str
    """
    from .paoflow_driver import (
        BASIS_GEN_BLOCK,
        HEADER,
        _format_upfs_line,
        _projection_lines,
        _uses_projwfc,
    )

    conductor = cfg['calculation'] == 'conductor'
    finite_bias = bool(cfg.get('finite_bias')) and conductor
    three_d = finite_bias and cfg.get('electrostatics') == '3d'
    header = HEADER.replace('python main.py', 'python main.landauer.py').replace(
        'mpirun -np <N> python main.py', 'mpirun -np <N> python main.landauer.py'
    )
    header = header.replace(
        'from PAOFLOW import PAOFLOW\n',
        'from PAOFLOW import PAOFLOW\nfrom PAOFLOW.Transport import Transport\n',
    )
    lines = [header]
    lines += [
        '',
        '# ----------------------------------------------------------------------- #',
        '# Configuration  (edit freely)                                            #',
        '# ----------------------------------------------------------------------- #',
        'HERE = os.path.dirname(os.path.abspath(__file__))',
        'SAVEDIR = os.path.join(HERE, {!r})'.format(cfg['savedir']),
        _format_upfs_line(cfg['upfs']),
        'BASISPATH = os.path.join(HERE, {!r}) + os.sep'.format(cfg['basisdir']),
        'OUTPUTDIR = {!r}'.format(cfg['outputdir']),
        'BASIS = {!r}   # configuration of the pseudo-atomic basis'.format(
            cfg.get('basis', 'standard')
        ),
        'PTHR = 0.95   # projectability threshold',
        '',
        '# Partition along the transport direction. bulk: the cell is the principal layer of the crystal;',
        '# conductor: the supercell is the conductor and its first and last LEAD_LAYERS atomic layers are',
        '# the principal layers of the leads (bulk-like: both leads take the block of the first ones).',
        'CALCULATION = {!r}'.format(cfg['calculation']),
        'TRANSPORT_DIRECTION = {!r}'.format(cfg['transport_direction']),
        'LEAD_LAYERS = {}'.format(cfg['lead_layers']),
        'DOUBLINGS = {}   # doublings of the cell along the transport direction'.format(
            cfg['doublings']
        ),
        'TRANSVERSE_NK = ({}, {})   # (0, 0): the transverse DFT mesh'.format(
            *cfg['transverse_nk']
        ),
        *(
            [
                'PRINCIPAL_LAYERS = {}   # bulk: atomic layers of the repeated unit, from the start'.format(
                    cfg['principal_layers']
                )
            ]
            if cfg.get('principal_layers')
            else []
        ),
        '',
        '# Energy grid of T(E) and of the conductor DOS (eV, relative to E_F) and the broadening.',
        'EMIN, EMAX, NE, DELTA = {}, {}, {}, {}'.format(
            cfg['emin'], cfg['emax'], cfg['ne'], cfg['delta']
        ),
        '',
        '# Current from the zero-bias transmission: mu_L, mu_R = +V/2, -V/2 (I > 0 for V > 0, in 2e/h eV).',
        'CURRENT = {}'.format(cfg['current']),
        'BIAS_MAX, NBIAS = {}, {}   # volt'.format(cfg['bias_max'], cfg['nbias']),
        'TEMPERATURE = {}   # K'.format(cfg['temperature']),
    ]
    if cfg.get('eigenchannels'):
        lines += [
            '',
            '# Transmission eigenchannels kept, and the energy (eV) of the eigenchannel wavefunctions',
            '# (None: no wavefunctions; written under OUTPUTDIR/eigenchannels).',
            'EIGENCHANNELS = {}'.format(cfg['eigenchannels']),
            'EIGENCHANNEL_ENERGY = {!r}'.format(cfg.get('eigenchannel_energy')),
        ]
    if conductor:
        lines += [
            '',
            '# Self-consistent finite bias (conductor only): NEGF density and the Hartree potential of the',
            "# bias-induced charge; ELECTROSTATICS 'planar' (layers) or '3d' (a Gaussian charge per atom).",
            'FINITE_BIAS = {}'.format(cfg['bias_values'] if finite_bias else []),
            'ELECTROSTATICS = {!r}'.format(cfg.get('electrostatics', 'planar')),
            'CHARGE_WIDTHS = {!r}   # bohr per species; None: from the pseudopotentials'.format(
                cfg.get('charge_widths')
            ),
            "PROFILES = {!r}   # 'linear': the ramp, no self-consistency".format(
                ('linear', 'self_consistent') if cfg.get('linear_ramp') else ('self_consistent',)
            ),
            'POTENTIAL_GRID_BIASES = {!r}   # U(r) as XSF (3d)'.format(
                [max(cfg['bias_values'])] if three_d and cfg.get('potential_grid') else None
            ),
            'WINDOW_SPACING, MIXING, TOLERANCE, MAX_ITERATIONS = {}, {}, {}, {}'.format(
                cfg['window_spacing'], cfg['mixing'], cfg['tolerance'], cfg['max_iterations']
            ),
        ]
    lines.append('')
    if not _uses_projwfc(cfg):
        lines.append(BASIS_GEN_BLOCK)
    lines += [
        '',
        'def build_hamiltonian():',
        '    """PAO Hamiltonian of the save (orthonormal: the Landauer blocks need no overlaps)."""',
        '    p = PAOFLOW.PAOFLOW(',
        '        workpath=HERE,',
        '        outputdir=OUTPUTDIR,',
        '        savedir=SAVEDIR,',
        "        smearing='gauss',",
        '        npool=1,',
        '        verbose=False,',
        '    )',
        *_projection_lines(cfg, configuration='BASIS'),
        '    p.projectability(pthr=PTHR)',
        '    p.pao_hamiltonian()',
        '    if DOUBLINGS:',
        "        axis = 'xyz'.index(TRANSPORT_DIRECTION)",
        '        p.doubling_Hamiltonian(*[DOUBLINGS if i == axis else 0 for i in range(3)])',
        '    return p',
        '',
        '',
        'def landauer(p):',
        '    """Transmission, conductor DOS and current."""',
        '    t = Transport(p.data_controller)',
        "    if CALCULATION == 'conductor':",
        '        t.define_partition(central_atoms="ALL", left_lead_layers=LEAD_LAYERS,',
        '                           right_lead_layers=LEAD_LAYERS, transport_direction=TRANSPORT_DIRECTION)',
        '    else:',
        (
            '        t.define_partition(central_atoms="ALL", central_layers=PRINCIPAL_LAYERS,\n'
            '                           transport_direction=TRANSPORT_DIRECTION)'
            if cfg.get('principal_layers')
            else '        t.define_partition(central_atoms="ALL", transport_direction=TRANSPORT_DIRECTION)'
        ),
        "    t.configure_outputs(output_dir=os.path.join(HERE, OUTPUTDIR), postfix='_' + CALCULATION)",
        '    t.configure_energy_grid(emin=EMIN, emax=EMAX, ne=NE, delta=DELTA, nk=TRANSVERSE_NK)',
        '    t.build_hamiltonian_blocks(calculation_type=CALCULATION, use_sym=False,',
        '                               do_overlap_transformation=False)',
        *(
            [
                '    plot_energy = EIGENCHANNEL_ENERGY is not None',
                '    index = round((EIGENCHANNEL_ENERGY - EMIN) / (EMAX - EMIN) * (NE - 1)) if plot_energy else 0',
                '    t.configure_eigenchannels(do_eigenchannels=True, neigchnx=EIGENCHANNELS,',
                '                              do_eigplot=plot_energy, ie_eigplot=index, ik_eigplot=0)',
            ]
            if cfg.get('eigenchannels')
            else []
        ),
        '    t.compute_transmission()   # conductance_<calculation>.dat: E, T(E)',
        '    t.compute_dos()            # doscond_<calculation>.dat',
        '    if CURRENT:',
        '        t.compute_current(bias_min=-BIAS_MAX, bias_max=BIAS_MAX, nbias=NBIAS, mu_L=0.5, mu_R=-0.5,',
        '                          sigma={!r} * TEMPERATURE)   # current.dat'.format(KB_EV),
    ]
    if conductor:
        lines += [
            '    if FINITE_BIAS:',
            '        t.configure_bias_self_consistency(',
            '            electrostatics=ELECTROSTATICS,',
            '            charge_widths=CHARGE_WIDTHS,',
            '            temperature=TEMPERATURE,',
            '            window_spacing=WINDOW_SPACING,',
            '            mixing=MIXING,',
            '            tolerance=TOLERANCE,',
            '            max_iterations=MAX_ITERATIONS,',
            '            verbose=True,',
            '        )',
            '        for profile in PROFILES:',
            '            # current_, potential_, charge_, transmission_<profile>_conductor.dat',
            '            t.compute_current_self_consistent(bias=FINITE_BIAS, mu_L=0.5, mu_R=-0.5, profile=profile,',
            '                                              potential_grid_biases=POTENTIAL_GRID_BIASES)',
        ]
    lines += [
        '    return t',
        '',
        '',
        'def main():',
        '    if not os.path.isdir(SAVEDIR):',
        '        print("{} not found. Run pw.x (scf then nscf) first.".format(SAVEDIR))',
        '        sys.exit(1)',
    ]
    if not _uses_projwfc(cfg):
        lines.append("    ensure_basis(preset='extended')")
    lines += [
        '    if "MPI" in globals():',
        '        MPI.COMM_WORLD.Barrier()',
        '    p = build_hamiltonian()',
        '    landauer(p)',
        '    p.finish_execution()',
        '',
        '',
        'if __name__ == "__main__":',
        '    # An error on one MPI rank stops every rank instead of hanging the job.',
        '    from PAOFLOW.utils.mpi_guard import run_main',
        '',
        '    run_main(main)',
        '',
    ]
    return '\n'.join(lines)
