"""``main.py`` and ``plot.py`` lines of the extra run properties (:mod:`PAOFLOW.gen.questions.run_extras`).

:func:`PAOFLOW.gen.paoflow_driver.build_run_script` and
:func:`~PAOFLOW.gen.paoflow_driver.build_plot_script` call these builders only when the matching
menu entry is chosen, so a run without them is written exactly as before.
"""

from __future__ import annotations

#: Menu entries that change how the run is made rather than add a property.
RUN_MODIFIERS = ('hamiltonian_changes', 'restart')

#: Topology searches on the real-space Hamiltonian (before the band path and the interpolation).
TOPOLOGY_SEARCHES = ('weyl_points', 'z2_invariant', 'mirror_chern')


def hubbard_vector(rows: list[tuple[int, int, float]]) -> list[float] | None:
    """Per-orbital U values up to the last shifted orbital (``add_external_fields(HubbardU=...)``).

    Parameters
    ----------
    rows : list of (int, int, float)
        First and last orbital (1-based, inclusive) and U (eV).

    Returns
    -------
    list of float or None
        U of orbitals 1, 2, ... up to the last one in ``rows`` (later rows win); None without
        rows.
    """
    if not rows:
        return None
    values = [0.0] * max(last for _first, last, _value in rows)
    for first, last, value in rows:
        for orbital in range(first - 1, last):
            values[orbital] = float(value)
    return values


def extra_constant_lines(cfg: dict) -> list[str]:
    """Configuration constants of the chosen extra properties (empty without them).

    Parameters
    ----------
    cfg : dict
        Run configuration.

    Returns
    -------
    list of str
    """
    props = cfg['properties']
    lines: list[str] = []
    if 'site_projected_bands' in props:
        lines += [
            '# Site-projected bands: atoms (0-based, in the order of the save).',
            'SITE_ATOMS = {!r}'.format(list(cfg['site_atoms'])),
        ]
    if cfg.get('shc_atoms'):
        lines += [
            '# Spin Hall conductivity of these atoms only (0-based); None: the whole cell.',
            'SHC_ATOMS = {!r}'.format(list(cfg['shc_atoms'])),
        ]
    if 'jdos' in props:
        lines += [
            '# Joint density of states: photon energies 0..JDOS_EMAX (eV), broadening (eV).',
            'JDOS_EMAX, JDOS_NE, JDOS_DELTA = {!r}, {!r}, {!r}'.format(
                float(cfg['jdos_emax']), int(cfg['jdos_ne']), float(cfg['jdos_delta'])
            ),
        ]
    if 'weyl_points' in props:
        lines += [
            '# Weyl point search: boxes along each reciprocal vector; symmetry images.',
            'WEYL_GRID = {!r}'.format(list(cfg['weyl_grid'])),
            'WEYL_SYMMETRIZE = {!r}'.format(bool(cfg['weyl_symmetrize'])),
        ]
    if 'z2_invariant' in props:
        lines += [
            '# Z2 invariant: time-reversal-invariant planes k3 = const (crystal units).',
            'Z2_PLANES = {!r}'.format(tuple(float(k3) for k3 in cfg['z2_planes'])),
        ]
    if 'doped_transport' in props:
        lines += [
            '# Doped semiconductor: carriers per cm^3 (negative: electrons, n type).',
            'DOPING = {!r}'.format(float(cfg['doping_conc'])),
            'DOPING_TMIN, DOPING_TMAX, DOPING_NT = {!r}, {!r}, {!r}'.format(
                float(cfg['doping_tmin']), float(cfg['doping_tmax']), int(cfg['doping_nt'])
            ),
            '# Window of the carrier count: every valence band, energies 3 meV apart (eV).',
            'DOPING_EMIN, DOPING_EMAX, DOPING_NE = {!r}, {!r}, {!r}'.format(
                float(cfg['doping_emin']), float(cfg['doping_emax']), int(cfg['doping_ne'])
            ),
            '# Built-in relaxation-time models ([]: the constant-relaxation-time tensors, per',
            '# second of relaxation time) and their material parameters.',
            'SCATTERING = {!r}'.format(list(cfg['scattering'])),
            'TAU_PARAMETERS = {!r}'.format(dict(cfg['tau_parameters'])),
        ]
    if 'hamiltonian_changes' in props:
        soc = {species: [lp, ld] for species, lp, ld in cfg['adhoc_soc']}
        lines += [
            '# Hamiltonian changes (modify_hamiltonian, after every pao_hamiltonian):',
            '# -U/2 (eV) on each orbital, in the order of the basis; None: no shift.',
            'HUBBARD_U = {!r}'.format(hubbard_vector(cfg['hubbard_shifts'])),
            'EFIELD = {!r}   # uniform electric field (V/A); None: none'.format(
                list(cfg['efield']) if cfg['efield'] else None
            ),
            'ADHOC_SOC = {!r}   # species: [lambda_p, lambda_d] (eV); {{}}: none'.format(soc),
            'SOC_THETA, SOC_PHI = {!r}, {!r}   # spin quantization axis (radians)'.format(
                *(float(angle) for angle in cfg['adhoc_soc_angles'])
            ),
            'ZEEMAN = {!r}   # exchange field B (eV), -B.S on site; None: none'.format(
                list(cfg['zeeman']) if cfg['zeeman'] else None
            ),
            'DOUBLING = {!r}   # the cell doubled nx, ny, nz times (2**n copies)'.format(
                tuple(cfg['doubling'])
            ),
            '# With HUBBARD_U, pao_eigh puts the top of this band (0-based) at zero.',
            'TOP_VALENCE_BAND = {!r}'.format(int(cfg['top_valence_band'])),
        ]
    if 'restart' in props:
        lines += [
            '# Restart files (one per MPI rank) written after the gradients; the next run',
            '# with the same number of ranks starts from them. Delete them to rebuild.',
            "RESTART = os.path.join(HERE, 'paoflow_dump')",
        ]
    return lines + [''] if lines else []


def modify_hamiltonian_lines(cfg: dict) -> list[str]:
    """``def modify_hamiltonian(p)`` of ``main.py`` (empty unless the Hamiltonian is changed).

    Parameters
    ----------
    cfg : dict
        Run configuration.

    Returns
    -------
    list of str
    """
    if 'hamiltonian_changes' not in cfg['properties']:
        return []
    return [
        '',
        'def modify_hamiltonian(p):',
        '    """The changes of the PAO Hamiltonian, applied after every pao_hamiltonian."""',
        '    if HUBBARD_U is not None or EFIELD is not None:',
        '        p.add_external_fields(Efield=EFIELD or [0.0], HubbardU=HUBBARD_U or [0.0])',
        '    if ADHOC_SOC:',
        '        p.adhoc_spin_orbit(theta=SOC_THETA, phi=SOC_PHI, soc_strengh=ADHOC_SOC)',
        '    if ZEEMAN is not None:',
        '        p.add_external_fields(Bfield=ZEEMAN)  # needs spinors: after the spin-orbit',
        '    if any(DOUBLING):',
        '        p.doubling_Hamiltonian(*DOUBLING)',
        '',
    ]


def apply_modifications(lines: list[str], cfg: dict) -> list[str]:
    """Insert ``modify_hamiltonian(p)`` after every ``p.pao_hamiltonian(...)`` line and pass the
    top valence band to ``pao_eigh`` (its Hubbard-shift reference).

    Parameters
    ----------
    lines : list of str
        Lines of ``main.py``.
    cfg : dict
        Run configuration.

    Returns
    -------
    list of str
        ``lines`` unchanged unless the Hamiltonian is changed.
    """
    if 'hamiltonian_changes' not in cfg['properties']:
        return lines
    out = []
    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if stripped == 'p.pao_eigh()':
            # the default bval (0) would put the top of the lowest band at zero
            line = indent + 'p.pao_eigh(bval=TOP_VALENCE_BAND)'
        out.append(line)
        if stripped.startswith('p.pao_hamiltonian(') and stripped.endswith(')'):
            out.append(indent + 'modify_hamiltonian(p)')
    return out


def topology_search_lines(cfg: dict) -> list[str]:
    """Weyl search, Z2 index and mirror Chern number on the real-space Hamiltonian.

    Parameters
    ----------
    cfg : dict
        Run configuration.

    Returns
    -------
    list of str
        Lines of ``run_properties`` (empty without these properties).
    """
    props = cfg['properties']
    lines: list[str] = []
    if 'weyl_points' in props:
        lines += [
            '    # Weyl points: hamiltonian.dat (read by Z2Pack for the chirality), then the search.',
            '    p.write_Hamiltonian()',
            '    p.find_weyl_points(search_grid=WEYL_GRID, symmetrize=WEYL_SYMMETRIZE)',
        ]
    if 'z2_invariant' in props:
        lines += [
            '    # Z2 index from the Wannier charge centres (Z2Pack; needs spin-orbit coupling).',
            '    p.z2_invariant(k3_planes=Z2_PLANES)',
        ]
    if 'mirror_chern' in props:
        lines += [
            '    # Mirror Chern number of the layer (Z2Pack; fully relativistic save).',
            '    mirror = p.mirror_chern_number()',
            '    if p.rank == 0 and mirror is not None:',
            '        import json',
            '',
            "        with open(os.path.join(HERE, OUTPUTDIR, 'mirror_chern.json'), 'w') as f:",
            '            json.dump(mirror, f, indent=1, default=str)',
        ]
    return lines + [''] if lines else []


def doped_transport_run_lines(
    cfg: dict, projection_lines: list[str], interpolation: str
) -> list[str]:
    """``def run_doped_transport()``: a run with ``smearing=None`` for the carrier count.

    Parameters
    ----------
    cfg : dict
        Run configuration.
    projection_lines : list of str
        The projection step of the standard basis (:func:`PAOFLOW.gen.paoflow_driver._projection_lines`).
    interpolation : str
        The ``interpolation=...`` argument (or ``''``).

    Returns
    -------
    list of str
    """
    if 'doped_transport' not in cfg['properties']:
        return []
    return [
        '',
        'def run_doped_transport():',
        '    """Doped semiconductor: chemical potential, conductivity and Seebeck coefficient.',
        '',
        '    smearing=None: the carrier count uses 0.01 eV Gaussians and the transport the',
        '    Fermi-Dirac derivative. Adaptive widths of the dispersive bands reach tenths of an eV,',
        '    and their tails in the gap swamp the electrons of a low doping.',
        '    """',
        '    p = PAOFLOW.PAOFLOW(',
        '        workpath=HERE,',
        '        outputdir=OUTPUTDIR,',
        '        savedir=SAVEDIR,',
        '        smearing=None,',
        '        npool=NPOOL,',
        '        save_overlaps=SAVE_OVERLAPS,',
        '        verbose=False,',
        '    )',
        '',
        *projection_lines,
        '    p.projectability(pthr=PTHR)',
        '    p.pao_hamiltonian()',
        '    if NFFT is not None:',
        '        p.interpolated_hamiltonian(nfft1=NFFT[0], nfft2=NFFT[1], nfft3=NFFT[2]%s)'
        % interpolation,
        '    p.pao_eigh()',
        '    p.gradient_and_momenta()',
        '    p.doped_transport(',
        '        DOPING, tmin=DOPING_TMIN, tmax=DOPING_TMAX, nt=DOPING_NT, emin=DOPING_EMIN,',
        '        emax=DOPING_EMAX, ne=DOPING_NE, scattering_channels=SCATTERING,',
        '        tau_dict=TAU_PARAMETERS,',
        '    )',
        '',
        '    p.finish_execution()',
        '',
    ]


def with_restart(body: list[str], cfg: dict) -> list[str]:
    """Wrap the build of ``run_properties`` (up to the gradients) in a restart branch.

    Parameters
    ----------
    body : list of str
        Lines of ``run_properties`` from the ``PAOFLOW.PAOFLOW(`` line on.
    cfg : dict
        Run configuration.

    Returns
    -------
    list of str
        ``body`` unchanged without the ``restart`` entry. With it: load the dump when it exists,
        otherwise build everything up to ``gradient_and_momenta`` and dump it.
    """
    if 'restart' not in cfg['properties']:
        return body
    split = next(
        i for i, line in enumerate(body) if line.strip().startswith('p.adaptive_smearing(')
    )
    build, rest = body[:split], body[split:]
    return (
        [
            "    if os.path.exists(RESTART + '_0.json'):",
            '        # the dump of an earlier run: Hamiltonian, eigenvalues and gradients',
            '        p = PAOFLOW.PAOFLOW(restart=True)',
            '        p.restart_load(fname_prefix=RESTART)',
            '    else:',
        ]
        + [('    ' + line) if line.strip() else line for line in build]
        + ['        p.restart_dump(fname_prefix=RESTART)', '']
        + rest
    )


def _plot_func(name: str, body: list[str]) -> list[str]:
    from .paoflow_driver import _plot_func as plot_func

    return plot_func(name, body)


def extra_plot_entries(cfg: dict, funcs: list[str], menu: list) -> tuple[list[str], list]:
    """``plot.py`` functions and menu entries of the extra properties.

    Parameters
    ----------
    cfg : dict
        Run configuration.
    funcs : list of str
        Source of the plot functions so far.
    menu : list of (str, str)
        Menu entries so far.

    Returns
    -------
    tuple
        ``(funcs, menu)`` with the entries of the chosen extra properties appended.
    """
    props = cfg['properties']
    if 'site_projected_bands' in props:
        atoms = ', '.join(str(atom + 1) for atom in cfg['site_atoms'])
        funcs += _plot_func(
            'plot_site_projected_bands',
            [
                "fb = _one('site-projected-bands_0.dat')",
                'if _missing(fb):',
                '    return',
                "sp = _one('*.kpath_points.txt')",
                'pplt.plot_weighted_bands(OUTPUTDIR, fb, sym_points=sp,',
                '                         title=%r,' % ('Bands projected on atoms ' + atoms),
                "                         cbar_label='weight', y_lim=_ewin())",
            ],
        )
        menu.append(('Site-projected bands', 'plot_site_projected_bands'))
    if 'jdos' in props:
        funcs += _plot_func(
            'plot_jdos',
            [
                'import numpy as np',
                'import matplotlib.pyplot as plt',
                '',
                "files = _many('jdos*.dat')",
                'if not files:',
                '    _missing(None)',
                '    return',
                'fig, ax = plt.subplots()',
                'for fn in files:',
                '    data = np.loadtxt(fn)',
                '    ax.plot(data[:, 0], data[:, 1], label=os.path.basename(fn))',
                "ax.set_xlabel('Photon energy (eV)')",
                "ax.set_ylabel('JDOS (1/eV)')",
                "ax.set_title('Joint density of states')",
                'ax.legend()',
                'plt.show()',
            ],
        )
        menu.append(('Joint density of states', 'plot_jdos'))
    if 'rashba_edelstein' in props:
        funcs += _plot_func(
            'plot_rashba_edelstein',
            [
                'import numpy as np',
                'import matplotlib.pyplot as plt',
                '',
                "files = _many('Ekai_*.dat')",
                'if not files:',
                '    _missing(None)',
                '    return',
                'fig, ax = plt.subplots()',
                'for fn in files:',
                '    data = np.atleast_2d(np.loadtxt(fn))',
                '    ax.plot(data[:, 0], data[:, 1], label=os.path.basename(fn)[5:-4])',
                'ax.set_xlim(*_ewin())',
                "ax.set_xlabel('Energy (eV)')",
                "ax.set_ylabel('Rashba-Edelstein response')",
                "ax.set_title('Current-induced spin density')",
                'ax.legend()',
                'plt.show()',
            ],
        )
        menu.append(('Rashba-Edelstein response', 'plot_rashba_edelstein'))
    if 'weyl_points' in props:
        funcs += _plot_func(
            'plot_weyl_points',
            [
                "f = _one('weyl_points.dat')",
                'if _missing(f):',
                '    return',
                'with open(f) as table:',
                '    print(table.read())',
            ],
        )
        menu.append(('Weyl points (table)', 'plot_weyl_points'))
    if 'z2_invariant' in props:
        funcs += _plot_func(
            'plot_z2_wannier_centres',
            [
                'import numpy as np',
                'import matplotlib.pyplot as plt',
                '',
                "files = _many('z2_wcc_k3_*.dat')",
                'if not files:',
                '    _missing(None)',
                '    return',
                'fig, axes = plt.subplots(1, len(files), sharey=True, squeeze=False)',
                'for ax, fn in zip(axes[0], files):',
                '    with open(fn) as handle:',
                "        title = handle.readline().split(';')[-1].strip()",
                '    data = np.atleast_2d(np.loadtxt(fn))',
                '    for column in range(1, data.shape[1]):',
                "        ax.plot(data[:, 0], data[:, column], 'k.', markersize=3)",
                "    ax.set_xlabel('$k_2$ (fraction of $b_2$)')",
                '    ax.set_title(title)',
                "axes[0][0].set_ylabel('Wannier charge centre (fraction of $a_1$)')",
                'plt.show()',
            ],
        )
        menu.append(('Z2: Wannier charge centres', 'plot_z2_wannier_centres'))
    if 'mirror_chern' in props:
        funcs += _plot_func(
            'plot_mirror_chern',
            [
                'import json',
                '',
                "f = _one('mirror_chern.json')",
                'if _missing(f):',
                '    return',
                'with open(f) as handle:',
                '    result = json.load(handle)',
                "for key in ('sigma_h', 'gap', 'C_plus', 'C_minus', 'C_M', 'nu', 'nu_z2'):",
                "    print('  %-8s %s' % (key, result.get(key)))",
            ],
        )
        menu.append(('Mirror Chern number (summary)', 'plot_mirror_chern'))
    if 'doped_transport' in props:
        funcs += _plot_func(
            'plot_doped_transport',
            [
                'import numpy as np',
                'import matplotlib.pyplot as plt',
                '',
                "files = _many('doped_transport_*.dat')",
                'if not files:',
                '    _missing(None)',
                '    return',
                'data = np.atleast_2d(np.loadtxt(files[0]))',
                'fig, (top, middle, bottom) = plt.subplots(3, 1, sharex=True, figsize=(6, 8))',
                "top.plot(data[:, 0], data[:, 1], 'o-')",
                "top.set_ylabel(r'$\\mu$ (eV)')",
                "middle.semilogy(data[:, 0], data[:, 5], 'o-')",
                "middle.set_ylabel(r'$\\rho$ ($\\Omega$ cm)')",
                "for column, label in zip((6, 7, 8), ('xx', 'yy', 'zz')):",
                "    bottom.plot(data[:, 0], data[:, column], 'o-', label=label)",
                "bottom.set_ylabel(r'$S$ ($\\mu$V/K)')",
                "bottom.set_xlabel('Temperature (K)')",
                'bottom.legend()',
                "fig.suptitle('Doped semiconductor: %s' % os.path.basename(files[0])[16:-4])",
                'plt.show()',
            ],
        )
        menu.append(('Doped semiconductor vs temperature', 'plot_doped_transport'))
    return funcs, menu
