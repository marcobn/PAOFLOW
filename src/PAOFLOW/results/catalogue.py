"""The catalogue of PAOFLOW outputs: one :class:`~PAOFLOW.results.model.Kind` per file type.

The patterns follow the writers (see ``docs/internals/web-gui-plan.md`` section 9).  Electronic
energies are relative to the Fermi level (PAOFLOW shifts them on output), so their axes carry a
reference line at zero.  Kinds are tried by ``priority`` and the first match wins; ambiguous names
(``alpha_xx.dat`` of the dielectric run against ``alpha_300K.dat`` of the phonon-assisted
absorption) are told apart by the pattern and the folder.
"""

from __future__ import annotations

from .model import Kind

E = ('Energy', 'eV')
PHOTON = ('Photon energy', 'eV')
TEMPERATURE = ('Temperature', 'K')

RUN = 'Run information'
ELECTRONIC = 'Electronic structure'
TRANSPORT = 'Boltzmann transport'
HALL = 'Topology and Hall'
OPTICS = 'Optics'
NLO = 'Non-linear optics'
PHONONS = 'Phonons'
ELPHON = 'Electron–phonon'
GREEN = "Green's-function transport"
ACBN0 = 'ACBN0'
SSCHA = 'Anharmonic phonons (SSCHA)'
QO = 'Quantum oscillations'
OTHER = 'Other files'

CATEGORY_ORDER = (
    RUN,
    ELECTRONIC,
    TRANSPORT,
    HALL,
    OPTICS,
    NLO,
    PHONONS,
    ELPHON,
    GREEN,
    ACBN0,
    SSCHA,
    QO,
    OTHER,
)

#: Linear optics: quantity -> (title, unit).
OPTICAL_QUANTITIES = {
    'epsi': ('Im ε (dielectric function)', ''),
    'epsr': ('Re ε (dielectric function)', ''),
    'ieps': ('Re ε from Kramers–Kronig', ''),
    'sigmar': ('Re σ (optical conductivity)', 'S/m'),
    'sigmai': ('Im σ (optical conductivity)', 'S/m'),
    'eels': ('Electron energy-loss function', ''),
    'nref': ('Refractive index n', ''),
    'kref': ('Extinction coefficient k', ''),
    'alpha': ('Absorption coefficient', '1/m'),
    'refl': ('Normal-incidence reflectivity', ''),
}

#: Non-linear optics: quantity -> (title, unit of the 3D run).
NLO_QUANTITIES = {
    'shift': ('Shift current σ', 'μA/V²'),
    'shiftm': ('Shift current (magnetic) σ', 'μA/V²'),
    'injectc': ('Injection current η (circular)', 'A V⁻² s⁻¹'),
    'injectm': ('Injection current η (linear)', 'A V⁻² s⁻¹'),
    'shgr': ('SHG χ⁽²⁾ (real part)', 'pm/V'),
    'shgi': ('SHG χ⁽²⁾ (imaginary part)', 'pm/V'),
    'shga': ('SHG |χ⁽²⁾|', 'pm/V'),
}

TRANSPORT_QUANTITIES = {
    'sigma': ('Electrical conductivity σ/τ', 'S m⁻¹ s⁻¹'),
    'Seebeck': ('Seebeck coefficient S', 'V/K'),
    'kappa': ('Electronic thermal conductivity κ/τ', 'W m⁻¹ K⁻¹ s⁻¹'),
    'PF': ('Power factor S²σ/τ', 'W m⁻¹ K⁻² s⁻¹'),
}


def _optics(groups: dict) -> tuple[str, str, str]:
    title, unit = OPTICAL_QUANTITIES[groups['q']]
    return title, title.split(' (')[0], unit


def _nlo(groups: dict) -> tuple[str, str, str]:
    title, unit = NLO_QUANTITIES[groups['q']]
    return title, title, unit


def _transport(groups: dict) -> tuple[str, str, str]:
    title, unit = TRANSPORT_QUANTITIES[groups['q']]
    return title, title, unit


def _hall(groups: dict) -> tuple[str, str, str]:
    q = groups['q']
    if q == 'ahc':
        return 'Anomalous Hall conductivity', 'σ_AH', 'S/cm'
    name = 'Spin' if q == 'shc' else 'Orbital'
    return '%s Hall conductivity' % name, 'σ', '(ħ/e) S/cm'


def _ac_hall(groups: dict) -> tuple[str, str, str]:
    part = 'Re' if groups['part'] == 'r' else 'Im'
    if groups['q'] == 'MCD':
        return 'Magnetic circular dichroism (%s σ_xy(ω))' % part, '%s σ(ω)' % part, ''
    name = 'Spin' if groups['q'] == 'SCD' else 'Orbital'
    return '%s Hall conductivity (AC, %s)' % (name, part), '%s σ(ω)' % part, ''


def _velocity(groups: dict) -> tuple[str, str, str]:
    direction = 'xyz'[int(groups['direction'])] if groups.get('direction', '').isdigit() else ''
    return 'Band velocity v_%s' % direction, 'v_%s' % direction, 'a.u.'


KINDS: tuple[Kind, ...] = (
    Kind(
        'manifest',
        'PAOFLOW runs of this folder',
        RUN,
        r'paoflow_outputs\.json',
        'manifest',
        priority=1,
    ),
    # ---------------------------------------------------------------- electronic structure
    Kind(
        'berry_bands',
        'Berry curvature on the bands',
        HALL,
        r'Omegaj?(?:_(?P<op>Spin|Orbital))?_(?P<spol>\w)?_?(?P<component>[xyz]{2})_bands_(?P<spin>\d+)\.dat',
        'bands',
        x=('k path', ''),
        y=('Ω', ''),
        priority=10,
    ),
    Kind(
        'bands',
        'Band structure',
        ELECTRONIC,
        r'(?:(?P<prefix>[^/]+?)\.)?bands(?:_(?P<tag>[A-Za-z]\w*?))?_(?P<spin>\d+)\.dat',
        'bands',
        x=('k path', ''),
        y=E,
        series=('tag', 'spin'),
        zero='y',
        gpao='plot_bands',
        exclude=r'.*(phonon|gruneisen|_band\.dat$).*',
        priority=20,
    ),
    Kind(
        'site_projected_bands',
        'Site-projected bands',
        ELECTRONIC,
        r'site-projected-bands_(?P<spin>\d+)\.dat',
        'weighted_bands',
        x=('k path', ''),
        y=E,
        zero='y',
        gpao='plot_weighted_bands',
    ),
    Kind(
        'velocity',
        '{title}',
        ELECTRONIC,
        r'velocity_(?P<direction>\d)_(?P<spin>\d+)\.dat',
        'bands',
        x=('k path', ''),
        series=('spin',),
        describe=_velocity,
    ),
    Kind(
        'pdos',
        'Projected density of states',
        ELECTRONIC,
        r'(?P<orbital>.+?)_pdos(?:dk)?_(?P<spin>\d+)\.dat',
        'pdos',
        x=E,
        y=('PDOS', 'states/eV'),
        series=('orbital',),
        exclude=r'(?:.*\.)?pdos(?:dk)?_sum_\d+\.dat',
        zero='x',
        gpao='plot_pdos',
        priority=15,
    ),
    Kind(
        'dos',
        'Density of states',
        ELECTRONIC,
        r'(?:(?P<prefix>[^/]+?)\.)?(?P<q>dos|dosdk|pdos_sum|pdosdk_sum)_(?P<spin>\d+)\.dat',
        'curve',
        x=E,
        y=('DOS', 'states/eV'),
        series=('q', 'spin'),
        zero='x',
        gpao='plot_dos',
        priority=30,
    ),
    Kind(
        'spin_texture',
        'Spin texture on the bands',
        ELECTRONIC,
        r'(?P<op>spin|orbital)-texture-bands\.dat',
        'texture',
        x=('k path', ''),
        y=E,
        zero='y',
    ),
    Kind(
        'fermi_surface',
        'Fermi surface',
        ELECTRONIC,
        r'FermiSurf_(?P<spin>\d+)\.bxsf',
        'volume',
        priority=10,
    ),
    Kind(
        'fermi_sheet',
        'Fermi-surface sheets (pyskeaf)',
        QO,
        r'Fermi_surf_band_(?P<band>\d+)\.bxsf',
        'volume',
        series=('band',),
    ),
    Kind(
        'effective_mass',
        'Effective masses',
        ELECTRONIC,
        r'effective_masses_(?P<spin>\d+)\.dat',
        'text',
    ),
    Kind(
        'projectability',
        'Projectability of the Kohn–Sham states',
        ELECTRONIC,
        r'projectability\.txt',
        'curve',
        x=E,
        y=('Projectability', ''),
    ),
    # ---------------------------------------------------------------- Boltzmann transport
    Kind(
        'boltzmann',
        '{title}',
        TRANSPORT,
        r'(?P<q>sigma|Seebeck|kappa|PF)(?P<smearing>[A-Za-z-]*)_(?P<spin>\d+)\.dat',
        'transport',
        x=('Chemical potential μ − E_F', 'eV'),
        series=('spin',),
        zero='x',
        describe=_transport,
        gpao='plot_electrical_conductivity',
    ),
    Kind(
        'hall_coefficient',
        'Hall coefficient',
        TRANSPORT,
        r'hall_trace(?:_(?P<smearing>[\w-]+?))?_(?P<spin>\d+)\.dat',
        'hall',
        x=('Chemical potential μ − E_F', 'eV'),
        y=('R_H', 'm³/C'),
        series=('spin',),
        zero='x',
    ),
    Kind(
        'doping',
        'Chemical potential at doping {type}{conc}',
        TRANSPORT,
        r'doping_(?P<type>[np])(?P<conc>.+)\.dat',
        'curve',
        x=TEMPERATURE,
        y=('Chemical potential μ', 'eV'),
    ),
    Kind(
        'conductivity_component',
        'Conductivity σ_{component}',
        TRANSPORT,
        r'cond_(?P<component>\w+?)_(?P<spin>\d+)\.dat',
        'curve',
        x=E,
        y=('σ', 'a.u.'),
        series=('spin',),
    ),
    # ---------------------------------------------------------------- topology and Hall
    Kind(
        'hall_dc',
        '{title}',
        HALL,
        r'(?P<q>shc|ohc|ahc)Ef_(?:(?P<spol>[xyz])_)?(?P<component>[xyz]{2})\.dat',
        'curve',
        x=E,
        series=('spol', 'component'),
        zero='x',
        describe=_hall,
        gpao='plot_shc',
    ),
    Kind(
        'hall_ac',
        '{title}',
        HALL,
        r'(?P<q>SCD|OCD|MCD)(?P<part>[ri])_(?:(?P<spol>[xyz])_)?(?P<component>[xyz]{2})\.dat',
        'curve',
        x=PHOTON,
        series=('spol', 'component'),
        describe=_ac_hall,
    ),
    Kind(
        'berry_curvature',
        'Berry curvature Ω along the path',
        HALL,
        r'Omega(?P<j>j?)(?:_(?P<op>Spin|Orbital))?_(?:(?P<spol>[xyz])_)?(?P<component>[xyz]{2})\.dat',
        'curve',
        x=('k path', ''),
        y=('Ω', 'a.u.'),
        series=('spol', 'component'),
        gpao='plot_berry',
    ),
    Kind(
        'linear_response',
        'Linear response {q} {parity}',
        HALL,
        r'(?P<q>SHC|REE|AHC|Cond)(?:_(?P<parity>EVEN|ODD))?_(?P<rest>[\w]+)\.dat',
        'curve',
        x=E,
        series=('rest',),
        zero='x',
    ),
    Kind(
        'rashba_edelstein',
        'Rashba–Edelstein response',
        HALL,
        r'(?P<op>spin|orbital)?_?(?:reeEf_(?P<rest>\w+)|Ekai_(?P<component>\w+))\.dat',
        'curve',
        x=E,
        y=('χ', 'ħ cm/V'),
        series=('rest', 'component'),
    ),
    Kind('weyl_points', 'Weyl points', HALL, r'weyl_points\.dat', 'text'),
    Kind('z2', 'Z2 invariants', HALL, r'Z2\.dim', 'text'),
    Kind('berry_phase', 'Berry phase', HALL, r'.*berry_phase.*\.dat', 'columns', x=('index', '')),
    Kind(
        'berry_bxsf',
        '{op} Berry curvature on the grid ({rest})',
        HALL,
        r'(?:(?P<op>Spin|Orbital)_)?Berry_(?P<rest>\w+)\.bxsf',
        'volume',
    ),
    # ---------------------------------------------------------------- optics
    Kind(
        'emissivity_spectral',
        'Spectral emissivity',
        OPTICS,
        r'(?P<q>emish)_(?P<component>[xyz]{2})(?:_(?P<spin>\d))?\.dat',
        'curve',
        x=PHOTON,
        y=('Hemispherical emissivity', ''),
        series=('component', 'spin'),
        priority=25,
    ),
    Kind(
        'emissivity_directional',
        'Directional {q}',
        OPTICS,
        r'(?P<q>refl|emis)_th(?P<angle>[\d.]+)_(?P<component>[xyz]{2})(?:_(?P<spin>\d))?\.dat',
        'curve',
        x=PHOTON,
        y=('Value', ''),
        series=('angle', 'component', 'spin'),
        priority=25,
    ),
    Kind(
        'emissivity_total',
        'Total hemispherical emissivity',
        OPTICS,
        r'emist_(?P<component>[xyz]{2})(?:_(?P<spin>\d))?\.dat',
        'curve',
        x=TEMPERATURE,
        y=('Total emissivity', ''),
        series=('component', 'spin'),
        priority=25,
    ),
    Kind(
        'optics',
        '{title}',
        OPTICS,
        r'(?P<q>epsi|epsr|ieps|sigmar|sigmai|eels|nref|kref|alpha|refl)_(?P<component>[xyz]{2})(?:_(?P<spin>\d))?\.dat',
        'curve',
        x=PHOTON,
        series=('component', 'spin'),
        describe=_optics,
        gpao='plot_optical',
        priority=30,
    ),
    Kind(
        'jdos',
        'Joint density of states',
        OPTICS,
        r'jdos(?:_(?P<spin>\d))?\.dat',
        'curve',
        x=PHOTON,
        y=('JDOS', '1/eV'),
        series=('spin',),
    ),
    # ---------------------------------------------------------------- non-linear optics
    Kind(
        'nlo',
        '{title}',
        NLO,
        r'(?P<q>shift|shiftm|injectc|injectm|shgr|shgi|shga)_(?P<component>[xyz]{3})(?:_(?P<spin>\d))?\.dat',
        'curve',
        x=PHOTON,
        series=('component', 'spin'),
        describe=_nlo,
        gpao='plot_nonlinear',
    ),
    Kind(
        'shg_terms',
        'SHG by term ({part}) χ⁽²⁾_{component}',
        NLO,
        r'shgt(?P<part>[ri])_(?P<term>\w+?)_(?P<component>[xyz]{3})\.dat',
        'curve',
        x=PHOTON,
        y=('χ⁽²⁾', 'pm/V'),
        series=('term',),
        gpao='plot_shg_terms',
    ),
    Kind(
        'coupled_dielectric',
        'Dielectric function at the kernel levels ({q})',
        NLO,
        r'(?P<q>cepsr|cepsi)_(?P<level>ipa|lf|alda|lrc|boot)_(?P<component>[xyz]{2})\.dat',
        'curve',
        x=PHOTON,
        y=('ε', ''),
        series=('level',),
        gpao='plot_coupled_response',
    ),
    Kind(
        'coupled_shg',
        'SHG at the kernel levels ({q}) χ⁽²⁾_{component}',
        NLO,
        r'(?P<q>cshgr|cshgi|cshga)_(?P<level>ipa|lf|alda|lrc|boot)_(?P<component>[xyz]{3})\.dat',
        'curve',
        x=PHOTON,
        y=('χ⁽²⁾', 'pm/V'),
        series=('level',),
    ),
    Kind(
        'molecular_beta',
        'Molecular hyperpolarizability β ({part})',
        NLO,
        r'beta_T_(?P<part>[ri])_(?P<component>[xyz]{3})\.dat',
        'curve',
        x=PHOTON,
        y=('β', 'a.u.'),
        series=('component',),
        gpao='plot_molecular_beta',
    ),
    Kind(
        'nlo_summary',
        'Non-linear optics summary ({name})',
        NLO,
        r'(?P<name>coupled_response|beta_molecular|beta_hrs|beta_depolarization|beta_coupled(?:_static)?)\.dat',
        'text',
    ),
    # ---------------------------------------------------------------- phonons
    Kind(
        'phonon_bands',
        'Phonon dispersion',
        PHONONS,
        r'(?P<f>.+?)_band\.dat',
        'phonon_bands',
        x=('q path', ''),
        y=('Frequency', ''),
        zero='y',
        exclude=r'qha_gruneisen_band\.dat',
        gpao='plot_phonons',
        priority=20,
    ),
    Kind(
        'phonon_dos',
        'Phonon density of states',
        PHONONS,
        r'(?P<f>.+?)_dos\.dat',
        'columns',
        x=('Frequency', ''),
        y=('DOS', 'states/unit'),
        priority=40,
    ),
    Kind(
        'phonon_thermal',
        'Phonon thermodynamics',
        PHONONS,
        r'(?P<f>.+?)_thermal\.dat',
        'columns',
        x=TEMPERATURE,
        columns=(
            ('Free energy', 'kJ/mol', (1,)),
            ('Entropy', 'J K⁻¹ mol⁻¹', (2,)),
            ('Heat capacity C_v', 'J K⁻¹ mol⁻¹', (3,)),
        ),
        gpao='plot_phonon_thermal',
    ),
    Kind(
        'ir_spectrum',
        'Infrared spectrum',
        PHONONS,
        r'(?P<f>.+?)_ir_spectrum\.dat',
        'columns',
        x=('Frequency', ''),
        gpao='plot_ir_spectrum',
    ),
    Kind(
        'raman_spectrum',
        'Raman spectrum',
        PHONONS,
        r'(?P<f>.+?)_raman_spectrum\.dat',
        'columns',
        x=('Frequency', ''),
        gpao='plot_raman_spectrum',
    ),
    Kind(
        'phonon_modes',
        'Phonon modes ({mode})',
        PHONONS,
        r'(?P<f>.+?)_(?P<mode>ir|raman)_modes\.dat',
        'text',
    ),
    Kind(
        'qha',
        'Quasi-harmonic: {q}',
        PHONONS,
        r'qha_(?P<q>ev|volume|thermal_expansion|bulk_modulus|gibbs|heat_capacity|gruneisen)\.dat',
        'columns',
        gpao='plot_qha',
    ),
    # ---------------------------------------------------------------- electron-phonon
    Kind(
        'alpha2f',
        'Eliashberg function α²F(ω)',
        ELPHON,
        r'alpha2F(?P<tr>_tr)?\.dat',
        'eliashberg',
        x=('ω', 'meV'),
        y=('α²F', ''),
        series=('tr',),
        gpao='plot_migdal_eliashberg',
    ),
    Kind(
        'gap_vs_t',
        'Superconducting gap vs temperature',
        ELPHON,
        r'gap_vs_T(?P<aniso>_aniso)?\.dat',
        'columns',
        x=TEMPERATURE,
        columns=(('Gap Δ', 'meV', (1, 2, 3)), ('Renormalisation Z', '', (4,))),
        column_names=('Δ(iω₀)', 'gap edge (Padé)', 'gap edge (analytic continuation)', 'Z(iω₀)'),
    ),
    Kind(
        'max_eigenvalue',
        'Linearised Migdal–Eliashberg kernel',
        ELPHON,
        r'max_eigenvalue(?P<aniso>_aniso)?\.dat',
        'columns',
        x=TEMPERATURE,
        y=('Largest eigenvalue', ''),
    ),
    Kind(
        'mobility',
        'Carrier mobility',
        ELPHON,
        r'mobility\.dat',
        'columns',
        x=TEMPERATURE,
        columns=(
            ('Mobility', 'cm²/(V s)', (4, 5, 6, 7)),
            ('Carrier density', 'cm⁻³', (2, 3)),
            ('Conductivity', 'S/m', (8, 9)),
            ('Seebeck coefficient', 'μV/K', (10,)),
            ('Fermi level μ_F', 'eV', (1,)),
        ),
        column_names=(
            'μ_F',
            'n electrons',
            'n holes',
            'electrons SERTA',
            'holes SERTA',
            'electrons MRTA',
            'holes MRTA',
            'σ SERTA',
            'σ MRTA',
            'S MRTA',
        ),
        gpao='plot_carrier_mobility',
    ),
    Kind(
        'resistivity',
        'Resistivity',
        ELPHON,
        r'resistivity\.dat',
        'columns',
        x=TEMPERATURE,
        y=('ρ', 'μΩ cm'),
        gpao='plot_metal_transport',
    ),
    Kind(
        'phonon_assisted_absorption',
        'Phonon-assisted absorption α(ω)',
        ELPHON,
        r'alpha_(?P<temperature>[\d.]+)K\.dat',
        'temperature_columns',
        x=PHOTON,
        y=('α', 'cm⁻¹'),
        series=('temperature',),
        gpao='plot_phonon_assisted_absorption',
    ),
    Kind(
        'indirect_eps2',
        'Indirect Im ε ({q})',
        ELPHON,
        r'epsilon2_(?P<q>indabs|indabs_lorenz|dirabs)_(?P<temperature>[\d.]+)K\.dat',
        'temperature_columns',
        x=PHOTON,
        y=('Im ε', ''),
        series=('temperature',),
    ),
    Kind(
        'thermal_emissivity',
        'Spectral emissivity vs temperature',
        ELPHON,
        r'(?P<q>emish|eps)_(?P<temperature>[\d.]+)K\.dat',
        'temperature_columns',
        x=PHOTON,
        y=('Value', ''),
        series=('temperature',),
        gpao='plot_thermal_emissivity',
    ),
    Kind(
        'thermal_emissivity_total',
        'Total hemispherical emissivity vs temperature',
        ELPHON,
        r'emist\.dat',
        'columns',
        x=TEMPERATURE,
        columns=(('Emissivity', '', (1, 2)), ('Planck fraction of the grid', '', (3,))),
        column_names=('slab', 'opaque half-space', 'Planck fraction'),
    ),
    Kind(
        'scattering_rates',
        'Scattering rates at {temperature} K',
        ELPHON,
        r'rates_(?P<temperature>[\d.]+)K\.dat',
        'columns',
        x=E,
    ),
    Kind(
        'coupling_path',
        'Coupling |g| along q',
        ELPHON,
        r'g_path\.dat',
        'columns',
        x=('|q|', '2π/a'),
    ),
    Kind(
        'thermal_summary',
        'Thermal renormalisation summary',
        ELPHON,
        r'(?P<name>thermal_summary|lambda_summary)\.dat',
        'text',
    ),
    Kind(
        'eliashberg_iso',
        'Migdal–Eliashberg {what} at {temperature} K',
        ELPHON,
        r'(?P<prefix>[^/]+?)\.(?P<what>imag|pade|acon|qdos)_iso_(?P<temperature>[\d.]+)',
        'columns',
    ),
    Kind(
        'arpes_map',
        'ARPES intensity ({what})',
        ELPHON,
        r'arpes_(?P<what>.+)\.npz',
        'map',
    ),
    Kind(
        'thermal_spectral',
        'Spectral function A(k, E; T) ({what})',
        ELPHON,
        r'thermal_spectral_(?P<what>.+)\.npz',
        'map',
    ),
    Kind(
        'elphon_npz',
        'Arrays of {name}',
        ELPHON,
        r'(?P<name>eliashberg|fs_coupling|migdal_eliashberg(?:_aniso)?|absorption|emissivity|scattering_rates|carrier_scattering|self_energy_fan_migdal|thermal_fc_.+)\.npz',
        'npz',
        priority=80,
    ),
    # ---------------------------------------------------------------- Green's-function transport
    Kind(
        'conductance',
        'Transmission ({post})',
        GREEN,
        r'conductance(?P<post>.*)\.dat',
        'columns',
        x=E,
        y=('Transmission', 'G₀'),
        zero='x',
    ),
    Kind(
        'doscond',
        'Density of states of the conductor ({post})',
        GREEN,
        r'doscond(?P<post>.*)\.dat',
        'columns',
        x=E,
        y=('DOS', 'states/eV'),
        zero='x',
    ),
    Kind(
        'current',
        'Current–voltage',
        GREEN,
        r'current\.dat',
        'curve',
        x=('Bias', 'V'),
        y=('I', 'a.u.'),
    ),
    # ---------------------------------------------------------------- ACBN0
    Kind('acbn0_u', 'Hubbard U of {orbital}', ACBN0, r'(?P<orbital>.+)_UJ\.txt', 'text'),
    Kind('acbn0_v', 'Intersite V {pair}', ACBN0, r'(?P<pair>.+)_V\.txt', 'text'),
    # ---------------------------------------------------------------- SSCHA
    Kind(
        'sscha',
        'SSCHA at {temperature} K',
        SSCHA,
        r'state\.json',
        'sscha',
        folder=r'sscha_T(?P<temperature>[\d.]+)$',
    ),
    # ---------------------------------------------------------------- quantum oscillations
    Kind(
        'dhva',
        'dHvA frequencies vs field angle',
        QO,
        r'(?P<name>.*freqvsangle.*)\.out',
        'dhva',
        x=('Field angle', 'deg'),
        y=('Frequency', 'kT'),
    ),
    # ---------------------------------------------------------------- anything else
    Kind('table', '{name}', OTHER, r'(?P<name>.+)\.(?:dat|txt|csv)', 'columns', priority=95),
    Kind(
        'grid_arrays',
        '{what} arrays on the k grid',
        ELECTRONIC,
        r'(?P<what>Fermi_surf|spin_text|orbital_text)_band_(?P<band>\d+)(?:_(?P<spin>\d+))?\.npz',
        'npz',
        series=('band',),
        priority=60,
    ),
    Kind('npz', 'Arrays of {name}', OTHER, r'(?P<name>.+)\.npz', 'npz', priority=95),
    Kind('json', '{name}', OTHER, r'(?P<name>.+)\.json', 'text', priority=95),
)

#: Folders never scanned (QE saves, basis tables, caches, version control).
SKIPPED_FOLDERS = (
    r'(\.save|\.git|__pycache__|\.paoflow-studio|_ph\d+|BASIS[\w-]*|tmp|cache|pseudo\w*)$'
)

#: File names never listed.
SKIPPED_FILES = r'(.*\.(py|pyc|in|out\.\d+|UPF|upf|xml|pkl|png|pdf|svg|jpg|sh|log|bak-\d+-\d+|md)|kpath_points\.txt|.*_band\.labels|k\.txt|wk\.txt|hamiltonian\.dat|pao_basis\.dat|.*\.bak-.*)'
