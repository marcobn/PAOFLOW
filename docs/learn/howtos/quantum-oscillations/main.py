from PAOFLOW.agentic.qo_analysis import QuantumOscillationAnalysis

qo = QuantumOscillationAnalysis(
    skeaf_dir='quantum-oscillations/pyskeaf',  # qo_EF_0_freqvsangle_<band>.out[.gz]
    output_dir='qo_analysis',
    bands=[11, 12, 13, 14, 15, 16, 17, 18],
    degenerate_groups=2,  # spin-orbit run: Kramers pairs (11,12), (13,14), ...
    sweep_angle='polar',
    fixed_angle=29.95,
    primitive_cell=[[0.5, 0.0, 1.690651], [0.5, 0.867802, 0.0], [0.0, 0.867802, 1.690651]],
    conventional_from_primitive=[[1, 1, -1], [-1, 1, 1], [1, -1, 1]],
    field_convention='reciprocal',
    system_name='NbSn2',
)
inventory = qo.fermi_surface_inventory()
print(inventory['copy_histogram'].to_string(index=False))

# Field directions in both conventions
print(
    qo.field_directions(
        [
            dict(label='[001]', hkl=[0, 0, 1], azimuthal=29.95, polar=0.0, day=3),
            dict(label='[11y]', hkl=[1, 1, 3.902], azimuthal=29.95, polar=45.0),
            dict(label='[110]', hkl=[1, 1, 0], azimuthal=29.95, polar=90.0, day=3),
        ]
    )
)

# The measured peaks and the LK table
qo.read_experiment(
    sweep_peaks='quantum-oscillations/experiment/exp_sweep_peaks.csv',
    lk_table='quantum-oscillations/experiment/exp_LK_table.csv',
    angle_scale=1.0,  # calculated angle = angle_scale * measured angle + angle_offset
    angle_offset=0.0,
)

# Matches against chance
palette = {'11,12': '#004488', '13,14': '#D55E00', '15,16': '#FFC20A', '17,18': '#009E73'}
sweeps = {}
for min_copies in (50, 100):
    sweeps[min_copies] = qo.sweep_comparison(
        min_copies=min_copies,
        angle_window=2.6,  # deg around each measured angle
        tolerance=0.05,  # |F_calc - F_exp| / F_exp
        weak_threshold=0.15,  # peaks below 15 % of the largest FFT magnitude are not tested
        colors=palette,
    )
    print(sweeps[min_copies]['text'])

# Anisotropy and mass enhancement
directions = qo.direction_analysis(min_copies=50, tolerance=0.05, anisotropy_window=0.35)
columns = ['label', 'group', 'F_ref_exp', 'F_ref_calc', 'F_cmp_exp', 'F_cmp_calc', 'ratio_exp',
           'ratio_calc', '1+lambda_ref', 'verdict']  # fmt: skip
for (first, second), best in directions['anisotropy'].items():
    print(f'{first} -> {second}')
    print(best[columns].round(3).to_string(index=False))
