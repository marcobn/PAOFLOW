"""NbSn2 (Fddd): torque rotation [001] -> [110] vs a spin-orbit pyskeaf calculation.

The experimental torque peaks were DIGITISED from a slide image (data/exp_sweep_peaks.csv
header) and the Lifshitz-Kosevich values TRANSCRIBED from slide text; they are not raw data.
Calculation: SOC, 49^3 Fermi-surface grid, pyskeaf num_interpolation = 320, rotation in the
plane containing c and the (110) normal (azimuthal 29.95 deg, polar 0 -> 90 deg, 90 angles).

Run:  python main.py            (about 2 minutes; set RUN_NETWORK = True for the heavy step)
"""

from PAOFLOW.agentic.qo_analysis import QuantumOscillationAnalysis

RUN_NETWORK = False   # breakdown network over 19 angles: ~15 min on 24 cores

qo = QuantumOscillationAnalysis(
    skeaf_dir='calc/soc',
    output_dir='qo_analysis_soc',
    bands=[11, 12, 13, 14, 15, 16, 17, 18],
    degenerate_groups='auto',
    sweep_angle='polar',
    fixed_angle=29.95,
    primitive_cell=[[0.5, 0.0, 1.690651], [0.5, 0.867802, 0.0], [0.0, 0.867802, 1.690651]],
    conventional_from_primitive=[[1, 1, -1], [-1, 1, 1], [1, -1, 1]],
    field_convention='reciprocal',
    parent_bands={11: 6, 12: 6, 13: 7, 14: 7, 15: 8, 16: 8, 17: 9, 18: 9},
    system_name='NbSn2 (SOC)',
)
qo.add_reference_run('SR 81^3/320', 'calc/sr_81_320', bands=[6, 7, 8, 9])

inventory = qo.fermi_surface_inventory()
print('degenerate groups:', inventory['groups'], '| bands missing:', inventory['missing_bands'])

print(qo.field_directions([
    dict(label='[001]', hkl=[0, 0, 1], azimuthal=29.95, polar=0.0, day=3),
    dict(label='[11y]', hkl=[1, 1, 3.902], azimuthal=29.95, polar=45.0),
    dict(label='[110]', hkl=[1, 1, 0], azimuthal=29.95, polar=90.0, day=3),
]))

qo.read_experiment(lk_table='data/exp_LK_table.csv')
digitised = qo.digitize_fft_map(
    'data/exp_torque_FFT_vs_angle_slide15.png',
    frequency_axis='x', frequency_pixels=(807, 2991), frequency_values=(10.0, 10000.0),
    angle_pixels=(1496, 272), angle_values=(0.0, 90.0), data_pixel_range=(630, 2991),
    colorbar=dict(x=3454, y0=1470, y1=228, vmax=0.001435),
    angles=list(range(0, 91, 5)), prominence=6.0e-5, min_frequency=15.0, pixel_nudges={'0': -4},
)
print(f"digitised peaks: {len(digitised['peaks'])}; median |dev| vs slide tables: "
      f"{digitised['validation'].dev_pct.abs().median():.2f} %")

colors = {'11,12': '#1f77b4', '13,14': '#d62728', '15,16': '#2ca02c', '17,18': '#9467bd'}
labels = {g: f'calc. bands {g} (SR band {s})' for g, s in
          (('11,12', 6), ('13,14', 7), ('15,16', 8), ('17,18', 9))}
axis_label = 'angle from [001] towards [110] (deg)'
for min_copies in (50, 100):
    sweep = qo.sweep_comparison(min_copies=min_copies, assignments='data/branch_assignments_soc.csv',
                                colors=colors, labels=labels, axis_label=axis_label,
                                title='NbSn2 (SOC): calculated extremal orbits vs torque FFT peaks')
    print(sweep['text'])

directions = qo.direction_analysis()
runs = qo.compare_runs()
print(runs['SR 81^3/320']['shifts'])

scan = qo.fermi_level_scan('calc/sr_band7_fermi_scan', bands=[7])
print(f"E_F scan: {len(scan['pairs'])} pairs, measured/Onsager {scan['median_ratio']:.2f}")

contacts = qo.breakdown_gaps(bands=[11, 13, 15, 17])
print(contacts.groupby('pair').k_gap_invA.min())

network = None
if RUN_NETWORK:
    result = qo.breakdown_network(bands=[11, 13, 15, 17], open_band=13)
    network = result['network']
    print(f"network: {result['rate']:.2f} matched vs chance {result['loguniform'][0]:.2f}")

weak = qo.weak_peak_analysis(network=network)
print('weak-peak combinations:', weak['weak'], '| parents:', weak.get('parents'))

qo.collaborator_package('qo_analysis_soc/collaborator_package', axis_label=axis_label,
                        title='NbSn2 (SOC)', colors=colors, labels=labels, network=network)
