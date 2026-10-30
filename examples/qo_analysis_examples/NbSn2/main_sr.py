"""NbSn2: scalar-relativistic pyskeaf run (81^3 grid, num_interpolation 320) vs the torque rotation,
with the coarse 49^3/200 run as convergence reference. BXSF files are not bundled for this run, so
the band groups are set explicitly (no degeneracy without SOC) and the gap map is not available.

Run:  python main_sr.py   (about 30 s; needs data/exp_sweep_peaks.csv from main.py or the bundled copy)
"""

from PAOFLOW.agentic.qo_analysis import QuantumOscillationAnalysis

qo = QuantumOscillationAnalysis(
    skeaf_dir='calc/sr_81_320',
    output_dir='qo_analysis_sr',
    bands=[6, 7, 8, 9],
    degenerate_groups=1,
    sweep_angle='polar',
    fixed_angle=29.95,
    system_name='NbSn2 (scalar-relativistic)',
)
qo.add_reference_run('SR 49^3/200', 'calc/sr_49_200', bands=[6, 7, 8, 9])
qo.read_experiment(sweep_peaks='data/exp_sweep_peaks.csv', lk_table='data/exp_LK_table.csv')

sweep = qo.sweep_comparison(min_copies=50, axis_label='angle from [001] towards [110] (deg)',
                            colors={'6': '#1f77b4', '7': '#d62728', '8': '#2ca02c', '9': '#9467bd'})
print(sweep['text'])

convergence = qo.compare_runs(frequency_windows=[(1000.0, 2600.0)])
print(convergence['SR 49^3/200']['shifts'])
print('band-7 heavy orbits (1-2.6 kT) at angles:',
      convergence['SR 49^3/200']['coverage'][(1000.0, 2600.0)]['this_run'])

scan = qo.fermi_level_scan('calc/sr_band7_fermi_scan', bands=[7])
print(f"E_F +/- 5 meV on band 7: {len(scan['pairs'])} pairs, measured/Onsager {scan['median_ratio']:.2f}")
for token, angles in scan['coverage'].items():
    print(f'  E_F {token}: heavy orbits at {angles}')
