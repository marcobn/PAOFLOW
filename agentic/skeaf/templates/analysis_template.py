"""Quantum-oscillation analysis of a pyskeaf calculation - template (copy and edit the <...> fields).

Stages follow the agent playbook (PAOFLOW/agentic/skeaf/AGENT_WORKFLOW.md).
Heavy steps (breakdown network, new pyskeaf runs) are commented out: ask before running them.
"""

from PAOFLOW.qo_analysis import QuantumOscillationAnalysis

qo = QuantumOscillationAnalysis(
    skeaf_dir='<folder with qo_EF_*_freqvsangle_*.out>',
    output_dir='qo_analysis',
    bxsf_dir=None,  # defaults to skeaf_dir
    bands=None,  # None = every band with a frequency file
    degenerate_groups='auto',  # Kramers pairs detected from the BXSF energy windows
    sweep_angle='polar',  # the angle varied in the pyskeaf rotation
    fixed_angle=None,  # the angle held fixed (deg); None = from the data
    primitive_cell=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # or scf_file='scf.in'
    conventional_from_primitive=None,  # centred cells: rows = conventional vectors from primitive
    field_convention='reciprocal',  # field normal to (hkl); 'real' = along [hkl]
    parent_bands=None,  # e.g. {11: 6, 12: 6} for SOC pairs -> SR bands
    system_name='<system>',
)
# qo.add_reference_run('<label>', '<older run folder>', bands=[...])

# Stage 0/2/4: inputs, bands crossing E_F, degenerate groups, Fermi-surface pieces, copy histogram
inventory = qo.fermi_surface_inventory()
print(inventory['groups'], inventory['missing_bands'], inventory['empty_bands'])

# Stage 1: measured directions (check both conventions)
print(
    qo.field_directions(
        [
            dict(label='[001]', hkl=[0, 0, 1], azimuthal=0.0, polar=0.0),
        ]
    )
)

# Stage 3: experiment (canonical tables; or digitise an FFT-vs-angle image)
qo.read_experiment(
    sweep_peaks='data/exp_sweep_peaks.csv',
    lk_table='data/exp_LK_table.csv',
    angle_scale=1.0,
    angle_offset=0.0,
)
# qo.digitize_fft_map('data/fft_map.png', frequency_axis='x', frequency_pixels=(..., ...),
#                     frequency_values=(10.0, 10000.0), angle_pixels=(..., ...), angle_values=(0.0, 90.0),
#                     data_pixel_range=(..., ...), colorbar=dict(x=..., y0=..., y1=..., vmax=...),
#                     angles=[0, 5, 10])

# Stage 6: continuous sweep comparison (repeat with a stricter copy cut)
for min_copies in (50, 100):
    print(
        qo.sweep_comparison(min_copies=min_copies, assignments=None, axis_label='<rotation angle>')[
            'text'
        ]
    )

# Stages 5 and 7: single directions, anisotropy, mass enhancement
qo.direction_analysis()

# Stage 8: validation
# qo.compare_runs(frequency_windows=[(1000.0, 3000.0)])
# qo.fermi_level_scan('<folder with shifted runs>', bands=[...], tokens=('-0.005', '0', '0.005'))
# qo.single_angle_check(band=..., azimuthal=..., polar=...)

# Stage 9: magnetic breakdown and weak peaks
# qo.breakdown_gaps(bands=[<one band per degenerate group>])
# qo.open_orbit_contacts(open_band=..., other_bands=[...], angles=[...])
# network = qo.breakdown_network(bands=[...], open_band=..., angles=[<one angle first>])['network']
# qo.weak_peak_analysis(network=None)

# Stage 10: collaborator package
# qo.collaborator_package('collaborator_package', axis_label='<rotation angle>')
