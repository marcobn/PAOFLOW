import pandas as pd
import pytest

from PAOFLOW.qo_analysis import QuantumOscillationAnalysis

pytestmark = pytest.mark.unit


def test_workflow_on_synthetic_run(frequency_run, tmp_path):
    peaks = pd.DataFrame(
        dict(
            angle_deg=[0.0, 45.0, 90.0, 45.0],
            freq_T=[500.0, 545.0, 590.0, 3000.0],
            rel_mag=[0.5, 0.5, 0.5, 0.05],
        )
    )
    peaks.to_csv(tmp_path / 'peaks.csv', index=False)
    qo = QuantumOscillationAnalysis(
        frequency_run,
        output_dir=tmp_path / 'out',
        degenerate_groups=1,
        primitive_cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    inventory = qo.fermi_surface_inventory()
    assert inventory['empty_bands'] == [3]
    qo.read_experiment(sweep_peaks=tmp_path / 'peaks.csv')
    result = qo.sweep_comparison(trials=20)
    assert result['rate'] == 1.0
    for name in ('fig_sweep_overlay.png', 'per_angle_listing.txt', 'summary.txt'):
        assert (tmp_path / 'out' / 'sweep' / name).exists()
    package = qo.collaborator_package(tmp_path / 'package')
    assert (package / 'data' / 'meta.json').exists() and (package / 'plot_figures.py').exists()
