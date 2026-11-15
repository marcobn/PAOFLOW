"""PDF vector ingestion, the PAOFLOW optics adapter and the figure helpers."""

import numpy as np
import pytest

from PAOFLOW.agentic.thermo_optics import digitize as dg
from PAOFLOW.agentic.thermo_optics.paoflow_optics import HBAR_C_EV_CM, OpticsModel


def test_vector_figure_exact_coordinates(tmp_path):
    pymupdf = pytest.importorskip('pymupdf')
    document = pymupdf.open()
    page = document.new_page(width=400, height=300)
    page.draw_rect(pymupdf.Rect(50, 50, 350, 250), color=(0, 0, 0))
    for k, x in enumerate(
        np.linspace(50, 350, 7)
    ):  # x ticks: values 0..3 every 0.5, majors at integers
        page.draw_line((x, 250), (x, 250 - (8 if k % 2 == 0 else 4)), color=(0, 0, 0))
    for y in (250, 150, 50):  # y majors 0, 1, 2
        page.draw_line((50, y), (58, y), color=(0, 0, 0))
    xs = np.linspace(0.0, 3.0, 61)
    points = [(50 + 100 * x, 250 - 100 * (x**2 / 4.5)) for x in xs]
    page.draw_polyline(points, color=(0, 0, 1))
    path = tmp_path / 'figure.pdf'
    document.save(path)

    from PAOFLOW.agentic.thermo_optics import ingest

    ticks = ingest.pdf_frame_ticks(path, 1)
    assert np.allclose(ticks['x_major'], [50, 150, 250, 350]) and np.allclose(
        ticks['y_major'], [50, 150, 250]
    )
    calibration = dg.AxisCalibration(
        dg.Axis(ticks['x_major'], [0, 1, 2, 3]), dg.Axis(ticks['y_major'], [2, 1, 0])
    )
    (line,) = ingest.pdf_vector_polylines(path, 1, (0, 0, 1))
    data = calibration.to_data(line)
    assert np.allclose(data[:, 1], data[:, 0] ** 2 / 4.5, atol=1e-4)


def synthetic_optics(folder):
    omega = np.arange(0.05, 3.0001, 0.05)
    temps = np.array([300.0, 600.0, 750.0])
    eps1 = np.tile(12.0 + 0.5 * omega, (3, 1))
    eps2 = np.exp(-0.5 / (8.617333262e-5 * temps))[:, None] * np.ones((1, omega.size)) * 1e3
    np.savez(folder / 'absorption.npz', omega_ev=omega, temps_k=temps, etas_ev=np.array([0.01, 0.05]),
             alpha_cm=np.ones((3, 2, omega.size)), nk_dense=48)  # fmt: skip
    np.savez(folder / 'emissivity.npz', omega_ev=omega, temps_k=temps, eps1=eps1, eps2=eps2,
             thickness_m=5e-4, planck_coverage=np.ones(3), direct_source='external')  # fmt: skip
    return omega, temps, eps1, eps2


def test_optics_model_alpha_and_temperature_interpolation(tmp_path):
    omega, temps, eps1, eps2 = synthetic_optics(tmp_path)
    model = OpticsModel(tmp_path)
    assert (
        model.eta_index == 1
        and model.bracket(693.0) == (600.0, 750.0)
        and model.bracket(600.0) == (600.0, 600.0)
    )
    kappa = np.sqrt(eps1[1] + 1j * eps2[1]).imag
    assert np.allclose(model.absorption_coefficient(600.0), 2 * omega * kappa / HBAR_C_EV_CM)
    e1, e2 = model.dielectric(693.0, interpolate=True)
    assert np.allclose(
        e2, 1e3 * np.exp(-0.5 / (8.617333262e-5 * 693.0))
    )  # exact for an activated eps2
    assert np.allclose(e1, eps1[1])
    with pytest.raises(KeyError):
        model.dielectric(693.0)  # off-grid without interpolation
    with pytest.raises(ValueError):
        model.bracket(1000.0)
    assert '| temperatures (K) | 300, 600, 750 |' in model.describe()


def test_optics_model_reports_electrons_provenance(tmp_path):
    synthetic_optics(tmp_path)
    with np.load(tmp_path / 'absorption.npz') as data:
        arrays = dict(data)
    arrays.update(electrons_save='acbn0/Si2.save', electrons_dft='DFT+U+V',
                  electrons_hubbard='DFT+U+V (ortho-atomic): Hubbard_U Si 3p 3.000 eV',
                  coupling_screening='ponce-resta', screening_eps_dft=12.97,
                  screening_eps_target=11.94)  # fmt: skip
    np.savez(tmp_path / 'absorption.npz', **arrays)
    model = OpticsModel(tmp_path)
    assert model.metadata['electrons_dft'] == 'DFT+U+V'
    text = model.describe()
    assert 'DFT+U+V' in text and 'acbn0/Si2.save' in text
    assert '| coupling_screening | ponce-resta (eps 12.97 -> 11.94) |' in text


def test_optics_model_from_system_folder(tmp_path):
    (tmp_path / 'system.py').write_text(
        'from PAOFLOW.elphon.workflows import ElphonSystem\n'
        "SYSTEM = ElphonSystem(prefix='Si2', root='.', masses_amu=[28.085], nelec=8, qgrid=(3, 3, 3))\n"
        "NAME = 'Si'\nOPTICS = dict(nk=12, nq=6)\n"
    )
    out = tmp_path / 'optics' / 'output'
    out.mkdir(parents=True)
    synthetic_optics(out)
    with np.load(out / 'absorption.npz') as data:
        arrays = dict(data)
    arrays.update(position_dipoles='full', scissor_mode='target', scissor_0k_ev=0.56)
    np.savez(out / 'absorption.npz', **arrays)
    model = OpticsModel.from_system(tmp_path)
    assert model.system['prefix'] == 'Si2' and model.system['optics_defaults'] == {
        'nk': 12,
        'nq': 6,
    }
    assert model.metadata['position_dipoles'] == 'full' and model.metadata['scissor_0k_ev'] == 0.56
    text = model.describe()
    assert '| coupling_source | dfpt |' in text and '| scissor_mode | target |' in text
    assert model.bracket(693.0) == (600.0, 750.0)


def test_temperature_colors_and_save(tmp_path):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from PAOFLOW.agentic.thermo_optics.plotting import (
        TEMPERATURE_RAMP,
        provenance_note,
        publication_style,
        save_figure,
        temperature_colors,
    )

    colors = temperature_colors([600, 300, 900])
    assert [colors[t] for t in (300.0, 600.0, 900.0)] == [
        TEMPERATURE_RAMP[0],
        TEMPERATURE_RAMP[2],
        TEMPERATURE_RAMP[4],
    ]
    with pytest.raises(ValueError):
        temperature_colors(range(6))
    with publication_style():
        fig, axis = plt.subplots()
        axis.plot([0, 1], [0, 1])
        provenance_note(fig, 'digitised ' * 40)
        paths = save_figure(fig, tmp_path / 'fig', dpi=100)
        plt.close(fig)
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
