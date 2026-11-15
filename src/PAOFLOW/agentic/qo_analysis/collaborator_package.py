"""Build a self-contained package for experimental collaborators."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

TEMPLATE_DIR = Path(__file__).resolve().parent / 'templates' / 'collaborator'


def build_package(
    destination: str | Path,
    orbits: pd.DataFrame,
    sweep_angle: str,
    sweep_peaks: pd.DataFrame,
    lk_table: pd.DataFrame | None,
    assignments: pd.DataFrame | None,
    network: pd.DataFrame | None,
    image: str | Path | None,
    title: str,
    axis_label: str,
    colors: dict[str, str],
    labels: dict[str, str],
    touching_groups: list[tuple[str, str]],
    contact_field: float,
    min_copies: int,
    figure_min_copies: int,
) -> Path:
    """Write ``data/`` (CSV tables and ``meta.json``), the standalone scripts and a README template.

    Returns
    -------
    Path
        ``destination``. The scripts need only Python 3, numpy and matplotlib.
    """
    destination = Path(destination)
    data = destination / 'data'
    data.mkdir(parents=True, exist_ok=True)
    orbits.assign(angle_deg=orbits[f'{sweep_angle}_deg'])[
        ['angle_deg', 'band', 'group', 'freq_T', 'mstar_abs', 'carrier', 'extremum', 'curv_kTA2', 'amp_factor',
         'num_orbit_copies']
    ].rename(columns={'mstar_abs': 'mstar_me'}).to_csv(data / 'calc_orbits.csv', index=False, float_format='%.6g')  # fmt: skip
    sweep_peaks[['calc_angle', 'freq_T', 'rel_mag']].rename(columns={'calc_angle': 'angle_deg'}).to_csv(
        data / 'exp_sweep_peaks.csv', index=False)  # fmt: skip
    if lk_table is not None:
        lk_table.to_csv(data / 'exp_LK_table.csv', index=False)
    if assignments is not None:
        assignments.to_csv(data / 'branch_assignments.csv', index=False)
    if network is not None:
        network.to_csv(data / 'network_orbits.csv', index=False)
    if image:
        shutil.copy(image, data / Path(image).name)
    meta = dict(
        title=title, axis_label=axis_label, colors=colors, labels=labels,
        contact_groups=[list(pair) for pair in touching_groups], contact_B0_max_T=contact_field,
        thresholds=dict(sig_mag=0.15, fig1_min_copies=figure_min_copies, direct_min_copies=min_copies,
                        angle_win=2.6, direct_tol=0.05, match_tol=0.03, combo_tol=0.015,
                        net_min_copies=20, net_min_R=0.05),
    )  # fmt: skip
    (data / 'meta.json').write_text(json.dumps(meta, indent=2))
    for script in ('plot_figures.py', 'weak_peak_tests.py'):
        shutil.copy(TEMPLATE_DIR / script, destination / script)
    if not (destination / 'README.md').exists():
        shutil.copy(TEMPLATE_DIR / 'README_template.md', destination / 'README.md')
    return destination
