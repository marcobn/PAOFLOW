"""User-facing orchestrator: compare pyskeaf extremal orbits with quantum-oscillation experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import experiment_data, fermi_sheets, field_geometry, orbit_matching, orbit_tables, qo_plots
from .skeaf_output import (
    fermi_tokens,
    find_bxsf_files,
    frequency_file_inventory,
    plain_bxsf_path,
    read_orbits,
    sweep_angle_name,
)


def markdown_table(table: pd.DataFrame | None, float_format: str = '{:.3g}') -> str:
    """Render a table as GitHub Markdown (no external dependency)."""
    if table is None or len(table) == 0:
        return '(none)'
    shown = table.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(
                lambda v: '' if pd.isna(v) else float_format.format(v)
            )
    header = '| ' + ' | '.join(map(str, shown.columns)) + ' |'
    rule = '|' + '|'.join('---' for _ in shown.columns) + '|'
    body = ['| ' + ' | '.join(map(str, row)) + ' |' for row in shown.itertuples(index=False)]
    return '\n'.join([header, rule] + body)


class QuantumOscillationAnalysis:
    """Interpret pyskeaf extremal orbits against measured quantum-oscillation frequencies.

    Parameters
    ----------
    skeaf_dir : str or Path
        Folder with ``qo_EF_<E>_freqvsangle_<band>.out[.gz]`` files.
    output_dir : str or Path, optional
        Where every step writes its tables and figures (one sub-folder per step).
    bxsf_dir : str or Path or None, optional
        Folder with ``Fermi_surf_band_<n>.bxsf[.gz]``; defaults to ``skeaf_dir``.
    bands : list[int] or None, optional
        Bands to analyse; ``None`` uses every band with a frequency file.
    fermi_token : str, optional
        Fermi-energy token of the main run (eV, as in the file names).
    degenerate_groups : {'auto'} or int, optional
        ``'auto'`` detects Kramers pairs (or larger multiplets) from the BXSF energy
        windows; an integer forces groups of that size.
    sweep_angle : {'auto', 'polar', 'azimuthal'}, optional
        The angle varied in the pyskeaf rotation.
    fixed_angle : float or None, optional
        The angle held fixed in the rotation (deg); inferred from the data if ``None``.
    primitive_cell : list or None, optional
        Primitive vectors (rows, any length unit); alternative to ``scf_file``.
    scf_file : str or Path or None, optional
        Quantum ESPRESSO input with ``CELL_PARAMETERS``.
    conventional_from_primitive : list or None, optional
        Rows = conventional vectors as integer combinations of the primitive ones.
    field_convention : {'reciprocal', 'real'}, optional
        Field normal to (hkl) or along [hkl].
    parent_bands : dict or None, optional
        Band of this run -> band of a reference run (e.g. SOC pair -> scalar-relativistic band).
    system_name : str, optional
        Used in titles and reports.

    Notes
    -----
    Methods follow the stages of the agent playbook
    (``agentic/skeaf/AGENT_WORKFLOW.md`` in the PAOFLOW repository). Every method writes its outputs
    to ``output_dir/<step>/`` and returns a summary; nothing is written into the
    calculation folders.
    """

    def __init__(
        self,
        skeaf_dir: str | Path,
        output_dir: str | Path = 'qo_analysis',
        bxsf_dir: str | Path | None = None,
        bands: list[int] | None = None,
        fermi_token: str = '0',
        degenerate_groups: str | int = 'auto',
        sweep_angle: str = 'auto',
        fixed_angle: float | None = None,
        primitive_cell: list | None = None,
        scf_file: str | Path | None = None,
        conventional_from_primitive: list | None = None,
        field_convention: str = 'reciprocal',
        parent_bands: dict[int, int] | None = None,
        system_name: str = 'system',
    ) -> None:
        self.skeaf_dir = Path(skeaf_dir)
        self.output_dir = Path(output_dir)
        self.bxsf_dir = Path(bxsf_dir) if bxsf_dir else self.skeaf_dir
        self.bands = list(bands) if bands else None
        self.fermi_token = fermi_token
        self.degenerate_groups = degenerate_groups
        self.requested_sweep_angle = sweep_angle
        self.fixed_angle = fixed_angle
        self.primitive_cell = primitive_cell
        self.scf_file = scf_file
        self.conventional_from_primitive = conventional_from_primitive
        self.field_convention = field_convention
        self.parent_bands = dict(parent_bands or {})
        self.system_name = system_name
        self.reference_runs: dict[str, dict] = {}
        self.sweep_peaks: pd.DataFrame | None = None
        self.lk_table: pd.DataFrame | None = None
        self.directions: list[dict] = []
        self.assignments: pd.DataFrame | None = None
        self._band_groups: dict[int, str] | None = None

    # ------------------------------------------------------------------ helpers
    def _step_dir(self, name: str) -> Path:
        folder = self.output_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def cache_dir(self) -> Path:
        """Folder for decompressed BXSF files."""
        return self._step_dir('bxsf_cache')

    def band_groups(self) -> dict[int, str]:
        """Band index to degenerate-group label (detected once, then cached)."""
        if self._band_groups is None:
            if self.degenerate_groups == 'auto' and find_bxsf_files(self.bxsf_dir):
                windows = fermi_sheets.bands_at_fermi_level(self.bxsf_dir, self.cache_dir)
                self._band_groups = fermi_sheets.group_labels(
                    fermi_sheets.degenerate_groups(windows)
                )
            else:
                size = 1 if self.degenerate_groups == 'auto' else int(self.degenerate_groups)
                present = sorted(
                    self.bands or read_orbits(self.skeaf_dir, None, self.fermi_token).band.unique()
                )
                groups = [present[i : i + size] for i in range(0, len(present), size)]
                self._band_groups = fermi_sheets.group_labels(groups)
        return self._band_groups

    def orbits(self, min_copies: int = 50, min_frequency: float = 5.0) -> pd.DataFrame:
        """Robust orbits of the main run (``num_orbit_copies >= min_copies``)."""
        raw = read_orbits(self.skeaf_dir, self.bands, self.fermi_token, self.band_groups())
        return orbit_tables.robust_orbits(raw, min_copies, min_frequency)

    def sweep_angle(self) -> str:
        """Name of the rotated angle (``'polar'`` or ``'azimuthal'``)."""
        raw = read_orbits(self.skeaf_dir, self.bands, self.fermi_token)
        return sweep_angle_name(raw, self.requested_sweep_angle)

    def _fixed_angle(self, sweep: str) -> float:
        if self.fixed_angle is not None:
            return float(self.fixed_angle)
        other = 'azimuthal' if sweep == 'polar' else 'polar'
        raw = read_orbits(self.skeaf_dir, self.bands, self.fermi_token)
        return float(raw[f'{other}_deg'].median())

    def conventional(self) -> np.ndarray:
        """Conventional lattice vectors (rows)."""
        if self.primitive_cell is not None:
            primitive = np.asarray(self.primitive_cell, float)
        elif self.scf_file:
            primitive = field_geometry.read_primitive_cell(self.scf_file)
        else:
            raise ValueError('give primitive_cell or scf_file')
        return field_geometry.conventional_cell(primitive, self.conventional_from_primitive)

    def add_reference_run(self, label: str, skeaf_dir: str | Path, bands: list[int] | None = None,
                          fermi_token: str = '0') -> None:  # fmt: skip
        """Register an older or alternative calculation (coarser grid, no SOC, ...) for comparisons."""
        self.reference_runs[label] = dict(
            skeaf_dir=Path(skeaf_dir), bands=bands, fermi_token=fermi_token
        )

    # ------------------------------------------------------------------ stages 0 and 2
    def fermi_surface_inventory(self) -> dict:
        """Frequency files, bands crossing E_F, degenerate groups, Fermi-surface pieces, copy histogram.

        Returns
        -------
        dict
            ``files``, ``band_windows``, ``groups``, ``missing_bands``, ``empty_bands``,
            ``pieces`` and ``copy_histogram``. Writes ``inventory/inventory.md``.
        """
        files = frequency_file_inventory(self.skeaf_dir, self.fermi_token)
        report = [f'# Inventory: {self.system_name}', '', f'`{self.skeaf_dir}`; Fermi tokens present: '
                  f'{fermi_tokens(self.skeaf_dir)}', '', markdown_table(files), '']  # fmt: skip
        empty = files.loc[files.is_empty, 'band'].tolist() if len(files) else []
        if empty:
            report.append(f'**Empty files for bands {empty}**: check with `single_angle_check` before '
                          'interpreting; an empty file usually means the job died before writing.')  # fmt: skip
        summary = dict(files=files, empty_bands=empty, band_windows=pd.DataFrame(), groups=[],
                       missing_bands=[], pieces=pd.DataFrame())  # fmt: skip
        if find_bxsf_files(self.bxsf_dir):
            windows = fermi_sheets.bands_at_fermi_level(self.bxsf_dir, self.cache_dir)
            groups = fermi_sheets.degenerate_groups(windows)
            requested = sorted(self.bands or files.band.tolist())
            missing = sorted(set(windows.loc[windows.crosses_EF, 'band']) - set(requested))
            pieces = []
            for group in groups:
                found = fermi_sheets.fermi_surface_pieces(
                    find_bxsf_files(self.bxsf_dir)[group[0]], self.cache_dir
                )
                pieces.append(dict(group=','.join(map(str, group)), kind=found['kind'],
                                   pieces=found['n_pieces'],
                                   largest=', '.join(map(str, found['sizes_voxels'][:6])),
                                   occupied=round(found['occupied_fraction'], 3)))  # fmt: skip
            summary.update(band_windows=windows, groups=groups, missing_bands=missing,
                           pieces=pd.DataFrame(pieces))  # fmt: skip
            report += ['## Bands crossing E_F (BXSF energies in Ry)', '',
                       markdown_table(windows.loc[windows.crosses_EF]), '',
                       f'degenerate groups: {groups}',
                       f'computed: {requested}; crossing but NOT computed: {missing or "none"}', '',
                       '## Fermi-surface pieces (one band per group; periodic connectivity)', '',
                       markdown_table(summary['pieces']), '']  # fmt: skip
        raw = read_orbits(self.skeaf_dir, self.bands, self.fermi_token)
        summary['copy_histogram'] = (
            orbit_tables.copy_count_histogram(raw) if len(raw) else pd.DataFrame()
        )
        if len(raw):
            report += ['## NumOrbCopy histogram (justify the robustness cut)', '',
                       markdown_table(summary['copy_histogram']), '',
                       f'extremum types: {raw.extremum.value_counts().to_dict()}']  # fmt: skip
        (self._step_dir('inventory') / 'inventory.md').write_text('\n'.join(report))
        return summary

    # ------------------------------------------------------------------ stage 1
    def field_directions(self, directions: list[dict]) -> str:
        """Check measured directions against SKEAF angles in both conventions.

        Parameters
        ----------
        directions : list[dict]
            ``label``, ``hkl`` and target ``azimuthal``/``polar`` (deg); optional ``day``
            selects one measurement session in the LK table.

        Returns
        -------
        str
            Report (also written to ``geometry/geometry.md``); stores ``directions``.
        """
        self.directions = list(directions)
        report = field_geometry.direction_report(
            self.conventional(), self.directions, self.field_convention
        )
        (self._step_dir('geometry') / 'geometry.md').write_text('```\n' + report + '\n```\n')
        return report

    # ------------------------------------------------------------------ stage 3
    def read_experiment(
        self,
        sweep_peaks: str | Path | None = None,
        lk_table: str | Path | None = None,
        angle_scale: float = 1.0,
        angle_offset: float = 0.0,
    ) -> None:
        """Load the canonical experimental tables (see :mod:`experiment_data`).

        ``calc_angle = angle_scale * measured_angle + angle_offset`` maps the experimental
        rotation angle onto the calculation's sweep angle.
        """
        self.angle_scale, self.angle_offset = angle_scale, angle_offset
        if sweep_peaks:
            self.sweep_peaks = experiment_data.read_sweep_peaks(
                sweep_peaks, angle_scale, angle_offset
            )
        if lk_table:
            self.lk_table = experiment_data.read_lk_table(lk_table)

    def digitize_fft_map(self, image: str | Path, **calibration) -> dict:
        """Digitise an FFT-vs-angle image, validate it against the LK table, and use it as sweep data.

        Parameters
        ----------
        image : str or Path
            The image.
        **calibration
            Arguments of :func:`experiment_data.digitize_fft_map` (axis pixels and values,
            colour bar, measured angles, ...).

        Returns
        -------
        dict
            ``peaks``, ``validation`` (against the configured directions) and
            ``striping``. Writes ``digitize/sweep_peaks_digitised.csv``.
        """
        peaks = experiment_data.digitize_fft_map(image, **calibration)
        folder = self._step_dir('digitize')
        peaks.to_csv(folder / 'sweep_peaks_digitised.csv', index=False, float_format='%.5g')
        self.image_calibration = dict(image=image, **{k: calibration[k] for k in (
            'frequency_axis', 'frequency_pixels', 'frequency_values', 'angle_pixels', 'angle_values')},
            log_frequency=calibration.get('log_frequency', True))  # fmt: skip
        validation = pd.DataFrame()
        if self.lk_table is not None and self.directions:
            rows = pd.concat([experiment_data.lk_rows_for_direction(self.lk_table, d) for d in self.directions],
                             ignore_index=True)  # fmt: skip
            rows = rows.loc[rows.freq_T >= calibration.get('min_frequency', 0.0)]
            validation = experiment_data.validate_digitization(peaks, rows)
            validation.to_csv(folder / 'digitisation_validation.csv', index=False)
        striping = experiment_data.regular_peak_runs(peaks)
        scale, offset = getattr(self, 'angle_scale', 1.0), getattr(self, 'angle_offset', 0.0)
        self.sweep_peaks = peaks.assign(calc_angle=scale * peaks.angle_deg + offset)
        return dict(peaks=peaks, validation=validation, striping=striping)

    # ------------------------------------------------------------------ stage 6
    def sweep_comparison(
        self,
        min_copies: int = 50,
        angle_window: float = 2.6,
        tolerance: float = 0.05,
        weak_threshold: float = 0.15,
        assignments: str | Path | None = None,
        colors: dict | None = None,
        labels: dict | None = None,
        title: str = '',
        axis_label: str | None = None,
        trials: int = 200,
        seed: int = 7,
    ) -> dict:
        """Calculated rotation vs measured sweep: figures, per-angle listing and chance tests.

        Returns
        -------
        dict
            ``rate``, ``n``, ``loguniform`` and ``shuffled`` baselines, ``table`` and
            ``listing``. Writes overlay figures, the listing and a summary to ``sweep/``.

        Notes
        -----
        If the match rate is within about two standard deviations of chance, frequency
        coincidences prove nothing: assign branches from their shape over the rotation.
        """
        if assignments:
            self.assignments = pd.read_csv(assignments, comment='#')
        sweep = self.sweep_angle()
        orbits = self.orbits(min_copies)
        peaks = self.sweep_peaks
        folder = self._step_dir('sweep')
        suffix = '' if min_copies == 50 else f'_N{min_copies}'
        qo_plots.plot_sweep_overlay(orbits, peaks, folder / f'fig_sweep_overlay{suffix}.png', sweep,
                                    weak_threshold, colors, labels, self.assignments,
                                    title or f'{self.system_name}: orbits (N >= {min_copies}) vs experiment',
                                    axis_label)  # fmt: skip
        if getattr(self, 'image_calibration', None):
            qo_plots.plot_on_measured_map(orbits, path=folder / f'fig_on_exp_map{suffix}.png', sweep_angle=sweep,
                                          colors=colors, angle_scale=getattr(self, 'angle_scale', 1.0),
                                          angle_offset=getattr(self, 'angle_offset', 0.0),
                                          **self.image_calibration)  # fmt: skip
        table, summary = orbit_matching.sweep_match(orbits, peaks, sweep, angle_window, tolerance, weak_threshold,
                                                    trials, seed)  # fmt: skip
        table.to_csv(folder / f'sweep_match_table{suffix}.csv', index=False)
        listing = orbit_tables.per_angle_listing(orbits, peaks, sorted(peaks.calc_angle.unique()), sweep,
                                                 weak_threshold)  # fmt: skip
        (folder / f'per_angle_listing{suffix}.txt').write_text(listing)
        text = (f'robust orbits: {len(orbits)} (NumOrbCopy >= {min_copies}); significant peaks: {summary["n"]}\n'
                f'matched within {100 * tolerance:.0f} % (+/-{angle_window} deg): {100 * summary["rate"]:.0f} %\n'
                f'chance: log-uniform {100 * summary["loguniform"][0]:.0f} +/- {100 * summary["loguniform"][1]:.0f} %, '
                f'angle-shuffled {100 * summary["shuffled"][0]:.0f} +/- {100 * summary["shuffled"][1]:.0f} %')  # fmt: skip
        (folder / f'summary{suffix}.txt').write_text(text)
        return dict(summary, table=table, listing=listing, text=text)

    # ------------------------------------------------------------------ stages 5 and 7
    def direction_analysis(
        self,
        min_copies: int = 50,
        merge_fraction: float = 0.015,
        tolerance: float = 0.05,
        anisotropy_window: float = 0.35,
        mass_weight: float = 0.2,
    ) -> dict:
        """Sheet tables at every measured direction, candidates, anisotropy ratios and 1 + lambda.

        Returns
        -------
        dict
            ``sheets`` (per direction) and ``anisotropy`` (per direction pair, best candidate
            per label with verdict and mass enhancement). Writes ``directions/directions.md``.
        """
        sweep = self.sweep_angle()
        orbits = self.orbits(min_copies)
        folder = self._step_dir('directions')
        sheets, report = {}, ['# Single-direction analysis', '']
        for direction in self.directions:
            angle = direction['polar'] if sweep == 'polar' else direction['azimuthal']
            here = orbit_tables.orbits_at_angle(orbits, angle, sweep)
            if here.empty:
                report += [f'## {direction["label"]}: no orbits at {sweep} = {angle}', '']
                continue
            table = orbit_tables.merge_into_sheets(here, merge_fraction)
            table['dFdE_T_per_meV'] = orbit_matching.onsager_slope(table.mstar_me, table.carrier)
            sheets[direction['label']] = table
            report += [f'## {direction["label"]} ({sweep} grid point {here[f"{sweep}_deg"].iloc[0]:.4f} deg)', '',
                       markdown_table(table), '']  # fmt: skip
            if self.lk_table is not None:
                rows = experiment_data.lk_rows_for_direction(self.lk_table, direction)
                if len(rows):
                    report += ['### Candidates (ambiguities kept)', '',
                               markdown_table(orbit_matching.direction_candidates(table, rows, tolerance)), '']  # fmt: skip
        anisotropy = {}
        labels = [d['label'] for d in self.directions if d['label'] in sheets]
        for i, first in enumerate(labels):
            for second in labels[i + 1 :]:
                direction_a = next(d for d in self.directions if d['label'] == first)
                direction_b = next(d for d in self.directions if d['label'] == second)
                ranked = orbit_matching.anisotropy_candidates(
                    sheets[first], sheets[second],
                    experiment_data.lk_rows_for_direction(self.lk_table, direction_a),
                    experiment_data.lk_rows_for_direction(self.lk_table, direction_b),
                    anisotropy_window, mass_weight)  # fmt: skip
                report += [f'## Anisotropy {second} / {first}', '']
                if ranked.empty:
                    report += [
                        'no label measured in both directions with a same-group candidate',
                        '',
                    ]
                    continue
                best = ranked.groupby('label').head(1).copy()
                best['verdict'] = [
                    orbit_matching.anisotropy_verdict(row) for row in best.itertuples()
                ]
                best['1+lambda_ref'] = orbit_matching.mass_enhancement(
                    best.m_ref_exp, best.m_ref_band
                )
                best['1+lambda_cmp'] = orbit_matching.mass_enhancement(
                    best.m_cmp_exp, best.m_cmp_band
                )
                anisotropy[(first, second)] = best
                report += [markdown_table(best), '']
        (folder / 'directions.md').write_text('\n'.join(report))
        return dict(sheets=sheets, anisotropy=anisotropy)

    # ------------------------------------------------------------------ stage 8
    def compare_runs(self, frequency_windows: list[tuple[float, float]] = ((1000.0, 2600.0),),
                     min_copies: int = 50) -> dict:  # fmt: skip
        """Orbit-by-orbit shifts of this run against every reference run, per reference band.

        Returns
        -------
        dict
            Per reference label: ``shifts`` table and frequency-window ``coverage``.
            Writes ``compare_runs/compare_runs.md`` and a panel figure.
        """
        sweep = self.sweep_angle()
        mine = self.orbits(min_copies)
        raw = read_orbits(self.skeaf_dir, self.bands, self.fermi_token)
        folder = self._step_dir('compare_runs')
        report, results, panels = ['# Comparison with reference runs', ''], {}, []
        for label, run in self.reference_runs.items():
            reference_raw = read_orbits(run['skeaf_dir'], run['bands'], run['fermi_token'])
            reference = orbit_tables.robust_orbits(reference_raw, min_copies)
            angles = sorted(
                set(np.round(raw[f'{sweep}_deg'], 4))
                & set(np.round(reference_raw[f'{sweep}_deg'], 4))
            )
            rows = []
            for reference_band in sorted(reference.band.unique()):
                own = [
                    b for b, parent in self.parent_bands.items() if parent == reference_band
                ] or [reference_band]
                ours = mine.loc[mine.band.isin(own)]
                theirs = reference.loc[reference.band == reference_band]
                shifts = orbit_tables.orbit_shifts(theirs, ours, angles, sweep)
                if len(shifts):
                    rows.append(dict(ref_band=reference_band, bands=','.join(map(str, own)), pairs=len(shifts),
                                     median_shift_pct=round(shifts.shift_pct.median(), 2),
                                     iqr=f'{shifts.shift_pct.quantile(.25):+.2f} ... {shifts.shift_pct.quantile(.75):+.2f}'))  # fmt: skip
                panels.append((f'{label} {reference_band} vs {",".join(map(str, own))}', theirs, ours,
                               f'{label} {reference_band}', f'this run {",".join(map(str, own))}'))  # fmt: skip
            coverage = {window: dict(reference=orbit_tables.angular_coverage(reference, *window, sweep),
                                     this_run=orbit_tables.angular_coverage(mine, *window, sweep))
                        for window in frequency_windows}  # fmt: skip
            results[label] = dict(shifts=pd.DataFrame(rows), coverage=coverage)
            report += [
                f'## {label}  (`{run["skeaf_dir"]}`)',
                '',
                markdown_table(pd.DataFrame(rows)),
                '',
            ]
            for window, values in coverage.items():
                report += [f'coverage {window[0]:.0f}-{window[1]:.0f} T: reference {values["reference"]}',
                           f'coverage {window[0]:.0f}-{window[1]:.0f} T: this run {values["this_run"]}', '']  # fmt: skip
        if panels:
            qo_plots.plot_run_comparison(
                panels, self.sweep_peaks, folder / 'fig_run_comparison.png', sweep
            )
        (folder / 'compare_runs.md').write_text('\n'.join(report))
        return results

    def fermi_level_scan(
        self,
        skeaf_dir: str | Path,
        bands: list[int],
        tokens: tuple[str, str, str] = ('-0.005', '0', '0.005'),
        energy_step_meV: float = 5.0,
        frequency_window: tuple[float, float] = (1000.0, 2600.0),
        min_copies: int = 50,
    ) -> dict:
        """Check Fermi-level-shifted runs against the Onsager relation and track orbit windows.

        Returns
        -------
        dict
            ``pairs`` (orbit pairs with measured and Onsager dF/dE), ``median_ratio``,
            ``median_shift_T`` per step, and ``coverage`` per token in ``frequency_window``.
        """
        runs = {token: orbit_tables.robust_orbits(read_orbits(skeaf_dir, bands, token), min_copies, 20.0)
                for token in tokens}  # fmt: skip
        sweep = sweep_angle_name(runs[tokens[1]], self.requested_sweep_angle)
        pairs = orbit_matching.fermi_shift_pairs(runs[tokens[0]], runs[tokens[1]], runs[tokens[2]],
                                                 energy_step_meV, sweep)  # fmt: skip
        pairs.to_csv(self._step_dir('fermi_level_scan') / 'fermi_shift_pairs.csv', index=False)
        return dict(pairs=pairs, median_ratio=float(pairs.ratio.median()),
                    median_shift_T=float((energy_step_meV * pairs.dFdE_meas.abs()).median()),
                    coverage={token: orbit_tables.angular_coverage(run, *frequency_window, sweep)
                              for token, run in runs.items()})  # fmt: skip

    # ------------------------------------------------------------------ stage 9
    def breakdown_gaps(
        self, bands: list[int], upsampling: int = 3, max_gap: float = 0.04
    ) -> pd.DataFrame:
        """Near-contacts between sheets of different degenerate groups (k-gap, B0, splitting).

        Writes ``breakdown/contacts.csv`` and returns the contact table.
        """
        contacts = fermi_sheets.sheet_contacts(
            self.bxsf_dir, bands, self.cache_dir, upsampling, max_gap
        )
        contacts.to_csv(self._step_dir('breakdown') / 'contacts.csv', index=False)
        self.contacts = contacts
        return contacts

    def _plain_bxsf(self, bands: list[int]) -> dict[int, Path]:
        files = find_bxsf_files(self.bxsf_dir)
        return {band: plain_bxsf_path(files[band], self.cache_dir) for band in bands}

    def single_angle_check(
        self, band: int, azimuthal: float, polar: float, num_interpolation: int = 200
    ) -> list:
        """In-process pyskeaf at one direction for one band (diagnoses empty production files)."""
        from .magnetic_breakdown import single_angle_orbits

        return single_angle_orbits(
            self._plain_bxsf([band])[band], azimuthal, polar, num_interpolation
        )

    def open_orbit_contacts(self, open_band: int, other_bands: list[int], angles: list[float],
                            num_interpolation: int = 100, n_jobs: int | None = -1) -> pd.DataFrame:  # fmt: skip
        """Slices where open trajectories of ``open_band`` approach other sheets, per angle."""
        from joblib import Parallel, delayed

        from PAOFLOW.utils.parallel_resources import resolve_n_jobs

        from .magnetic_breakdown import open_trajectory_contacts

        sweep = self.sweep_angle()
        fixed = self._fixed_angle(sweep)
        paths = self._plain_bxsf([open_band] + list(other_bands))
        workers = resolve_n_jobs(n_jobs, n_tasks=len(angles))
        rows = Parallel(n_jobs=workers, backend='loky')(
            delayed(open_trajectory_contacts)(paths, open_band, list(other_bands),
                                              *((fixed, a) if sweep == 'polar' else (a, fixed)),
                                              num_interpolation)
            for a in angles)  # fmt: skip
        table = pd.DataFrame(rows)
        table.to_csv(self._step_dir('breakdown') / 'open_contacts.csv', index=False)
        return table

    def breakdown_network(
        self,
        bands: list[int],
        open_band: int | None = None,
        angles: list[float] | None = None,
        num_interpolation: int = 150,
        junction_cutoff: float = 0.03,
        reference_field: float = 30.0,
        n_jobs: int | None = -1,
        min_copies: int = 50,
        tolerance: float = 0.03,
        trials: int = 200,
        seed: int = 7,
    ) -> dict:
        """Magnetic-breakdown network orbits over the rotation, compared with unexplained peaks.

        Heavy: about 3-4 minutes per angle and core at ``num_interpolation=150`` for a
        multi-sheet metal. Run one angle first.

        Returns
        -------
        dict
            ``network`` (grouped network frequencies), ``statistics`` and the comparison
            summary (``rate`` vs both chance baselines). Writes ``breakdown/network_*.csv``
            and ``breakdown/fig_network_orbits.png``.
        """
        from .magnetic_breakdown import network_orbits_over_sweep

        sweep = self.sweep_angle()
        angles = angles or sorted(self.sweep_peaks.calc_angle.unique())
        labels = {band: self.parent_bands.get(band, band) for band in bands}
        network, statistics = network_orbits_over_sweep(
            self._plain_bxsf(bands), angles, sweep, self._fixed_angle(sweep), n_jobs,
            num_interpolation=num_interpolation, junction_cutoff=junction_cutoff,
            reference_field=reference_field, band_labels=labels, open_band=open_band)  # fmt: skip
        folder = self._step_dir('breakdown')
        network.to_csv(folder / 'network_orbits.csv', index=False)
        statistics.to_csv(folder / 'network_stats.csv', index=False)
        return dict(network=network, statistics=statistics,
                    **self.network_comparison(network, min_copies, tolerance, trials, seed))  # fmt: skip

    def _unexplained_mask(self, orbits: pd.DataFrame, sweep: str, angle_window: float, tolerance: float,
                          weak_threshold: float, weak: bool = False) -> np.ndarray:  # fmt: skip
        peaks = self.sweep_peaks
        selected = (peaks.rel_mag < weak_threshold) if weak else (peaks.rel_mag >= weak_threshold)
        angles_c, freqs_c = orbits[f'{sweep}_deg'].to_numpy(), orbits.freq_T.to_numpy()
        explained = np.array([abs(orbit_matching.nearest_deviation(angles_c, freqs_c, a, f, angle_window)[0]) <= tolerance
                              for a, f in zip(peaks.calc_angle, peaks.freq_T)])  # fmt: skip
        return (selected & ~explained).to_numpy()

    def network_comparison(self, network: pd.DataFrame, min_copies: int = 50, tolerance: float = 0.03,
                           trials: int = 200, seed: int = 7, angle_window: float = 2.6,
                           direct_tolerance: float = 0.05, weak_threshold: float = 0.15) -> dict:  # fmt: skip
        """Robust network orbits vs significant peaks that no direct orbit explains, with chance baselines."""
        sweep = self.sweep_angle()
        orbits = self.orbits(min_copies)
        peaks = self.sweep_peaks
        unexplained = self._unexplained_mask(
            orbits, sweep, angle_window, direct_tolerance, weak_threshold
        )
        robust = network.loc[(network.copies >= 20) & (network.R_ref >= 0.05)]
        rate, hits = orbit_matching.match_rate(robust.angle.to_numpy(), robust.freq_T.to_numpy(),
                                               peaks.calc_angle[unexplained].to_numpy(),
                                               peaks.freq_T[unexplained].to_numpy(), 1e-3, tolerance)  # fmt: skip
        baselines = orbit_matching.chance_baselines(robust.angle.to_numpy(), robust.freq_T.to_numpy(),
                                                    peaks.calc_angle[unexplained].to_numpy(),
                                                    peaks.freq_T[unexplained].to_numpy(), 1e-3, tolerance,
                                                    np.unique(peaks.calc_angle), trials, seed)  # fmt: skip
        matched = np.zeros(len(peaks), dtype=bool)
        matched[np.flatnonzero(unexplained)] = hits
        qo_plots.plot_network_orbits(orbits, network, peaks, unexplained, matched,
                                     self._step_dir('breakdown') / 'fig_network_orbits.png', sweep, weak_threshold,
                                     title=f'{self.system_name}: magnetic-breakdown network orbits')  # fmt: skip
        return dict(rate=rate, n_unexplained=int(unexplained.sum()), n_robust=len(robust),
                    n_robust_open=int(robust.uses_open.sum()), **baselines)  # fmt: skip

    def weak_peak_analysis(self, min_copies: int = 50, angle_window: float = 2.6, tolerance: float = 0.05,
                           combination_tolerance: float = 0.015, weak_threshold: float = 0.15,
                           contact_field: float = 40.0, network: pd.DataFrame | None = None) -> dict:  # fmt: skip
        """Weak peaks: harmonics/combinations, touching-sheet parents, and network orbits, vs chance.

        Returns
        -------
        dict
            ``weak`` and ``significant`` combination summaries, ``parents`` relation counts
            (requires :meth:`breakdown_gaps` first) and, if ``network`` is given, the weak
            peaks vs network-orbit rates. Writes tables to ``weak_peaks/``.
        """
        sweep = self.sweep_angle()
        orbits = self.orbits(min_copies)
        peaks = self.sweep_peaks.reset_index(drop=True)
        as_calc = peaks.assign(angle_deg=peaks.calc_angle)
        folder = self._step_dir('weak_peaks')
        result = {}
        angles_c, freqs_c, groups_c = (
            orbits[f'{sweep}_deg'].to_numpy(),
            orbits.freq_T.to_numpy(),
            orbits.group.to_numpy(),
        )

        def group_of(angle: float, frequency: float):
            deviation, index = orbit_matching.nearest_deviation(
                angles_c, freqs_c, angle, frequency, angle_window
            )
            return groups_c[index] if index >= 0 and abs(deviation) <= tolerance else None

        for subset in ('weak', 'significant'):
            table, summary = orbit_matching.combination_test(as_calc, weak_threshold, combination_tolerance,
                                                             subset=subset)  # fmt: skip
            table.to_csv(folder / f'combinations_{subset}.csv', index=False)
            result[subset] = summary
            if (
                subset == 'weak'
                and getattr(self, 'contacts', None) is not None
                and len(self.contacts)
            ):
                labels = self.band_groups()
                touching = {frozenset((labels.get(int(p.split('-')[0]), p.split('-')[0]),
                                       labels.get(int(p.split('-')[1]), p.split('-')[1])))
                            for p in self.contacts.loc[self.contacts.B0_T < contact_field, 'pair']}  # fmt: skip
                parents = orbit_matching.combination_parents(table, as_calc, group_of, touching)
                parents.to_csv(folder / 'combination_parents.csv', index=False)
                result['parents'] = (
                    parents.relation.value_counts().to_dict() if len(parents) else {}
                )
                result['touching_groups'] = sorted(tuple(sorted(p)) for p in touching)
        if network is not None:
            unexplained_weak = self._unexplained_mask(
                orbits, sweep, angle_window, tolerance, weak_threshold, weak=True
            )
            for name, selection in (('robust', (network.copies >= 20) & (network.R_ref >= 0.05)),
                                    ('open-sheet any', network.uses_open & (network.copies >= 3)),
                                    ('all', network.copies >= 3)):  # fmt: skip
                subset = network.loc[selection]
                rate, _ = orbit_matching.match_rate(subset.angle.to_numpy(), subset.freq_T.to_numpy(),
                                                    peaks.calc_angle[unexplained_weak].to_numpy(),
                                                    peaks.freq_T[unexplained_weak].to_numpy(), 1e-3, 0.03)  # fmt: skip
                base = orbit_matching.chance_baselines(subset.angle.to_numpy(), subset.freq_T.to_numpy(),
                                                       peaks.calc_angle[unexplained_weak].to_numpy(),
                                                       peaks.freq_T[unexplained_weak].to_numpy(), 1e-3, 0.03,
                                                       np.unique(peaks.calc_angle))  # fmt: skip
                result[f'network_{name}'] = dict(rate=rate, **base)
        striping = experiment_data.regular_peak_runs(peaks)
        striping.to_csv(folder / 'striping.csv', index=False)
        result['striping'] = striping
        return result

    # ------------------------------------------------------------------ stage 10
    def collaborator_package(self, destination: str | Path, min_copies: int = 50, figure_min_copies: int = 100,
                             axis_label: str = 'angle (deg)', title: str | None = None, colors: dict | None = None,
                             labels: dict | None = None, network: pd.DataFrame | None = None,
                             contact_field: float = 40.0) -> Path:  # fmt: skip
        """Self-contained package for experimental collaborators (data, numpy+matplotlib scripts, README template).

        Returns
        -------
        Path
            The package folder. Run ``plot_figures.py`` and ``weak_peak_tests.py`` inside it,
            then fill ``README.md`` (keep the provenance caveat box).
        """
        from .collaborator_package import build_package

        sweep = self.sweep_angle()
        orbits = self.orbits(min_copies)
        touching = []
        if getattr(self, 'contacts', None) is not None and len(self.contacts):
            groups = self.band_groups()
            touching = sorted({tuple(sorted((groups.get(int(p.split('-')[0]), p.split('-')[0]),
                                             groups.get(int(p.split('-')[1]), p.split('-')[1]))))
                               for p in self.contacts.loc[self.contacts.B0_T < contact_field, 'pair']})  # fmt: skip
        return build_package(destination, orbits, sweep, self.sweep_peaks, self.lk_table, self.assignments,
                             network, getattr(self, 'image_calibration', {}).get('image'),
                             title or self.system_name, axis_label,
                             qo_plots.group_colors(orbits.group, colors), labels or {}, touching, contact_field,
                             min_copies, figure_min_copies)  # fmt: skip
