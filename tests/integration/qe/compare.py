from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SKIP_COMPARE_PATTERNS = {
    'hamiltonian.dat',
    'weyl_points.dat',
    'Omega_z_xy.dat',
    'effmass*',
}


@dataclass(frozen=True)
class CompareFailure(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _load_dat(path: Path) -> np.ndarray:
    try:
        data = np.loadtxt(path)
    except ValueError:
        rows = []
        expected_cols = None
        with path.open('r', encoding='utf-8', errors='ignore') as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue

                parts = stripped.split()
                try:
                    values = [float(token) for token in parts]
                except ValueError:
                    continue

                if expected_cols is None:
                    expected_cols = len(values)
                if len(values) != expected_cols:
                    continue

                rows.append(values)

        if not rows:
            raise ValueError(f'No numeric data rows found in {path}')

        data = np.asarray(rows, dtype=float)
    if data.ndim == 1:
        data = np.atleast_2d(data)
    return data.T


def _plot_columns(data: np.ndarray) -> list[int]:
    n_col = data.shape[0]
    if n_col <= 1:
        return []
    if n_col <= 4:
        return list(range(1, n_col))
    cols = [1, 2, n_col - 1]
    return sorted(set(cols))


def _write_comparison_plot(out: np.ndarray, ref: np.ndarray, out_name: str, plot_dir: Path) -> Path:
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f'{Path(out_name).stem}.png'

    x_ref = ref[0, :]
    x_out = out[0, :]
    cols = _plot_columns(ref)

    fig, (ax_data, ax_diff) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for col in cols:
        ax_data.plot(x_ref, ref[col, :], linewidth=1.5, label=f'ref c{col}')
        ax_data.plot(x_out, out[col, :], '--', linewidth=1.2, label=f'out c{col}')
        ax_diff.plot(
            x_out, np.abs(out[col, :] - ref[col, :]), linewidth=1.2, label=f'|delta| c{col}'
        )

    ax_data.set_ylabel('Value')
    ax_data.set_title(f'Output vs Reference: {out_name}')
    ax_data.grid(True, alpha=0.25)
    ax_data.legend(loc='best', fontsize=8)

    ax_diff.set_xlabel('X (first column)')
    ax_diff.set_ylabel('|delta|')
    ax_diff.grid(True, alpha=0.25)
    ax_diff.legend(loc='best', fontsize=8)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    return plot_path


def compare_dat_dirs(
    outdir: Path,
    refdir: Path,
    *,
    tolerance: float = 0.01,
    plot_dir: Path | None = None,
) -> None:
    out_files = sorted(
        p
        for p in outdir.glob('*.dat')
        if not any(fnmatch(p.name, pattern) for pattern in SKIP_COMPARE_PATTERNS)
    )
    ref_files = sorted(
        p
        for p in refdir.glob('*.dat')
        if not any(fnmatch(p.name, pattern) for pattern in SKIP_COMPARE_PATTERNS)
    )

    if not refdir.exists() or len(ref_files) == 0:
        raise CompareFailure(f'Reference directory missing or empty: {refdir}')

    if not outdir.exists() or len(out_files) == 0:
        raise CompareFailure(f'No output files found in: {outdir}')

    out_names = [p.name for p in out_files]
    ref_names = [p.name for p in ref_files]
    if out_names != ref_names:
        missing = sorted(set(ref_names) - set(out_names))
        extra = sorted(set(out_names) - set(ref_names))
        raise CompareFailure(
            'Output/reference file list mismatch.\n'
            f'  outdir: {outdir}\n'
            f'  refdir: {refdir}\n'
            f'  missing outputs: {missing}\n'
            f'  unexpected outputs: {extra}'
        )

    worst = ('', -1.0, -1.0)
    for outp, refp in zip(out_files, ref_files):
        out = _load_dat(outp)
        ref = _load_dat(refp)

        if out.shape != ref.shape:
            raise CompareFailure(
                f'Shape mismatch for {outp.name}: out {out.shape} vs ref {ref.shape}'
            )

        plot_path = None
        if plot_dir is not None:
            plot_path = _write_comparison_plot(out, ref, outp.name, plot_dir)

        n_col = out.shape[0]
        n_row = out.shape[1]
        if n_col < 2:
            raise CompareFailure(f'Not enough columns in {outp.name} for comparison')

        out_cmp = np.abs(out[1:n_col, :])
        ref_cmp = np.abs(ref[1:n_col, :])

        absolute_error = np.sum(np.abs(out_cmp - ref_cmp), axis=1) / n_row
        data_range = np.amax(np.amax(out[1:n_col, :], axis=1), axis=0) - np.amin(
            np.amin(out[1:n_col, :], axis=1), axis=0
        )

        relative_error = []
        valid_data = True
        for idx in range(n_col - 1):
            rel_error = absolute_error[idx] / data_range if data_range != 0 else np.nan
            relative_error.append(rel_error)

            if np.isnan(rel_error) or rel_error > tolerance:
                valid_data = False

        max_abs = float(np.max(absolute_error)) if absolute_error.size else 0.0
        max_rel = float(np.max(relative_error)) if relative_error else 0.0
        if max_rel > worst[2]:
            worst = (outp.name, max_abs, max_rel)

        if not valid_data:
            raise CompareFailure(
                f'Data mismatch in {outp.name}\n'
                f'  out: {outp}\n'
                f'  ref: {refp}\n'
                f'  plot: {plot_path}\n'
                f'  max_abs_err: {max_abs:.6e}\n'
                f'  max_rel_err: {max_rel:.6e}\n'
                f'  tolerance={tolerance}'
            )

    if worst[2] < 0:
        raise CompareFailure('No comparable data found.')
