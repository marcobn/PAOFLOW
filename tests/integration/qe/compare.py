from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CompareFailure(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _load_dat(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = np.atleast_2d(data)
    return data.T


def compare_dat_dirs(
    outdir: Path,
    refdir: Path,
    *,
    tolerance: float = 0.01,
) -> None:
    out_files = sorted(outdir.glob('*.dat'))
    ref_files = sorted(refdir.glob('*.dat'))

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
                f'  max_abs_err: {max_abs:.6e}\n'
                f'  max_rel_err: {max_rel:.6e}\n'
                f'  tolerance={tolerance}'
            )

    if worst[2] < 0:
        raise CompareFailure('No comparable data found.')
