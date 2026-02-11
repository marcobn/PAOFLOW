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
    # Match your original transpose behavior
    return data.T


def compare_dat_dirs(
    outdir: Path,
    refdir: Path,
    *,
    rtol: float = 1e-2,
    atol: float = 0.0,
    ignore_sign: bool = False,
) -> None:
    """
    Compare all *.dat files in outdir to those in refdir.
    Raises CompareFailure on mismatch.
    """
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

    worst = ('', 0.0)  # (filename, max_abs_diff)
    for outp, refp in zip(out_files, ref_files):
        out = _load_dat(outp)
        ref = _load_dat(refp)

        if out.shape != ref.shape:
            raise CompareFailure(
                f'Shape mismatch for {outp.name}: out {out.shape} vs ref {ref.shape}'
            )

        if ignore_sign:
            out = np.abs(out)
            ref = np.abs(ref)

        # If you want to ignore the first column like your original script:
        if out.shape[0] > 1:
            out_cmp = out[1:, :]
            ref_cmp = ref[1:, :]
        else:
            out_cmp = out
            ref_cmp = ref

        diff = np.max(np.abs(out_cmp - ref_cmp))
        if diff > worst[1]:
            worst = (outp.name, float(diff))

        if not np.allclose(out_cmp, ref_cmp, rtol=rtol, atol=atol, equal_nan=False):
            # Provide a helpful scalar summary
            abs_err = np.abs(out_cmp - ref_cmp)
            max_abs = float(np.max(abs_err))
            mean_abs = float(np.mean(abs_err))
            raise CompareFailure(
                f'Data mismatch in {outp.name}\n'
                f'  out: {outp}\n'
                f'  ref: {refp}\n'
                f'  max_abs_err: {max_abs:.6e}\n'
                f'  mean_abs_err: {mean_abs:.6e}\n'
                f'  rtol={rtol}, atol={atol}, ignore_sign={ignore_sign}'
            )

    # Optional: you could log worst-case info if desired, but tests should be quiet on pass.
