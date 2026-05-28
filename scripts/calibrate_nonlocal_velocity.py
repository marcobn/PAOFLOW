"""Phase 4 step 3 — empirical sign+magnitude calibration for the NL velocity.

Runs PAOFLOW on the Si / Al / Cu dielectric benchmarks in three
configurations:

* ``baseline``  — ``nonlocal_velocity = False`` (legacy, no correction)
* ``nlv_neg``   — ``nonlocal_velocity = True``, ``inject = True``, ``sign = -1``
* ``nlv_pos``   — ``nonlocal_velocity = True``, ``inject = True``, ``sign = +1``

and prints a table comparing each configuration's ``eps2`` peak height
against the QE epsilon.x reference (``epsi_<prefix>.dat`` in the same
directory).  The sign that drives the Cu PF/QE ratio from ~0.48 toward
1.0 is the empirically-pinned sign for Phase 4.

Usage
-----
    python scripts/calibrate_nonlocal_velocity.py \\
        --root /Users/marco/Workspace/PAOFLOW/examples/qe_examples \\
        --examples example15 example17_Cu_epsilon

Run with no ``--examples`` to do all three benchmarks.  Each PAOFLOW
invocation is a subprocess so MPI/global state cannot leak between runs.
Outputs land in ``<example>/output_calib_<label>/`` and are NOT cleaned
up automatically so the spectra can be inspected.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Add the repo root to sys.path so we can import the metric helpers from
# the integration test suite without installing it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from tests.integration.qe.dielectric_metrics import (  # noqa: E402
    metrics_from_paoflow_output,
    metrics_from_qe_outputs,
)

# (example_subdir, qe_prefix, eps2 window, eels window)  — windows mirror
# tests/integration/qe/test_dielectric_vs_qe.py::_BENCHMARKS so the
# extracted peaks correspond to the documented features.
EXAMPLES = [
    ('example15', 'silicon', (3.0, 4.0), (3.0, 4.0)),
    ('example17_Al_epsilon', 'aluminum', (0.5, 3.0), (6.0, 9.5)),
    ('example17_Cu_epsilon', 'copper', (1.5, 4.5), (5.0, 9.5)),
]

# Calibration configurations.  Each tuple is (label, attr-overrides) where
# attr-overrides is a dict written into ``paoflow.data_controller.data_attributes``
# *before* ``gradient_and_momenta`` runs.
CONFIGS: list[tuple[str, dict]] = [
    ('baseline', {'nonlocal_velocity': False}),
    (
        'nlv_neg',
        {
            'nonlocal_velocity': True,
            'nonlocal_velocity_inject': True,
            'nonlocal_velocity_sign': -1,
        },
    ),
    (
        'nlv_pos',
        {
            'nonlocal_velocity': True,
            'nonlocal_velocity_inject': True,
            'nonlocal_velocity_sign': 1,
        },
    ),
]


_PATCH_TEMPLATE = """
# >>> nlv calibration patch >>> (auto-generated; safe to delete)
_arry, _attr = paoflow.data_controller.data_dicts()
_attr.update({attr_overrides!r})
# <<< nlv calibration patch <<<
"""


def _patch_main(src: Path, dst: Path, *, label: str, attr_overrides: dict) -> None:
    """Copy ``src`` to ``dst`` after redirecting outputdir + injecting attrs."""
    text = src.read_text()

    # Redirect output to a per-label subdir so concurrent runs don't clobber.
    out_target = f"'./output_calib_{label}/'"
    for old in ("'./output/'", '"./output/"'):
        if old in text:
            text = text.replace(old, out_target, 1)
            break
    else:
        raise RuntimeError(
            f"Could not find './output/' literal in {src}; please rerun "
            'after pinning the outputdir kwarg in main.py.'
        )

    # Insert the attribute-override patch right before gradient_and_momenta.
    hook = 'paoflow.gradient_and_momenta()'
    if hook not in text:
        raise RuntimeError(f'{src}: expected the literal call ``{hook}`` to hook the calibration')
    patch = _PATCH_TEMPLATE.format(attr_overrides=attr_overrides)
    text = text.replace(hook, patch + hook, 1)

    dst.write_text(text)


def _run_one(example_dir: Path, label: str, attr_overrides: dict) -> Path:
    """Run a patched copy of ``example_dir/main.py`` and return its output dir."""
    patched = example_dir / f'_calib_main_{label}.py'
    _patch_main(example_dir / 'main.py', patched, label=label, attr_overrides=attr_overrides)
    env = os.environ.copy()
    env.setdefault('MPLBACKEND', 'Agg')
    try:
        proc = subprocess.run(
            [sys.executable, patched.name],
            cwd=example_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    finally:
        try:
            patched.unlink()
        except FileNotFoundError:
            pass
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f'{example_dir.name}/{label} failed with rc={proc.returncode}')
    out = example_dir / f'output_calib_{label}'
    if not out.is_dir():
        raise RuntimeError(f'{example_dir.name}/{label}: expected {out} to exist after run')
    return out


def _row(label: str, m_pao: dict, m_qe: dict) -> str:
    ratio = m_pao['eps2_peak_height'] / m_qe['eps2_peak_height']
    return (
        f'  {label:10s}  eps2_peak: E={m_pao["eps2_peak_energy"]:6.3f} eV  '
        f'H={m_pao["eps2_peak_height"]:9.4f}  '
        f'PF/QE = {ratio:6.4f}'
    )


def calibrate(root: Path, want: list[str] | None) -> dict:
    results: dict = {}
    for sub, qe_prefix, eps2_win, eels_win in EXAMPLES:
        if want and sub not in want:
            continue
        ex = root / sub
        if not ex.is_dir():
            print(f'[skip] {ex} not a directory')
            continue
        qe_files = {k: ex / f'{k}_{qe_prefix}.dat' for k in ('epsr', 'epsi', 'eels')}
        missing = [str(p) for p in qe_files.values() if not p.exists()]
        if missing:
            print(f'[skip] {sub}: QE reference missing: {", ".join(missing)}')
            continue

        qe = metrics_from_qe_outputs(
            epsr_path=qe_files['epsr'],
            epsi_path=qe_files['epsi'],
            eels_path=qe_files['eels'],
            eps2_peak_window=eps2_win,
            eels_peak_window=eels_win,
        ).asdict()

        results[sub] = {'qe': qe, 'paoflow': {}}
        print(f'\n=== {sub}  (QE eps2 peak H = {qe["eps2_peak_height"]:.4f}) ===')
        for label, attr_overrides in CONFIGS:
            print(f'  [run] {label}: {attr_overrides}')
            outdir = _run_one(ex, label, attr_overrides)
            m = metrics_from_paoflow_output(
                outdir,
                eps2_peak_window=eps2_win,
                eels_peak_window=eels_win,
            ).asdict()
            results[sub]['paoflow'][label] = m
            print(_row(label, m, qe))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--root',
        type=Path,
        default=Path('/Users/marco/Workspace/PAOFLOW/examples/qe_examples'),
        help='Directory containing example15/, example17_Al_epsilon/, example17_Cu_epsilon/.',
    )
    ap.add_argument(
        '--examples',
        nargs='*',
        default=None,
        help='Subset of example dirs to run (default: all).',
    )
    ap.add_argument(
        '--json',
        type=Path,
        default=None,
        help='Optional path to dump the full results JSON.',
    )
    args = ap.parse_args()

    results = calibrate(args.root, args.examples)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f'\n[json] wrote {args.json}')

    # Final summary
    print('\n================ summary (eps2 peak height, PF/QE ratio) ================')
    print(f'  {"example":24s}  {"baseline":>10s}  {"nlv_neg":>10s}  {"nlv_pos":>10s}')
    for sub, data in results.items():
        qe_h = data['qe']['eps2_peak_height']
        cells = []
        for label, _ in CONFIGS:
            m = data['paoflow'].get(label)
            cells.append(f'{m["eps2_peak_height"] / qe_h:10.4f}' if m else f'{"-":>10s}')
        print(f'  {sub:24s}  ' + '  '.join(cells))


if __name__ == '__main__':
    main()
