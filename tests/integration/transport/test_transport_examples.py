from __future__ import annotations

from pathlib import Path

import pytest

from .runner import run_example_in_sandbox
from .compare import compare_dat_dirs, CompareFailure


HERE = Path(__file__).resolve().parent
EXAMPLES_ROOT = HERE  # because examples live in tests/integration/transport/


def _discover_examples() -> list[Path]:
    return sorted([p for p in EXAMPLES_ROOT.glob('example*') if p.is_dir()])


@pytest.mark.parametrize('example_dir', _discover_examples(), ids=lambda p: p.name)
def test_transport_example(example_dir: Path, tmp_path: Path) -> None:
    # Put sandbox under tmp_path (pytest cleans it up automatically)
    sandbox_root = tmp_path / 'sandbox'
    sandbox_root.mkdir(parents=True, exist_ok=True)

    result = run_example_in_sandbox(example_dir, sandbox_root)

    # Compare outputs to references inside each example directory
    try:
        compare_dat_dirs(
            result.outdir,
            result.refdir,
            rtol=1e-2,  # same spirit as your tolerance=0.01
            atol=0.0,
            ignore_sign=False,  # set True if you really mean to ignore sign
        )
    except CompareFailure as e:
        # Keep sandbox on failure? tmp_path will be kept by pytest if you run with:
        # pytest --basetemp=... or -s; but easiest is to include the path in error:
        raise AssertionError(
            f'{result.example_name} failed.\nSandbox: {result.workdir}\n{e}'
        ) from e
