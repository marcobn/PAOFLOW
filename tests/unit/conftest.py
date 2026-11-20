"""Shared fixtures of the unit tests."""

import pytest


@pytest.fixture(autouse=True)
def _run_in_a_temporary_folder(tmp_path_factory, monkeypatch):
    """Run every unit test in its own empty folder.

    ``PAOFLOW(...)`` creates ``./output`` and writes the run record there: started in the folder
    ``pytest`` was launched from, a test would write into the repository (or into the run record
    of a calculation the developer keeps there).
    """
    monkeypatch.chdir(tmp_path_factory.mktemp('cwd'))
