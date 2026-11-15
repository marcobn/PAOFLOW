"""The legacy import path PAOFLOW.qo_analysis aliases PAOFLOW.agentic.qo_analysis."""

import importlib

import pytest


def test_legacy_qo_analysis_imports_alias_the_new_package():
    with pytest.warns(FutureWarning):
        legacy = importlib.import_module('PAOFLOW.qo_analysis.field_geometry')
    new = importlib.import_module('PAOFLOW.agentic.qo_analysis.field_geometry')
    assert legacy is new
    from PAOFLOW.agentic.qo_analysis import QuantumOscillationAnalysis
    from PAOFLOW.qo_analysis import QuantumOscillationAnalysis as Legacy

    assert Legacy is QuantumOscillationAnalysis
