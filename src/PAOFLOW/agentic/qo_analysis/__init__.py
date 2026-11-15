"""Agentic analysis of pyskeaf quantum-oscillation calculations against experiments.

The user-facing class is :class:`QuantumOscillationAnalysis`. The agent playbook,
lessons learned, prompts and the analysis-script template live outside the package, in
the repository folder ``agentic/skeaf/``.
"""

from .analysis_driver import QuantumOscillationAnalysis

__all__ = ['QuantumOscillationAnalysis']
