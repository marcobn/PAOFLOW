"""Input and driver-script generators for PAOFLOW workflows.

Two command-line generators are provided:

- :func:`PAOFLOW.gen.aflow_qe.main` (console script ``paoflow-gen-qe``):
  build a Quantum ESPRESSO ``scf`` input from a materials-database entry
  (AFLOW or C2DB; the database is auto-detected from the identifier).
- :func:`PAOFLOW.gen.paoflow_driver.main` (console script ``paoflow-gen``):
  interactively generate a PAOFLOW ``main.py`` driver script (and an optional
  companion ``plot.py``) from the output of a Quantum ESPRESSO run.
"""

from PAOFLOW.gen.aflow_qe import main as aflow_qe_main
from PAOFLOW.gen.paoflow_driver import main as paoflow_driver_main

__all__ = [
    'aflow_qe_main',
    'paoflow_driver_main',
]
