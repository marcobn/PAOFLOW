"""Anharmonic phonons at temperature (SSCHA): running, monitoring and interpreting campaigns.

Serves the repository workflow ``agentic/sscha/`` (playbook, lessons, prompts, templates). The engine is PAOFLOW's
SSCHA interface (``PAOFLOW.sscha_*``, :mod:`PAOFLOW.phonon.sscha_bridge`) driven by ``main.sscha.py`` drivers from
``paoflow-gen`` (workflow ``sscha``; launchers ``local``, ``slurm``, ``manual``). This package reads the run
directories only (no python-sscha import) and holds

* :mod:`~PAOFLOW.agentic.sscha.run` -- :class:`~PAOFLOW.agentic.sscha.run.SSCHARun`: one run directory (stage and
  next step, population history, Hessian verdict, exported matrices, failed configurations, convergence and
  stability diagnostics) and :func:`~PAOFLOW.agentic.sscha.run.find_runs`;
* :mod:`~PAOFLOW.agentic.sscha.scan` -- temperature scans: the table of the softest Hessian modes, the stability
  temperature from :math:`\\omega^2 = a (T - T_s)` and the next temperature of a bracketing scan;
* :mod:`~PAOFLOW.agentic.sscha.dispersion` -- dispersions along a path and the softest mode on a fine q-mesh from
  any set of QE dynamical matrices (harmonic or exported SSCHA);
* :mod:`~PAOFLOW.agentic.sscha.plotting` -- convergence, scan and dispersion figures.
"""

from .dispersion import dispersion, grid_minimum, phonon_model
from .run import SSCHARun, find_runs
from .scan import next_temperature, scan_table, signed_square, stability_temperature

__all__ = [
    'SSCHARun',
    'dispersion',
    'find_runs',
    'grid_minimum',
    'next_temperature',
    'phonon_model',
    'scan_table',
    'signed_square',
    'stability_temperature',
]
