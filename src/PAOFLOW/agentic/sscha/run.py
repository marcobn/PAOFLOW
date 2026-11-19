"""Read-only adapter of one SSCHA run directory (``PAOFLOW.sscha_*`` / ``main.sscha.py``).

The run directory holds ``state.json`` (progress, settings), ``pop<p>/`` (configurations, pw.x outputs, matrices,
``summary.json`` of the minimisation), ``hessian/`` (free-energy Hessian, ``summary.json``) and
``force_constants_<kind>/`` (matrices on the harmonic q-grid). Nothing here imports python-sscha: the adapter reads
files only, so an agent can monitor runs on any machine where the directories are visible.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

import numpy as np

from ...phonon.sscha_bridge import UNSTABLE_CM1, completed_outputs, run_status


class SSCHARun:
    """One SSCHA run at one temperature.

    Parameters
    ----------
    sscha_dir : str
        Directory of the run (``output/sscha_T<T>`` of ``main.sscha.py``).

    Raises
    ------
    FileNotFoundError
        If the directory has no ``state.json``.
    """

    def __init__(self, sscha_dir: str) -> None:
        self.sscha_dir = os.path.abspath(sscha_dir)
        path = os.path.join(self.sscha_dir, 'state.json')
        if not os.path.isfile(path):
            raise FileNotFoundError('%s: not an SSCHA run (no state.json)' % self.sscha_dir)
        with open(path) as handle:
            self.state: dict[str, Any] = json.load(handle)

    # ------------------------------------------------------------------ basic facts
    @property
    def temperature(self) -> float:
        """Ionic temperature (K)."""
        return float(self.state['temperature'])

    @property
    def electronic_temperature(self) -> float | None:
        """Fermi-Dirac temperature of the force runs (K), ``None`` for the parent's occupations."""
        value = self.state.get('electronic_temperature')
        return None if value is None else float(value)

    @property
    def supercell(self) -> tuple[int, ...]:
        """Diagonal supercell of the configurations."""
        return tuple(int(n) for n in self.state['supercell'])

    def status(self, min_fraction: float = 0.95) -> dict:
        """Stage and next step (:func:`~PAOFLOW.phonon.sscha_bridge.run_status`)."""
        return run_status(self.sscha_dir, min_fraction=min_fraction)

    def history(self) -> list[dict]:
        """``summary.json`` of every minimised population (population, size, free energy and error, Kong-Liu
        ratio, convergence, lowest auxiliary frequencies in cm^-1)."""
        return self.status()['history']

    def hessian(self) -> dict | None:
        """The free-energy Hessian verdict (``stable``, ``lowest_cm1`` as ``(q, frequency)`` pairs), or ``None``."""
        path = os.path.join(self.sscha_dir, 'hessian', 'summary.json')
        if not os.path.isfile(path):
            return None
        with open(path) as handle:
            return json.load(handle)

    def lowest_hessian(self) -> tuple[list[float], float] | None:
        """``(q, frequency)`` of the softest Hessian mode on the supercell grid (cm^-1, imaginary negative)."""
        hessian = self.hessian()
        if hessian is None:
            return None
        q, w = min(hessian['lowest_cm1'], key=lambda row: row[1])
        return list(q), float(w)

    def force_constants(self, kind: str = 'hessian') -> str | None:
        """Path prefix ``<dir>/<prefix>.dyn`` of the exported matrices (``'auxiliary'`` or ``'hessian'``)."""
        found = sorted(
            glob.glob(os.path.join(self.sscha_dir, 'force_constants_%s' % kind, '*.dyn0'))
        )
        return found[0][:-1] if found else None

    def failed_configurations(self) -> dict[int, tuple[int, int]]:
        """``{population: (failed, total)}`` of the configurations without a finished pw.x output."""
        rows = {}
        for folder in sorted(glob.glob(os.path.join(self.sscha_dir, 'pop[0-9]*'))):
            population = int(os.path.basename(folder)[3:])
            if population == 0:
                continue
            outputs, total = completed_outputs(folder)
            rows[population] = (total - len(outputs), total)
        return rows

    # ------------------------------------------------------------------ diagnosis
    def diagnostics(self, fine_mesh: tuple[int, int, int] | None = None) -> list[dict]:
        """Findings on the convergence and the stability of the run, worst first.

        Parameters
        ----------
        fine_mesh : tuple of int, optional
            Also evaluate the exported Hessian force constants on this q-mesh
            (:func:`~PAOFLOW.agentic.sscha.dispersion.grid_minimum`): the stability verdict of the supercell grid
            alone can miss an instability between its q-points.

        Returns
        -------
        list of dict
            ``level`` (``'error'``, ``'warning'``, ``'info'``), ``code`` and ``message`` per finding.
        """
        findings: list[dict] = []
        status = self.status()
        history = status['history']
        ratio = float(self.state.get('kong_liu_ratio', 0.5))
        budget = len(self.state.get('population_sizes') or [])

        if status['stage'] == 'failed':
            findings.append(dict(level='error', code='not_converged', message=(
                'not converged after %d populations: extend population_sizes (larger populations if every '
                'minimisation stopped on the Kong-Liu ratio)' % budget)))  # fmt: skip
        stopped = [
            row
            for row in history
            if not row['converged'] and row.get('kong_liu', 1.0) <= ratio + 0.02
        ]
        if len(history) >= 3 and len(stopped) == len(history) and not status['converged']:
            findings.append(dict(level='warning', code='kong_liu_every_population', message=(
                'every population stopped on the Kong-Liu ratio %.2f: the Gaussian moves too far per population; '
                'use larger populations or a smaller min_step_dyn' % ratio)))  # fmt: skip
        if len(history) >= 2 and not status['converged']:
            last, previous = history[-1], history[-2]
            error = np.hypot(last['free_energy_error_ry'], previous['free_energy_error_ry'])
            drift = abs(last['free_energy_ry'] - previous['free_energy_ry'])
            if error > 0 and drift > 3.0 * error:
                findings.append(dict(level='info', code='free_energy_drift', message=(
                    'free energy still drifting: %.2e Ry between the last populations (3 sigma = %.2e Ry)'
                    % (drift, 3.0 * error))))  # fmt: skip
            w_last, w_prev = (
                np.asarray(last['lowest_cm1'][:3]),
                np.asarray(previous['lowest_cm1'][:3]),
            )
            change = np.max(np.abs(w_last - w_prev) / np.maximum(np.abs(w_prev), 1.0))
            if change > 0.05:
                findings.append(
                    dict(
                        level='info',
                        code='frequencies_moving',
                        message=(
                            'lowest auxiliary frequencies moved %.0f %% between the last populations'
                            % (100 * change)
                        ),
                    )
                )
        for population, (failed, total) in self.failed_configurations().items():
            if total and failed > 0.05 * total:
                findings.append(dict(level='warning', code='failed_configurations', message=(
                    'population %d: %d of %d pw.x runs without forces (scf convergence, wall time, memory)'
                    % (population, failed, total))))  # fmt: skip
        hessian = self.hessian()
        if hessian is not None:
            q, w = self.lowest_hessian()
            if not hessian['stable']:
                findings.append(dict(level='info', code='unstable', message=(
                    'free-energy Hessian unstable at %g K (%.1f cm^-1 at q = %s): the lattice is not stabilised '
                    'at this temperature' % (self.temperature, w, np.round(q, 4).tolist()))))  # fmt: skip
            fc = self.force_constants('hessian')
            if fine_mesh is not None and fc is not None:
                from .dispersion import grid_minimum

                q_fine, w_fine = grid_minimum(fc, fine_mesh)
                if w_fine < UNSTABLE_CM1 and hessian['stable']:
                    findings.append(dict(level='warning', code='unstable_between_grid_points', message=(
                        'stable on the supercell grid but %.1f cm^-1 at q = %s of the %s mesh: the supercell is '
                        'too small or the harmonic support misses the instability'
                        % (w_fine, np.round(q_fine, 4).tolist(), 'x'.join(map(str, fine_mesh))))))  # fmt: skip
        if self.electronic_temperature is None and self.temperature >= 1000.0:
            findings.append(dict(level='info', code='cold_electrons', message=(
                'force runs at %g K use the occupations of the parent scf; for a metal at high temperature '
                'consider electronic_temperature=True (Fermi-Dirac at k_B T)' % self.temperature)))  # fmt: skip
        order = {'error': 0, 'warning': 1, 'info': 2}
        return sorted(findings, key=lambda row: order[row['level']])

    def describe(self) -> str:
        """Text summary: settings, populations, Hessian, exported matrices, findings."""
        status = self.status()
        lines = ['SSCHA run %s' % self.sscha_dir,
                 '  T = %g K, electrons %s, supercell %s, stage %s' % (
                     self.temperature,
                     'parent occupations' if self.electronic_temperature is None
                     else 'Fermi-Dirac %g K' % self.electronic_temperature,
                     'x'.join(map(str, self.supercell)), status['stage'])]  # fmt: skip
        for row in status['history']:
            lines.append('  pop %2d  %4s cfg  F = %.6f +- %.6f Ry  Kong-Liu %4s  converged %-5s  lowest %s cm-1' % (
                row['population'], row.get('size', '?'), row['free_energy_ry'], row['free_energy_error_ry'],
                '%.2f' % row['kong_liu'] if 'kong_liu' in row else '?', row['converged'],
                [round(w, 1) for w in row['lowest_cm1'][:3]]))  # fmt: skip
        lowest = self.lowest_hessian()
        if lowest is not None:
            lines.append(
                '  Hessian: %s, softest %.1f cm^-1 at q = %s'
                % (
                    'stable' if self.hessian()['stable'] else 'UNSTABLE',
                    lowest[1],
                    np.round(lowest[0], 4).tolist(),
                )
            )
        for kind in ('auxiliary', 'hessian'):
            if self.force_constants(kind):
                lines.append('  %s matrices: %s' % (kind, self.force_constants(kind)))
        for finding in self.diagnostics():
            lines.append('  [%s] %s' % (finding['level'], finding['message']))
        return '\n'.join(lines)


def find_runs(root: str) -> list[SSCHARun]:
    """Every SSCHA run below ``root`` (directories with a ``state.json``), sorted by temperature."""
    runs = [SSCHARun(os.path.dirname(path)) for path in glob.glob(os.path.join(root, '**', 'state.json'),
                                                                   recursive=True)]  # fmt: skip
    return sorted(runs, key=lambda run: (run.temperature, run.sscha_dir))
