"""Model adapter (template): a PAOFLOW output folder -> observables at experimental conditions.

Copy into the study (or propose it for ``PAOFLOW.agentic.thermo_optics`` with a test
when it is reusable).  The adapter only *reads* the calculation output; every method
returns the observable on the model abscissa, at one condition, so that
``metrics.curve_deviation`` can interpolate it to the experimental points.  The built-in
adapter is ``paoflow_optics.OpticsModel`` (absorption, refractive index, emissivity); the
sibling workflows have ``PAOFLOW.agentic.transport.MobilityModel`` / ``ResistivityModel``,
``PAOFLOW.agentic.superconductivity.EliashbergModel`` / ``GapModel`` and
``PAOFLOW.agentic.arpes.ArpesModel``.  All offer ``from_system(folder)``, which reads
``system.py`` of an electron-phonon system folder with
``PAOFLOW.elphon.workflows.load_system`` and records it for ``describe()`` -- copy that
pattern for a new workflow output (``<workflow>/<output>/<results>.npz``).
"""

from pathlib import Path

import numpy as np


class ModelAdapter:
    """<Observable> of a PAOFLOW run in ``output_dir`` (read-only)."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        with np.load(self.output_dir / '<results>.npz', allow_pickle=True) as data:
            self.data = {key: np.asarray(data[key]) for key in data.files}
        self.temperatures = np.asarray(self.data.get('temps_k', []), dtype=float)
        self.x = np.asarray(self.data['<abscissa key>'], dtype=float)  # state its unit

    def bracket(self, temperature_k):
        """The model temperatures enclosing ``temperature_k``."""
        temps = np.sort(self.temperatures)
        k = int(np.searchsorted(temps, temperature_k))
        if np.isclose(temps[min(k, temps.size - 1)], temperature_k):
            return float(temperature_k), float(temperature_k)
        if k == 0 or k == temps.size:
            raise ValueError('%g K outside the model range' % temperature_k)
        return float(temps[k - 1]), float(temps[k])

    def observable(self, temperature_k, interpolate=False, activated=False, **geometry):
        """<Observable> on ``self.x`` at ``temperature_k`` and the experimental ``geometry``.

        With ``interpolate`` a temperature between two model ones is interpolated:
        ln y linear in 1/T if ``activated``, y linear in T otherwise.  Label the result.
        """
        values = np.asarray(self.data['<observable key>'], dtype=float)  # (ntemp, nx)
        low, high = self.bracket(temperature_k) if interpolate else (temperature_k, temperature_k)
        a = int(np.argmin(np.abs(self.temperatures - low)))
        b = int(np.argmin(np.abs(self.temperatures - high)))
        if a == b:
            return values[a]
        if activated:
            w = (1 / temperature_k - 1 / low) / (1 / high - 1 / low)
            return np.exp((1 - w) * np.log(values[a]) + w * np.log(values[b]))
        w = (temperature_k - low) / (high - low)
        return (1 - w) * values[a] + w * values[b]

    def describe(self):
        """Markdown table of the run parameters (for the report)."""
        lines = ['| parameter | value |', '|---|---|']
        lines.append('| temperatures (K) | %s |' % ', '.join('%g' % t for t in self.temperatures))
        return '\n'.join(lines)
