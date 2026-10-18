"""Adapter: PAOFLOW optical-absorption and thermal-emissivity outputs as model curves.

Reads the ``absorption.npz`` and ``emissivity.npz`` written by the phonon-assisted
absorption driver (:mod:`PAOFLOW.elphon.phonon_assisted_absorption`) and evaluates
the observables at experimental conditions:

* :meth:`OpticsModel.absorption_coefficient` -- ``alpha(E)`` from the full complex
  dielectric function, :math:`\\alpha = 2\\omega\\kappa/c` with
  :math:`n + i\\kappa = \\sqrt{\\varepsilon_1 + i\\varepsilon_2}` (or, without the
  emissivity file, the driver's ``alpha_cm`` with its constant refractive index);
* :meth:`OpticsModel.refractive_index` -- ``n`` and ``kappa``;
* :meth:`OpticsModel.emissivity` -- the directional slab emissivity recomputed at
  the experimental thickness and angle (Kirchhoff, incoherent slab);
* :meth:`OpticsModel.free_carrier_absorption` -- the sub-gap absorption, the part
  of ``alpha`` below a photon-energy cut where only free carriers absorb.

Every method works on the model temperatures; :func:`~PAOFLOW.agentic.thermo_optics.metrics.match_conditions`
pairs them with the measured ones, and ``interpolate=True`` evaluates a temperature
between two model ones (:meth:`OpticsModel.dielectric`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

HBAR_C_EV_CM = 1.97326980e-5  # hbar c in eV cm


class OpticsModel:
    """Model optical observables from a PAOFLOW absorption/emissivity output folder.

    Parameters
    ----------
    output_dir : str or Path
        Folder with ``absorption.npz`` and (optionally) ``emissivity.npz``.
    eta_index : int, optional
        Intermediate-state broadening of the phonon-assisted term
        (index into ``etas_ev``; default the EPW-style 0.05 eV entry if present).

    Attributes
    ----------
    temperatures : ndarray
        Model temperatures (K).
    omega_ev : ndarray
        Photon energies (eV).
    metadata : dict
        Run parameters (grids, gaps, scissor, thickness, ...), for the report.
    """

    def __init__(self, output_dir: str | Path, eta_index: int | None = None) -> None:
        self.output_dir = Path(output_dir)
        with np.load(self.output_dir / 'absorption.npz', allow_pickle=True) as data:
            self.absorption = {key: np.asarray(data[key]) for key in data.files}
        emissivity_file = self.output_dir / 'emissivity.npz'
        self.emissivity_data = None
        if emissivity_file.exists():
            with np.load(emissivity_file, allow_pickle=True) as data:
                self.emissivity_data = {key: np.asarray(data[key]) for key in data.files}
        self.temperatures = np.asarray(self.absorption['temps_k'], dtype=float)
        self.omega_ev = np.asarray(self.absorption['omega_ev'], dtype=float)
        etas = np.asarray(self.absorption.get('etas_ev', [0.05]))
        if eta_index is None:
            eta_index = int(np.argmin(np.abs(etas - 0.05)))
        self.eta_index = eta_index
        self.metadata = self._metadata()

    def _metadata(self) -> dict:
        a = self.absorption
        keys = ('nk_dense', 'nq_dense', 'refractive_index', 'indirect_gap_dft_ev', 'direct_gap_dft_ev',
                'scissor_velocity', 'cell_volume_bohr3', 'nelec', 'coupling_save', 'electrons_save',
                'electrons_dft', 'electrons_hubbard')  # fmt: skip
        out = {key: a[key].item() if np.ndim(a[key]) == 0 else a[key] for key in keys if key in a}
        if np.ndim(out.get('refractive_index', 0.0)) > 0:  # n(omega) of the direct dielectric run
            n = np.asarray(out['refractive_index'], dtype=float)
            out['refractive_index'] = 'n(omega) %.3g-%.3g' % (n.min(), n.max())
        for key in ('indirect_gap_ev_t', 'direct_gap_ev_t', 'scissor_ev', 'fermi_levels_ev'):
            if key in a:
                out[key] = dict(zip(self.temperatures.tolist(), np.asarray(a[key]).tolist()))
        out['eta_ev'] = float(np.asarray(a.get('etas_ev', [np.nan]))[self.eta_index])
        out['omega_grid_ev'] = (float(self.omega_ev[0]), float(self.omega_ev[-1]),
                                float(np.diff(self.omega_ev).mean()))  # fmt: skip
        if self.emissivity_data is not None:
            e = self.emissivity_data
            out['emissivity_thickness_m'] = float(e['thickness_m'])
            out['direct_source'] = str(e.get('direct_source', ''))
            out['planck_coverage'] = dict(
                zip(self.temperatures.tolist(), e['planck_coverage'].tolist())
            )
        return out

    def bracket(self, temperature_k: float) -> tuple[float, float]:
        """The model temperatures enclosing ``temperature_k`` (equal if on the grid)."""
        temps = np.sort(self.temperatures)
        if temperature_k < temps[0] or temperature_k > temps[-1]:
            raise ValueError(
                '%g K outside the model range %g-%g K' % (temperature_k, temps[0], temps[-1])
            )
        k = int(np.searchsorted(temps, temperature_k))
        if np.isclose(temps[min(k, temps.size - 1)], temperature_k):
            return float(temperature_k), float(temperature_k)
        return float(temps[k - 1]), float(temps[k])

    def temperature_index(self, temperature_k: float) -> int:
        """Index of a model temperature (exact match required)."""
        k = int(np.argmin(np.abs(self.temperatures - temperature_k)))
        if abs(self.temperatures[k] - temperature_k) > 1e-6:
            raise KeyError(
                'no model temperature %g K (have %s)' % (temperature_k, self.temperatures)
            )
        return k

    def dielectric(
        self, temperature_k: float, interpolate: bool = False
    ) -> tuple[NDArray, NDArray]:
        """``(eps1, eps2)`` on ``omega_ev`` (from ``emissivity.npz``).

        With ``interpolate`` a temperature between two model temperatures
        :math:`T_a < T < T_b` gets :math:`\\varepsilon_1` linear in :math:`T` and
        :math:`\\ln\\varepsilon_2` linear in :math:`1/T` (thermally activated
        carriers and phonon populations), an *interpolated model* to be labelled
        as such.
        """
        if self.emissivity_data is None:
            raise ValueError('eps1 needs emissivity.npz (run the driver with --emissivity)')
        low, high = self.bracket(temperature_k) if interpolate else (temperature_k, temperature_k)
        if low == high:
            k = self.temperature_index(low)
            return self.emissivity_data['eps1'][k], self.emissivity_data['eps2'][k]
        a, b = self.temperature_index(low), self.temperature_index(high)
        eps1, eps2 = self.emissivity_data['eps1'], self.emissivity_data['eps2']
        weight_t = (temperature_k - low) / (high - low)
        weight_inv = (1.0 / temperature_k - 1.0 / low) / (1.0 / high - 1.0 / low)
        log2 = (1 - weight_inv) * np.log(np.clip(eps2[a], 1e-300, None)) + weight_inv * np.log(
            np.clip(eps2[b], 1e-300, None)
        )
        return (1 - weight_t) * eps1[a] + weight_t * eps1[b], np.exp(log2)

    def refractive_index(
        self, temperature_k: float, interpolate: bool = False
    ) -> tuple[NDArray, NDArray]:
        """``(n, kappa)`` with :math:`n + i\\kappa = \\sqrt{\\varepsilon}`."""
        eps1, eps2 = self.dielectric(temperature_k, interpolate)
        root = np.sqrt(eps1 + 1j * eps2)
        return root.real, root.imag

    def absorption_coefficient(
        self, temperature_k: float, constant_index: bool = False, interpolate: bool = False
    ) -> NDArray:
        """``alpha`` (cm^-1) on ``omega_ev``.

        With ``emissivity.npz`` and ``constant_index=False``:
        :math:`\\alpha = 2\\omega\\kappa/c` from the full dielectric function (the
        consistent choice; the dispersion of ``n`` matters above ~2 eV in Si).
        Otherwise the driver's ``alpha_cm`` (constant ``n_r``, chosen ``eta``).
        """
        if self.emissivity_data is None or constant_index:
            alpha = np.asarray(self.absorption['alpha_cm'])[:, self.eta_index]
            low, high = (
                self.bracket(temperature_k) if interpolate else (temperature_k, temperature_k)
            )
            if low == high:
                return alpha[self.temperature_index(low)]
            a, b = self.temperature_index(low), self.temperature_index(high)
            w = (1.0 / temperature_k - 1.0 / low) / (1.0 / high - 1.0 / low)
            logs = np.log(np.clip(alpha[[a, b]], 1e-300, None))
            return np.exp((1 - w) * logs[0] + w * logs[1])
        _, kappa = self.refractive_index(temperature_k, interpolate)
        return 2.0 * self.omega_ev * kappa / HBAR_C_EV_CM

    def emissivity(
        self,
        temperature_k: float,
        thickness_m: float,
        angle_deg: float = 0.0,
        interpolate: bool = False,
    ) -> NDArray[np.float64]:
        """Directional slab emissivity on ``omega_ev`` at a given thickness and angle."""
        from ...response.emissivity import emissivity_from_dielectric

        eps1, eps2 = self.dielectric(temperature_k, interpolate)
        out = emissivity_from_dielectric(
            self.omega_ev, eps1[None], eps2[None], [temperature_k], thickness_m, angles_deg=(angle_deg,),
        )  # fmt: skip
        return out['directional_slab'][0, 0]

    def free_carrier_absorption(
        self, temperature_k: float, max_energy_ev: float, interpolate: bool = False
    ) -> tuple[NDArray, NDArray]:
        """``(omega, alpha)`` below ``max_energy_ev`` (sub-gap: free carriers only)."""
        alpha = self.absorption_coefficient(temperature_k, interpolate=interpolate)
        keep = self.omega_ev <= max_energy_ev
        return self.omega_ev[keep], alpha[keep]

    def describe(self) -> str:
        """Markdown summary of the run parameters (for the report)."""
        m = self.metadata
        lines = ['| parameter | value |', '|---|---|']
        lines.append('| temperatures (K) | %s |' % ', '.join('%g' % t for t in self.temperatures))
        lines.append('| photon grid (eV) | %.3f-%.3f, step %.3f |' % m['omega_grid_ev'])
        for key in ('nk_dense', 'nq_dense', 'indirect_gap_dft_ev', 'direct_gap_dft_ev', 'refractive_index',
                    'scissor_velocity', 'eta_ev', 'emissivity_thickness_m', 'direct_source',
                    'electrons_dft', 'electrons_hubbard', 'electrons_save'):  # fmt: skip
            if key in m:
                value = m[key]
                lines.append(
                    '| %s | %s |' % (key, ('%.4g' % value) if isinstance(value, float) else value)
                )
        if 'indirect_gap_ev_t' in m:
            gaps = ', '.join('%g K: %.3f' % (t, g) for t, g in m['indirect_gap_ev_t'].items())
            lines.append('| indirect gap used (eV) | %s |' % gaps)
        if 'planck_coverage' in m:
            cover = ', '.join('%g K: %.2f' % (t, c) for t, c in m['planck_coverage'].items())
            lines.append('| Planck fraction on the grid | %s |' % cover)
        return '\n'.join(lines)


def photon_energy_window(model: OpticsModel) -> tuple[float, float]:
    """Photon-energy range of the model grid (eV)."""
    return float(model.omega_ev[0]), float(model.omega_ev[-1])


def convert_alpha_to_sqrt_alpha_hnu(
    energy_ev: ArrayLike, alpha_cm: ArrayLike
) -> NDArray[np.float64]:
    """:math:`\\sqrt{\\alpha h\\nu}` ((cm^-1 eV)^1/2), the Macfarlane plot of indirect edges."""
    return np.sqrt(np.clip(np.asarray(alpha_cm) * np.asarray(energy_ev), 0.0, None))
