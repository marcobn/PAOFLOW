"""Adapter: the ARPES outputs of an electron-phonon system folder as model observables.

Reads ``arpes/<output>/`` written by ``arpes/arpes.py`` (:mod:`PAOFLOW.arpes` maps and the
self-energies of :mod:`PAOFLOW.elphon.workflows.arpes`):

* ``arpes_bare.npz`` and ``arpes_<kind>_<T>K.npz`` (``kind`` = ``isotropic``, ``anisotropic``,
  ``fan_migdal``): the band maps (:meth:`IntensityMap.save <PAOFLOW.arpes.IntensityMap.save>`
  layout) on a k-path, bare and dressed with the electron-phonon self-energy;
* ``sigma_<kind>_<T>K.dat``: the isotropic (Engelsberg-Schrieffer) or state-resolved
  ``Sigma(omega, T)`` in meV;
* ``self_energy_fan_migdal.npz``: the band-resolved Fan-Migdal ``Sigma_nk(omega, T)`` on the path.

From these it gives what ARPES measures: the dressed dispersion from MDC peaks, the Fermi
momentum, the bare and renormalised Fermi velocities (the kink, :math:`\\lambda`), the
self-energy, the kink energy, and the phonon linewidths of every state versus temperature.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from . import curves

MAP_PATTERN = re.compile(r'^arpes_(?P<kind>[a-z_]+?)_(?P<temp>[0-9]+(?:\.[0-9]+)?)K\.npz$')
SIGMA_PATTERN = re.compile(r'^sigma_(?P<kind>[a-z_\-]+?)_(?P<temp>[0-9]+(?:\.[0-9]+)?)K\.dat$')
FAN_MIGDAL_FILE = 'self_energy_fan_migdal.npz'
BARE_FILE = 'arpes_bare.npz'


def parse_metadata(text: str) -> dict:
    """The ``metadata`` of a saved map (``repr`` of a dict with NumPy scalars) as a dict."""
    cleaned = re.sub(r'np\.\w+\(([^()]*)\)', r'\1', str(text))
    try:
        value = ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        return {'raw': str(text)}
    return value if isinstance(value, dict) else {'raw': str(text)}


def load_map(path: str | Path) -> dict:
    """The arrays of an ``arpes_*.npz`` map, with ``metadata`` parsed and ``distance`` added."""
    with np.load(path, allow_pickle=True) as data:
        out = {key: np.asarray(data[key]) for key in data.files}
    out['metadata'] = parse_metadata(out['metadata'].item() if 'metadata' in out else '{}')
    out['distance'] = path_distance(out['kpoints'])
    return out


def path_distance(kpoints: NDArray) -> NDArray[np.float64]:
    """Cumulative distance (same unit as ``kpoints``, 1/Å) along a k-path."""
    kp = np.atleast_2d(np.asarray(kpoints, dtype=float))
    if kp.shape[0] < 2:
        return np.zeros(kp.shape[0])
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(kp, axis=0), axis=1))])


def read_sigma_table(path: str | Path) -> dict:
    """A ``sigma_<kind>_<T>K.dat`` file: ``w_ev``, the complex ``sigmas`` (eV) and their ``labels``.

    The isotropic file has one ``Re, Im`` pair; the anisotropic one has a pair per selected
    Fermi-surface state (``crossing``, ``weakest``, ``strongest``) and a last pair for the
    isotropic average.
    """
    path = Path(path)
    with open(path) as handle:
        header = handle.readline().lstrip('#').strip()
    data = np.loadtxt(path, ndmin=2)
    npair = (data.shape[1] - 1) // 2
    sigmas = [(data[:, 1 + 2 * i] + 1j * data[:, 2 + 2 * i]) * 1e-3 for i in range(npair)]
    tokens = [t.strip() for t in re.split(r'\s{2,}', header) if t.strip()][1:]
    labels = []
    for i in range(npair):
        token = tokens[i] if i < len(tokens) else 'column_%d' % i
        token = re.sub(r',\s*T\s*=.*$', '', token)
        token = re.sub(r'\((?:meV|eV)\)', '', token)
        token = re.sub(r'^Re/Im\s+', '', token)
        token = re.sub(r'^Re\s+', '', token)
        token = re.sub(r'\bSigma\b', '', token)
        token = re.sub(r'\bstate\b', '', token)
        labels.append(token.strip() or 'isotropic')
    return {'w_ev': data[:, 0] * 1e-3, 'sigmas': sigmas, 'labels': labels}


class ArpesModel:
    """Model ARPES observables from a PAOFLOW ``arpes/output`` folder.

    Parameters
    ----------
    output_dir : str or Path
        Folder with ``arpes_bare.npz``, ``arpes_<kind>_<T>K.npz``, ``sigma_<kind>_<T>K.dat``
        and/or ``self_energy_fan_migdal.npz``.
    system : dict, optional
        Configuration of the system folder (:meth:`from_system`), reported by :meth:`describe`.

    Attributes
    ----------
    map_files : dict
        ``(kind, temperature_k) -> path`` of the dressed maps (``kind`` with underscores).
    sigma_files : dict
        ``(kind, temperature_k) -> path`` of the self-energy tables.
    fan_migdal : dict or None
        The arrays of ``self_energy_fan_migdal.npz`` (``sigma_ev`` of shape
        ``(ntemp, nk, nband, nw)``, ``energies_ev``, ``bands``, ``w_ev``, ``temps_k``, ``k_cart``).
    metadata : dict
        The experiment settings of the maps (photon energy, polarisation, resolution, ...).
    """

    def __init__(self, output_dir: str | Path, system: dict | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.system = dict(system or {})
        self.map_files: dict[tuple[str, float], Path] = {}
        self.sigma_files: dict[tuple[str, float], Path] = {}
        self._maps: dict[tuple[str, float | None], dict] = {}
        for path in sorted(self.output_dir.iterdir()) if self.output_dir.is_dir() else []:
            match = MAP_PATTERN.match(path.name)
            if match:
                self.map_files[(match['kind'], float(match['temp']))] = path
                continue
            match = SIGMA_PATTERN.match(path.name)
            if match:
                self.sigma_files[(match['kind'].replace('-', '_'), float(match['temp']))] = path
        self.bare_file = self.output_dir / BARE_FILE
        if not self.bare_file.exists():
            self.bare_file = None
        self.fan_migdal = None
        fan = self.output_dir / FAN_MIGDAL_FILE
        if fan.exists():
            with np.load(fan, allow_pickle=True) as data:
                self.fan_migdal = {key: np.asarray(data[key]) for key in data.files}
        if self.bare_file is None and not self.map_files and self.fan_migdal is None:
            raise FileNotFoundError('%s holds no ARPES output' % self.output_dir)
        self.metadata: dict = {}
        if self.bare_file is not None:
            self.metadata = self.map('bare')['metadata']
        elif self.map_files:
            self.metadata = self.map(*next(iter(self.map_files)))['metadata']

    @classmethod
    def from_system(
        cls, system_dir: str | Path, output: str = 'output', workflow: str = 'arpes'
    ) -> ArpesModel:
        """Open the ARPES output of an electron-phonon system folder.

        Parameters
        ----------
        system_dir : str or Path
            A folder holding ``system.py``.
        output : str, optional
            Output directory of ``arpes/arpes.py``.
        workflow : str, optional
            Workflow folder name.

        Returns
        -------
        ArpesModel
        """
        from ...elphon.workflows import load_system

        config = load_system(str(system_dir))
        system = config.SYSTEM
        info = {
            'system_dir': str(Path(system_dir).resolve()),
            'name': getattr(config, 'NAME', system.prefix),
            'prefix': system.prefix,
            'coupling_source': system.coupling_source,
            'coarse_grid': 'x'.join(str(n) for n in system.coarse_grid),
            'qgrid': 'x'.join(str(n) for n in system.qgrid),
            'basis': '%s (%s, pthr %g)'
            % (system.basis_dir or 'BASIS', system.configuration, system.pthr),
            'arpes_defaults': dict(getattr(config, 'ARPES', {}) or {}),
        }
        return cls(Path(system_dir) / workflow / output, system=info)

    # ----------------------------------------------------------------- inventory
    @property
    def kinds(self) -> list[str]:
        """Self-energy kinds with a dressed map (``isotropic``, ``anisotropic``, ``fan_migdal``)."""
        return sorted({kind for kind, _ in self.map_files})

    def temperatures(self, kind: str | None = None) -> list[float]:
        """Temperatures of the dressed maps (of one ``kind`` if given)."""
        return sorted({t for k, t in self.map_files if kind is None or k == kind})

    def map(self, kind: str = 'bare', temperature_k: float | None = None) -> dict:
        """The arrays of a map (cached): ``kind='bare'`` or a dressed ``(kind, temperature_k)``."""
        if kind == 'bare':
            if self.bare_file is None:
                raise FileNotFoundError('%s has no %s' % (self.output_dir, BARE_FILE))
            key: tuple[str, float | None] = ('bare', None)
            path = self.bare_file
        else:
            if temperature_k is None:
                raise ValueError('a dressed map needs its temperature')
            kind = kind.replace('-', '_')
            key = (kind, self._match_temperature(temperature_k, self.temperatures(kind)))
            path = self.map_files[(kind, key[1])]
        if key not in self._maps:
            self._maps[key] = load_map(path)
        return self._maps[key]

    @staticmethod
    def _match_temperature(temperature_k: float, available: list[float]) -> float:
        if not available:
            raise KeyError('no map at any temperature')
        nearest = min(available, key=lambda t: abs(t - temperature_k))
        if abs(nearest - temperature_k) > 1e-6:
            raise KeyError('no map at %g K (have %s)' % (temperature_k, available))
        return nearest

    # ----------------------------------------------------------------- bare band
    def band_energies(self) -> tuple[NDArray, NDArray, NDArray]:
        """``(distance, energies (nk, nb), bands)`` of the window bands on the path (bare map)."""
        bare = self.map('bare')
        return bare['distance'], bare['energies'], bare['bands']

    def crossing_band(self) -> int:
        """Column of the window band that comes closest to E_F on the path."""
        _, energies, _ = self.band_energies()
        return int(np.argmin(np.min(np.abs(energies), axis=0)))

    def bare_dispersion(self, band: int | None = None) -> dict:
        """``k`` (distance along the path) and ``energy_ev`` of one window band (column)."""
        distance, energies, bands = self.band_energies()
        ib = self.crossing_band() if band is None else int(band)
        return {'k': distance, 'energy_ev': energies[:, ib], 'band': int(bands[ib])}

    def bare_fermi_momentum(self, band: int | None = None) -> float:
        """Distance along the path where the bare band crosses E_F."""
        bare = self.bare_dispersion(band)
        return curves.fermi_momentum(bare['energy_ev'], bare['k'])

    # ----------------------------------------------------------------- dressed maps
    def mdc(self, kind: str, temperature_k: float | None, energy_ev: float, broadened: bool = True):
        """``(distance, I(k))`` at the energy closest to ``energy_ev``."""
        amap = self.map(kind, temperature_k)
        iw = int(np.argmin(np.abs(amap['w'] - energy_ev)))
        return amap['distance'], (amap['intensity'] if broadened else amap['raw'])[:, iw]

    def edc(self, kind: str, temperature_k: float | None, ik: int, broadened: bool = True):
        """``(w, I(w))`` at path point ``ik``."""
        amap = self.map(kind, temperature_k)
        return amap['w'], (amap['intensity'] if broadened else amap['raw'])[ik]

    def dispersion(
        self,
        kind: str,
        temperature_k: float | None = None,
        energy_window: tuple[float, float] | None = None,
        k_window: tuple[float, float] | None = None,
        method: str = 'peak',
        broadened: bool = True,
        **kwargs,
    ) -> dict[str, NDArray[np.float64]]:
        """MDC dispersion of a map (:func:`curves.mdc_dispersion` on the path distance)."""
        amap = self.map(kind, temperature_k)
        return curves.mdc_dispersion(
            amap['distance'], amap['w'], amap['intensity'] if broadened else amap['raw'],
            energy_window=energy_window, k_window=k_window, method=method, **kwargs,
        )  # fmt: skip

    def fermi_momentum(self, kind: str, temperature_k: float | None = None, **kwargs) -> float:
        """Distance along the path where the MDC dispersion crosses E_F."""
        d = self.dispersion(kind, temperature_k, **kwargs)
        return curves.fermi_momentum(d['energy_ev'], d['k'])

    def velocities(
        self,
        kind: str,
        temperature_k: float | None = None,
        window_ev: tuple[float, float] = (-0.03, -0.005),
        band: int | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """Bare and dressed Fermi velocities and the kink mass enhancement.

        Parameters
        ----------
        kind, temperature_k
            The dressed map.
        window_ev : (min, max), optional
            Energies of the linear fits (between the kink and E_F for a metal).
        band : int, optional
            Column of the bare band (default: the one crossing E_F).
        **kwargs
            Passed to :meth:`dispersion`.

        Returns
        -------
        dict
            ``v_bare``, ``v_dressed`` (eV·Å), ``lambda`` (:math:`|v_b/v| - 1`), ``k_fermi_bare``,
            ``k_fermi_dressed`` and the fit rms values.
        """
        dressed = self.dispersion(kind, temperature_k, **kwargs)
        fit_d = curves.linear_band(dressed['energy_ev'], dressed['k'], window_ev)
        bare = self.bare_dispersion(band)
        fit_b = curves.linear_band(bare['energy_ev'], bare['k'], window_ev)
        return {
            'v_bare': fit_b['velocity'],
            'v_dressed': fit_d['velocity'],
            'lambda': curves.mass_enhancement_from_velocities(fit_b['velocity'], fit_d['velocity']),
            'k_fermi_bare': fit_b['k_fermi'],
            'k_fermi_dressed': fit_d['k_fermi'],
            'rms_bare': fit_b['rms'],
            'rms_dressed': fit_d['rms'],
        }

    def self_energy_from_map(
        self, kind: str, temperature_k: float | None = None, band: int | None = None, **kwargs
    ) -> dict[str, NDArray[np.float64]]:
        """Self-energy extracted from the dressed map as an experimentalist would (MDC analysis)."""
        dressed = self.dispersion(kind, temperature_k, **kwargs)
        bare = self.bare_dispersion(band)
        return curves.self_energy_from_dispersion(
            dressed['energy_ev'], dressed['k'], dressed['fwhm_k'], (bare['k'], bare['energy_ev'])
        )

    # ----------------------------------------------------------------- self-energy tables
    def self_energy(self, kind: str, temperature_k: float, column: int | str = 0) -> dict:
        """``Sigma(omega)`` of ``sigma_<kind>_<T>K.dat``: ``w_ev``, ``sigma`` (complex, eV) and ``label``.

        ``column`` selects the state of an anisotropic table (index or label).
        """
        kind = kind.replace('-', '_')
        key = (
            kind,
            self._match_temperature(temperature_k, [t for k, t in self.sigma_files if k == kind]),
        )
        table = read_sigma_table(self.sigma_files[key])
        if isinstance(column, str):
            if column not in table['labels']:
                raise KeyError('no column %r (have %s)' % (column, table['labels']))
            column = table['labels'].index(column)
        return {
            'w_ev': table['w_ev'],
            'sigma': table['sigmas'][column],
            'label': table['labels'][column],
        }

    def mass_enhancement(
        self, kind: str, temperature_k: float, column: int | str = 0
    ) -> tuple[float, float]:
        """``(lambda, Z)`` from the slope of ``Re Sigma`` at E_F.

        See :func:`PAOFLOW.elphon.self_energy.mass_enhancement`.
        """
        from ...elphon.self_energy import mass_enhancement

        table = self.self_energy(kind, temperature_k, column)
        return mass_enhancement(table['w_ev'], table['sigma'])

    def kink_energy(
        self, kind: str, temperature_k: float, column: int | str = 0, side: str = 'occupied'
    ) -> float:
        """Energy of the extremum of ``Re Sigma`` of a self-energy table."""
        table = self.self_energy(kind, temperature_k, column)
        return curves.kink_energy(table['w_ev'], table['sigma'].real, side)

    # ----------------------------------------------------------------- Fan-Migdal linewidths
    def _need_fan_migdal(self) -> dict:
        if self.fan_migdal is None:
            raise FileNotFoundError('%s has no %s' % (self.output_dir, FAN_MIGDAL_FILE))
        return self.fan_migdal

    def fan_migdal_temperatures(self) -> list[float]:
        """Temperatures of the Fan-Migdal self-energy."""
        return [float(t) for t in self._need_fan_migdal()['temps_k']]

    def linewidths(self, temperature_k: float) -> dict:
        """On-shell widths ``-Im Sigma_nk(e_nk)`` and shifts ``Re Sigma_nk(e_nk)`` of the path states.

        Returns
        -------
        dict
            ``distance`` (nk,), ``energies_ev`` (nk, nband), ``widths_ev``, ``shifts_ev``,
            ``bands`` and ``temperature_k``.
        """
        fm = self._need_fan_migdal()
        temps = [float(t) for t in fm['temps_k']]
        it = temps.index(self._match_temperature(temperature_k, temps))
        w, e, sigma = fm['w_ev'], fm['energies_ev'], fm['sigma_ev']
        widths = np.zeros_like(e)
        shifts = np.zeros_like(e)
        for ik in range(e.shape[0]):
            for ib in range(e.shape[1]):
                iw = int(np.argmin(np.abs(w - e[ik, ib])))
                widths[ik, ib] = -sigma[it, ik, ib, iw].imag
                shifts[ik, ib] = sigma[it, ik, ib, iw].real
        return {
            'distance': path_distance(fm['k_cart'])
            if 'k_cart' in fm
            else np.arange(e.shape[0], dtype=float),
            'energies_ev': e,
            'widths_ev': widths,
            'shifts_ev': shifts,
            'bands': fm['bands'],
            'temperature_k': temps[it],
        }

    def linewidth_vs_temperature(self, band: int, ik: int) -> tuple[NDArray, NDArray]:
        """``(temps_k, widths_ev)`` of one state (``band``: column of ``bands``; ``ik``: path point)."""
        temps = np.asarray(self.fan_migdal_temperatures())
        widths = np.array([self.linewidths(t)['widths_ev'][ik, band] for t in temps])
        return temps, widths

    def state_self_energy(self, temperature_k: float, band: int, ik: int) -> dict:
        """``Sigma_nk(omega)`` of one state: ``w_ev``, ``sigma`` (complex, eV), ``energy_ev``."""
        fm = self._need_fan_migdal()
        temps = [float(t) for t in fm['temps_k']]
        it = temps.index(self._match_temperature(temperature_k, temps))
        return {
            'w_ev': fm['w_ev'],
            'sigma': fm['sigma_ev'][it, ik, band],
            'energy_ev': float(fm['energies_ev'][ik, band]),
        }

    # ----------------------------------------------------------------- report
    def describe(self) -> str:
        """Markdown summary of the run parameters (for the report)."""
        lines = ['| parameter | value |', '|---|---|']
        for key in (
            'system_dir',
            'name',
            'prefix',
            'coupling_source',
            'coarse_grid',
            'qgrid',
            'basis',
        ):
            if key in self.system:
                lines.append('| %s | %s |' % (key, self.system[key]))
        for key in (
            'photon_ev',
            'work_function_ev',
            'polarization',
            'resolution',
            'energy',
            'kz',
            'radial',
        ):
            if key in self.metadata:
                lines.append('| %s | %s |' % (key, self.metadata[key]))
        if self.bare_file is not None:
            distance, energies, bands = self.band_energies()
            lines.append('| path points | %d (%.4f 1/A) |' % (distance.size, distance[-1]))
            lines.append('| window bands | %s |' % ', '.join(str(int(b)) for b in bands))
        for kind in self.kinds:
            lines.append(
                '| maps %s (K) | %s |'
                % (kind, ', '.join('%g' % t for t in self.temperatures(kind)))
            )
        if self.fan_migdal is not None:
            fm = self.fan_migdal
            lines.append(
                '| Fan-Migdal temperatures (K) | %s |' % ', '.join('%g' % t for t in fm['temps_k'])
            )
            for key in ('nq_dense', 'eta_ev', 'adaptive_smearing', 'metal'):
                if key in fm:
                    lines.append(
                        '| %s | %s |' % (key, fm[key].item() if np.ndim(fm[key]) == 0 else fm[key])
                    )
        return '\n'.join(lines)
