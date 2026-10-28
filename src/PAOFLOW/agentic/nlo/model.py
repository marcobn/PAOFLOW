"""Adapter: the non-linear optics outputs of a PAOFLOW run as model observables.

Reads the two-column files of :meth:`PAOFLOW.nonlinear_optics` (``shift_<abc>.dat``,
``shiftm_<abc>.dat`` in µA/V², ``injectc_/injectm_<abc>.dat`` in A V⁻² s⁻¹,
``shgr_/shgi_/shga_<abc>.dat`` in pm/V; sheet units nm µA/V², nm A V⁻² s⁻¹ and pm²/V for 2D
runs) and gives the spectra, the full tensors at a photon energy, static values, peaks, the
point-group and Kleinman checks and the unit conversions needed to compare with experiments
(``d_ijk = chi/2``, 2D sheet vs effective 3D).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from . import symmetry as sym

KINDS = ('shift', 'shiftm', 'injectc', 'injectm', 'shgr', 'shgi', 'shga')
UNITS_3D = {
    'shift': 'uA/V^2', 'shiftm': 'uA/V^2', 'injectc': 'A V^-2 s^-1', 'injectm': 'A V^-2 s^-1',
    'shgr': 'pm/V', 'shgi': 'pm/V', 'shga': 'pm/V',
}  # fmt: skip
UNITS_2D = {
    'shift': 'nm uA/V^2', 'shiftm': 'nm uA/V^2', 'injectc': 'nm A V^-2 s^-1',
    'injectm': 'nm A V^-2 s^-1', 'shgr': 'pm^2/V', 'shgi': 'pm^2/V', 'shga': 'pm^2/V',
}  # fmt: skip
FILE = re.compile(
    r'^(?P<kind>shift|shiftm|injectc|injectm|shgr|shgi|shga)_(?P<abc>[xyz]{3})(?P<spin>_[01])?\.dat$'
)


class NonlinearModel:
    """Model second-order responses from a PAOFLOW output folder.

    Parameters
    ----------
    output_dir : str or Path
        Folder with the ``<kind>_<abc>.dat`` files.
    two_d : bool, optional
        Whether the run used ``twoD=True`` (sheet units).
    slab_height_nm : float, optional
        Height of the 2D cell used by the run (for the sheet / effective-3D conversions).
    system : dict, optional
        Free-form description (structure, grids, scissor, basis) reported by :meth:`describe`.

    Attributes
    ----------
    energy_ev : ndarray
        Photon energies.
    data : dict
        ``(kind, abc) -> ndarray`` of the spectra (spin-summed files only).
    kinds, components : list of str
    """

    def __init__(
        self,
        output_dir: str | Path,
        two_d: bool = False,
        slab_height_nm: float | None = None,
        system: dict | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.two_d = bool(two_d)
        self.slab_height_nm = slab_height_nm
        self.system = dict(system or {})
        self.data: dict[tuple[str, str], NDArray[np.float64]] = {}
        self.energy_ev = None
        for path in sorted(self.output_dir.glob('*.dat')):
            match = FILE.match(path.name)
            if not match or match['spin']:
                continue
            table = np.loadtxt(path, ndmin=2)
            if self.energy_ev is None:
                self.energy_ev = table[:, 0]
            elif table.shape[0] != self.energy_ev.size:
                raise ValueError('%s has a different energy grid' % path)
            self.data[(match['kind'], match['abc'])] = table[:, 1]
        if not self.data:
            raise FileNotFoundError('%s holds no non-linear optics output' % self.output_dir)
        self.kinds = sorted({k for k, _ in self.data}, key=KINDS.index)
        self.components = sorted({c for _, c in self.data})

    # ----------------------------------------------------------------- spectra
    def unit(self, kind: str) -> str:
        """Unit of a response kind for this run."""
        return (UNITS_2D if self.two_d else UNITS_3D)[kind]

    def spectrum(self, kind: str, abc: str) -> NDArray[np.float64]:
        """The spectrum of one component (``'xyz'``) of one kind (``'shga'``, ``'shift'``, ...)."""
        if (kind, abc) not in self.data:
            raise KeyError('no %s_%s.dat in %s' % (kind, abc, self.output_dir))
        return self.data[(kind, abc)]

    def shg(self, abc: str) -> NDArray[np.complex128]:
        """Complex :math:`\\chi^{abc}(\\omega)` from the real and imaginary files."""
        return self.spectrum('shgr', abc) + 1j * self.spectrum('shgi', abc)

    def value(self, kind: str, abc: str, energy_ev: float) -> float:
        """Linear interpolation of a spectrum at a photon energy."""
        return float(np.interp(energy_ev, self.energy_ev, self.spectrum(kind, abc)))

    def tensor(self, kind: str, energy_ev: float, fill=np.nan) -> NDArray:
        """``(3, 3, 3)`` tensor of a kind at a photon energy (``'shg'`` gives the complex one)."""
        if kind == 'shg':
            values = {c: complex(np.interp(energy_ev, self.energy_ev, self.shg(c))) for c in self.components
                      if ('shgr', c) in self.data}  # fmt: skip
        else:
            values = {
                c: self.value(kind, c, energy_ev) for c in self.components if (kind, c) in self.data
            }
        return sym.tensor_from_components(values, fill)

    def static(self, kind: str = 'shga', abc: str | None = None) -> dict[str, float]:
        """Values at the lowest photon energy of the grid (the static limit for SHG)."""
        labels = [abc] if abc else [c for c in self.components if (kind, c) in self.data]
        return {c: float(self.spectrum(kind, c)[0]) for c in labels}

    def peak(
        self, kind: str, abc: str, window: tuple[float, float] | None = None
    ) -> dict[str, float]:
        """Largest ``|value|`` of a spectrum inside ``window`` (eV): ``value``, ``energy_ev``."""
        y = self.spectrum(kind, abc)
        keep = (
            np.ones_like(y, dtype=bool)
            if window is None
            else (self.energy_ev >= window[0]) & (self.energy_ev <= window[1])
        )
        idx = np.flatnonzero(keep)
        i = idx[np.argmax(np.abs(y[idx]))]
        return {'value': float(y[i]), 'energy_ev': float(self.energy_ev[i])}

    def onset(self, kind: str, abc: str, fraction: float = 0.02) -> float:
        """First photon energy where ``|value|`` exceeds ``fraction`` of its maximum (NaN if never)."""
        y = np.abs(self.spectrum(kind, abc))
        above = np.flatnonzero(y > fraction * y.max())
        return float(self.energy_ev[above[0]]) if above.size else np.nan

    # ----------------------------------------------------------------- symmetry
    def symmetry_check(self, point_group: str, kind: str, energy_ev: float) -> dict:
        """Forbidden components, their leakage and the symmetry error of a tensor at one energy.

        Returns
        -------
        dict
            ``forbidden`` (labels), ``leakage`` (max |forbidden| / max |allowed|), ``relations``
            (:func:`symmetry.independent_relations`), ``error`` (:func:`symmetry.symmetry_error`
            on the components present).
        """
        ops = sym.point_group_operations(point_group)
        t = self.tensor(kind, energy_ev, fill=0.0)
        present = np.zeros((3, 3, 3), dtype=bool)
        for c in self.components:
            if (kind if kind != 'shg' else 'shgr', c) in self.data:
                present[tuple(sym.LABELS.index(ch) for ch in c)] = True
        forbidden = sym.forbidden_components(ops)
        allowed = [c for c in self.components if c not in forbidden]

        def amp(labels):
            return max(
                [abs(t[tuple(sym.LABELS.index(ch) for ch in c)]) for c in labels], default=0.0
            )

        return {
            'forbidden': forbidden,
            'forbidden_present': [c for c in forbidden if c in self.components],
            'leakage': amp([c for c in forbidden if c in self.components])
            / (amp(allowed) or np.inf),
            'relations': sym.independent_relations(ops),
            'error': sym.symmetry_error(np.where(present, t, 0.0), ops)
            if np.all(present)
            else None,
        }

    def kleinman_check(self, energy_ev: float | None = None) -> dict[str, float]:
        """Kleinman (full permutation) symmetry error of the SHG tensor (default: the static limit)."""
        e = self.energy_ev[0] if energy_ev is None else energy_ev
        return sym.kleinman_error(self.tensor('shg', e, fill=0.0))

    # ----------------------------------------------------------------- units
    @staticmethod
    def d_coefficient(chi_pm_v: float) -> float:
        """``d_ijk = chi_ijk / 2`` (the SHG coefficient quoted by experiments, pm/V)."""
        return 0.5 * chi_pm_v

    def effective_3d(self, kind: str, abc: str, layer_thickness_nm: float) -> NDArray[np.float64]:
        """Sheet response divided by a layer thickness: the effective 3D value (µA/V² or pm/V)."""
        if not self.two_d:
            raise ValueError('effective_3d applies to 2D (sheet) runs')
        scale = (
            layer_thickness_nm
            if kind in ('shift', 'shiftm', 'injectc', 'injectm')
            else 1.0e3 * layer_thickness_nm
        )
        return self.spectrum(kind, abc) / scale

    # ----------------------------------------------------------------- report
    def describe(self) -> str:
        """Markdown summary (for the report)."""
        lines = ['| parameter | value |', '|---|---|']
        for key, value in self.system.items():
            lines.append('| %s | %s |' % (key, value))
        lines.append('| output | %s |' % self.output_dir)
        lines.append(
            '| photon energies (eV) | %d points, %.3g-%.3g |'
            % (self.energy_ev.size, self.energy_ev[0], self.energy_ev[-1])
        )
        lines.append('| kinds | %s |' % ', '.join(self.kinds))
        lines.append('| components | %s |' % ', '.join(self.components))
        lines.append('| units | %s |' % ('2D sheet' if self.two_d else '3D'))
        if ('shga', self.components[0]) in self.data:
            lines.append(
                '| \\|chi\\| static (%s) | %s |'
                % (
                    self.unit('shga'),
                    ', '.join('%s %.4g' % (c, v) for c, v in self.static('shga').items()),
                )
            )
        if ('shift', self.components[0]) in self.data:
            lines.append(
                '| shift peaks (%s) | %s |'
                % (
                    self.unit('shift'),
                    ', '.join(
                        '%s %.4g at %.2f eV' % (c, p['value'], p['energy_ev'])
                        for c, p in (
                            (c, self.peak('shift', c))
                            for c in self.components
                            if ('shift', c) in self.data
                        )
                    ),
                )
            )
        return '\n'.join(lines)
