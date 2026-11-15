"""Scattering channels: relaxation rates for transport and optical response.

A *scattering channel* returns a relaxation **rate** :math:`1/\\tau` (1/s) as a
function of temperature and, where meaningful, of the carrier energy (DC
transport) or of the photon frequency (optical response).  Channels combine by
Matthiessen's rule (:func:`matthiessen`), and the same objects feed

* the Boltzmann transport tensors of :meth:`PAOFLOW.PAOFLOW.transport`
  (``scattering_channels=[...]``, through
  :func:`~PAOFLOW.boltzmann.do_Boltz_tensors.get_tau`), and
* the (extended) Drude dielectric function of metals
  (:mod:`PAOFLOW.response.drude`).

Channels:

* :class:`ConstantRate`, :class:`ResidualResistivity` -- energy- and
  temperature-independent rates (impurities, defects);
* :class:`FermiLiquidRate` -- electron-electron scattering,
  :math:`\\hbar/\\tau = b[(E-\\mu)^2 + (\\pi k_BT)^2]` (DC) or
  :math:`b[(\\hbar\\omega)^2 + (2\\pi k_BT)^2]` (optical);
* :class:`PhenomenologicalChannel` -- the semiconductor models of
  :mod:`~PAOFLOW.boltzmann.do_tau_models` (acoustic and optical deformation
  potential, polar optical, piezoelectric, ionized impurity) or any
  ``f(kT_eV, |E|_eV, params) -> tau`` function;
* :class:`TransportSpectralChannel` -- electron-phonon scattering from the
  transport Eliashberg function :math:`\\alpha^2F_{tr}(\\omega)` (Allen), with the
  frequency-dependent rate and mass enhancement of the extended Drude model;
* :class:`EnergyResolvedRate` -- a tabulated :math:`1/\\tau(E, T)`, e.g. the
  ab initio SERTA/MRTA electron-phonon rates of semiconductors from
  :mod:`PAOFLOW.elphon.carrier_scattering`.

Units at the interface: temperatures in K, energies and frequencies in eV
(:math:`\\hbar\\omega`), rates in 1/s.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

HBAR_EV_S = 6.582119569e-16  # eV s
KB_EV_K = 8.617333262e-5  # eV / K
EPSILON0_SI = 8.8541878128e-12  # F / m


def _broadcast(value: float | NDArray, energy_ev, omega_ev) -> NDArray[np.float64]:
    """``value`` broadcast to the shape of ``omega_ev`` or ``energy_ev`` (scalar otherwise)."""
    target = omega_ev if omega_ev is not None else energy_ev
    if target is None:
        return np.asarray(value, dtype=float)
    return np.broadcast_to(np.asarray(value, dtype=float), np.shape(target)).astype(float)


class ScatteringChannel:
    """Interface of a scattering channel: a relaxation rate in 1/s.

    Subclasses implement :meth:`rate`.  ``energy_ev`` (carrier energy relative to
    the chemical potential) is used by energy-resolved (DC) channels and
    ``omega_ev`` (photon energy) by frequency-resolved (optical) channels; a
    channel that depends on neither broadcasts its value to the shape of the one
    given.
    """

    name = 'channel'

    def rate(
        self,
        temperature_k: float,
        energy_ev: ArrayLike | None = None,
        omega_ev: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Relaxation rate :math:`1/\\tau` (1/s).

        Parameters
        ----------
        temperature_k : float
            Temperature (K).
        energy_ev : array_like, optional
            Carrier energies relative to the chemical potential (eV).
        omega_ev : array_like, optional
            Photon energies (eV), for the optical rate.

        Returns
        -------
        ndarray
            Rates (1/s), shaped like ``omega_ev`` or ``energy_ev``.
        """
        raise NotImplementedError

    def mass_enhancement(self, omega_ev: ArrayLike, temperature_k: float) -> NDArray[np.float64]:
        """Optical mass enhancement :math:`\\lambda(\\omega, T)` (0 unless overridden)."""
        return np.zeros(np.shape(omega_ev))

    def bind(self, data_controller) -> None:
        """Adapt the channel to the band structure of a Boltzmann run (no-op by default).

        Called by :func:`~PAOFLOW.boltzmann.do_Boltz_tensors.get_tau` on every
        rank before the rates are evaluated.
        """

    def __repr__(self) -> str:
        return '%s(%s)' % (type(self).__name__, self.name)


class ConstantRate(ScatteringChannel):
    """Energy-, frequency- and temperature-independent rate.

    Parameters
    ----------
    rate_per_s : float
        Relaxation rate (1/s).
    name : str, optional
        Label.
    """

    def __init__(self, rate_per_s: float, name: str = 'constant') -> None:
        self.rate_per_s = float(rate_per_s)
        self.name = name

    def rate(self, temperature_k, energy_ev=None, omega_ev=None):
        return _broadcast(self.rate_per_s, energy_ev, omega_ev)


class ResidualResistivity(ConstantRate):
    """Constant rate that gives a residual resistivity :math:`\\rho_0` with a plasma frequency.

    Parameters
    ----------
    rho0_ohm_m : float
        Residual resistivity (Ohm m).
    plasma_ev : float
        Plasma frequency :math:`\\hbar\\omega_p` (eV).

    Notes
    -----
    From the Drude resistivity :math:`\\rho = 1/(\\varepsilon_0\\omega_p^2\\tau)`,
    :math:`1/\\tau_0 = \\varepsilon_0\\,\\omega_p^2\\,\\rho_0`.
    """

    def __init__(self, rho0_ohm_m: float, plasma_ev: float) -> None:
        omega_p = float(plasma_ev) / HBAR_EV_S
        super().__init__(EPSILON0_SI * omega_p**2 * float(rho0_ohm_m), name='residual')
        self.rho0_ohm_m = float(rho0_ohm_m)


class FermiLiquidRate(ScatteringChannel):
    """Electron-electron (Fermi-liquid) scattering.

    Parameters
    ----------
    coefficient_per_ev : float
        :math:`b` (1/eV) in :math:`\\hbar/\\tau = b\\,[\\ldots]`.

    Notes
    -----
    .. math::

        \\frac{\\hbar}{\\tau_{DC}} = b\\left[(E-\\mu)^2 + (\\pi k_BT)^2\\right],
        \\qquad
        \\frac{\\hbar}{\\tau(\\omega)} = b\\left[(\\hbar\\omega)^2 + (2\\pi k_BT)^2\\right]

    (Gurzhi); the optical form applies when ``omega_ev`` is given.
    """

    def __init__(self, coefficient_per_ev: float, name: str = 'electron-electron') -> None:
        self.coefficient_per_ev = float(coefficient_per_ev)
        self.name = name

    def rate(self, temperature_k, energy_ev=None, omega_ev=None):
        kt = KB_EV_K * float(temperature_k)
        if omega_ev is not None:
            hbar_rate = self.coefficient_per_ev * (
                np.asarray(omega_ev) ** 2 + (2.0 * np.pi * kt) ** 2
            )
        else:
            energy = 0.0 if energy_ev is None else np.asarray(energy_ev)
            hbar_rate = self.coefficient_per_ev * (energy**2 + (np.pi * kt) ** 2)
        return np.asarray(hbar_rate, dtype=float) / HBAR_EV_S


class PhenomenologicalChannel(ScatteringChannel):
    """Adapter for the relaxation-time models of :mod:`~PAOFLOW.boltzmann.do_tau_models`.

    Parameters
    ----------
    model : str, callable or TauModel
        A built-in label (``'acoustic'``, ``'optical'``, ``'polar_optical'``,
        ``'polar_acoustic'``, ``'impurity'``), a function
        ``f(kT_eV, |E|_eV, params) -> tau (s)`` or a
        :class:`~PAOFLOW.boltzmann.TauModel.TauModel`.
    params : dict, optional
        Model parameters (see :mod:`~PAOFLOW.boltzmann.do_tau_models`).
    weight : float, optional
        Weight :math:`w` of the channel, :math:`1/\\tau = w/\\tau_{model}`.

    Notes
    -----
    The models depend on :math:`|E|` measured from the energy zero of the
    calculation and on :math:`k_BT`; they need ``energy_ev`` (they have no
    optical form).
    """

    def __init__(self, model, params: dict | None = None, weight: float = 1.0) -> None:
        from .do_tau_models import builtin_tau_model
        from .TauModel import TauModel

        if isinstance(model, str):
            built = builtin_tau_model(model, params, weight)
            if built is None:
                raise ValueError('unknown scattering model %r' % model)
            self.function, self.params, self.name = built.function, params, model
        elif isinstance(model, TauModel):
            self.function, self.params = model.function, model.params
            self.name = getattr(model.function, '__name__', 'TauModel')
            weight = model.weight if weight == 1.0 else weight
        elif callable(model):
            self.function, self.params = model, params
            self.name = getattr(model, '__name__', 'model')
        else:
            raise TypeError('model must be a label, a function or a TauModel')
        self.weight = float(weight)

    def rate(self, temperature_k, energy_ev=None, omega_ev=None):
        if energy_ev is None:
            raise ValueError('%s is an energy-resolved (DC) channel: give energy_ev' % self.name)
        tau = self.function(
            KB_EV_K * float(temperature_k), np.abs(np.asarray(energy_ev)), self.params
        )
        return self.weight / np.asarray(tau, dtype=float)


def _y_coth(y: NDArray[np.float64], kt: float) -> NDArray[np.float64]:
    """:math:`y\\coth(y/2k_BT)`, finite at :math:`y = 0` (:math:`2k_BT`) and :math:`T = 0` (:math:`|y|`)."""
    y = np.asarray(y, dtype=float)
    if kt <= 0.0:
        return np.abs(y)
    x = y / (2.0 * kt)
    small = np.abs(x) < 1.0e-8
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        value = y / np.tanh(np.where(small, 1.0, x))
    return np.where(small, 2.0 * kt, value)


class TransportSpectralChannel(ScatteringChannel):
    """Electron-phonon scattering from the transport Eliashberg function.

    Parameters
    ----------
    omega_ev : array_like, shape ``(nw,)``
        Phonon energies (eV), increasing.
    a2f_tr : array_like, shape ``(nw,)``
        :math:`\\alpha^2F_{tr}(\\omega)` (dimensionless), e.g. from
        :func:`~PAOFLOW.elphon.do_pao_eph_dense_q.eliashberg_dense_q` with
        ``transport=True``.
    name : str, optional
        Label.

    Notes
    -----
    DC rate (Allen; Grimvall), with :math:`x = \\hbar\\omega/2k_BT`:

    .. math::

        \\frac{1}{\\tau_{tr}(T)} = \\frac{4\\pi k_BT}{\\hbar}\\int\\frac{d\\omega}{\\omega}\\,
            \\alpha^2F_{tr}(\\omega)\\left[\\frac{x}{\\sinh x}\\right]^2
        \\;\\xrightarrow{\\;k_BT \\gg \\hbar\\omega\\;}\\; \\frac{2\\pi\\lambda_{tr}k_BT}{\\hbar}.

    Optical rate (Allen 1971; Shulga, Dolgov and Maksimov 1991):

    .. math::

        \\frac{1}{\\tau(\\omega, T)} = \\frac{\\pi}{\\hbar\\omega}\\int d\\Omega\\,
            \\alpha^2F_{tr}(\\Omega)\\Big[2\\omega\\coth\\frac{\\Omega}{2k_BT}
            - (\\omega+\\Omega)\\coth\\frac{\\omega+\\Omega}{2k_BT}
            + (\\omega-\\Omega)\\coth\\frac{\\omega-\\Omega}{2k_BT}\\Big],

    whose :math:`\\omega \\to 0` limit is the DC rate and which at :math:`T = 0`
    is :math:`(2\\pi/\\hbar\\omega)\\int_0^\\omega(\\omega-\\Omega)\\alpha^2F_{tr}`.
    The mass enhancement :math:`\\lambda(\\omega, T)` follows by causality
    (:meth:`mass_enhancement`).
    """

    def __init__(self, omega_ev: ArrayLike, a2f_tr: ArrayLike, name: str = 'e-ph') -> None:
        omega = np.asarray(omega_ev, dtype=float)
        a2f = np.asarray(a2f_tr, dtype=float)
        keep = omega > 0.0
        self.omega_ev, self.a2f_tr = omega[keep], a2f[keep]
        self.name = name

    @classmethod
    def from_npz(cls, path: str, key: str = 'a2F_tr') -> 'TransportSpectralChannel':
        """Channel from an ``eliashberg.npz`` (keys ``omega`` in eV and ``a2F_tr``)."""
        with np.load(path) as data:
            return cls(data['omega'], data[key], name='e-ph (%s)' % key)

    @property
    def lambda_tr(self) -> float:
        """:math:`\\lambda_{tr} = 2\\int\\alpha^2F_{tr}(\\omega)/\\omega\\,d\\omega`."""
        return float(2.0 * np.trapezoid(self.a2f_tr / self.omega_ev, self.omega_ev))

    def dc_rate(self, temperature_k: float) -> float:
        """DC transport rate :math:`1/\\tau_{tr}(T)` (1/s)."""
        kt = KB_EV_K * float(temperature_k)
        if kt <= 0.0:
            return 0.0
        x = self.omega_ev / (2.0 * kt)
        with np.errstate(over='ignore'):
            ratio = np.where(x < 350.0, x / np.sinh(np.minimum(x, 350.0)), 0.0)
        integral = np.trapezoid(self.a2f_tr / self.omega_ev * ratio**2, self.omega_ev)
        return float(4.0 * np.pi * kt * integral / HBAR_EV_S)

    def optical_rate_ev(self, omega_ev: ArrayLike, temperature_k: float) -> NDArray[np.float64]:
        """:math:`\\hbar/\\tau(\\omega, T)` (eV) of the extended Drude model."""
        omega = np.atleast_1d(np.asarray(omega_ev, dtype=float))
        kt = KB_EV_K * float(temperature_k)
        big = self.omega_ev[None, :]
        w = omega[:, None]
        bracket = 2.0 * w * (_y_coth(big, kt) / big)  # 2 w coth(Omega / 2kT)
        bracket = bracket - _y_coth(w + big, kt) + _y_coth(w - big, kt)
        integral = np.trapezoid(self.a2f_tr[None, :] * bracket, self.omega_ev, axis=1)
        out = np.empty_like(omega)
        tiny = omega < 1.0e-9
        out[~tiny] = np.pi * integral[~tiny] / omega[~tiny]
        out[tiny] = self.dc_rate(temperature_k) * HBAR_EV_S
        return out.reshape(np.shape(omega_ev)) if np.ndim(omega_ev) else out[0]

    def rate(self, temperature_k, energy_ev=None, omega_ev=None):
        if omega_ev is not None:
            return np.asarray(self.optical_rate_ev(omega_ev, temperature_k)) / HBAR_EV_S
        return _broadcast(self.dc_rate(temperature_k), energy_ev, None)

    def mass_enhancement(
        self,
        omega_ev: ArrayLike,
        temperature_k: float,
        omega_max_ev: float | None = None,
        max_points: int = 20000,
    ) -> NDArray[np.float64]:
        """Optical mass enhancement :math:`\\lambda(\\omega, T)` of the extended Drude model.

        Parameters
        ----------
        omega_ev : array_like
            Photon energies (eV).
        temperature_k : float
            Temperature (K).
        omega_max_ev : float, optional
            Upper end of the Kramers-Kronig integral (default: 40 times the
            largest phonon energy, at least twice the largest ``omega_ev``).
        max_points : int, optional
            Largest number of samples of the integration grid.

        Returns
        -------
        ndarray
            :math:`\\lambda(\\omega, T)`, shaped like ``omega_ev``.

        Notes
        -----
        The memory function :math:`M = 1/\\tau(\\omega) - i\\omega\\lambda(\\omega)` is
        causal, so with :math:`r(\\omega) = \\hbar/\\tau(\\omega)`

        .. math::

            \\lambda(\\omega) = \\frac{2}{\\pi}\\int_0^\\infty
                \\frac{r(\\omega') - r(\\omega)}{\\omega'^2 - \\omega^2}\\,d\\omega',

        (a constant has zero principal value, which removes the singularity);
        beyond :math:`W` the rate is taken constant, which adds
        :math:`(2/\\pi)\\,[r(W) - r(\\omega)]\\,\\mathrm{artanh}(\\omega/W)/\\omega`.
        At :math:`T = 0`, :math:`\\lambda(0) = \\lambda_{tr}`.
        """
        omega = np.atleast_1d(np.asarray(omega_ev, dtype=float))
        top = omega_max_ev or max(40.0 * float(self.omega_ev.max()), 2.0 * float(omega.max()))
        spacing = float(np.min(np.diff(self.omega_ev))) if self.omega_ev.size > 1 else top
        step = max(min(spacing, top) / 2.0, top / max_points)  # at most ~max_points samples
        grid = np.arange(0.0, top + 0.5 * step, step)
        rates = self.optical_rate_ev(grid, temperature_k)
        r_w = rates[-1]
        rate_at = np.interp(omega, grid, rates)
        lam = np.empty_like(omega)
        for i, (w, r0) in enumerate(zip(omega, rate_at)):
            with np.errstate(divide='ignore', invalid='ignore'):
                integrand = (rates - r0) / (grid**2 - w**2)
            singular = ~np.isfinite(integrand)
            if np.any(singular):  # removable: r'(w) / 2w, from the neighbours
                integrand[singular] = np.interp(
                    grid[singular], grid[~singular], integrand[~singular]
                )
            tail = (r_w - r0) * (np.arctanh(w / top) / w if w > 0.0 else 1.0 / top)
            lam[i] = 2.0 / np.pi * (np.trapezoid(integrand, grid) + tail)
        return lam.reshape(np.shape(omega_ev)) if np.ndim(omega_ev) else lam[0]


class EnergyResolvedRate(ScatteringChannel):
    """Tabulated rate :math:`1/\\tau(E, T)`, e.g. ab initio electron-phonon scattering.

    Parameters
    ----------
    energy_ev : array_like, shape ``(ne,)``
        Carrier energies (eV), increasing, on the scale of PAOFLOW's band
        energies ``E_k`` (zero at the DFT Fermi level of the PAO Hamiltonian).
    temps_k : array_like, shape ``(nt,)``
        Temperatures (K), increasing.
    rate_per_s : array_like, shape ``(nt, ne)``
        Rates (1/s).
    name : str, optional
        Label.
    energy_shift_ev : float, optional
        Added to ``energy_ev``, to align a table computed with a different
        energy zero.
    band_edges : tuple of float, optional
        ``(vbm, cbm)`` (eV) of the band structure of the table, for a gapped
        material.  The rates of electrons are then a function of the energy above
        the conduction-band minimum and those of holes of the energy below the
        valence-band maximum, aligned to the band edges of the band structure
        that uses them (:meth:`align_to`, or :meth:`bind` in
        :meth:`PAOFLOW.PAOFLOW.transport`).

    Notes
    -----
    The rate is interpolated linearly in energy (held constant beyond the
    tabulated range) and in temperature; a temperature outside the table
    raises.  :func:`PAOFLOW.elphon.carrier_scattering.energy_resolved_rates`
    builds the table as the transport-weighted average of the state-resolved
    SERTA/MRTA rates,

    .. math::

        \\tau_{eff}(E) = \\frac{\\sum_{n\\mathbf k} v^2_{n\\mathbf k}\\tau_{n\\mathbf k}
            \\delta(E - \\varepsilon_{n\\mathbf k})}
            {\\sum_{n\\mathbf k} v^2_{n\\mathbf k}\\delta(E - \\varepsilon_{n\\mathbf k})},

    which reproduces the trace of the k-resolved RTA conductivity in an
    energy-only relaxation-time model (:meth:`PAOFLOW.PAOFLOW.transport`).
    Aligning the band edges matters because the Boltzmann run usually has a
    different band structure (PAOFLOW's zero-padding interpolation, or another
    QE run) from the Wigner-Seitz interpolation of the electron-phonon run:
    a few tens of meV in the gap change the carrier densities, not the rates.
    """

    def __init__(
        self,
        energy_ev: ArrayLike,
        temps_k: ArrayLike,
        rate_per_s: ArrayLike,
        name: str = 'e-ph (energy-resolved)',
        energy_shift_ev: float = 0.0,
        band_edges: tuple[float, float] | None = None,
    ) -> None:
        shift = float(energy_shift_ev)
        self.energy_ev = np.asarray(energy_ev, dtype=float) + shift
        self.temps_k = np.atleast_1d(np.asarray(temps_k, dtype=float))
        self.rate_per_s = np.asarray(rate_per_s, dtype=float).reshape(
            self.temps_k.size, self.energy_ev.size
        )
        self.name = name
        self.band_edges = None
        if band_edges is not None:
            vbm, cbm = (float(e) + shift for e in band_edges)
            if cbm <= vbm:
                raise ValueError('band_edges need a gap (vbm < cbm).')
            self.band_edges = (vbm, cbm)
        self.target_edges = None

    @classmethod
    def from_npz(
        cls, path: str, kind: str = 'mrta', energy_shift_ev: float = 0.0, align: bool = True
    ) -> 'EnergyResolvedRate':
        """Channel from a ``scattering_rates.npz`` (``energy_ev``, ``temps_k``, ``rate_<kind>``).

        With ``align`` (default) and a gapped table (``vbm_ev``, ``cbm_ev``,
        ``metal`` in the file), the energies are aligned to the band edges of the
        band structure that uses the channel.
        """
        with np.load(path) as data:
            edges = None
            if align and 'vbm_ev' in data and not bool(data.get('metal', False)):
                edges = (float(data['vbm_ev']), float(data['cbm_ev']))
            return cls(
                data['energy_ev'], data['temps_k'], data['rate_%s' % kind],
                name='e-ph (%s)' % kind.upper(), energy_shift_ev=energy_shift_ev,
                band_edges=edges,
            )  # fmt: skip

    def align_to(self, vbm_ev: float, cbm_ev: float) -> None:
        """Use the band edges ``(vbm_ev, cbm_ev)`` of the target band structure."""
        if self.band_edges is None:
            raise ValueError('%s has no band edges to align.' % self.name)
        self.target_edges = (float(vbm_ev), float(cbm_ev))

    def bind(self, data_controller) -> None:
        """Align to the band edges of a PAOFLOW run (``E_k``, ``nelec``; MPI-reduced)."""
        if self.band_edges is None:
            return
        from mpi4py import MPI

        arrays, attributes = data_controller.data_dicts()
        nelec = int(round(attributes['nelec']))
        nocc = nelec if attributes.get('dftSO', False) else nelec // 2
        energies = np.asarray(arrays['E_k'])
        vbm = np.array(energies[:, nocc - 1].max() if energies.size else -np.inf)
        cbm = np.array(energies[:, nocc].min() if energies.size else np.inf)
        comm = MPI.COMM_WORLD
        if comm.Get_size() > 1:
            vbm = comm.allreduce(float(vbm), op=MPI.MAX)
            cbm = comm.allreduce(float(cbm), op=MPI.MIN)
        if cbm <= vbm:
            raise ValueError(
                '%s: the band structure has no gap above band %d; use align=False.'
                % (self.name, nocc)
            )
        self.align_to(float(vbm), float(cbm))

    def rate(self, temperature_k, energy_ev=None, omega_ev=None):
        if energy_ev is None:
            raise ValueError('%s is a DC (energy-resolved) channel: give energy_ev.' % self.name)
        temp = float(temperature_k)
        low, high = self.temps_k[0], self.temps_k[-1]
        tol = 1.0e-6 * max(1.0, high)
        if temp < low - tol or temp > high + tol:
            raise ValueError(
                '%s: T = %g K outside the tabulated range [%g, %g] K.'
                % (self.name, temp, low, high)
            )
        energy = np.asarray(energy_ev, dtype=float)
        if self.band_edges is not None and self.target_edges is not None:
            (vbm, cbm), (vbm_t, cbm_t) = self.band_edges, self.target_edges
            conduction = energy > 0.5 * (vbm_t + cbm_t)
            energy = np.where(conduction, energy - cbm_t + cbm, energy - vbm_t + vbm)
        rows = [np.interp(energy, self.energy_ev, r) for r in self.rate_per_s]
        if self.temps_k.size == 1:
            return rows[0]
        j = int(np.clip(np.searchsorted(self.temps_k, temp) - 1, 0, self.temps_k.size - 2))
        t0, t1 = self.temps_k[j], self.temps_k[j + 1]
        x = min(max((temp - t0) / (t1 - t0), 0.0), 1.0)
        return (1.0 - x) * rows[j] + x * rows[j + 1]


def matthiessen(
    channels: Sequence[ScatteringChannel],
    temperature_k: float,
    energy_ev: ArrayLike | None = None,
    omega_ev: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Total relaxation rate by Matthiessen's rule.

    Parameters
    ----------
    channels : sequence of ScatteringChannel
        Channels to combine.
    temperature_k : float
        Temperature (K).
    energy_ev, omega_ev : array_like, optional
        Carrier energies (DC) or photon energies (optical), eV.

    Returns
    -------
    ndarray
        :math:`1/\\tau = \\sum_c 1/\\tau_c` (1/s).
    """
    total = 0.0
    for channel in channels:
        total = total + channel.rate(temperature_k, energy_ev=energy_ev, omega_ev=omega_ev)
    return np.asarray(total, dtype=float)


def total_mass_enhancement(
    channels: Sequence[ScatteringChannel], omega_ev: ArrayLike, temperature_k: float
) -> NDArray[np.float64]:
    """Sum of the channels' optical mass enhancements :math:`\\lambda(\\omega, T)`."""
    total = np.zeros(np.shape(omega_ev))
    for channel in channels:
        total = total + channel.mass_enhancement(omega_ev, temperature_k)
    return total
