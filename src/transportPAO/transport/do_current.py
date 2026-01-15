import numpy as np
from pathlib import Path
from typing import Tuple
from mpi4py import MPI

from transportPAO.utils.locate import locate
from transportPAO.utils.memusage import MemoryTracker
from transportPAO.workspace.prepare_data import prepare_current

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class CurrentCalculator:
    def __init__(self, data: dict):
        self.data = data
        self.vgrid = self.build_bias_grid(data["Vmin"], data["Vmax"], data["nV"])
        self.egrid, self.transm = self.read_transmittance(data["filein"])
        self.currents = None

    def read_transmittance(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read the transmittance data file.

        Parameters
        ----------
        `file_path` : str
            Path to the file containing energy and transmittance values.

        Returns
        -------
        `egrid` : ndarray
            Array of energy values.
        `transm` : ndarray
            Corresponding transmittance values.
        """
        data = np.loadtxt(file_path)
        return data[:, 0], data[:, 1]

    def build_bias_grid(self, vmin: float, vmax: float, nv: int) -> np.ndarray:
        """
        Construct a linear bias voltage grid.

        Parameters
        ----------
        `vmin` : float
            Minimum bias value.
        `vmax` : float
            Maximum bias value.
        `nv` : int
            Number of bias points.

        Returns
        -------
        `vgrid` : ndarray
            Bias voltage values.
        """
        return np.linspace(vmin, vmax, nv)

    def fermi_dirac(self, E: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """
        Compute the Fermi-Dirac distribution.

        Parameters
        ----------
        `E` : ndarray
            Energy values.
        `mu` : float
            Chemical potential.
        `sigma` : float
            Broadening factor (eV).

        Returns
        -------
        `f` : ndarray
            Fermi-Dirac values.
        """
        return 1.0 / (np.exp(-(E - mu) / sigma) + 1.0)

    def interpolate_transmittance(
        self,
        egrid: np.ndarray,
        transm: np.ndarray,
        i_start: int,
        i_end: int,
        ndiv: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate transmittance linearly onto a finer energy grid.

        Parameters
        ----------
        `egrid` : ndarray
            Original energy grid.
        `transm` : ndarray
            Original transmittance values.
        `i_start` : int
            Start index of the energy window.
        `i_end` : int
            End index of the energy window.
        `ndiv` : int
            Number of subdivisions per interval.

        Returns
        -------
        `egrid_new` : ndarray
            Finer energy grid.
        `transm_new` : ndarray
            Interpolated transmittance.
        """
        if ndiv == 1:
            return egrid[i_start : i_end + 1], transm[i_start : i_end + 1]

        ndim = i_end - i_start + 1
        ndim_new = (ndim - 1) * ndiv + 1

        egrid_new = np.linspace(egrid[i_start], egrid[i_end], ndim_new)
        transm_new = np.interp(
            egrid_new, egrid[i_start : i_end + 1], transm[i_start : i_end + 1]
        )
        return egrid_new, transm_new

    def compute_current_vs_bias(
        self,
        egrid: np.ndarray,
        transm: np.ndarray,
        vgrid: np.ndarray,
        mu_L: float,
        mu_R: float,
        sigma: float,
    ) -> np.ndarray:
        r"""
        Compute current I(V) as a function of bias using Landauer formula.

        Parameters
        ----------
        `egrid` : ndarray
            Energy grid.
        `transm` : ndarray
            Transmittance on the energy grid.
        `vgrid` : ndarray
            Bias voltages.
        `mu_L` : float
            Left chemical potential coefficient.
        `mu_R` : float
            Right chemical potential coefficient.
        `sigma` : float
            Broadening parameter (smearing width, in eV).

        Returns
        -------
        `currents` : ndarray
            Current at each bias voltage.

        Notes
        -----
        Implements:
        .. math::
            I(V) = \int dE \; T(E) [f(E - \mu_L) - f(E - \mu_R)]

        The integration mesh is refined to resolve the energy window around the chemical potentials.
        """
        ne = len(egrid)
        de_old = (egrid[-1] - egrid[0]) / (ne - 1)
        currents = np.zeros_like(vgrid)

        for iv, V in enumerate(vgrid):
            muL_v = mu_L * V
            muR_v = mu_R * V

            try:
                i_start = locate(egrid, min(muL_v, muR_v) - sigma - 3 * de_old)
                i_end = locate(egrid, max(muL_v, muR_v) + sigma + 3 * de_old)
            except ValueError:
                continue

            ndim = i_end - i_start + 1
            if ndim < 2:
                continue

            if ndim % 2 == 0:
                i_end -= 1
                ndim -= 1

            de = (egrid[i_end] - egrid[i_start]) / (ndim - 1)
            ndiv = max(1, int(round(de / (2 * sigma))))

            egrid_new, transm_new = self.interpolate_transmittance(
                egrid, transm, i_start, i_end, ndiv
            )
            fL = self.fermi_dirac(egrid_new, muL_v, sigma)
            fR = self.fermi_dirac(egrid_new, muR_v, sigma)

            de_new = egrid_new[1] - egrid_new[0]

            integral = 0.0
            for i in range(len(egrid_new) - 1):
                fL_i = fL[i]
                fL_ip1 = fL[i + 1]
                fR_i = fR[i]
                fR_ip1 = fR[i + 1]

                fdiff_i = fL_i - fR_i
                fdiff_ip1 = fL_ip1 - fR_ip1

                t_i = transm_new[i]
                t_ip1 = transm_new[i + 1]

                integral += (
                    fdiff_i * t_i * de_new / 3.0
                    + fdiff_ip1 * t_ip1 * de_new / 3.0
                    + fdiff_i * t_ip1 * de_new / 6.0
                    + fdiff_ip1 * t_i * de_new / 6.0
                )

            currents[iv] = integral

        return currents

    def run(self) -> None:
        self.currents = self.compute_current_vs_bias(
            self.egrid,
            self.transm,
            self.vgrid,
            self.data["mu_L"],
            self.data["mu_R"],
            self.data["sigma"],
        )

    def write_output(self) -> None:
        outpath = Path(self.data["fileout"])
        outpath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(outpath, np.column_stack([self.vgrid, self.currents]))
        print(f"Saved current vs bias to {outpath}")


class CurrentRunner:
    @classmethod
    def from_yaml(cls, yaml_file: str):
        data = prepare_current(yaml_file)
        memory_tracker = MemoryTracker()

        calculator = CurrentCalculator(data)

        return cls(calculator, memory_tracker)

    def __init__(self, calculator: CurrentCalculator, memory_tracker: MemoryTracker):
        self.calculator = calculator
        self.memory_tracker = memory_tracker
