from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from PAOFLOW.transport.partition.directions import direction_index


class SurfaceKPath:
    """Container for a surface-projected transverse k-path.

    Attributes
    ----------
    vkpt_par3D : NDArray[np.float64]
        k-points in fractional (crystal) coordinates, shape ``(nkpts, 3)``, with
        the transport-direction component set to zero so that Bloch phases pick
        up only transverse ``R`` contributions.
    wk_par : NDArray[np.float64]
        k-point weights, shape ``(nkpts,)``. All ones: the surface spectral
        function is reported raw, not BZ-averaged.
    kdist : NDArray[np.float64]
        Cumulative in-plane distance along the path, shape ``(nkpts,)``. Used as
        the abscissa when plotting the spectral map.
    ticks : NDArray[np.int64]
        Indices of the high-symmetry points along the path, shape ``(nlabels,)``.
    labels : list[str]
        High-symmetry point labels aligned with ``ticks``.
    """

    __slots__ = ('vkpt_par3D', 'wk_par', 'kdist', 'ticks', 'labels')

    def __init__(
        self,
        vkpt_par3D: NDArray[np.float64],
        wk_par: NDArray[np.float64],
        kdist: NDArray[np.float64],
        ticks: NDArray[np.int64],
        labels: list[str],
    ) -> None:
        self.vkpt_par3D = vkpt_par3D
        self.wk_par = wk_par
        self.kdist = kdist
        self.ticks = ticks
        self.labels = labels

    @property
    def nkpts(self) -> int:
        """Number of k-points on the path."""
        return int(self.vkpt_par3D.shape[0])


def _parse_path_labels(path_file: str) -> tuple[NDArray[np.int64], list[str]]:
    """Convert the ``get_path`` label summary into tick indices and labels.

    Parameters
    ----------
    path_file : str
        Newline-separated ``"<label> <npoints>"`` records emitted by
        :func:`PAOFLOW.spectrum.kpnts_interpolation_mesh.get_path`. The number
        is how many k-points that segment contributes before the next label.

    Returns
    -------
    tuple[NDArray[np.int64], list[str]]
        ``(ticks, labels)`` where ``ticks[i]`` is the k-point index at which
        ``labels[i]`` sits.
    """
    ticks: list[int] = []
    labels: list[str] = []
    index = 0

    for line in path_file.strip().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        label, count = parts[0], int(parts[1])
        labels.append(label)
        ticks.append(index)
        index += count

    return np.asarray(ticks, dtype=np.int64), labels


def build_surface_kpath(
    data_controller: Any,
    *,
    transport_direction: str,
    band_path: Optional[str] = None,
    high_sym_points: Optional[Mapping[str, Sequence[float]]] = None,
    ibrav: Optional[int] = None,
    dk: float = 0.01,
    nk_path: Optional[int] = None,
) -> SurfaceKPath:
    r"""Build a surface-projected transverse k-path for surface band structures.

    The bulk high-symmetry path is generated with the standard PAOFLOW band-path
    machinery and then **projected onto the surface plane** by discarding the
    component along the transport (surface-normal) direction. This yields the
    transverse :math:`k_\perp` sampling used by the NEGF surface Green's
    function, whose spectral function
    :math:`A(k_\perp, E) = -\frac{1}{\pi}\,\mathrm{Im}\,\mathrm{Tr}\,G_s`
    is the surface-projected bulk band structure.

    Parameters
    ----------
    data_controller : DataController
        Shared PAOFLOW data store providing ``alat``, ``ibrav``, ``band_path``,
        ``a_vectors``, ``b_vectors``, and ``high_sym_points``.
    transport_direction : {'x', 'y', 'z'}
        Transport / surface-normal direction. Its k-component is projected out.
    band_path : str, optional
        Path string such as ``'gG-X-M-gG'``. Falls back to the value stored on
        the data controller, then to the default path for ``ibrav``.
    high_sym_points : Mapping[str, Sequence[float]], optional
        Explicit label -> fractional coordinate mapping. Falls back to the data
        controller, then to the tabulated points for ``ibrav``.
    ibrav : int, optional
        Quantum ESPRESSO Bravais lattice index. Falls back to the data
        controller. Required (non-zero) unless both ``band_path`` and
        ``high_sym_points`` are supplied.
    dk : float, optional
        Spacing between consecutive k-points along the path. Ignored when
        ``nk_path`` is given. Default ``0.01``.
    nk_path : int, optional
        Target number of k-points on the path. When given, ``dk`` is rescaled in
        a two-pass sweep so the path lands close to this count.

    Returns
    -------
    SurfaceKPath
        Projected path with weights, plotting abscissa, and tick metadata.

    Raises
    ------
    ValueError
        If the lattice information needed to build the path is unavailable.

    Notes
    -----
    The projection uses the *bulk* high-symmetry path restricted to the surface
    plane rather than a dedicated 2D surface-BZ path. Choose ``band_path`` so the
    segments lie in the surface plane (for example ``'gG-X'`` for a (001)
    surface with transport along ``z``); a segment running purely along the
    transport direction projects onto a single point and produces a degenerate
    abscissa.
    """
    from PAOFLOW.spectrum.kpnts_interpolation_mesh import _getHighSymPoints, get_path

    arrays, attr = data_controller.data_dicts()

    ibrav_value = ibrav if ibrav is not None else attr.get('ibrav')
    path_string = band_path if band_path is not None else attr.get('band_path')

    if high_sym_points is not None:
        points_map: Optional[Mapping[str, Sequence[float]]] = high_sym_points
    else:
        stored = arrays.get('high_sym_points')
        points_map = stored if stored else None

    if ibrav_value is None and (path_string is None or points_map is None):
        raise ValueError(
            'Cannot build a surface k-path: supply `ibrav`, or provide both '
            '`band_path` and `high_sym_points`.'
        )
    if ibrav_value in (0, None) and (path_string is None or points_map is None):
        raise ValueError(
            'ibrav=0 does not define a default high-symmetry path. Pass an '
            'explicit `band_path` together with `high_sym_points`, or pass the '
            '`ibrav` of the equivalent Bravais lattice.'
        )

    alat = attr['alat']
    a_vectors = arrays['a_vectors']
    b_vectors = arrays['b_vectors']

    # `get_path` needs an explicit special-points dict whenever a path string is
    # supplied, so resolve the tabulated defaults for `ibrav` up front.
    if points_map is None:
        points_map, default_path = _getHighSymPoints(ibrav_value, alat, a_vectors)
        if path_string is None:
            path_string = default_path

    if nk_path is not None:
        if nk_path < 2:
            raise ValueError(f'`nk_path` must be at least 2, got {nk_path}.')
        probe_dk = 1.0e-5
        probe, _ = get_path(
            ibrav_value, alat, a_vectors, probe_dk, b_vectors, path_string, points_map
        )
        dk_used = probe_dk * (probe.shape[1] / nk_path)
    else:
        dk_used = dk

    points, path_file = get_path(
        ibrav_value, alat, a_vectors, dk_used, b_vectors, path_string, points_map
    )

    # `points` is (3, nkpts) in fractional (crystal) coordinates.
    vkpt_par3D = np.ascontiguousarray(points.T, dtype=np.float64)

    # Project onto the surface plane: drop the surface-normal component so the
    # Bloch phase table only accumulates transverse R contributions.
    axis = direction_index(transport_direction)
    vkpt_par3D[:, axis] = 0.0

    # Raw spectral function: unit weights, no time-reversal folding.
    wk_par = np.ones(vkpt_par3D.shape[0], dtype=np.float64)

    # Plotting abscissa: cumulative distance in the projected (in-plane) BZ.
    kcart = vkpt_par3D @ np.asarray(b_vectors, dtype=np.float64)
    steps = np.linalg.norm(np.diff(kcart, axis=0), axis=1)
    kdist = np.concatenate(([0.0], np.cumsum(steps)))

    ticks, labels = _parse_path_labels(path_file)

    return SurfaceKPath(
        vkpt_par3D=vkpt_par3D,
        wk_par=wk_par,
        kdist=kdist,
        ticks=ticks,
        labels=labels,
    )
