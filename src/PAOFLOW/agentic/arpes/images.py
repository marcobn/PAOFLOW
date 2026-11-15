"""Experimental ARPES band maps from images: a calibrated bitmap as a ``(k, E, intensity)`` map.

A published band map is a bitmap with a momentum axis along the columns and an energy axis
along the rows. With the axis calibrations of :mod:`PAOFLOW.agentic.thermo_optics.digitize`
(:class:`~PAOFLOW.agentic.thermo_optics.digitize.Axis`), the pixels inside the frame become
an intensity map in the layout of the model maps (``intensity`` of shape ``(nk, nw)``), so the
same MDC analysis (:mod:`PAOFLOW.agentic.arpes.curves`) applies to both.  The intensity is a
grey level (or one colour channel): a published colour scale is monotonic in brightness only
for sequential colour maps, so check the scale bar and state the assumption.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..thermo_optics.digitize import Axis, load_image


def band_map_from_image(
    image: str | Path | NDArray,
    k_axis: Axis,
    energy_axis: Axis,
    rows: tuple[int, int] | None = None,
    cols: tuple[int, int] | None = None,
    channel: str | Callable[[NDArray], NDArray] = 'gray',
    dark_is_high: bool = True,
) -> dict[str, NDArray[np.float64]]:
    """Intensity map of a calibrated band-map image.

    Parameters
    ----------
    image : path or ndarray
        Image file, or an array ``(rows, cols, 3)`` in [0, 1] (or ``(rows, cols)`` grey).
    k_axis : Axis
        Pixel column -> momentum (1/Å) calibration.
    energy_axis : Axis
        Pixel row -> energy (eV, relative to E_F) calibration.
    rows, cols : (start, stop), optional
        Pixel range of the plot frame interior (default: the whole image).
    channel : {'gray', 'r', 'g', 'b'} or callable, optional
        Grey level, one colour channel, or a function of the RGB array giving the value.
    dark_is_high : bool, optional
        Dark pixels are high intensity (grey-scale maps on white); ``False`` for bright-is-high.

    Returns
    -------
    dict
        ``k`` (increasing), ``energy_ev`` (increasing) and ``intensity`` of shape ``(nk, nw)``
        in [0, 1].
    """
    data = load_image(image) if isinstance(image, (str, Path)) else np.asarray(image, dtype=float)
    if data.ndim == 2:
        value = data
    elif callable(channel):
        value = np.asarray(channel(data), dtype=float)
    else:
        index = {'gray': None, 'r': 0, 'g': 1, 'b': 2}[channel]
        value = data.mean(axis=2) if index is None else data[..., index]
    if dark_is_high:
        value = 1.0 - value
    r0, r1 = rows if rows is not None else (0, value.shape[0])
    c0, c1 = cols if cols is not None else (0, value.shape[1])
    crop = value[r0:r1, c0:c1]
    k = k_axis.to_value(np.arange(c0, c1, dtype=float))
    energy = energy_axis.to_value(np.arange(r0, r1, dtype=float))
    intensity = crop.T  # (ncols, nrows) = (nk, nw)
    korder, eorder = np.argsort(k), np.argsort(energy)
    return {
        'k': k[korder],
        'energy_ev': energy[eorder],
        'intensity': intensity[korder][:, eorder],
    }
