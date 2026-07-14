"""``fermi-plotter`` — interactive 3-D Fermi-surface viewer for PAOFLOW BXSF output.

Reads a Fermi-surface BXSF file written by PAOFLOW (``FermiSurf_{ispin}.bxsf``,
produced by :func:`PAOFLOW.writers.write2bxsf.write2bxsf`) and renders the
Fermi sheets as interactive iso-surfaces with Mayavi.  The surfaces are
extracted with marching cubes and coloured by the Fermi velocity
:math:`|\\nabla_{\\mathbf k} E|`.  The Mayavi window supports rotation, zoom and
pan out of the box.

This module backs the ``fermi-plotter`` console script (see ``[project.scripts]``
in ``pyproject.toml``).

Usage examples
--------------
Composite image (all bands crossing the Fermi window)::

    fermi-plotter FermiSurf_0.bxsf

A single band (by its BXSF band label)::

    fermi-plotter FermiSurf_0.bxsf --band 58

A subset of bands, upsampled 2x for smoother sheets, saved to PNG::

    fermi-plotter FermiSurf_0.bxsf --band 57,58 --interp 2 --save fermi.png

Only one spin channel is handled per run; point the tool at the desired
``FermiSurf_{ispin}.bxsf`` file directly.

Requires the ``fermisurface`` extra (``pip install "PAOFLOW[fermisurface]"`` or
``pip install mayavi scikit-image``).
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# BXSF parsing
# --------------------------------------------------------------------------- #

_FLOAT_RE = re.compile(r'[-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?')
_FERMI_RE = re.compile(r'fermi\s*energy', re.IGNORECASE)
_BANDGRID_RE = re.compile(r'BANDGRID_3D_BANDS', re.IGNORECASE)
_BAND_RE = re.compile(r'BAND\s*:\s*(\d+)', re.IGNORECASE)
_END_RE = re.compile(r'END_BANDGRID_3D', re.IGNORECASE)


def _to_float(token: str) -> float:
    return float(token.replace('D', 'E').replace('d', 'e'))


@dataclass
class FermiSurfData:
    """Parsed contents of a (possibly multi-band) PAOFLOW BXSF file.

    Attributes
    ----------
    fermi_energy : float
        Fermi energy in eV, as written in the ``BEGIN_INFO`` block.
    dims : tuple[int, int, int]
        Grid dimensions ``(Nx, Ny, Nz)`` as stored in the file.  PAOFLOW writes
        the periodic wrap-around grid, so ``Nx = nk1 + 1`` etc.
    recip : np.ndarray, shape (3, 3)
        Reciprocal-lattice (spanning) vectors as stored in the file; row ``i``
        is the ``i``-th vector (units of :math:`2\\pi/a_{\\rm lat}`).
    bands : dict[int, np.ndarray]
        Mapping of BXSF band label -> energy grid of shape ``dims`` (eV).
    """

    fermi_energy: float
    dims: tuple[int, int, int]
    recip: np.ndarray
    bands: dict[int, np.ndarray]


def read_fermi_bxsf(path: str | Path) -> FermiSurfData:
    """Read a PAOFLOW Fermi-surface BXSF file (single or multi-band).

    Unlike ``PAOFLOW.pyskeaf.io_bxsf.read_bxsf`` (which targets single-band,
    non-periodic SKEAF grids), this reader accepts the multi-band,
    periodic-wrapped grid that ``write2bxsf`` emits.

    Parameters
    ----------
    path : str or Path
        Path to the ``.bxsf`` file.

    Returns
    -------
    FermiSurfData
        Parsed Fermi energy, grid dimensions, reciprocal vectors and per-band
        energy grids.
    """
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines()

    # 1. Fermi energy.
    fermi_energy = None
    for line in lines:
        if _FERMI_RE.search(line):
            toks = _FLOAT_RE.findall(line)
            if toks:
                fermi_energy = _to_float(toks[-1])
            break
    if fermi_energy is None:
        raise ValueError(f'{path}: no "Fermi Energy" line found.')

    # 2. Locate the BANDGRID_3D_BANDS keyword and read the header block.
    i = 0
    n = len(lines)
    while i < n and not _BANDGRID_RE.search(lines[i]):
        i += 1
    if i >= n:
        raise ValueError(f'{path}: no BANDGRID_3D_BANDS block found.')
    i += 1

    def _next_nonblank(idx: int) -> int:
        while idx < n and not lines[idx].strip():
            idx += 1
        return idx

    i = _next_nonblank(i)
    nbnd = int(_FLOAT_RE.findall(lines[i])[0])
    i = _next_nonblank(i + 1)
    dims = tuple(int(v) for v in _FLOAT_RE.findall(lines[i])[:3])
    i += 1  # origin
    origin = np.array([_to_float(v) for v in _FLOAT_RE.findall(lines[i])[:3]])
    recip = np.empty((3, 3), dtype=float)
    for r in range(3):
        i += 1
        recip[r] = [_to_float(v) for v in _FLOAT_RE.findall(lines[i])[:3]]

    npts = int(np.prod(dims))

    # 3. Read each BAND block.
    bands: dict[int, np.ndarray] = {}
    i += 1
    while i < n:
        line = lines[i]
        m = _BAND_RE.search(line)
        if m:
            label = int(m.group(1))
            vals: list[float] = []
            i += 1
            while i < n and not _BAND_RE.search(lines[i]) and not _END_RE.search(lines[i]):
                vals.extend(_to_float(t) for t in _FLOAT_RE.findall(lines[i]))
                i += 1
            if len(vals) < npts:
                raise ValueError(f'{path}: band {label} has {len(vals)} values, expected {npts}.')
            bands[label] = np.asarray(vals[:npts], dtype=float).reshape(dims)
            continue
        if _END_RE.search(line):
            break
        i += 1

    if len(bands) != nbnd:
        # Not fatal, just informational.
        print(
            f'warning: header declares {nbnd} bands, parsed {len(bands)}.',
            file=sys.stderr,
        )
    if not np.allclose(origin, 0.0):
        print(f'warning: non-zero grid origin {origin} ignored.', file=sys.stderr)

    return FermiSurfData(fermi_energy, dims, recip, bands)


# --------------------------------------------------------------------------- #
# Interpolation
# --------------------------------------------------------------------------- #


def fft_upsample(grid: np.ndarray, factor: int) -> np.ndarray:
    """Fourier zero-padding upsampling of a periodic energy grid.

    The PAOFLOW grid is periodic with a duplicated endpoint plane in each
    direction.  This routine strips the wrap plane, upsamples the base
    (period) grid by ``factor`` via Fourier zero-padding (exact for periodic
    band energies), and re-appends the wrap plane so the result stays closed
    for marching cubes.

    Parameters
    ----------
    grid : np.ndarray, shape (Nx, Ny, Nz)
        Wrap-closed energy grid (``N_i = n_i + 1``).
    factor : int
        Integer upsampling factor (``>= 1``).  ``1`` returns the input.

    Returns
    -------
    np.ndarray
        Upsampled, wrap-closed grid of shape
        ``(factor*n_x + 1, factor*n_y + 1, factor*n_z + 1)`` along axes with
        ``n_i > 1`` (axes of length 1 are left unchanged).
    """
    if factor <= 1:
        return grid

    active = tuple(s > 1 for s in grid.shape)  # axes carrying a periodic wrap
    base = grid[
        tuple(slice(0, s - 1) if a else slice(None) for s, a in zip(grid.shape, active))
    ]  # strip periodic wrap -> period grid
    ft = np.fft.fftn(base)
    old = base.shape
    # Only Fourier-upsample axes whose period has >1 sample; a constant axis
    # (period length 1, e.g. a 2-D k-grid) cannot be refined.
    new = tuple(s * factor if s > 1 else s for s in old)

    padded = np.zeros(new, dtype=complex)

    # Copy the frequency components into the enlarged spectrum, splitting the
    # Nyquist-symmetric halves so the inverse transform stays real.
    def _halves(o: int):
        pos = (o + 1) // 2  # number of non-negative freqs (incl. DC)
        neg = o - pos
        return pos, neg

    # General N-D placement of frequency blocks.
    def _place(src, dst):
        for combo in itertools.product(range(2), repeat=3):
            src_sl, dst_sl = [], []
            ok = True
            for ax in range(3):
                o = old[ax]
                ns = new[ax]
                pos, neg = _halves(o)
                if combo[ax] == 0:  # positive freqs
                    src_sl.append(slice(0, pos))
                    dst_sl.append(slice(0, pos))
                else:  # negative freqs
                    if neg == 0:
                        ok = False
                        break
                    src_sl.append(slice(o - neg, o))
                    dst_sl.append(slice(ns - neg, ns))
            if ok:
                dst[tuple(dst_sl)] = src[tuple(src_sl)]

    _place(ft, padded)

    scale = np.prod([ns / o for o, ns in zip(old, new)])
    up = np.fft.ifftn(padded).real * scale

    # Re-append the periodic wrap plane on every originally-active axis so the
    # grid stays closed for marching cubes (restores Nz=2 for 2-D k-grids too).
    out_shape = tuple(s + 1 if a else s for s, a in zip(up.shape, active))
    out = np.empty(out_shape, dtype=float)
    sx, sy, sz = up.shape
    out[:sx, :sy, :sz] = up
    if active[0]:
        out[sx, :sy, :sz] = up[0, :, :]
    if active[1]:
        out[:, sy, :sz] = out[:, 0, :sz]
    if active[2]:
        out[:, :, sz] = out[:, :, 0]
    return out


# --------------------------------------------------------------------------- #
# Geometry / velocity
# --------------------------------------------------------------------------- #


def fermi_velocity_field(energy: np.ndarray, recip: np.ndarray) -> np.ndarray:
    """Return the Fermi-velocity magnitude ``|grad_k E|`` on the grid.

    Parameters
    ----------
    energy : np.ndarray, shape (Nx, Ny, Nz)
        Wrap-closed energy grid (eV).
    recip : np.ndarray, shape (3, 3)
        Reciprocal spanning vectors (rows).

    Returns
    -------
    np.ndarray, shape (Nx, Ny, Nz)
        ``|grad_k E|`` at every grid node (eV per reciprocal-length unit).
    """
    dims = np.array(energy.shape)
    # Fractional spacing between nodes (period spans indices 0..N-1).
    span = np.where(dims > 1, dims - 1, 1)
    dfrac = 1.0 / span
    # np.gradient needs >=3 points for edge_order=2 and >=2 for edge_order=1;
    # constant axes (length 1) contribute a zero derivative.
    g_frac = np.zeros(energy.shape + (3,), dtype=float)
    for axis in range(3):
        if energy.shape[axis] < 2:
            continue
        eo = 2 if energy.shape[axis] >= 3 else 1
        g_frac[..., axis] = np.gradient(energy, dfrac[axis], axis=axis, edge_order=eo)
    binv = np.linalg.inv(recip)  # frac_i = k_j * binv[j, i]
    # grad_k_j = sum_i binv[j, i] * dE/dfrac_i
    grad_k = np.einsum('ji,...i->...j', binv, g_frac)
    return np.linalg.norm(grad_k, axis=-1)


def _cell_edges(recip: np.ndarray):
    """Return line segments (list of (p0, p1)) for the reciprocal cell box."""
    corners = []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                corners.append(a * recip[0] + b * recip[1] + c * recip[2])
    corners = np.array(corners)
    idx = {(a, b, c): a * 4 + b * 2 + c for a in (0, 1) for b in (0, 1) for c in (0, 1)}
    edges = []
    for (a, b, c), i0 in idx.items():
        for axis in range(3):
            nb = [a, b, c]
            if nb[axis] == 0:
                nb[axis] = 1
                edges.append((corners[i0], corners[idx[tuple(nb)]]))
    return edges


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def plot_fermi_surface(
    data: FermiSurfData,
    band_labels: list[int],
    interp: int = 1,
    fermi_shift: float = 0.0,
    cmap: str = 'jet',
    opacity: float = 1.0,
    show_bz: bool = True,
    figsize: tuple[int, int] = (900, 700),
    save: str | None = None,
) -> None:
    """Render the selected Fermi sheets with Mayavi.

    Parameters
    ----------
    data : FermiSurfData
        Parsed BXSF contents.
    band_labels : list[int]
        BXSF band labels to render.  A single element gives a one-band image;
        multiple elements give the composite image.
    interp : int, optional
        Integer Fourier upsampling factor for smoother sheets (default 1).
    fermi_shift : float, optional
        Energy offset (eV) added to the Fermi level defining the iso-surface.
    cmap : str, optional
        Mayavi/VTK colormap name for the velocity colouring.
    opacity : float, optional
        Surface opacity in ``[0, 1]``.
    show_bz : bool, optional
        Draw the reciprocal-cell parallelepiped wireframe.
    figsize : tuple[int, int], optional
        Render window size in pixels.
    save : str, optional
        If given, render and save a PNG to this path instead of opening an
        interactive window.  (VTK on macOS has no OSMesa off-screen backend, so
        this still renders through a normal on-screen window.)
    """
    from mayavi import mlab
    from scipy.ndimage import map_coordinates
    from skimage import measure

    level = data.fermi_energy + fermi_shift
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=figsize)

    # Global velocity range across selected bands for a shared colour scale.
    speeds_min, speeds_max = np.inf, -np.inf
    prepared = []
    for label in band_labels:
        energy = data.bands[label]
        if interp > 1:
            energy = fft_upsample(energy, interp)
        emin, emax = energy.min(), energy.max()
        if not (emin <= level <= emax):
            print(
                f'band {label}: does not cross E={level:.4f} eV '
                f'(range [{emin:.3f}, {emax:.3f}]); skipping.',
                file=sys.stderr,
            )
            continue
        try:
            verts, faces, _, _ = measure.marching_cubes(energy, level=level)
        except (ValueError, RuntimeError) as exc:
            print(f'band {label}: marching cubes failed ({exc}); skipping.', file=sys.stderr)
            continue
        speed = fermi_velocity_field(energy, data.recip)
        prepared.append((label, energy, verts, faces, speed))
        speeds_min = min(speeds_min, speed.min())
        speeds_max = max(speeds_max, speed.max())

    if not prepared:
        print('No band crosses the Fermi level; nothing to plot.', file=sys.stderr)
        return

    for label, energy, verts, faces, speed in prepared:
        dims = np.array(energy.shape)
        span = np.where(dims > 1, dims - 1, 1)
        frac = verts / span
        cart = frac @ data.recip
        vspeed = map_coordinates(speed, verts.T, order=1, mode='nearest')
        mesh = mlab.triangular_mesh(
            cart[:, 0],
            cart[:, 1],
            cart[:, 2],
            faces,
            scalars=vspeed,
            colormap=cmap,
            opacity=opacity,
            vmin=speeds_min,
            vmax=speeds_max,
            figure=fig,
        )
        mesh.name = f'band_{label}'

    if show_bz:
        for p0, p1 in _cell_edges(data.recip):
            mlab.plot3d(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color=(0.3, 0.3, 0.3),
                tube_radius=None,
                line_width=1.5,
                figure=fig,
            )

    mlab.colorbar(title='|grad E|  (Fermi velocity)', orientation='vertical', nb_labels=5)
    mlab.orientation_axes()

    if save is not None:
        mlab.savefig(save, size=figsize)
        mlab.close(fig)
        print(f'saved {save}')
    else:
        mlab.show()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_bands(spec: str | None, available: list[int]) -> list[int]:
    if spec is None or spec.lower() in ('all', 'composite'):
        return sorted(available)
    labels = [int(x) for x in re.split(r'[,\s]+', spec.strip()) if x]
    missing = [b for b in labels if b not in available]
    if missing:
        raise SystemExit(f'band(s) {missing} not in file; available: {sorted(available)}')
    return labels


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='fermi-plotter',
        description='Interactive Mayavi Fermi-surface viewer for PAOFLOW BXSF files.',
    )
    p.add_argument('bxsf', help='Path to a FermiSurf_{ispin}.bxsf file.')
    p.add_argument(
        '--band',
        default=None,
        help='Band label(s) to plot: e.g. "58" (single) or "57,58" (subset). '
        'Default: composite (all bands in the file).',
    )
    p.add_argument(
        '--interp',
        type=int,
        default=1,
        metavar='N',
        help='Integer Fourier upsampling factor for smoother sheets (default 1).',
    )
    p.add_argument(
        '--fermi-shift',
        type=float,
        default=0.0,
        metavar='dE',
        help='Energy offset (eV) added to the Fermi level (default 0).',
    )
    p.add_argument('--cmap', default='jet', help='Colormap for velocity colouring (default jet).')
    p.add_argument('--opacity', type=float, default=1.0, help='Surface opacity 0..1 (default 1).')
    p.add_argument('--no-bz', action='store_true', help='Hide the reciprocal-cell box.')
    p.add_argument(
        '--size', default='900x700', metavar='WxH', help='Window size in pixels (default 900x700).'
    )
    p.add_argument(
        '--save',
        default=None,
        metavar='PNG',
        help='Render and save this PNG instead of opening an interactive window.',
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point for ``fermi-plotter``."""
    args = _build_parser().parse_args(argv)
    path = Path(args.bxsf)
    if not path.exists():
        print(f'error: file not found: {path}', file=sys.stderr)
        return 2

    data = read_fermi_bxsf(path)
    labels = _parse_bands(args.band, list(data.bands.keys()))
    try:
        w, h = (int(v) for v in args.size.lower().split('x'))
    except ValueError:
        print(f'error: bad --size "{args.size}", expected WxH.', file=sys.stderr)
        return 2

    print(
        f'{path.name}: Efermi={data.fermi_energy:.4f} eV, grid={data.dims}, '
        f'bands={sorted(data.bands.keys())}; plotting {labels}.'
    )
    try:
        plot_fermi_surface(
            data,
            labels,
            interp=args.interp,
            fermi_shift=args.fermi_shift,
            cmap=args.cmap,
            opacity=args.opacity,
            show_bz=not args.no_bz,
            figsize=(w, h),
            save=args.save,
        )
    except ImportError as exc:
        print(
            f'Missing dependency: {exc}.\n'
            'Install the viewer requirements with:\n'
            '    pip install "PAOFLOW[fermisurface]"   # or: pip install mayavi scikit-image',
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
