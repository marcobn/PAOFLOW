from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .arpes.experiment import IntensityMap

# A colour matplotlib accepts: a name or hex string, or an (R, G, B) tuple in [0, 1].
Color = str | tuple[float, float, float]
SymPoints = str | tuple[list[int], list[str]]


class GPAO:
    """Plot PAOFLOW output files with matplotlib.

    Each method reads the text files that a PAOFLOW property writes to
    ``outputdir`` (``bands_0.dat``, ``dosdk_0.dat``, ``sigmagauss_0.dat``, ...),
    opens a new figure and calls ``matplotlib.pyplot.show()``. No ``PAOFLOW``
    object is needed, so a finished run can be plotted on another machine.
    GPAO computes nothing: a figure shows what the files contain.

    Examples
    --------
    >>> from PAOFLOW import GPAO
    >>> plotter = GPAO.GPAO()
    >>> plotter.plot_bands('output/bands_0.dat', sym_points='output/kpath_points.txt')
    """

    def __init__(self) -> None:
        pass

    def plot_dos(
        self,
        fname: str,
        title: str | None = None,
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
        vertical: bool = False,
        col: Color = 'black',
    ) -> None:
        """Plot the density of states.

        Parameters
        ----------
        fname : str
            File name (including relative path), for example ``dosdk_0.dat``:
            energy (eV) and DOS (states/eV) in two columns.
        title : str, optional
            A title for the plot (default ``'DoS'``).
        x_lim : tuple of float, optional
            Pair of axis limits ``(x_min, x_max)``.
        y_lim : tuple of float, optional
            Pair of axis limits ``(y_min, y_max)``. By default the DOS axis
            starts at 0.
        vertical : bool, optional
            Set ``True`` to plot energy on the y-axis and states/eV on the x-axis.
        col : str or tuple of float, optional
            A colour recognized by matplotlib or a 3-tuple ``(R, G, B)``.

        Returns
        -------
        None
            Shows the figure.
        """
        from .graphics.plot_functions import plot_dos
        from .inputs.read_pao_output import read_dos_PAO

        es, dos = read_dos_PAO(fname)

        plot_dos(es, dos, title, x_lim, y_lim, vertical, col)

    def plot_pdos(
        self,
        fnames: list[str],
        title: str | None = None,
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
        vertical: bool = False,
        cols: Color | list[Color] | None = None,
        labels: list[str] | None = None,
        legend: bool = True,
        psum_inds: list[list[int]] | None = None,
    ) -> None:
        """Plot the projected density of states.

        Parameters
        ----------
        fnames : list of str
            List of file names (including relative path), for example the
            ``*_pdosdk_0.dat`` files of ``dos()``, one per orbital.
        title : str, optional
            A title for the plot (default ``'PDoS'``).
        x_lim : tuple of float, optional
            Pair of axis limits ``(x_min, x_max)``.
        y_lim : tuple of float, optional
            Pair of axis limits ``(y_min, y_max)``. By default the PDOS axis
            starts at 0.
        vertical : bool, optional
            Set ``True`` to plot energy on the y-axis and states/eV on the x-axis.
        cols : str, tuple or list, optional
            One colour for every curve, or a list of colours recognized by
            matplotlib (strings or 3-tuples ``(R, G, B)``) with one entry per
            curve. ``None`` uses the matplotlib colour cycle.
        labels : list of str, optional
            Legend labels, one per curve. The default labels are the curve
            numbers.
        legend : bool, optional
            Show the legend (default ``True``).
        psum_inds : list of list of int, optional
            A list of lists. Each inner list contains the positions in
            ``fnames`` of the files to sum together. There is one line plotted
            for each inner list, so ``cols`` and ``labels`` then take one entry
            per inner list.

        Returns
        -------
        None
            Shows the figure.
        """
        import numpy as np

        from .graphics.plot_functions import plot_pdos
        from .inputs.read_pao_output import read_dos_PAO

        es = None
        dos = []
        for fn in fnames:
            es, ds = read_dos_PAO(fn)
            dos.append(ds)

        if psum_inds is not None:
            ndos = np.zeros((len(psum_inds), len(es)), dtype=float)
            for i, p in enumerate(psum_inds):
                for v in p:
                    ndos[i, :] += dos[v]
            dos = ndos

        plot_pdos(es, dos, title, x_lim, y_lim, vertical, cols, labels, legend)

    def plot_weighted_bands(
        self,
        outputdir: str | None,
        fname: str,
        sym_points: SymPoints | None = None,
        title: str | None = None,
        cbar_label: str | None = None,
        label: str | None = None,
        filename: str | None = None,
        y_lim: tuple[float, float] | None = None,
        col: Color = 'black',
    ) -> None:
        """Plot the site-projected band structure, coloured by the weight on the chosen sites.

        Parameters
        ----------
        outputdir : str or None
            Prefix joined to ``filename`` by plain string concatenation, so it
            needs a trailing ``/``. ``None`` saves ``filename`` as given.
        fname : str
            File name (including relative path) of the
            ``site-projected-bands_<ispin>.dat`` file written by
            ``site_projected_bands()``: k-point index, band energy (eV) and site
            weight in three columns.
        sym_points : str or tuple, optional
            File name for the ``kpath_points.txt`` produced by PAOFLOW.
            Otherwise, provide a tuple of two lists. The first contains indices
            of the high-symmetry points, the second contains their labels. The
            plot needs it: the line at :math:`E = 0` is drawn across these points.
        title : str, optional
            A title for the plot.
        cbar_label : str, optional
            Label of the colour bar (default ``'Weight'``).
        label : str, optional
            Energy-axis label (default :math:`\\epsilon(\\mathbf{k})` (eV)).
        filename : str, optional
            If given, also save the figure to this path at 300 dpi.
        y_lim : tuple of float, optional
            Pair of axis limits ``(y_min, y_max)``.
        col : str or tuple of float, optional
            Not used: the points are coloured by their weight.

        Returns
        -------
        None
            Shows the figure and, with ``filename``, writes it to disk.

        Notes
        -----
        Weights outside :math:`[0, 1]` are rescaled to the 1st–99th percentile
        range of the file before they set the colour and the marker size.
        """
        from .graphics.plot_functions import plot_weighted_bands
        from .inputs.read_pao_output import read_site_projected

        if sym_points is not None:
            if type(sym_points) is str:
                from .inputs.read_pao_output import read_band_path_PAO

                sym_points = read_band_path_PAO(sym_points)
        plot_weighted_bands(
            outputdir,
            read_site_projected(fname),
            sym_points,
            title,
            cbar_label,
            label,
            filename,
            y_lim,
            col,
        )

    def plot_bands(
        self,
        fnames: str | list[str],
        sym_points: SymPoints | None = None,
        title: str | None = None,
        label: str | None = None,
        labels: list[str] | None = None,
        y_lim: tuple[float, float] | None = None,
        cols: Color | list[Color] | None = None,
        legend: bool = True,
        show: bool = True,
    ) -> Axes:
        """Plot one or more band structures for comparison.

        Parameters
        ----------
        fnames : str or list of str
            File name or list of file names (including relative path) of
            ``bands_0.dat`` files. Several files are overlaid on the k axis of
            the first one.
        sym_points : str or tuple, optional
            File name for the ``kpath_points.txt`` produced by PAOFLOW.
            Otherwise, provide a tuple of two lists. The first contains indices
            of the high-symmetry points, the second contains their labels.
            Without it the k axis is hidden.
        title : str, optional
            A title for the plot (default ``'Band Structure'``).
        label : str, optional
            y-axis label (default :math:`\\epsilon(\\mathbf{k})` (eV)).
        labels : list of str, optional
            Legend labels, one per file (e.g. ``['DFT', 'SK model']``).
        y_lim : tuple of float, optional
            Pair of axis limits ``(y_min, y_max)``.
        cols : str, tuple or list, optional
            Colour(s) for each dataset: one colour, or a list of strings or
            3-tuples recognized by matplotlib with one entry per file. With
            several files and a single colour, that colour goes to the first
            file and the others get black, red, blue, ...
        legend : bool, optional
            Show legend when labels are provided (default ``True``).
        show : bool, optional
            Show the figure (default ``True``). With ``False`` the figure stays
            open, so that you can draw more data on the returned axes, for
            example reference eigenvalues, and show it with ``plt.show()``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes of the band plot. Shows the figure when ``show`` is
            ``True``.
        """
        from .graphics.plot_functions import plot_bands
        from .inputs.read_pao_output import read_bands_PAO

        if isinstance(fnames, str):
            fnames = [fnames]

        bands_list = [read_bands_PAO(fn) for fn in fnames]

        if sym_points is not None:
            if type(sym_points) is str:
                from .inputs.read_pao_output import read_band_path_PAO

                sym_points = read_band_path_PAO(sym_points)
        return plot_bands(bands_list, sym_points, title, label, y_lim, cols, labels, legend, show)

    def plot_arpes(
        self,
        intensity_map: IntensityMap,
        fname: str | None = None,
        cut: str = 'band_map',
        value: float = 0.0,
        path_labels: tuple[list[int], list[str]] | None = None,
        title: str | None = None,
        cmap: str = 'Greys_r',
        broadened: bool = True,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot an ARPES simulation (``PAOFLOW.arpes.IntensityMap``).

        Parameters
        ----------
        intensity_map : IntensityMap
            Result of ``PAOFLOW.arpes`` / ``PAOFLOW.arpes.simulate``.
        fname : str, optional
            File name of the figure (``None`` shows it). Ignored when ``ax`` is
            given.
        cut : {'band_map', 'constant_energy', 'edc'}, optional
            ``'band_map'`` (energy vs the path index, or vs ``k_x`` at
            ``k_y = value`` for a mesh), ``'constant_energy'`` (``k_x``-``k_y``
            map at the energy ``value``), ``'edc'`` (energy distribution curve
            at k-point index ``value``).
        value : float or int, optional
            Energy (eV), ``k_y`` (1/Å) or k index of the cut.
        path_labels : tuple, optional
            ``(indices, labels)`` of the high-symmetry points of a path.
        title : str, optional
            A title.
        cmap : str, optional
            Colour map.
        broadened : bool, optional
            Resolution-broadened (``True``) or raw intensity.
        ax : matplotlib.axes.Axes, optional
            Draw into an existing axis instead of a new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axis drawn into.

        Raises
        ------
        ValueError
            If ``cut`` is unknown, or ``'constant_energy'`` is asked for a path
            instead of a momentum mesh.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        own_figure = ax is None
        if own_figure:
            fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
        data = intensity_map.intensity if broadened else intensity_map.raw
        w = intensity_map.w
        if cut == 'edc':
            ax.plot(w, data[int(value)], color='k')
            ax.set_xlabel('E - E$_F$ (eV)')
            ax.set_ylabel('intensity (arb. units)')
        elif cut == 'constant_energy':
            if intensity_map.grid is None:
                raise ValueError('a constant-energy map needs a momentum mesh')
            iw = int(np.argmin(np.abs(w - value)))
            cube = intensity_map.as_grid(broadened)
            kx, ky = intensity_map.axes['kx'], intensity_map.axes['ky']
            ax.pcolormesh(kx, ky, cube[:, :, iw].T, cmap=cmap, shading='auto')
            ax.set_xlabel('k$_x$ (1/Å)')
            ax.set_ylabel('k$_y$ (1/Å)')
            ax.set_aspect('equal')
        elif cut == 'band_map':
            if intensity_map.grid is not None:
                ky = intensity_map.axes['ky']
                iy = int(np.argmin(np.abs(ky - value)))
                image = intensity_map.as_grid(broadened)[:, iy, :]
                x = intensity_map.axes['kx']
                ax.set_xlabel('k$_x$ (1/Å)')
            else:
                image = data
                x = np.arange(data.shape[0])
                ax.set_xlabel('k-path')
                if path_labels is not None:
                    ax.set_xticks(path_labels[0])
                    ax.set_xticklabels(path_labels[1])
                    for idx in path_labels[0]:
                        ax.axvline(idx, color='0.6', lw=0.6)
            ax.pcolormesh(x, w, image.T, cmap=cmap, shading='auto')
            ax.axhline(0.0, color='w', lw=0.5, ls='--')
            ax.set_ylabel('E - E$_F$ (eV)')
        else:
            raise ValueError("cut must be 'band_map', 'constant_energy' or 'edc'")
        if title is not None:
            ax.set_title(title)
        if own_figure:
            if fname is None:
                plt.show()
            else:
                fig.savefig(fname, dpi=200)
                plt.close(fig)
        return ax

    def plot_berry(
        self,
        fname: str,
        sym_points: SymPoints | None = None,
        title: str | None = None,
        label: str | None = None,
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
        col: Color = 'black',
        dos_ticks: bool = False,
    ) -> None:
        """Plot the Berry curvature along the band path.

        Parameters
        ----------
        fname : str
            File name (including relative path) of an ``Omega_*.dat`` or
            ``Omegaj_*.dat`` file of ``topology(Berry=True)``: k-point index and
            curvature (bohr²) in two columns.
        sym_points : str or tuple, optional
            File name for the ``kpath_points.txt`` produced by PAOFLOW.
            Otherwise, provide a tuple of two lists. The first contains indices
            of the high-symmetry points, the second contains their labels.
        title : str, optional
            A title for the plot (default ``'Berry curvature vs k-point'``).
        label : str, optional
            y-axis label (default :math:`\\Omega(\\mathbf{k})`).
        x_lim : tuple of float, optional
            Not used: the k axis always spans the path.
        y_lim : tuple of float, optional
            Pair of axis limits ``(y_min, y_max)`` of the curvature.
        col : str or tuple of float, optional
            A colour recognized by matplotlib or a 3-tuple ``(R, G, B)``.
        dos_ticks : bool, optional
            Not used.

        Returns
        -------
        None
            Shows the figure.
        """
        import numpy as np

        from .graphics.plot_functions import plot_bands
        from .inputs.read_pao_output import read_dos_PAO

        if title is None:
            title = 'Berry curvature vs k-point'
        if label is None:
            label = r'$\Omega(\mathbf{k})$'
        if sym_points is not None:
            if type(sym_points) is str:
                from .inputs.read_pao_output import read_band_path_PAO

                sym_points = read_band_path_PAO(sym_points)
        path, omega = read_dos_PAO(fname)
        plot_bands(np.array([omega]), sym_points, title, label, y_lim, col)

    def plot_dos_beside_bands(
        self,
        fn_dos: str,
        fn_bands: str,
        sym_points: SymPoints | None = None,
        title: str | None = None,
        band_label: str | None = None,
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
        col: Color = 'black',
        dos_ticks: bool = False,
    ) -> None:
        """Plot the density of states beside the band structure, on the same energy axis.

        Parameters
        ----------
        fn_dos : str
            File name for dos (including relative path), e.g. ``dosdk_0.dat``.
        fn_bands : str
            File name for bands (including relative path), e.g. ``bands_0.dat``.
        sym_points : str or tuple, optional
            File name for the ``kpath_points.txt`` produced by PAOFLOW.
            Otherwise, provide a tuple of two lists. The first contains indices
            of the high-symmetry points, the second contains their labels.
        title : str, optional
            A title for the plot (default ``'Band Structure and DoS'``).
        band_label : str, optional
            Energy-axis label of the band panel (default
            :math:`\\epsilon(\\mathbf{k})` (eV)).
        x_lim : tuple of float, optional
            Pair of limits ``(x_min, x_max)`` of the DOS axis of the DOS panel;
            by default it starts at 0.
        y_lim : tuple of float, optional
            Pair of energy limits ``(y_min, y_max)``, shared by both panels.
            Keep it inside the energy range of the DOS file.
        col : str or tuple of float, optional
            A colour recognized by matplotlib or a 3-tuple ``(R, G, B)``.
        dos_ticks : bool, optional
            Show the axes of the DOS panel (default ``False`` hides them).

        Returns
        -------
        None
            Shows the figure.
        """
        from .graphics.plot_functions import plot_dos_beside_bands
        from .inputs.read_pao_output import read_bands_PAO, read_dos_PAO

        if sym_points is not None:
            if type(sym_points) is str:
                from .inputs.read_pao_output import read_band_path_PAO

                sym_points = read_band_path_PAO(sym_points)

        bands = read_bands_PAO(fn_bands)
        es, dos = read_dos_PAO(fn_dos)

        plot_dos_beside_bands(
            es, dos, bands, sym_points, title, band_label, x_lim, y_lim, col, dos_ticks
        )

    def plot_berry_under_bands(
        self,
        fn_berry: str,
        fn_bands: str,
        sym_points: SymPoints | None = None,
        title: str | None = None,
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
        col: Color = 'black',
        dos_ticks: bool = False,
        band_label: str | None = None,
        berry_label: str | None = None,
    ) -> None:
        """Plot the Berry curvature in a panel below the band structure, on the same k axis.

        Parameters
        ----------
        fn_berry : str
            File name for the Berry curvature (including relative path), an
            ``Omega_*.dat`` file of ``topology(Berry=True)``.
        fn_bands : str
            File name for bands (including relative path), from the same
            ``bands()`` path as ``fn_berry``.
        sym_points : str or tuple, optional
            File name for the ``kpath_points.txt`` produced by PAOFLOW.
            Otherwise, provide a tuple of two lists. The first contains indices
            of the high-symmetry points, the second contains their labels.
        title : str, optional
            A title for the plot (default ``'Band Structure and Berry Phase'``).
        x_lim : tuple of float, optional
            Not used: the k axis always spans the path.
        y_lim : tuple of float, optional
            Pair of energy limits ``(y_min, y_max)`` of the band panel. The
            curvature panel spans the range of the data; use :meth:`plot_berry`
            to choose it.
        col : str or tuple of float, optional
            A colour recognized by matplotlib or a 3-tuple ``(R, G, B)``.
        dos_ticks : bool, optional
            Not used.
        band_label : str, optional
            Energy-axis label (default :math:`\\epsilon(\\mathbf{k})` (eV)).
        berry_label : str, optional
            Curvature-axis label (default :math:`\\Omega(\\mathbf{k})`).

        Returns
        -------
        None
            Shows the figure.
        """
        from .graphics.plot_functions import plot_berry_under_bands
        from .inputs.read_pao_output import read_bands_PAO, read_dos_PAO

        if sym_points is not None:
            if type(sym_points) is str:
                from .inputs.read_pao_output import read_band_path_PAO

                sym_points = read_band_path_PAO(sym_points)

        bands = read_bands_PAO(fn_bands)
        path, omega = read_dos_PAO(fn_berry)

        plot_berry_under_bands(
            omega,
            bands,
            sym_points,
            title,
            band_label,
            berry_label,
            x_lim,
            y_lim,
            col,
            dos_ticks,
        )

    def plot_electrical_conductivity(
        self,
        fname: str,
        t_ele: list[tuple[int, int]] = [(0, 0), (1, 1), (2, 2)],
        vE: float | None = None,
        title: str = 'Sigma vs Energy',
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
        col: Color | list[Color] = [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        legend: bool = True,
    ) -> None:
        """Plot the electrical conductivity against the chemical potential.

        Each temperature in the file gets its own figure, over the full energy
        range of the file, with ``', T=<temperature>'`` appended to ``title``.

        Parameters
        ----------
        fname : str
            File name (including relative path) of a ``sigma*_0.dat`` file of
            ``transport()``: temperature (K), chemical potential (eV) and the
            ``xx, yy, zz, xy, xz, yz`` components in
            :math:`(\\Omega\\,\\mathrm{m\\,s})^{-1}`, that is :math:`\\sigma/\\tau`
            for a constant relaxation time.
        t_ele : list of tuple of int, optional
            Tensor elements as index pairs (e.g. ``(1, 2)`` for ``yz``). The
            default plots the 3 diagonal elements separately. Providing an
            empty list plots the average of the diagonal components.
        vE : float, optional
            Not used: the conductivity is always plotted against energy.
        title : str, optional
            A title for the plot.
        x_lim : tuple of float, optional
            Pair of axis limits ``(x_min, x_max)``.
        y_lim : tuple of float, optional
            Pair of axis limits ``(y_min, y_max)``. By default the axis starts
            at 0 and covers the whole energy range of the file, not only
            ``x_lim``.
        col : str, tuple or list, optional
            One colour for every element, or a list of colours (strings or
            3-tuples ``(R, G, B)``) with one entry per tensor element. A shorter
            list falls back to the matplotlib colour cycle.
        legend : bool, optional
            Show the legend (default ``True``).

        Returns
        -------
        None
            Shows one figure per temperature.
        """
        from .graphics.plot_functions import plot_tensor
        from .inputs.read_pao_output import read_transport_PAO

        x_label = 'Energy (eV)'
        y_label = r'Conductivity $(\Omega\, m\, s)^{-1}$'
        enes, temps, tensors = read_transport_PAO(fname)
        for i, temp in enumerate(temps):
            ttitle = title + ', T={}'.format(temp)
            plot_tensor(
                enes,
                tensors[i],
                t_ele,
                ttitle,
                x_lim,
                y_lim,
                x_label,
                y_label,
                col,
                legend,
                min_zero=True,
            )

    def plot_seebeck(
        self,
        fname: str,
        t_ele: list[tuple[int, int]] = [(0, 0), (1, 1), (2, 2)],
        vE: float | None = None,
        title: str = 'Seebeck vs Energy',
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
        col: Color | list[Color] = [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        legend: bool = True,
    ) -> None:
        """Plot the Seebeck coefficient against the chemical potential.

        Each temperature in the file gets its own figure, over the full energy
        range of the file, with ``', T=<temperature>'`` appended to ``title``.
        The values are converted from V/K to :math:`\\mu`\\ V/K.

        Parameters
        ----------
        fname : str
            File name (including relative path) of a ``Seebeck*_0.dat`` file of
            ``transport()``: temperature (K), chemical potential (eV) and the
            ``xx, yy, zz, xy, xz, yz`` components in V/K.
        t_ele : list of tuple of int, optional
            Tensor elements as index pairs (e.g. ``(1, 2)`` for ``yz``). The
            default plots the 3 diagonal elements separately. Providing an
            empty list plots the average of the diagonal components.
        vE : float, optional
            Not used: the coefficient is always plotted against energy.
        title : str, optional
            A title for the plot.
        x_lim : tuple of float, optional
            Pair of axis limits ``(x_min, x_max)``.
        y_lim : tuple of float, optional
            Pair of axis limits ``(y_min, y_max)`` in :math:`\\mu`\\ V/K. The
            automatic range covers the whole energy range of the file, not only
            ``x_lim``.
        col : str, tuple or list, optional
            One colour for every element, or a list of colours (strings or
            3-tuples ``(R, G, B)``) with one entry per tensor element. A shorter
            list falls back to the matplotlib colour cycle.
        legend : bool, optional
            Show the legend (default ``True``).

        Returns
        -------
        None
            Shows one figure per temperature.
        """
        from .graphics.plot_functions import plot_tensor
        from .inputs.read_pao_output import read_transport_PAO

        x_label = 'Energy (eV)'
        y_label = r'Seebeck ($\mu$V/K)'
        enes, temps, tensors = read_transport_PAO(fname)
        for i, temp in enumerate(temps):
            ttitle = title + ', T={}'.format(temp)
            plot_tensor(
                enes,
                tensors[i] * 1e6,
                t_ele,
                ttitle,
                x_lim,
                y_lim,
                x_label,
                y_label,
                col,
                legend,
                min_zero=False,
            )

    def plot_shc(
        self,
        fname: str | list[str],
        title: str = 'SHC vs Energy',
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
        cols: Color | list[Color] | None = None,
        legend: bool = True,
        label: str | None = None,
    ) -> None:
        """Plot the spin Hall conductivity against the Fermi energy.

        Parameters
        ----------
        fname : str or list of str
            File name (including relative path) of a ``shcEf_*.dat`` file of
            ``spin_Hall()``, or a list of files plotted together with one legend
            entry each. The legend entry is the last four characters of the
            file name before the extension (``z_xy``).
        title : str, optional
            A title for the plot.
        x_lim : tuple of float, optional
            Pair of axis limits ``(x_min, x_max)``.
        y_lim : tuple of float, optional
            Pair of axis limits ``(y_min, y_max)``. For a single file the
            default is 1.1 times the range of the data.
        cols : str, tuple or list, optional
            One colour, or a list of colours (strings or 3-tuples ``(R, G, B)``)
            with one entry per file.
        legend : bool, optional
            Show the legend when several files are plotted (default ``True``).
        label : str, optional
            y-axis label; the default is for a bulk calculation,
            :math:`(\\Omega\\,\\mathrm{cm})^{-1}`. A ``twoD=True`` run writes the
            sheet conductivity in atomic units (:math:`\\hbar = e = 1`), where the
            quantum :math:`e/2\\pi` is :math:`1/2\\pi`.

        Returns
        -------
        None
            Shows the figure.
        """
        import numpy as np

        from .inputs.read_pao_output import read_dos_PAO

        x_label = 'Energy (eV)'
        y_label = r'$\sigma$ ($\Omega$cm)$^{-1}$' if label is None else label

        if isinstance(fname, str):
            from .graphics.plot_functions import plot_dos

            es, shc = read_dos_PAO(fname)
            if y_lim is None:
                y_lim = 1.1 * np.array([np.min(shc), np.max(shc)])
            plot_dos(es, shc, title, x_lim, y_lim, False, cols, x_label, y_label)

        elif isinstance(fname, list):
            from .graphics.plot_functions import plot_shc_tensor

            es = None
            data = []
            labels = []
            if not isinstance(cols, list):
                cols = [cols] * len(fname)
            for fn in fname:
                tag = fn.split('.')[-2][-4:]
                es, shc = read_dos_PAO(fn)
                data.append(shc)
                labels.append(tag)
            plot_shc_tensor(es, data, title, x_lim, y_lim, x_label, y_label, cols, labels, legend)

    def plot_ohc(
        self,
        fname: str | list[str],
        title: str = 'OHC vs Energy',
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
        cols: Color | list[Color] | None = None,
        legend: bool = True,
    ) -> None:
        """Plot the orbital Hall conductivity vs energy.

        PAOFLOW writes one ``ohcEf_{spol}_{ipol}{jpol}.dat`` file per requested
        tensor element (two columns: energy and conductivity), the orbital
        counterpart of the spin Hall files.  Pass a single file name to plot one
        component, or a list of file names to overlay several components.

        Parameters
        ----------
        fname : str or list of str
            File name or list of file names (including relative path).
        title : str, optional
            A title for the plot.
        x_lim : tuple of float, optional
            Pair of axis limits ``(x_min, x_max)``.
        y_lim : tuple of float, optional
            Pair of axis limits ``(y_min, y_max)``. For a single file the
            default is 1.1 times the range of the data.
        cols : str, tuple or list, optional
            A colour or 3-tuple ``(R, G, B)``, or a list of them, one for each
            tensor element.
        legend : bool, optional
            Show the legend when several components are plotted.

        Returns
        -------
        None
            Shows the figure.
        """
        import numpy as np

        from .inputs.read_pao_output import read_dos_PAO

        x_label = 'Energy (eV)'
        y_label = r'$\sigma^{L}$ ($\Omega$cm)$^{-1}$'

        if isinstance(fname, str):
            from .graphics.plot_functions import plot_dos

            es, ohc = read_dos_PAO(fname)
            if y_lim is None:
                y_lim = 1.1 * np.array([np.min(ohc), np.max(ohc)])
            plot_dos(es, ohc, title, x_lim, y_lim, False, cols, x_label, y_label)

        elif isinstance(fname, list):
            from .graphics.plot_functions import plot_shc_tensor

            es = None
            data = []
            labels = []
            if not isinstance(cols, list):
                cols = [cols] * len(fname)
            for fn in fname:
                tag = fn.split('.')[-2][-4:]
                es, ohc = read_dos_PAO(fn)
                data.append(ohc)
                labels.append(tag)
            plot_shc_tensor(es, data, title, x_lim, y_lim, x_label, y_label, cols, labels, legend)

    def plot_ahc(
        self,
        fname: str | list[str],
        title: str = 'AHC vs Energy',
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
        cols: Color | list[Color] | None = None,
        legend: bool = True,
    ) -> None:
        """Plot the anomalous Hall conductivity vs energy.

        PAOFLOW writes one ``ahcEf_{ipol}{jpol}.dat`` file per requested tensor
        element (two columns: energy and conductivity).  Pass a single file name
        to plot one component, or a list of file names to overlay several
        components on the same axes.

        Parameters
        ----------
        fname : str or list of str
            File name or list of file names (including relative path).
        title : str, optional
            A title for the plot.
        x_lim : tuple of float, optional
            Pair of axis limits ``(x_min, x_max)``.
        y_lim : tuple of float, optional
            Pair of axis limits ``(y_min, y_max)``. For a single file the
            default is 1.1 times the range of the data.
        cols : str, tuple or list, optional
            A colour or 3-tuple ``(R, G, B)``, or a list of them, one for each
            tensor element.
        legend : bool, optional
            Show the legend when several components are plotted.

        Returns
        -------
        None
            Shows the figure.
        """
        import numpy as np

        from .inputs.read_pao_output import read_dos_PAO

        x_label = 'Energy (eV)'
        y_label = r'$\sigma^{A}$ ($\Omega$cm)$^{-1}$'

        if isinstance(fname, str):
            from .graphics.plot_functions import plot_dos

            es, ahc = read_dos_PAO(fname)
            if y_lim is None:
                y_lim = 1.1 * np.array([np.min(ahc), np.max(ahc)])
            plot_dos(es, ahc, title, x_lim, y_lim, False, cols, x_label, y_label)

        elif isinstance(fname, list):
            from .graphics.plot_functions import plot_shc_tensor

            es = None
            data = []
            labels = []
            if not isinstance(cols, list):
                cols = [cols] * len(fname)
            for fn in fname:
                tag = fn.split('.')[-2][-2:]
                es, ahc = read_dos_PAO(fn)
                data.append(ahc)
                labels.append(tag)
            plot_shc_tensor(es, data, title, x_lim, y_lim, x_label, y_label, cols, labels, legend)

    def plot_dielectric(
        self,
        fname: str | list[str],
        title: str | None = None,
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
        cols: Color | list[Color] | None = None,
        labels: list[str] | None = None,
        legend: bool = True,
        legend_outside: bool = False,
    ) -> None:
        """Plot an optical / dielectric spectrum vs energy.

        The dielectric-tensor module writes two-column ``.dat`` files such as
        ``epsi_xx.dat`` (Im eps), ``epsr_xx.dat`` (Re eps), ``eels_xx.dat``,
        ``sigmar_xx.dat`` etc.  Pass a single file name to plot one spectrum,
        or a list of file names to overlay several on the same axes.

        Parameters
        ----------
        fname : str or list of str
            File name or list of file names (including relative path).
        title : str, optional
            A title for the plot (default ``'Dielectric function'``).
        x_lim : tuple of float, optional
            Pair of axis limits ``(x_min, x_max)``.
        y_lim : tuple of float, optional
            Pair of axis limits ``(y_min, y_max)``.
        cols : str, tuple or list, optional
            Color or list of colors, one per file.
        labels : list of str, optional
            Legend labels, one per file (defaults to the file name without its
            extension).
        legend : bool, optional
            Show the legend when several spectra are plotted.
        legend_outside : bool, optional
            Place the legend in a panel to the right of the axes instead of
            inside (avoids overlapping crowded curves).

        Returns
        -------
        None
            Shows the figure.
        """
        from .inputs.read_pao_output import read_dos_PAO

        x_label = 'Energy (eV)'
        y_label = 'Dielectric response'

        if isinstance(fname, str):
            from .graphics.plot_functions import plot_dos

            if title is None:
                title = 'Dielectric function'
            es, eps = read_dos_PAO(fname)
            plot_dos(es, eps, title, x_lim, y_lim, False, cols, x_label, y_label)

        elif isinstance(fname, list):
            from .graphics.plot_functions import plot_shc_tensor

            if title is None:
                title = 'Dielectric function'
            es = None
            data = []
            auto_labels = []
            if not isinstance(cols, list):
                cols = [cols] * len(fname)
            for fn in fname:
                tag = fn.split('/')[-1].split('.')[0]
                es, eps = read_dos_PAO(fn)
                data.append(eps)
                auto_labels.append(tag)
            if labels is None:
                labels = auto_labels
            plot_shc_tensor(
                es,
                data,
                title,
                x_lim,
                y_lim,
                x_label,
                y_label,
                cols,
                labels,
                legend,
                legend_outside,
            )

    # Property prefix -> (default legend label, y-axis label) for plot_optical.
    OPTICAL_PROPERTIES = {
        'epsi': (r'$\varepsilon_2$', r'$\varepsilon_2$ (Im $\varepsilon$)'),
        'epsr': (r'$\varepsilon_1$', r'$\varepsilon_1$ (Re $\varepsilon$)'),
        'ieps': (r'$\varepsilon(i\omega)$', r'$\varepsilon(i\omega)$'),
        'eels': ('EELS', r'$-$Im $\varepsilon^{-1}$'),
        'nref': ('n', 'Refractive index'),
        'kref': (r'$\kappa$', 'Extinction coefficient'),
        'alpha': (r'$\alpha$', r'Absorption $\alpha$ (1/m)'),
        'refl': ('Reflectivity', 'Reflectivity'),
        'sigmar': (r'Re $\sigma$', r'$\sigma$ (S/m)'),
        'sigmai': (r'Im $\sigma$', r'$\sigma$ (S/m)'),
        'emish': ('Hemispherical emissivity', 'Emissivity'),
        'emist': ('Total emissivity', 'Emissivity'),
        'shift': (r'$\sigma_{\rm shift}$', r'Shift current ($\mu$A/V$^2$)'),
        'shiftm': (r'$\sigma_{\rm shift}^{M}$', r'Magnetic shift current ($\mu$A/V$^2$)'),
        'injectc': (r'$\eta_C$', r'Circular injection (A V$^{-2}$ s$^{-1}$)'),
        'injectm': (r'$\eta_M$', r'Magnetic injection (A V$^{-2}$ s$^{-1}$)'),
        'shgr': (r'Re $\chi^{(2)}$', r'$\chi^{(2)}$ (pm/V)'),
        'shgi': (r'Im $\chi^{(2)}$', r'$\chi^{(2)}$ (pm/V)'),
        'shga': (r'$|\chi^{(2)}|$', r'$\chi^{(2)}$ (pm/V)'),
    }

    def plot_optical(
        self,
        properties: str | list[str],
        path: str = '.',
        component: str | list[str] = 'xx',
        spin: int | None = None,
        title: str | None = None,
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
        cols: Color | list[Color] | None = None,
        labels: list[str] | None = None,
        legend: bool = True,
        y_label: str | None = None,
    ) -> None:
        """Plot a user-selected set of optical / emissivity spectra together.

        Resolves each requested property to the two-column ``.dat`` file
        written by :meth:`PAOFLOW.dielectric_tensor` and overlays the curves on
        one figure. This lets the user choose exactly which optical quantities
        to display instead of plotting raw file names.

        Recognized property keys (file prefixes):
        ``epsi``, ``epsr``, ``ieps``, ``eels``, ``nref``, ``kref``, ``alpha``,
        ``refl``, ``sigmar``, ``sigmai`` (dielectric / optical), plus the
        emissivity outputs ``emish`` (spectral hemispherical), ``emist`` (total
        hemispherical vs temperature) and the directional spectra written per
        incidence angle, e.g. ``refl_th30`` and ``emis_th30``.

        Parameters
        ----------
        properties : str or list of str
            One key or a list of keys to overlay.
        path : str, optional
            Directory containing the ``.dat`` files. Default ``'.'``.
        component : str or list of str, optional
            Diagonal tensor component to read (``'xx'``, ``'yy'``, ``'zz'``). A
            list/tuple overlays several components on one figure (e.g.
            ``['xx', 'yy', 'zz']``).
        spin : int, optional
            Spin channel (0 or 1) for spin-polarized runs; ``None`` for the
            spin-unpolarized files.
        title : str, optional
            A title for the plot (default ``'Optical properties'``).
        x_lim : tuple of float, optional
            Pair of axis limits ``(x_min, x_max)``.
        y_lim : tuple of float, optional
            Pair of axis limits ``(y_min, y_max)``.
        cols : str, tuple or list, optional
            Color or list of colors, one per property.
        labels : list of str, optional
            Legend labels, one per property (defaults to the property names).
        legend : bool, optional
            Show the legend.
        y_label : str, optional
            y-axis label. The default is the label of the property, in the bulk
            units of the file; a ``twoD=True`` run writes sheet units (for
            example nm µA/V² for the shift current), so pass the label here.

        Returns
        -------
        None
            Shows the figure.

        Raises
        ------
        Exception
            If ``labels`` does not have one entry per property.

        Notes
        -----
        ``emist`` (total emissivity) is tabulated versus temperature, not
        photon energy; do not overlay it with energy-axis spectra in the
        same call.
        """
        import os

        from .graphics.plot_functions import plot_optical
        from .inputs.read_pao_output import read_dos_PAO

        if isinstance(properties, str):
            properties = [properties]
        if labels is not None and len(labels) != len(properties):
            raise Exception('Must provide one label for each property')

        if isinstance(component, str):
            components = [component]
        else:
            components = list(component)
        multi_comp = len(components) > 1

        spin_tag = '' if spin is None else '_%d' % spin

        curves = []
        y_labels = set()
        temperature_axis = []
        for i, prop in enumerate(properties):
            meta = self.OPTICAL_PROPERTIES.get(prop)
            for comp in components:
                fn = os.path.join(path, '%s_%s%s.dat' % (prop, comp, spin_tag))
                x, y = read_dos_PAO(fn)
                if labels is not None:
                    label = labels[i]
                elif meta is not None:
                    label = meta[0]
                else:
                    label = prop
                if multi_comp:
                    label = '%s (%s)' % (label, comp)
                curves.append((x, y, label))
                if meta is not None:
                    y_labels.add(meta[1])
                temperature_axis.append(prop.startswith('emist'))

        if all(temperature_axis) and temperature_axis:
            x_label = 'Temperature (K)'
        else:
            x_label = 'Energy (eV)'
            if any(temperature_axis):
                print(
                    'Warning: mixing total emissivity (vs temperature) with '
                    'energy-axis spectra; x-axis is labeled as energy.'
                )

        if y_label is None:
            y_label = y_labels.pop() if len(y_labels) == 1 else 'Optical response'

        if title is None:
            title = 'Optical properties'

        plot_optical(curves, title, x_lim, y_lim, x_label, y_label, cols, legend)

    def optical_color(
        self,
        path: str = '.',
        component: str = 'avg',
        spin: int | None = None,
        illuminant: str | float = 'E',
        title: str | None = None,
        label: str | None = None,
        show: bool = True,
    ) -> tuple[tuple[float, float, float], tuple[int, int, int], str]:
        """Derive the perceived visible color (sRGB) of a material from its normal-incidence reflectivity.

        The reflectivity spectra ``refl_<component>.dat`` written by
        :meth:`PAOFLOW.dielectric_tensor` are passed through the CIE 1931
        color-matching functions under the chosen illuminant to obtain a CIE
        XYZ tristimulus, which is converted to an sRGB color and displayed as
        a color swatch.

        Parameters
        ----------
        path : str, optional
            Directory containing the ``refl_*.dat`` files. Default ``'.'``.
        component : str, optional
            Diagonal tensor component (``'xx'``, ``'yy'``, ``'zz'``) or
            ``'avg'`` to average the available diagonal components (default).
        spin : int, optional
            Spin channel (0 or 1) for spin-polarized runs; ``None`` for the
            spin-unpolarized files.
        illuminant : str or float, optional
            ``'E'`` equal-energy (intrinsic color, default), ``'D65'``
            daylight, or a blackbody temperature in kelvin.
        title : str, optional
            A title for the swatch figure.
        label : str, optional
            Optional text drawn on the swatch (e.g. the material).
        show : bool, optional
            Render the swatch (set ``False`` to only return the color).

        Returns
        -------
        tuple
            ``(rgb01, rgb255, hexstr)``: sRGB in [0, 1], in [0, 255], and as a
            ``'#rrggbb'`` string. Also prints the color.

        Raises
        ------
        Exception
            If no ``refl_*.dat`` file of the requested component is found.
        """
        import os

        from .graphics.color import reflectance_to_srgb, visible_grid_covered
        from .inputs.read_pao_output import read_dos_PAO

        spin_tag = '' if spin is None else '_%d' % spin
        if component == 'avg':
            comps = ['xx', 'yy', 'zz']
        else:
            comps = [component]

        ene = None
        refl_sum = None
        nfound = 0
        for comp in comps:
            fn = os.path.join(path, 'refl_%s%s.dat' % (comp, spin_tag))
            if not os.path.isfile(fn):
                continue
            x, y = read_dos_PAO(fn)
            import numpy as np

            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            if refl_sum is None:
                ene = x
                refl_sum = y
            else:
                refl_sum = refl_sum + y
            nfound += 1

        if nfound == 0:
            raise Exception(
                'No reflectivity files found for component %r in %s' % (component, path)
            )

        refl = refl_sum / nfound

        if not visible_grid_covered(ene):
            print(
                'Warning: the reflectivity energy grid does not span the full '
                'visible range (~1.59-3.26 eV); the derived color is biased. '
                'Re-run the optical calculation with emax >= ~3.3 eV.'
            )

        rgb01, rgb255, hexstr = reflectance_to_srgb(ene, refl, illuminant)
        print(
            'Perceived color (sRGB): RGB={}  hex={}'.format(tuple(int(c) for c in rgb255), hexstr)
        )

        if show:
            from .graphics.plot_functions import plot_color_swatch

            plot_color_swatch(rgb01, hexstr=hexstr, title=title, label=label)

        return rgb01, rgb255, hexstr

    def plot_phonons(
        self,
        band_file: str,
        dos_file: str | None = None,
        labels_file: str | None = None,
        title: str | None = None,
        y_lim: tuple[float, float] | None = None,
        col: Color = 'black',
        units: str = 'THz',
        filename: str | None = None,
    ) -> None:
        """Plot a phonon dispersion (and optional DOS) from PAOFLOW output files.

        Parameters
        ----------
        band_file : str
            Path to ``<fname>_band.dat`` (distance + frequencies).
        dos_file : str, optional
            Path to ``<fname>_dos.dat`` (frequency + DOS).
        labels_file : str, optional
            Path to ``<fname>_band.labels`` produced alongside the band file;
            supplies the high-symmetry tick marks.
        title : str, optional
            A title for the plot.
        y_lim : tuple of float, optional
            Pair of frequency-axis limits ``(y_min, y_max)``.
        col : str or tuple of float, optional
            Line colour recognised by matplotlib.
        units : str, optional
            Frequency unit string for the axis label.
        filename : str, optional
            If given, save the figure to this path.

        Returns
        -------
        None
            Shows the figure; with ``filename`` it is also saved there.
        """
        import numpy as np

        from .graphics.plot_functions import plot_phonons

        band = np.loadtxt(band_file)
        distances = band[:, 0]
        frequencies = band[:, 1:]

        ticks = None
        if labels_file is not None:
            positions, labels = [], []
            with open(labels_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(None, 1)
                    positions.append(float(parts[0]))
                    labels.append(parts[1] if len(parts) > 1 else '')
            ticks = (positions, labels)

        dos = None
        if dos_file is not None:
            d = np.loadtxt(dos_file)
            dos = (d[:, 0], d[:, 1])

        plot_phonons(
            distances,
            frequencies,
            ticks=ticks,
            dos=dos,
            title=title,
            y_lim=y_lim,
            col=col,
            units=units,
            filename=filename,
        )

    def plot_gruneisen_band(
        self,
        band_file: str,
        labels_file: str | None = None,
        title: str | None = None,
        y_lim: tuple[float, float] | None = None,
        col: Color = 'black',
        filename: str | None = None,
    ) -> None:
        """Plot the mode Grueneisen dispersion from PAOFLOW output files.

        Parameters
        ----------
        band_file : str
            Path to ``<fname>_gruneisen_band.dat`` (distance + per-branch
            [gruneisen, frequency] columns).
        labels_file : str, optional
            Path to ``<fname>_gruneisen_band.labels`` supplying the
            high-symmetry tick marks.
        title : str, optional
            A title for the plot.
        y_lim : tuple of float, optional
            Pair of Grueneisen-axis limits ``(y_min, y_max)``.
        col : str or tuple of float, optional
            Line colour recognised by matplotlib.
        filename : str, optional
            If given, save the figure to this path.

        Returns
        -------
        None
            Shows the figure; with ``filename`` it is also saved there.
        """
        import numpy as np

        from .graphics.plot_functions import plot_gruneisen_band

        band = np.loadtxt(band_file)
        distances = band[:, 0]
        # Columns after the distance alternate gruneisen, frequency per branch.
        gruneisen = band[:, 1::2]

        ticks = None
        if labels_file is not None:
            positions, labels = [], []
            with open(labels_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(None, 1)
                    positions.append(float(parts[0]))
                    labels.append(parts[1] if len(parts) > 1 else '')
            ticks = (positions, labels)

        plot_gruneisen_band(
            distances,
            gruneisen,
            ticks=ticks,
            title=title,
            y_lim=y_lim,
            col=col,
            filename=filename,
        )

    def plot_phonon_thermal(
        self,
        thermal_file: str,
        title: str | None = None,
        filename: str | None = None,
    ) -> None:
        """Plot harmonic thermal properties from a PAOFLOW output file.

        Parameters
        ----------
        thermal_file : str
            Path to ``<fname>_thermal.dat`` with columns temperature (K), free
            energy (kJ/mol), entropy (J/K/mol) and constant-volume heat
            capacity (J/K/mol).
        title : str, optional
            A title for the plot.
        filename : str, optional
            If given, save the figure to this path.

        Returns
        -------
        None
            Shows the figure; with ``filename`` it is also saved there.
        """
        import numpy as np

        from .graphics.plot_functions import plot_phonon_thermal

        data = np.loadtxt(thermal_file)
        plot_phonon_thermal(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            data[:, 3],
            title=title,
            filename=filename,
        )

    def plot_ir_spectrum(
        self,
        spectrum_file: str,
        modes_file: str | None = None,
        title: str | None = None,
        x_lim: tuple[float, float] | None = None,
        col: Color = 'black',
        units: str = 'cm-1',
        filename: str | None = None,
    ) -> None:
        """Plot an infrared spectrum from PAOFLOW output files.

        Parameters
        ----------
        spectrum_file : str
            Path to ``<fname>_ir_spectrum.dat`` (frequency + broadened
            intensity).
        modes_file : str, optional
            Path to ``<fname>_ir_modes.dat``; when given the discrete mode
            intensities are drawn as vertical sticks.
        title : str, optional
            A title for the plot.
        x_lim : tuple of float, optional
            Pair of frequency-axis limits ``(x_min, x_max)``.
        col : str or tuple of float, optional
            Line colour recognised by matplotlib.
        units : str, optional
            Frequency unit string for the axis label.
        filename : str, optional
            If given, save the figure to this path.

        Returns
        -------
        None
            Shows the figure; with ``filename`` it is also saved there.
        """
        import numpy as np

        from .graphics.plot_functions import plot_ir_spectrum

        spec = np.loadtxt(spectrum_file)
        frequencies = spec[:, 0]
        intensities = spec[:, 1]

        modes = None
        if modes_file is not None:
            m = np.loadtxt(modes_file, usecols=(1, 2))
            modes = (m[:, 0], m[:, 1])

        plot_ir_spectrum(
            frequencies,
            intensities,
            modes=modes,
            title=title,
            x_lim=x_lim,
            col=col,
            units=units,
            filename=filename,
        )

    def plot_qha(
        self,
        volume_file: str | None = None,
        thermal_expansion_file: str | None = None,
        bulk_modulus_file: str | None = None,
        heat_capacity_file: str | None = None,
        gruneisen_file: str | None = None,
        ev_file: str | None = None,
        title: str | None = None,
        filename: str | None = None,
    ) -> None:
        """Plot quasi-harmonic quantities from PAOFLOW output files.

        Each argument is the path to one ``qha_*.dat`` table written by
        :meth:`PAOFLOW.PAOFLOW.quasi_harmonic`.  Missing / ``None`` files are
        skipped, so a subset can be plotted.  All temperature-resolved tables
        share the same temperature grid (column 0).

        Parameters
        ----------
        volume_file : str, optional
            ``<fname>_volume.dat`` (T, V in Å³).
        thermal_expansion_file : str, optional
            ``<fname>_thermal_expansion.dat`` (T, alpha in 1/K).
        bulk_modulus_file : str, optional
            ``<fname>_bulk_modulus.dat`` (T, B in GPa).
        heat_capacity_file : str, optional
            ``<fname>_heat_capacity.dat`` (T, Cp in J/K/mol).
        gruneisen_file : str, optional
            ``<fname>_gruneisen.dat`` (T, gamma).
        ev_file : str, optional
            ``<fname>_ev.dat`` (volume in Å³, static energy eV).
        title : str, optional
            A title for the figure.
        filename : str, optional
            If given, save the figure to this path.

        Returns
        -------
        None
            Shows the figure, also saved to ``filename`` if given; prints a note and
            returns if none of the files exists.
        """
        import os

        import numpy as np

        from .graphics.plot_functions import plot_qha

        def _load(path):
            if path is None or not os.path.isfile(path):
                return None
            data = np.loadtxt(path)
            if data.ndim != 2 or data.shape[0] == 0:
                return None
            return data

        temps = None
        kwargs = {}
        for key, path in (
            ('volume', volume_file),
            ('thermal_expansion', thermal_expansion_file),
            ('bulk_modulus', bulk_modulus_file),
            ('heat_capacity', heat_capacity_file),
            ('gruneisen', gruneisen_file),
        ):
            data = _load(path)
            if data is None:
                continue
            if temps is None:
                temps = data[:, 0]
            kwargs[key] = data[:, 1]

        ev = _load(ev_file)
        if ev is not None:
            kwargs['ev'] = (ev[:, 0], ev[:, 1])

        if temps is None and 'ev' not in kwargs:
            print('  (no QHA output files found to plot)')
            return

        plot_qha(
            temps if temps is not None else np.array([0.0]),
            title=title,
            filename=filename,
            **kwargs,
        )

    def plot_raman_spectrum(
        self,
        spectrum_file: str,
        modes_file: str | None = None,
        title: str | None = None,
        x_lim: tuple[float, float] | None = None,
        col: Color = 'black',
        units: str = 'cm-1',
        filename: str | None = None,
    ) -> None:
        """Plot a Raman spectrum from PAOFLOW output files.

        Parameters
        ----------
        spectrum_file : str
            Path to ``<fname>_raman_spectrum.dat`` (frequency + broadened
            intensity).
        modes_file : str, optional
            Path to ``<fname>_raman_modes.dat``; when given the discrete mode
            intensities are drawn as vertical sticks.
        title : str, optional
            A title for the plot.
        x_lim : tuple of float, optional
            Pair of frequency-axis limits ``(x_min, x_max)``.
        col : str or tuple of float, optional
            Line colour recognised by matplotlib.
        units : str, optional
            Frequency unit string for the axis label.
        filename : str, optional
            If given, save the figure to this path.

        Returns
        -------
        None
            Shows the figure; with ``filename`` it is also saved there.
        """
        import numpy as np

        from .graphics.plot_functions import plot_raman_spectrum

        spec = np.loadtxt(spectrum_file)
        frequencies = spec[:, 0]
        intensities = spec[:, 1]

        modes = None
        if modes_file is not None:
            # columns: mode, frequency, activity, activity_norm, intensity, ...
            m = np.loadtxt(modes_file, usecols=(1, 4))
            modes = (m[:, 0], m[:, 1])

        plot_raman_spectrum(
            frequencies,
            intensities,
            modes=modes,
            title=title,
            x_lim=x_lim,
            col=col,
            units=units,
            filename=filename,
        )

    def plot_migdal_eliashberg(
        self,
        npz_file: str,
        temps: list[float] | None = None,
        title: str | None = None,
        filename: str | None = None,
        real_axis_max_mev: float = 60.0,
    ) -> None:
        """Plot the isotropic Migdal-Eliashberg results (gap, Z, real axis, Delta(T), Tc).

        Parameters
        ----------
        npz_file : str
            ``migdal_eliashberg.npz`` written by
            :func:`PAOFLOW.elphon.migdal_eliashberg.write_me_outputs`.
        temps : list of float, optional
            Temperatures (K) shown in the frequency-resolved panels; defaults to
            up to four temperatures with a non-zero gap.
        title : str, optional
            A title for the figure.
        filename : str, optional
            If given, the figure is also saved to this path.
        real_axis_max_mev : float, optional
            Upper frequency (meV) of the real-axis panels (default 60), capped
            at ``wscut``.

        Returns
        -------
        None
            Shows the figure (:func:`PAOFLOW.graphics.plot_functions.plot_migdal_eliashberg`).
        """
        import numpy as np

        from .graphics.plot_functions import plot_migdal_eliashberg

        with np.load(npz_file) as data:
            plot_migdal_eliashberg(
                dict(data),
                temps=temps,
                title=title,
                filename=filename,
                real_axis_max_mev=real_axis_max_mev,
            )

    def plot_phonon_assisted_absorption(
        self,
        npz_file: str,
        eta_ev: float = 0.05,
        temperature: float | None = None,
        epw_indabs_file: str | None = None,
        title: str | None = None,
        filename: str | None = None,
    ) -> None:
        """Plot the phonon-assisted and direct optical absorption.

        Parameters
        ----------
        npz_file : str
            ``absorption.npz`` written by
            :func:`PAOFLOW.elphon.phonon_assisted_absorption.write_absorption_outputs`.
        eta_ev : float, optional
            Intermediate-state broadening (eV) shown (closest computed value).
        temperature : float, optional
            Temperature (K) of the dielectric-function panel (default: the first).
        epw_indabs_file : str, optional
            EPW ``epsilon2_indabs_<T>K.dat`` overlaid for comparison.
        title : str, optional
            A title for the figure.
        filename : str, optional
            If given, the figure is also saved to this path.

        Returns
        -------
        None
            Shows the figure
            (:func:`PAOFLOW.graphics.plot_functions.plot_phonon_assisted_absorption`).
        """
        import numpy as np

        from .graphics.plot_functions import plot_phonon_assisted_absorption

        with np.load(npz_file) as data:
            plot_phonon_assisted_absorption(
                dict(data), eta_ev=eta_ev, temperature=temperature,
                epw_indabs_file=epw_indabs_file, title=title, filename=filename,
            )  # fmt: skip

    def plot_thermal_emissivity(
        self,
        npz_file: str,
        temps: list[float] | None = None,
        title: str | None = None,
        filename: str | None = None,
    ) -> None:
        """Plot the spectral and total hemispherical emissivity versus temperature.

        Parameters
        ----------
        npz_file : str
            ``emissivity.npz`` written by
            :func:`PAOFLOW.elphon.phonon_assisted_absorption.write_emissivity_outputs`.
        temps : list of float, optional
            Temperatures (K) of the spectral panel (default: up to five).
        title : str, optional
            A title for the figure.
        filename : str, optional
            If given, the figure is also saved to this path.

        Returns
        -------
        None
            Shows the figure
            (:func:`PAOFLOW.graphics.plot_functions.plot_thermal_emissivity`).
        """
        import numpy as np

        from .graphics.plot_functions import plot_thermal_emissivity

        with np.load(npz_file) as data:
            plot_thermal_emissivity(dict(data), temps=temps, title=title, filename=filename)

    def plot_metal_transport(
        self,
        eliashberg_npz: str,
        resistivity_file: str,
        experiment_file: str | None = None,
        title: str | None = None,
        filename: str | None = None,
    ) -> None:
        """Plot alpha^2F, alpha^2F_tr and the resistivity of a metal.

        Parameters
        ----------
        eliashberg_npz : str
            ``eliashberg.npz`` of ``main.elphon.py --transport``.
        resistivity_file : str
            ``resistivity.dat`` of ``transport.elphon.py`` (T, total and per-channel
            resistivity in micro-ohm cm; channel names in the header).
        experiment_file : str, optional
            Two columns: temperature (K), measured resistivity (micro-ohm cm).
        title : str, optional
            A title for the figure.
        filename : str, optional
            If given, the figure is also saved to this path.

        Returns
        -------
        None
            Shows the figure
            (:func:`PAOFLOW.graphics.plot_functions.plot_metal_transport`).
        """
        import numpy as np

        from .graphics.plot_functions import plot_metal_transport

        names = None
        with open(resistivity_file, encoding='utf-8') as handle:
            header = handle.readline()
        if ':' in header:
            names = [n.strip() for n in header.split(':', 1)[1].split(',') if n.strip()]
        experiment = np.loadtxt(experiment_file) if experiment_file else None
        with np.load(eliashberg_npz) as data:
            plot_metal_transport(
                dict(data), np.loadtxt(resistivity_file), names, experiment, title, filename
            )

    def plot_carrier_mobility(
        self,
        mobility_file: str,
        scattering_npz: str | None = None,
        temperature_k: float = 300.0,
        experiment_file: str | None = None,
        title: str | None = None,
        filename: str | None = None,
    ) -> None:
        """Plot the phonon-limited mobilities (SERTA/MRTA) and the scattering rates.

        Parameters
        ----------
        mobility_file : str
            ``mobility.dat`` of :func:`PAOFLOW.elphon.carrier_scattering.write_mobility_outputs`.
        scattering_npz : str, optional
            ``carrier_scattering.npz`` of the same run; adds the rates panel.
        temperature_k : float, optional
            Temperature of the rates panel.
        experiment_file : str, optional
            Three columns: temperature (K), measured electron and hole mobilities
            (cm^2/(V s)); NaN skips a value.
        title : str, optional
            A title for the figure.
        filename : str, optional
            If given, the figure is also saved to this path.

        Returns
        -------
        None
            Shows the figure
            (:func:`PAOFLOW.graphics.plot_functions.plot_carrier_mobility`).
        """
        import numpy as np

        from .graphics.plot_functions import plot_carrier_mobility

        experiment = np.loadtxt(experiment_file) if experiment_file else None
        scattering = None
        if scattering_npz is not None:
            with np.load(scattering_npz) as data:
                scattering = dict(data)
        plot_carrier_mobility(
            np.loadtxt(mobility_file), scattering, temperature_k, experiment, title, filename
        )

    def plot_migdal_eliashberg_aniso(
        self,
        npz_file: str,
        temps: list[float] | None = None,
        iso_npz_file: str | None = None,
        title: str | None = None,
        filename: str | None = None,
        distribution_scale: float = 3.0e-3,
    ) -> None:
        """Plot the anisotropic Migdal-Eliashberg results (gap distributions, DOS, lambda_nk).

        Parameters
        ----------
        npz_file : str
            ``migdal_eliashberg_aniso.npz`` written by
            :func:`PAOFLOW.elphon.anisotropic_eliashberg.write_me_aniso_outputs`.
        temps : list of float, optional
            Temperatures (K) shown in the distribution and quasiparticle-DOS
            panels; defaults to up to four temperatures with a non-zero gap.
        iso_npz_file : str, optional
            Isotropic ``migdal_eliashberg.npz``, whose gap is drawn for comparison.
        title : str, optional
            A title for the figure.
        filename : str, optional
            If given, the figure is also saved to this path.
        distribution_scale : float, optional
            Horizontal scale (K per 1/eV^2) of the gap distributions drawn at
            each temperature (default 3e-3, as EPW tutorial 04).

        Returns
        -------
        None
            Shows the figure
            (:func:`PAOFLOW.graphics.plot_functions.plot_migdal_eliashberg_aniso`).
        """
        import numpy as np

        from .graphics.plot_functions import plot_migdal_eliashberg_aniso

        iso_data = None
        if iso_npz_file is not None:
            with np.load(iso_npz_file) as iso:
                iso_data = dict(iso)
        with np.load(npz_file) as data:
            plot_migdal_eliashberg_aniso(
                dict(data), temps=temps, iso_data=iso_data, title=title, filename=filename,
                distribution_scale=distribution_scale,
            )  # fmt: skip
