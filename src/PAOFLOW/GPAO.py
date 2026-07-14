class GPAO:
    def __init__(self):
        pass

    def plot_dos(self, fname, title=None, x_lim=None, y_lim=None, vertical=False, col='black'):
        """
        Plot the density of states.

        Arguments:
          fname (str): File name (including relative path)
          title (str): A title for the plot
          x_lim (tuple): Pair of axis limits (x_min, x_max)
          y_lim (tuple): Pair of axis limits (y_min, y_max)
          vertical (bool): Set True to plot energy on the y-axis and elec/eV on the x-axis.
          col (str or tuple): A string recognized by matplotlib or a 3-tuple (R,G,B)
        """
        from .inputs.read_pao_output import read_dos_PAO
        from .graphics.plot_functions import plot_dos

        es, dos = read_dos_PAO(fname)

        plot_dos(es, dos, title, x_lim, y_lim, vertical, col)

    def plot_pdos(
        self,
        fnames,
        title=None,
        x_lim=None,
        y_lim=None,
        vertical=False,
        cols=None,
        labels=None,
        legend=True,
        psum_inds=None,
    ):
        """
        Plot the projected density of states.

        Arguments:
          fnames (list): List of file names (including relative path)
          title (str): A title for the plot
          x_lim (tuple): Pair of axis limits (x_min, x_max)
          y_lim (tuple): Pair of axis limits (y_min, y_max)
          vertical (bool): Set True to plot energy on the y-axis and elec/eV on the x-axis.
          cols (str or tuple): A list of string recognized by matplotlib or 3-tuples (R,G,B) with the same dimension as the number of files
          labels (list): List of strings with same dimension as the number of files
          psum_inds (list): A list of lists. Each inner list contains indices for the dos files elements to sum together. There is one line plotted for each inner list.
        """
        import numpy as np

        from .inputs.read_pao_output import read_dos_PAO
        from .graphics.plot_functions import plot_pdos

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
        outputdir,
        fname,
        sym_points=None,
        title=None,
        cbar_label=None,
        label=None,
        filename=None,
        y_lim=None,
        col='black',axes=None,bar=True):
        """
        Plot the band structure

        Arguments:
          fname (str): File name (including relative path)
          sym_points (str or tuple): File name for the kpath_points produced by PAOFLOW. Otherwise, provide a tuple of two lists. The first contains indices of the high sym points, the second contains labels for the high sym points.
          title (str): A title for the plot
          y_lim (tuple): Pair of axis limits (y_min, y_max)
          col (str or tuple): A string recognized by matplotlib or a 3-tuple (R,G,B)
        """
        from .inputs.read_pao_output import read_site_projected
        from .graphics.plot_functions import plot_weighted_bands

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
            col,axes,bar
        )

    def plot_bands(
        self,
        fnames,
        sym_points=None,
        title=None,
        label=None,
        labels=None,
        y_lim=None,
        cols=None,
        legend=True,ax=None
    ):
        """
        Plot one or more band structures for comparison.

        Arguments:
          fnames (str or list): File name or list of file names (including relative path).
          sym_points (str or tuple): File name for the kpath_points produced by PAOFLOW. Otherwise, provide a tuple of two lists. The first contains indices of the high sym points, the second contains labels for the high sym points.
          title (str): A title for the plot
          label (str): y-axis label
          labels (list): Legend labels, one per file (e.g. ['DFT','SK model'])
          y_lim (tuple): Pair of axis limits (y_min, y_max)
          cols (list or str): Color(s) for each dataset. A list of strings/3-tuples recognized by matplotlib.
          legend (bool): Show legend when labels are provided (default True)
        """
        from .inputs.read_pao_output import read_bands_PAO
        from .graphics.plot_functions import plot_bands

        if isinstance(fnames, str):
            fnames = [fnames]

        bands_list = [read_bands_PAO(fn) for fn in fnames]

        if sym_points is not None:
            if type(sym_points) is str:
                from .inputs.read_pao_output import read_band_path_PAO

                sym_points = read_band_path_PAO(sym_points)
        plot_bands(bands_list, sym_points, title, label, y_lim, cols, labels, legend,ax)

    def plot_berry(
        self,
        fname,
        sym_points=None,
        title=None,
        label=None,
        x_lim=None,
        y_lim=None,
        col='black',
        dos_ticks=False,
    ):
        """
        Plot the band structure

        Arguments:
          fname (str): File name (including relative path)
          sym_points (str or tuple): File name for the kpath_points produced by PAOFLOW. Otherwise, provide a tuple of two lists. The first contains indices of the high sym points, the second contains labels for the high sym points.
          title (str): A title for the plot
          y_lim (tuple): Pair of axis limits (y_min, y_max)
          col (str or tuple): A string recognized by matplotlib or a 3-tuple (R,G,B)
        """
        import numpy as np

        from .inputs.read_pao_output import read_dos_PAO
        from .graphics.plot_functions import plot_bands

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
        fn_dos,
        fn_bands,
        sym_points=None,
        title=None,
        band_label=None,
        x_lim=None,
        y_lim=None,
        col='black',
        dos_ticks=False,
    ):
        """
        Plot the density of states beside the band structure

        Arguments:
          fn_dos (str): File name for dos (including relative path)
          fn_bands (str): File name for bands (including relative path)
          sym_points (str or tuple): File name for the kpath_points produced by PAOFLOW. Otherwise, provide a tuple of two lists. The first contains indices of the high sym points, the second contains labels for the high sym points.
          title (str): A title for the plot
          x_lim (tuple): Pair of axis limits (x_min, x_max)
          y_lim (tuple): Pair of axis limits (y_min, y_max)
          col (str or tuple): A string recognized by matplotlib or a 3-tuple (R,G,B)
        """
        from .inputs.read_pao_output import read_bands_PAO, read_dos_PAO
        from .graphics.plot_functions import plot_dos_beside_bands

        if sym_points is not None:
            if type(sym_points) is str:
                from .inputs.read_pao_output import read_band_path_PAO

                sym_points = read_band_path_PAO(sym_points)

        bands = read_bands_PAO(fn_bands)
        es, dos = read_dos_PAO(fn_dos)

        plot_dos_beside_bands(
            es, dos, bands, sym_points, band_label, title, x_lim, y_lim, col, dos_ticks
        )

    def plot_berry_under_bands(
        self,
        fn_berry,
        fn_bands,
        sym_points=None,
        title=None,
        x_lim=None,
        y_lim=None,
        col='black',
        dos_ticks=False,
        band_label=None,
        berry_label=None,
    ):
        """
        Plot the berry phase below the band structure

        Arguments:
          fn_berry (str): File name for berry (including relative path)
          fn_bands (str): File name for bands (including relative path)
          sym_points (str or tuple): File name for the kpath_points produced by PAOFLOW. Otherwise, provide a tuple of two lists. The first contains indices of the high sym points, the second contains labels for the high sym points.
          title (str): A title for the plot
          x_lim (tuple): Pair of axis limits (x_min, x_max)
          y_lim (tuple): Pair of axis limits (y_min, y_max)
          col (str or tuple): A string recognized by matplotlib or a 3-tuple (R,G,B)
        """
        from .inputs.read_pao_output import read_bands_PAO, read_dos_PAO
        from .graphics.plot_functions import plot_berry_under_bands

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
        fname,
        t_ele=[(0, 0), (1, 1), (2, 2)],
        vE=None,
        title='Sigma vs Energy',
        x_lim=None,
        y_lim=None,
        col=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        legend=True,
    ):
        """
        Plot the electrical conductivity. If multiple Temperatures are computed the default behavior is to plot the full energy range for every temperature. If a conductivity vs temperature plot is desired, set vE to the energy at which conductivity should be collected for each temperature.

        Arguments:
          fname (str): File name (including relative path)
          t_ele (list): Tensor elements as tuple pairs (e.g. (1,2) for (y,z)). Default behavior is to plot the 3 diagonal elements seprately. Providing an empty list will average the diagonal components
          title (str): A title for the plot
          x_lim (tuple): Pair of axis limits (x_min, x_max)
          y_lim (tuple): Pair of axis limits (y_min, y_max)
          col (list): A list of 3-tuples (R,G,B), one for each tensor element.
          vE (float): Set to an energy to plot the conductivity vs temperature. The value of conductivity is taken at the provided energy for each temperature.
        """
        from .inputs.read_pao_output import read_transport_PAO
        from .graphics.plot_functions import plot_tensor

        x_label = '$Energy (eV)$'
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
        fname,
        t_ele=[(0, 0), (1, 1), (2, 2)],
        vE=None,
        title='Seebeck vs Energy',
        x_lim=None,
        y_lim=None,
        col=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        legend=True,
    ):
        """
        Plot the Seebeck coefficient. If multiple Temperatures are computed the default behavior is to plot the full energy range for every temperature. If a conductivity vs temperature plot is desired, set vE to the energy at which conductivity should be collected for each temperature.

        Arguments:
          fname (str): File name (including relative path)
          t_ele (list): Tensor elements as tuple pairs (e.g. (1,2) for (y,z)). Default behavior is to plot the 3 diagonal elements seprately. Providing an empty list will average the diagonal components
          vE (float): Set to an energy to plot the conductivity vs temperature. The value of conductivity is taken at the provided energy for each temperature.
          title (str): A title for the plot
          x_lim (tuple): Pair of axis limits (x_min, x_max)
          y_lim (tuple): Pair of axis limits (y_min, y_max)
          col (list): A list of 3-tuples (R,G,B), one for each tensor element.
        """
        from .inputs.read_pao_output import read_transport_PAO
        from .graphics.plot_functions import plot_tensor

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
        fname,
        title='SHC vs Energy',
        x_lim=None,
        y_lim=None,
        cols=None,
        legend=True,
    ):
        """
        Plot the Seebeck coefficient. If multiple Temperatures are computed the default behavior is to plot the full energy range for every temperature. If a conductivity vs temperature plot is desired, set vE to the energy at which conductivity should be collected for each temperature.

        Arguments:
          fname (str): File name (including relative path)
          title (str): A title for the plot
          x_lim (tuple): Pair of axis limits (x_min, x_max)
          y_lim (tuple): Pair of axis limits (y_min, y_max)
          cols (list): A list of 3-tuples (R,G,B), one for each tensor element.
        """
        import numpy as np

        from .inputs.read_pao_output import read_dos_PAO

        x_label = 'Energy (eV)'
        y_label = r'$\sigma$ ($\Omega$cm)$^{-1}$'

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

    def plot_ahc(
        self,
        fname,
        title='AHC vs Energy',
        x_lim=None,
        y_lim=None,
        cols=None,
        legend=True,
    ):
        """
        Plot the anomalous Hall conductivity vs energy.

        PAOFLOW writes one ``ahcEf_{ipol}{jpol}.dat`` file per requested tensor
        element (two columns: energy and conductivity).  Pass a single file name
        to plot one component, or a list of file names to overlay several
        components on the same axes.

        Arguments:
          fname (str or list): File name or list of file names (including relative path)
          title (str): A title for the plot
          x_lim (tuple): Pair of axis limits (x_min, x_max)
          y_lim (tuple): Pair of axis limits (y_min, y_max)
          cols (list): A 3-tuple (R,G,B) or list of them, one for each tensor element.
          legend (bool): Show the legend when several components are plotted.
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
        fname,
        title=None,
        x_lim=None,
        y_lim=None,
        cols=None,
        labels=None,
        legend=True,
    ):
        """
        Plot an optical / dielectric spectrum vs energy.

        The dielectric-tensor module writes two-column ``.dat`` files such as
        ``epsi_xx.dat`` (Im eps), ``epsr_xx.dat`` (Re eps), ``eels_xx.dat``,
        ``sigmar_xx.dat`` etc.  Pass a single file name to plot one spectrum,
        or a list of file names to overlay several on the same axes.

        Arguments:
          fname (str or list): File name or list of file names (including relative path)
          title (str): A title for the plot
          x_lim (tuple): Pair of axis limits (x_min, x_max)
          y_lim (tuple): Pair of axis limits (y_min, y_max)
          cols (str/tuple or list): Color or list of colors, one per file.
          labels (list): Legend labels, one per file (defaults to the file tags).
          legend (bool): Show the legend when several spectra are plotted.
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
            plot_shc_tensor(es, data, title, x_lim, y_lim, x_label, y_label, cols, labels, legend)
