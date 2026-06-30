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
        from .graphics.plot_functions import plot_dos
        from .inputs.read_pao_output import read_dos_PAO

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
        outputdir,
        fname,
        sym_points=None,
        title=None,
        cbar_label=None,
        label=None,
        filename=None,
        y_lim=None,
        col='black',
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
        fnames,
        sym_points=None,
        title=None,
        label=None,
        labels=None,
        y_lim=None,
        cols=None,
        legend=True,
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
        from .graphics.plot_functions import plot_bands
        from .inputs.read_pao_output import read_bands_PAO

        if isinstance(fnames, str):
            fnames = [fnames]

        bands_list = [read_bands_PAO(fn) for fn in fnames]

        if sym_points is not None:
            if type(sym_points) is str:
                from .inputs.read_pao_output import read_band_path_PAO

                sym_points = read_band_path_PAO(sym_points)
        plot_bands(bands_list, sym_points, title, label, y_lim, cols, labels, legend)

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
        from .graphics.plot_functions import plot_dos_beside_bands
        from .inputs.read_pao_output import read_bands_PAO, read_dos_PAO

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
        from .graphics.plot_functions import plot_tensor
        from .inputs.read_pao_output import read_transport_PAO

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
        legend_outside=False,
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
          legend_outside (bool): Place the legend in a panel to the right of
            the axes instead of inside (avoids overlapping crowded curves).
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
    }

    def plot_optical(
        self,
        properties,
        path='.',
        component='xx',
        spin=None,
        title=None,
        x_lim=None,
        y_lim=None,
        cols=None,
        labels=None,
        legend=True,
    ):
        """
        Plot a user-selected set of optical / emissivity spectra together.

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

        Arguments:
          properties (str or list): One key or a list of keys to overlay.
          path (str): Directory containing the ``.dat`` files. Default '.'.
          component (str or list): Diagonal tensor component to read ('xx',
            'yy', 'zz'). A list/tuple overlays several components on one figure
            (e.g. ``['xx', 'yy', 'zz']``).
          spin (int): Spin channel (0 or 1) for spin-polarized runs; ``None``
            for the spin-unpolarized files.
          title (str): A title for the plot.
          x_lim (tuple): Pair of axis limits (x_min, x_max).
          y_lim (tuple): Pair of axis limits (y_min, y_max).
          cols (str/tuple or list): Color or list of colors, one per property.
          labels (list): Legend labels, one per property (defaults to the
            property names).
          legend (bool): Show the legend.

        Note:
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

        y_label = y_labels.pop() if len(y_labels) == 1 else 'Optical response'

        if title is None:
            title = 'Optical properties'

        plot_optical(curves, title, x_lim, y_lim, x_label, y_label, cols, legend)

    def optical_color(
        self,
        path='.',
        component='avg',
        spin=None,
        illuminant='E',
        title=None,
        label=None,
        show=True,
    ):
        """
        Derive the perceived visible color (sRGB) of a material from its
        normal-incidence reflectivity and display it as a color swatch.

        The reflectivity spectra ``refl_<component>.dat`` written by
        :meth:`PAOFLOW.dielectric_tensor` are passed through the CIE 1931
        color-matching functions under the chosen illuminant to obtain a CIE
        XYZ tristimulus, which is converted to an sRGB color.

        Arguments:
          path (str): Directory containing the ``refl_*.dat`` files. Default '.'.
          component (str): Diagonal tensor component ('xx', 'yy', 'zz') or
            'avg' to average the available diagonal components (default).
          spin (int): Spin channel (0 or 1) for spin-polarized runs; ``None``
            for the spin-unpolarized files.
          illuminant (str or float): 'E' equal-energy (intrinsic color, default),
            'D65' daylight, or a blackbody temperature in kelvin.
          title (str): A title for the swatch figure.
          label (str): Optional text drawn on the swatch (e.g. the material).
          show (bool): Render the swatch (set False to only return the color).

        Returns:
          tuple: ``(rgb01, rgb255, hexstr)`` -- sRGB in [0, 1], in [0, 255], and
          as a ``'#rrggbb'`` string.
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
        band_file,
        dos_file=None,
        labels_file=None,
        title=None,
        y_lim=None,
        col='black',
        units='THz',
        filename=None,
    ):
        """Plot a phonon dispersion (and optional DOS) from PAOFLOW output files.

        Arguments:
          band_file (str): Path to ``<fname>_band.dat`` (distance + frequencies).
          dos_file (str): Optional path to ``<fname>_dos.dat`` (frequency + DOS).
          labels_file (str): Optional path to ``<fname>_band.labels`` produced
            alongside the band file; supplies the high-symmetry tick marks.
          title (str): A title for the plot.
          y_lim (tuple): Pair of frequency-axis limits (y_min, y_max).
          col (str or tuple): Line colour recognised by matplotlib.
          units (str): Frequency unit string for the axis label.
          filename (str): If given, save the figure to this path.
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

    def plot_phonon_thermal(
        self,
        thermal_file,
        title=None,
        filename=None,
    ):
        """Plot harmonic thermal properties from a PAOFLOW output file.

        Arguments:
          thermal_file (str): Path to ``<fname>_thermal.dat`` with columns
            temperature (K), free energy (kJ/mol), entropy (J/K/mol) and
            constant-volume heat capacity (J/K/mol).
          title (str): A title for the plot.
          filename (str): If given, save the figure to this path.
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
        spectrum_file,
        modes_file=None,
        title=None,
        x_lim=None,
        col='black',
        units='cm-1',
        filename=None,
    ):
        """Plot an infrared spectrum from PAOFLOW output files.

        Arguments:
          spectrum_file (str): Path to ``<fname>_ir_spectrum.dat`` (frequency +
            broadened intensity).
          modes_file (str): Optional path to ``<fname>_ir_modes.dat``; when given
            the discrete mode intensities are drawn as vertical sticks.
          title (str): A title for the plot.
          x_lim (tuple): Pair of frequency-axis limits (x_min, x_max).
          col (str or tuple): Line colour recognised by matplotlib.
          units (str): Frequency unit string for the axis label.
          filename (str): If given, save the figure to this path.
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
