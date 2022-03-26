from matplotlib import pyplot as plt

class GPAO:

  def __init__ ( self ):
    pass

  def plot_dos ( self, fname, title=None, x_lim=None, y_lim=None, vertical=False, col='black' ):
    '''
      Plot the density of states.

      Arguments:
        fname (str): File name (including relative path)
        title (str): A title for the plot
        x_lim (tuple): Pair of axis limits (x_min, x_max)
        y_lim (tuple): Pair of axis limits (y_min, y_max)
        vertical (bool): Set True to plot energy on the y-axis and elec/eV on the x-axis.
        col (str or tuple): A string recognized by matplotlib or a 3-tuple (R,G,B)
    '''
    from .defs.read_pao_output import read_dos_PAO
    from .graphics.plot_functions import plot_dos

    es, dos = read_dos_PAO(fname)

    plot_dos(es, dos, title, x_lim, y_lim, vertical, col)


  def plot_pdos ( self, fnames, title=None, x_lim=None, y_lim=None, vertical=False, cols=None, labels=None, legend=True ):
    from .defs.read_pao_output import read_dos_PAO
    from .graphics.plot_functions import plot_pdos

    es = None
    dos = []
    for fn in fnames:
      es,ds = read_dos_PAO(fn)
      dos.append(ds)

    plot_pdos(es, dos, title, x_lim, y_lim, vertical, cols, labels, legend)


  def plot_bands ( self, fname, sym_points=None, title=None, y_lim=None, col='black' ):
    '''
      Plot the band structure

      Arguments:
        fname (str): File name (including relative path)
        sym_points (str or tuple): File name for the kpath_points produced by PAOFLOW. Otherwise, provide a tuple of two lists. The first contains indices of the high sym points, the second contains labels for the high sym points.
        title (str): A title for the plot
        y_lim (tuple): Pair of axis limits (y_min, y_max)
        col (str or tuple): A string recognized by matplotlib or a 3-tuple (R,G,B)
    '''
    from .graphics.plot_functions import plot_bands
    from .defs.read_pao_output import read_bands_PAO

    if not sym_points is None:
      if type(sym_points) is str:
        from .defs.read_pao_output import read_band_path_PAO
        sym_points = read_band_path_PAO(sym_points)
    plot_bands(read_bands_PAO(fname), sym_points, title, y_lim, col)


  def plot_dos_beside_bands ( self, fn_dos, fn_bands, sym_points=None, title=None, x_lim=None, y_lim=None, col='black', dos_ticks=False ):
    '''
      Plot the density of states beside the band structure

      Arguments:
        fn_dos (str): File name for dos (including relative path)
        fn_bands (str): File name for bands (including relative path)
        sym_points (str or tuple): File name for the kpath_points produced by PAOFLOW. Otherwise, provide a tuple of two lists. The first contains indices of the high sym points, the second contains labels for the high sym points.
        title (str): A title for the plot
        x_lim (tuple): Pair of axis limits (x_min, x_max)
        y_lim (tuple): Pair of axis limits (y_min, y_max)
        col (str or tuple): A string recognized by matplotlib or a 3-tuple (R,G,B)
        vT (float): Set to an energy to plot the conductivity vs temperature. The value of conductivity is taken at the provided energy for each temperature.
    '''
    from .defs.read_pao_output import read_dos_PAO,read_bands_PAO
    from .graphics.plot_functions import plot_dos_beside_bands

    if not sym_points is None:
      if type(sym_points) is str:
        from .defs.read_pao_output import read_band_path_PAO
        sym_points = read_band_path_PAO(sym_points)

    bands = read_bands_PAO(fn_bands)
    es, dos = read_dos_PAO(fn_dos)

    plot_dos_beside_bands(es, dos, bands, sym_points, title, x_lim, y_lim, col, dos_ticks)    


  def plot_electrical_conductivity ( self, fname, t_ele=[(0,0),(1,1),(2,2)], vE=None, title='Sigma vs Energy', x_lim=None, y_lim=None, col=[(1,0,0),(0,1,0),(0,0,1)], legend=True ):
    '''
      Plot the electrical conductivity. If multiple Temperatures are computed the default behavior is to plot the full energy range for every temperature. If a conductivity vs temperature plot is desired, set vE to the energy at which conductivity should be collected for each temperature.

      Arguments:
        fname (str): File name (including relative path)
        t_ele (list): Tensor elements as tuple pairs (e.g. (1,2) for (y,z)). Default behavior is to plot the 3 diagonal elements seprately. Providing an empty list will average the diagonal components
        title (str): A title for the plot
        x_lim (tuple): Pair of axis limits (x_min, x_max)
        y_lim (tuple): Pair of axis limits (y_min, y_max)
        col (list): A list of 3-tuples (R,G,B), one for each tensor element.
        vE (float): Set to an energy to plot the conductivity vs temperature. The value of conductivity is taken at the provided energy for each temperature.
    '''
    from .defs.read_pao_output import read_transport_PAO
    from .graphics.plot_functions import plot_tensor

    x_label = '$Energy (eV)$'
    y_label = 'Conductivity $(\Omega m s)^{-1}$'
    enes, temps, tensors = read_transport_PAO(fname)
    for i,temp in enumerate(temps):
      ttitle = title + ', T={}'.format(temp)
      plot_tensor(enes, tensors[i], t_ele, ttitle, x_lim, y_lim, x_label, y_label, col, legend, min_zero=True)


  def plot_seebeck ( self, fname, t_ele=[(0,0),(1,1),(2,2)], vE=None, title='Seebeck vs Energy', x_lim=None, y_lim=None, col=[(1,0,0),(0,1,0),(0,0,1)], legend=True ):
    '''
      Plot the Seebeck coefficient. If multiple Temperatures are computed the default behavior is to plot the full energy range for every temperature. If a conductivity vs temperature plot is desired, set vE to the energy at which conductivity should be collected for each temperature.

      Arguments:
        fname (str): File name (including relative path)
        t_ele (list): Tensor elements as tuple pairs (e.g. (1,2) for (y,z)). Default behavior is to plot the 3 diagonal elements seprately. Providing an empty list will average the diagonal components
        vE (float): Set to an energy to plot the conductivity vs temperature. The value of conductivity is taken at the provided energy for each temperature.
        title (str): A title for the plot
        x_lim (tuple): Pair of axis limits (x_min, x_max)
        y_lim (tuple): Pair of axis limits (y_min, y_max)
        col (list): A list of 3-tuples (R,G,B), one for each tensor element.
    '''
    from .defs.read_pao_output import read_transport_PAO
    from .graphics.plot_functions import plot_tensor

    x_label = 'Energy (eV)'
    y_label = 'Seebeck ($\mu$V/K)'
    enes, temps, tensors = read_transport_PAO(fname)
    for i,temp in enumerate(temps):
      ttitle = title + ', T={}'.format(temp)
      plot_tensor(enes, tensors[i]*1e6, t_ele, ttitle, x_lim, y_lim, x_label, y_label, col, legend, min_zero=False)

