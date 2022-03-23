from matplotlib import pyplot as plt

def plot_dos ( es, dos, title, x_lim, y_lim, vertical, col ):
  '''
  '''

  fig = plt.figure()

  tit = 'DoS' if title is None else title
  fig.suptitle(tit)

  ax = fig.add_subplot(111)

  if vertical:
    ax.plot(dos, es, color=col)
  else:
    ax.plot(es, dos, color=col)
  if not x_lim is None:
    ax.set_xlim(*x_lim)
  elif vertical:
    ax.set_xlim(0, ax.get_xlim()[1])
  if not y_lim is None:
    ax.set_ylim(*y_lim)
  elif not vertical:
    ax.set_ylim(0, ax.get_ylim()[1])

  el = 'Energy (eV)'
  dl = 'electrons/eV'
  xl = el if not vertical else dl
  yl = dl if not vertical else el

  ax.set_xlabel(xl, fontsize=12)
  ax.set_ylabel(yl, fontsize=12)

  plt.show()


def plot_bands ( bands, sym_points, title, y_lim, col ):
  '''
  '''

  fig = plt.figure()

  tit = 'Band Structure' if title is None else title
  fig.suptitle(tit)

  ax = fig.add_subplot(111)

  for b in bands:
    ax.plot(b, color=col)
  if y_lim is None:
    y_lim = ax.get_ylim()
  ax.set_xlim(0, bands.shape[1])
  ax.set_ylim(*y_lim)
  if sym_points is None:
    ax.xaxis.set_visible(False)
  else:
    ax.set_xticks(sym_points[0])
    ax.set_xticklabels(sym_points[1])
    ax.vlines(sym_points[0], y_lim[0], y_lim[1], color='gray')
  ax.set_ylabel('Energy (eV)', fontsize=12)

  plt.show()


def plot_dos_beside_bands ( es, dos, bands, sym_points, title, x_lim, y_lim, col, dos_ticks ):
  '''
  '''
  from matplotlib import gridspec

  fig = plt.figure()
  spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[5,1])

  tit = 'Band Structure and DoS' if title is None else title
  fig.suptitle(tit)

  ax_b = fig.add_subplot(spec[0])
  ax_d = fig.add_subplot(spec[1])

  for b in bands:
    ax_b.plot(b, color=col)
  if y_lim is None:
    y_lim = ax_b.get_ylim()
  ax_b.set_xlim(0, bands.shape[1]-1)
  ax_b.set_ylim(*y_lim)
  if sym_points is None:
    ax_b.xaxis.set_visible(False)
  else:
    ax_b.set_xticks(sym_points[0])
    ax_b.set_xticklabels(sym_points[1])
    ax_b.vlines(sym_points[0], y_lim[0], y_lim[1], color='gray')
  ax_b.set_ylabel('Energy (eV)', fontsize=12)
  
  ax_d.plot(dos, es, color=col)
  if not x_lim is None:
    ax_d.set_xlim(*x_lim)
  else:
    ax_d.set_xlim(0, ax_d.get_xlim()[1])
  if not y_lim is None:
    ax_d.set_ylim(*y_lim)
  if not dos_ticks:
    ax_d.yaxis.set_visible(False)
    plt.tight_layout()

  plt.show()


def plot_tensor ( enes, tensors, eles, title, x_lim, y_lim, x_lab, y_lab, col, min_zero=False ):
  '''
  '''
  import numpy as np

  fig = plt.figure()

  if title is None:
    raise ValueError('\'title\' cannot be None in plot_tensor')
  fig.suptitle(title)

  ax = fig.add_subplot(111)

  if len(eles) == 0:
    tval = np.empty(tensors.shape[0], dtype=float)
    for i,v in enumerate(tensors):
      tval[i] = np.sum([v[j,j] for j in range(3)])
    col = col if type(col) is str else col[0]
    ax.plot(enes, tval, color=col)
  else:
    if len(col) >= len(eles):
      for i,e in enumerate(eles):
        ax.plot(enes, tensors[:,e[0],e[1]], color=col[i])
    else:
      for e in eles:
        ax.plot(enes, tensors[:,e[0],e[1]])

  if not x_lim is None:
    ax.set_xlim(*x_lim)
  if not y_lim is None:
    ax.set_ylim(*y_lim)
  elif min_zero:
    ax.set_ylim(0, ax.get_ylim()[1])

  ax.set_xlabel(x_lab)
  ax.set_ylabel(y_lab)

  plt.show()

  pass


def plot_tensor_vs_temperature( temps, temp, enes, tensors, eles, x_lim, y_lim, col ):
  '''
  '''
  pass

  fig.suptitle(tit)

  ax = fig.add_subplot(111)

  enes,temps,tensors = read_transport_PAO(fname)
  

  fig = plt.figure()

  tit = 'Band Structure and DoS' if title is None else title
  fig.suptitle(tit)

  ax = fig.add_subplot(111)


  print(temps, enes, tensors.shape)
