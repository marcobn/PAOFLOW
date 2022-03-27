from matplotlib import pyplot as plt

def plot_dos ( es, dos, title, x_lim, y_lim, vertical, col, x_label=None, y_label=None ):
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

  el = 'Energy (eV)' if x_label is None else x_label
  dl = 'electrons/eV' if y_label is None else y_label
  xl = el if not vertical else dl
  yl = dl if not vertical else el

  ax.set_xlabel(xl, fontsize=12)
  ax.set_ylabel(yl, fontsize=12)

  plt.show()


def plot_pdos ( es, dos, title, x_lim, y_lim, vertical, cols, labels, legend ):
  '''
  '''
  import numpy as np

  if labels is None:
    labels = list(range(len(dos)))
  else:
    if len(labels) != len(dos):
      raise Exception('Must provide one label for each pdos file')

  if cols is None or isinstance(cols, str):
    cols = [cols] * len(dos)
  else:
    cols = np.array(cols)
    cs = cols.shape
    if len(cs) == 1:
      cols = [cols] * len(dos)
    elif cs[0] != len(labels):
      raise Exception('Must provide one color for each pdos file')

  fig = plt.figure()

  tit = 'PDoS' if title is None else title
  fig.suptitle(tit)

  ax = fig.add_subplot(111)

  if vertical:
    for i,d in enumerate(dos):
      ax.plot(d, es, color=cols[i], label=labels[i])
  else:
    for i,d in enumerate(dos):
      ax.plot(es, d, color=cols[i], label=labels[i])
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

  if legend:
    ax.legend()

  plt.show()


def plot_bands ( bands, sym_points, title, label, y_lim, col ):
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
  if label is None:
    label = '$\epsilon$($\\bf{k}$)'
  ax.set_ylabel(label, fontsize=12)

  plt.show()


def plot_dos_beside_bands ( es, dos, bands, sym_points, title, band_label, x_lim, y_lim, col, dos_ticks ):
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
  if band_label is None:
    band_label = '$\epsilon$($\\bf{k}$)'
  ax_b.set_ylabel(band_label, fontsize=12)
  
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


def plot_berry_under_bands ( berry, bands, sym_points, title, band_label, berry_label, x_lim, y_lim, col, dos_ticks ):
  '''
  '''
  from matplotlib import gridspec

  fig = plt.figure()
  spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[3,1])

  tit = 'Band Structure and Berry Phase' if title is None else title
  fig.suptitle(tit)

  ax_ba = fig.add_subplot(spec[0])
  ax_be = fig.add_subplot(spec[1])

  ax_be.plot(berry, color=col)
  for b in bands:
    ax_ba.plot(b, color=col)
  if y_lim is None:
    y_lim = ax_ba.get_ylim()
  ax_be.set_xlim(0, bands.shape[1]-1)
  ax_ba.set_xlim(0, bands.shape[1]-1)
  ax_ba.set_ylim(*y_lim)
  if sym_points is None:
    ax_be.xaxis.set_visible(False)
    ax_ba.xaxis.set_visible(False)
  else:
    tlim = ax_be.get_ylim()
    ax_be.set_ylim(*tlim)
    ax_be.set_xticks(sym_points[0])
    ax_be.set_xticklabels(sym_points[1])
    ax_be.vlines(sym_points[0], tlim[0], tlim[1], color='gray')
    ax_ba.set_xticks(sym_points[0])
    ax_ba.set_xticklabels(sym_points[1])
    ax_ba.vlines(sym_points[0], y_lim[0], y_lim[1], color='gray')

  if berry_label is None:
    berry_label = '$\Omega$($\\bf{k}$)'
  if band_label is None:
    band_label = '$\epsilon$($\\bf{k}$)'
  ax_be.set_ylabel(berry_label, fontsize=12)
  ax_ba.set_ylabel(band_label, fontsize=12)

  plt.show()
  quit()
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


def plot_tensor ( enes, tensors, eles, title, x_lim, y_lim, x_lab, y_lab, col, legend, min_zero=False ):
  '''
  '''
  import numpy as np

  fig = plt.figure()

  if title is None:
    raise ValueError('\'title\' cannot be None in plot_tensor')
  fig.suptitle(title)

  ax = fig.add_subplot(111)

  lmap = {0:'x', 1:'y', 2:'z'}
  lkey = lambda a,b : lmap[a] + lmap[b]
  if len(eles) == 0:
    tval = np.empty(tensors.shape[0], dtype=float)
    for i,v in enumerate(tensors):
      tval[i] = np.sum([v[j,j] for j in range(3)])
    col = col if type(col) is str else col[0]
    ax.plot(enes, tval, color=col, label='Avg.')
  else:
    if type(col) is str:
      for e in eles:
        ax.plot(enes, tensors[:,e[0],e[1]], color=col, label=lkey(*e))
    elif len(col) >= len(eles):
      for i,e in enumerate(eles):
        ax.plot(enes, tensors[:,e[0],e[1]], color=col[i], label=lkey(*e))
    else:
      for e in eles:
        ax.plot(enes, tensors[:,e[0],e[1]], label=lkey(*e))

  if not x_lim is None:
    ax.set_xlim(*x_lim)
  if not y_lim is None:
    ax.set_ylim(*y_lim)
  elif min_zero:
    ax.set_ylim(0, ax.get_ylim()[1])

  ax.set_xlabel(x_lab)
  ax.set_ylabel(y_lab)

  if legend:
    ax.legend()

  plt.show()


def plot_shc_tensor ( enes, shc, title, x_lim, y_lim, x_lab, y_lab, cols, labels, legend ):
  '''
  '''
  import numpy as np

  fig = plt.figure()

  if title is None:
    raise ValueError('\'title\' cannot be None in plot_tensor')
  fig.suptitle(title)

  ax = fig.add_subplot(111)

  if len(cols) >= len(shc):
    for i,s in enumerate(shc):
      ax.plot(enes, s, color=cols[i], label=labels[i])
  else:
    raise Exception('Dimensions of colors are incorrect. Blame GPAO.py')

  if not x_lim is None:
    ax.set_xlim(*x_lim)
  if not y_lim is None:
    ax.set_ylim(*y_lim)

  ax.set_xlabel(x_lab)
  ax.set_ylabel(y_lab)

  if legend:
    ax.legend()

  plt.show()

