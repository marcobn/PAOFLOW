
if __name__ == '__main__':
  from PAOFLOW import GPAO
  from glob import glob

  f_pdos_pat = '../example01/Reference/*_pdosdk*'

  fnames = sorted(glob(f_pdos_pat))

  # Create labels and colros, to override default indices and color wheel
  cols = []
  labels = []
  for i,a in enumerate(['Si', 'Si']):
    for j,o in enumerate(['s', 'p', 'd']):
      for _ in range(2*j+1):
        cols.append([i/2+.5, j/3+.2, 0])
        labels.append('{}{}_{}'.format(a,i,o))

  pplt = GPAO.GPAO()

  # Plot all dos files in the list fnames
  pplt.plot_pdos(fnames, cols=cols, labels=labels)
