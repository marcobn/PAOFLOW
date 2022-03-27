
if __name__ == '__main__':
  from PAOFLOW import GPAO
  from glob import glob

  f_pdos_pat = '../example01/Reference/*_pdosdk*'

  fnames = sorted(glob(f_pdos_pat))

  # Create labels and colros, to override default indices and color wheel
  ind = 0
  cols = []
  labels = []
  # Also create a table indicating how the pdos should be summed prior to plotting
  psums = []

  # Iterate over the shell levels for each shell of each atom
  for i,a in enumerate(['Si', 'Si']):
    for j,o in enumerate(['s', 'p', 'd']):
      ps = []
      for _ in range(2*j+1):
        ps.append(ind)
        ind += 1
      psums.append(ps)
      cols.append([i/2+.5, j/3+.2, 0])
      labels.append('{}{}_{}'.format(a,i,o))

  # Add the elements for summing total dos
  psums.append(list(range(ind)))
  cols.append([0,0,0])
  labels.append('Total')

  # Plot each shell for each atom
  pplt = GPAO.GPAO()
  pplt.plot_pdos(fnames, cols=cols, labels=labels, psum_inds=psums)
