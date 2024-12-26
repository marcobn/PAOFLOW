
if __name__ == '__main__':
  from PAOFLOW import GPAO
  from glob import glob

  f_pdos_pat = '../example01/Reference/*_pdosdk*'
  fnames = sorted(glob(f_pdos_pat))

  pplt = GPAO.GPAO()

  # Display a default plot of all pdos files
  pplt.plot_pdos(fnames)

  # Create a table indicating how the pdos should be summed prior to plotting
  #  A list for s, p and d
  ind = 0
  psums = [[], [], []]

  # Create labels and colros to override defaults
  labels = ['s', 'p', 'd']
  cols = [[1,0,0], [.8,.2,0], [.6,.4,0]]

  # Iterate over the shell levels for each shell of each atom
  #   appending the pdos index to appropriate shell array
  for i,atom in enumerate(['Si', 'Si']):
    for j,orbital in enumerate(labels):
      for _ in range(2*j+1):
        psums[j].append(ind)
        ind += 1

  # Add the elements for summing total dos
  psums.append(list(range(ind)))
  cols.append([0,0,0])
  labels.append('Total')


  # Plot each shell for each atom
  pplt.plot_pdos(fnames, cols=cols, labels=labels, psum_inds=psums)

