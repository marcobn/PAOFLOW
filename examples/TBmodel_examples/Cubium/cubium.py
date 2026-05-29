from PAOFLOW import PAOFLOW
import numpy as np

def main():

  model = {'label':'cubium', 't':1.0}
  paoflow = PAOFLOW.PAOFLOW(model=model, outputdir='./output', verbose=True)

  path = 'G-X-M-G-R'
  special_points = {'G':[0.0, 0.0, 0.0],'X':[0.0, 0.5, 0.0],'M':[0.5, 0.5, 0.0],'R':[0.5,0.5,0.5]}
  paoflow.bands(ibrav=1, nk=100, band_path=path, high_sym_points=special_points)


  paoflow.finish_execution()

if __name__== '__main__':
  main()
