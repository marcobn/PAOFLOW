from PAOFLOW import PAOFLOW
import numpy as np

def main():

  model = {'label':'graphene', 't':2.7}
  paoflow = PAOFLOW.PAOFLOW(model=model, outputdir='./output', verbose=True)

  path = 'G-M-K-G'
  special_points = {'G':[0.0, 0.0, 0.0],'K':[2.0/3.0, 1.0/3.0, 0.0],'M':[1.0/2.0, 0.0/2.0, 0.0]}
  paoflow.bands(ibrav=4, nk=100, band_path=path, high_sym_points=special_points)

  paoflow.finish_execution()

if __name__== '__main__':
  main()
