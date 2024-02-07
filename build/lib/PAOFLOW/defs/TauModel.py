
class TauModel:
  '''
  Object containing a functional form for Tau, depending on termperature and an energy eigenvalue (both in units of eV).
  All TauModel's included in the scattering_channels arugment of PAOFLOW's transport routine
   are homically summed to obtain the effective relaxation time for each (temp,eig) combination.
  An additional argument, params (a dictionary), is required. Further parameters can be passed into    the routine by including them in this dictionary.
  '''

  def __init__ ( self, function=None, params=None, weight=1. ):
    '''
    Arguments:
      function (func): A function requiring 3 arguments, (temp,eig,params)
      params (dict): A dictionary with any additional constants of variables the function may require
      weight (float): A weight, w_i, incorporated in the harmonic sum of tau. 1/Tau = Sum(w_i/Tau_i)
    '''
    self.function = function
    self.params = params
    self.weight = weight

  def evaluate ( self, temp, eigs ):
    return self.function(temp, eigs, self.params)
