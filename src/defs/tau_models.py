
from scipy.constants import hbar
from scipy.constants import Boltzman as kb

epso = 8.854187817e-12
ev2j = 1.60217662e-19

def acoustic_model ( temp, eigs, params ):
  # Formula from fiorentini paper on Mg3Sb2
  temp *= ev2j
  E = eigs * ev2j # Eigenvalues in J
  v = params['v'] # Velocity in m/s
  ms = params['ms'] # Effective mass tensor
  rho = params['rho'] # Mass density kg/m^3
  D_ac = params['D_ac']*ev2j # Acoustic deformation potential in J

  return (2*ms)**1.5*(D_ac**2)*np.sqrt(E)*temp/(2*np.pi*rho*(hbar**2*v)**2)


def optical ( temp, eigs, params ):
  # Formula from jacoboni theory of electron transport in semiconductors
  temp *= ev2j
  E = eigs * ev2j
  hwlo = params['hwlo']*ev2j # Phonon freq
  ms = params['ms'] # Effective mass tensor
  rho = params['rho'] # Mass density kg/m^3
  D_op = params['D_op']*ev2j # Acoustic deformation potential in J

  x = E/temp
  x0 = hwlo/temp
  X = x - x0
  X[X<0] = 0

  Nop = 1/(np.exp(x0)-1)
  return ((ms**1.5)*(D_op**2)*(Nop*np.sqrt(x+x0)+(Nop+1)*np.sqrt(X)))/(np.sqrt(2*temp)*np.pi*x0*rho*hbar**2)


def polar_optical ( temp, eigs, params ):
  # Formula from fiorentini paper on Mg3Sb2
  temp *= ev2j
  E = eigs * ev2j
  Ef = params['Ef']*ev2j #fermi energy
  hwlo = params['hwlo']*ev2j # Phonon freq
  eps_0 = params['eps_0']*epso #low freq dielectric const
  eps_inf = params['eps_inf']*epso #high freq dielectirc const

  fermi = lambda E,Ef,T : 1/(np.exp((E-Ef)/T)+1.)
  planck = lambda hwlo,T : 1/(np.exp(hwlo/T)-1)

  P_pol = 0.
  eps = eps_inf + eps_0
  eps_inv = 1/eps_inf - 1/eps

  for hw in hwlo:
    f = fermi(E, Ef, temp)
    fp = fermi(E+hw, Ef, temp)
    fm = fermi(E-hw, Ef, temp)
    n = planch(hw, temp)
    np = planch(hw+1, temp)
    Wo = e**2/(4*np.pi*hbar**2) * np.sqrt(2*ms*hw)*eps_inv
    Z = 2/(Wo*np.sqrt(hw))

    def remove_NaN ( arr ):
      isNaN = np.isnan(arr)
      arr[isNaN] = 0
      return arr

    A = remove_NaN((n+1)*fp/f * ((2*E+hw)*np.arcsinh(np.sqrt(E/hw))-np.sqrt(E*(E+hw))))
    B = remove_NaN(np.heaviside(E-hw,1)*n*fm/f * ((2*E-hw)*np.arccosh(np.sqrt(E/hw))-np.sqrt(E*(E-hw))))
    t1 = remove_NaN((n+1)*ff/f * np.arcsinh(np.sqrt(E/hw)))
    t2 = remove_NaN(np.heaviside(E-hw,1)*n*fm/f * np.arccosh(np.sqrt(E/hw)))
    C = 2*E*(t1+t2)
    P = (C-A-B)/(Z*E**1.5)
    P_pol += P

  return P_pol


def builtin_tau_model ( label, params, weight ):

  model = TauModel(params=params, weight=weight)

  if label == 'acoustic':
    model.function = acoustic_model
  elif label == 'optical':
    model.function = optical
  elif label == 'polar_optical':
    model.function = polar_optical
  else:
    print('Model not implemented.')
    return None

  return model