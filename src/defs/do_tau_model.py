def get_tau (temp,data_controller, channels ):

  import numpy as np
  import scipy.constants as cp
  h = cp.hbar
  kb = cp.Boltzmann
  hw0 = 0.0136*1.60217662e-19 #joules
  rho = 8.24e3   #kg/m^3
  a = 6.34e-10 #metres
  Ea = 15*1.60217662e-19 #joules
  Eo = 26*1.60217662e-19
  temp *= 1.60217662e-19
  nI = 1e23 #no.of impuritites/m^3
  e = 1.60217662e-19
  n = 1e19 #electron density /m^3
  nd = 5e25 #doping in /m^3
  arry,attr = data_controller.data_dicts()
  snktot = arry['E_k'].shape[0]
  nspin = arry['E_k'].shape[2]
  bnd = attr['bnd']
  taus = []
  ml = 0.24
  mt = 0.02459   + ((8.659341e-5)*(temp/kb))
  DtK = 1e11*1.60217662e-19 #J/m
  #eps = 8.854187817e-12
  di_inf = 32.6*8.854187817e-12 #PbTe
  di_0 = 400*8.854187817e-12 #PbTe
  et =di_inf*8.854187817e-12 # dielectric constant*permitivtty of free space
  Zf = 6 #number of equivalent valleys
  v = 1770
  rv2 = 0.71e11
  ms = (ml*(mt**2))**(1./3)
  me = ms*9.10938e-31*np.ones((snktot,bnd,nspin), dtype=float) #effective mass tensor in kg 
  E = abs(1.60217662e-19*(arry['E_k'][:,:bnd]))

  for c in channels:

      #if c == 'impurity':
        #  qo = np.sqrt(((e**2)*n)/(et*temp))
        #  epso = ((h**2)*(qo**2))/(2*me)
        #  i_tau = (16*np.pi*np.sqrt(2*me)*(et**2)*(E**1.5))/((np.log(1+(4*E/epso))-((4*E/epso)/(1+(4*E/epso))))*(e**4)*nI)
        #  taus.append(i_tau)

      if c == 'accoustic':
          #a_tau = (2*np.pi*(h**4)*rho*v**2*((E/temp)**-0.5))/((np.power(2*me*temp,1.5)*Ea**2))
          a_tau = (2*np.pi*(h**4)*rv2*((E/temp)**-0.5))/((np.power(2*me*temp,1.5)*Ea**2))
          taus.append(a_tau)

      if c == 'optical':
         # Nop = (temp/hw0)-0.5
         # x = E/temp
         # xo = hw0/temp
         # X = x-xo
         # X[X<0] = 0
         # o_tau = (np.sqrt(2*temp)*np.pi*xo*(h**2)*rho)/((me**1.5)*(DtK**2)*(Nop*np.sqrt(x+xo)+(Nop+1)*np.sqrt(X)))#elastic +inelastic
          #o_tau_no_inels = (2/np.pi)*((hw0/Eo)**2)*(h**2*a**2*rho)*((E/temp)**-0.5)/((2*me*temp)**1.5)
          o_tau = (2/np.pi)*((hw0/Eo)**2)*(h**2*a**2*rho)*((E/temp)**-0.5)/((2*me*temp)**1.5)
          taus.append(o_tau)


      if c == 'polar optical':
          ro = ((di_inf*temp)/(4*np.pi*e**2*nd))**0.5
          deltap = (2*me*E*(2*ro)**2)/h**2
          F_scr = 1 - ((2/deltap)*np.log(deltap+1))+1/(deltap+1)
          F_scr_2 = 1
          di = ((1/di_inf)-(1/di_0))**-1
          po_tau = (di*h**2*np.power(E/(temp),(0.5)))/((np.power(2*me*temp,0.5))*F_scr*e**2)
          po_tau_no_scr = (di*h**2*np.power(E/(temp),(0.5)))/((np.power(2*me*temp,0.5))*F_scr_2*e**2)
          taus.append(po_tau)

      tau = np.zeros((snktot,bnd,nspin), dtype=float)
      for t in taus:
          tau += 1./t
      tau = 1/tau
  E_re = np.reshape(E/1.60217662e-19,(snktot,bnd)) #i do this because i am not able to saave 3d arrays to a file
  tau_new = np.reshape(tau,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  o_tau_new = np.reshape(o_tau,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  a_tau_new = np.reshape(a_tau,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  po_scr_tau_new = np.reshape(po_tau,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  po_no_scr_tau_new = np.reshape(po_tau_no_scr,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  np.savetxt('E.dat',E_re)
  np.savetxt('tau.dat',tau_new)
  np.savetxt('o_tau.dat',o_tau_new)
  np.savetxt('a_tau.dat',a_tau_new)
  np.savetxt('po_scr_tau.dat',po_scr_tau_new)
  np.savetxt('po_no_scr_tau.dat',po_no_scr_tau_new)
  return tau


