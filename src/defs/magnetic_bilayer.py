
import numpy as np
from scipy import fftpack as FFT

def magnetic_bilayer ( data_controller, params ):

  # Numpy, cmath shorts
  cos = np.cos
  sin = np.sin
  pi = np.pi
  i = complex(0,1)
  sqrt = np.sqrt

  ## Parameters
  hbar = 1
  Es = 3.2
  Epx = -0.5
  Epy = -0.5
  Epz = -0.5

  ts = 0.5     # t_s
  tps = 0.5    # t_p,sigma
  tpp = 0.2    # t_p,pi
  tss = 0.1    # Is it same as t_s? this is between s orbitals not sigma,
             #as in eq.A8. I think it should be either 0.2 or 0.5.
  a = 1

  gsp = 0.5
  alpha_nm_so = 0.05

  Edxy = -0.5
  Edyz = -0.5
  Edzx = -0.5
  Edx2y2 = -0.5
  Edz2 = -0.5

  tds = 0.1
  tdp = 0.05
  tdd = 0.02

  J = 0.5
  alpha_fm_so = 0.1

  gpdp = 0.1
  gpds = 0.4

  arry,attr = data_controller.data_dicts()

  alat = attr['alat'] = 1
  arry['a_vectors'] = alat*np.array([[1,0,0],[0,1,0],[0,0,1]])
  arry['b_vectors'] = np.zeros((3,3),dtype=float)

  volume = np.dot(np.cross(arry['a_vectors'][0,:],arry['a_vectors'][1,:]),arry['a_vectors'][2,:])
  arry['b_vectors'][0,:] = (np.cross(arry['a_vectors'][1,:],arry['a_vectors'][2,:]))/volume
  arry['b_vectors'][1,:] = (np.cross(arry['a_vectors'][2,:],arry['a_vectors'][0,:]))/volume
  arry['b_vectors'][2,:] = (np.cross(arry['a_vectors'][0,:],arry['a_vectors'][1,:]))/volume 

  nawf = attr['nawf'] = 36
  nspin = attr['nspin'] = 1

  nk1 = attr['nk1'] = 11
  nk2 = attr['nk2'] = 11
  nk3 = attr['nk3'] = 1

  nx,ny = (nk1,nk2)
  kxGM = np.linspace(-pi,pi,nx)
  kyGM = np.linspace(-pi,pi,ny)

  #Diagonal components of H^NM_2d(0)
  def Es_k(kx,ky):
    return Es - 2*ts*(np.cos(kx*a) + np.cos(ky*a))

  def Epx_k(kx,ky):
    return Epx + 2*tps*cos(kx*a) - 2*tpp*cos(ky*a)

  def Epy_k(kx,ky):
    return Epy - 2*tpp*cos(kx*a) + 2*tps*cos(ky*a)

  def Epz_k(kx,ky):
    return Epz - 2*tpp*(cos(kx*a)+cos(ky*a))

  #Diagonal components of H^fm_2d(0)
  def Ed_xy(kx,ky):
    return Edxy + 2*tdp*(cos(kx*a) + cos(ky*a))
  def Ed_yz(kx,ky):
    return Edyz - 2*tdd*cos(kx*a) + 2*tdp*cos(ky*a)
  def Ed_zx(kx,ky):
    return Edzx + 2*tdp*cos(kx*a) - 2*tdd*cos(ky*a)
  def Ed_x2_y2(kx,ky):
    return Edx2y2 - ((3.0/2)*tds + (1.0/2)*tdd)*(cos(kx*a) + cos(ky*a))
  def Ed_z2(kx,ky):
    return Edz2 - ((1/2)*tds + (3/2)*tdd)*(cos(kx*a) + cos(ky*a))

  ### Hamiltonian for normal metal

  def H_nm_2d0(kx, ky):
    r1 = np.array([Es_k(kx, ky), 2*i*gsp*sin(kx*a), 2*i*gsp*sin(ky*a), 0])
    r2 = np.array([-2*i*gsp*sin(kx*a), Epx_k(kx, ky), 0, 0])
    r3 = np.array([-2*i*gsp*sin(ky*a), 0, Epy_k(kx, ky), 0])
    r4 = np.array([0, 0, 0, Epz_k(kx, ky)])
    h1 = np.array([r1, r2, r3, r4]).T
    h2 = np.array(np.eye(2))
    h = np.kron(h2,h1)
    hnm = np.array(h).astype(complex)
    return hnm

  def H_nm(kx,ky):
    return (H_nm_2d0(kx,ky) + LS_sp)
    
  def En_nm_2d(kx,ky):
    t1 = np.array((H_nm_2d0(kx,ky) + LS_sp), dtype = 'complex64')
    w1,v1 = LA.eig(t1) #calculate eigen-values and eigen-vectors
    return np.sort(w1.real)

  def H_fm_2d0(kx, ky):
    r1f = np.array([Ed_xy(kx, ky), 0, 0, 0, 0])
    r2f = np.array([0, Ed_yz(kx, ky), 0, 0, 0])
    r3f = np.array([0, 0, Ed_zx(kx, ky), 0, 0])
    r4f = np.array([0, 0, 0, Ed_x2_y2(kx, ky), 0])
    r5f = np.array([0, 0, 0, 0, Ed_z2(kx, ky)])
    h1 = np.array([r1f.T, r2f.T, r3f.T, r4f.T, r5f.T])
    h2 = np.array(np.identity(2))
    h4 = np.kron(h1,h2)
    hfm = np.array(h4)
    return hfm


  Lpx = hbar* np.array([[0, 0, 0],
                        [0, 0, -i],
                        [0, i, 0]])

  Lpy = hbar* np.array([[0, 0, i],
                        [0, 0, 0],
                        [-i, 0, 0]])
            
  Lpz = hbar* np.array([[0, -i, 0],
                        [i, 0, 0],
                        [0, 0, 0]])

  Lp = np.array([Lpx, Lpy, Lpz])

  Sx =(hbar/2)*np.array([[0, 1],
                         [1, 0]])

  Sy =(hbar/(2*i))*np.array([[0, 1],
                             [-1, 0]])

  Sz = (hbar/2)*np.array([[1, 0],
                          [0, -1]])

  S = np.array([Sx, Sy, Sz])

  F = np.kron(Lpx,Sx)-np.kron(Sx,Lpx)

  LpS = (alpha_nm_so/hbar**2)*(np.kron(Lpx, Sx) + np.kron(Lpy, Sy) + np.kron(Lpz, Sz))
  LpS = np.array(LpS).astype(complex)

  LS_sp = np.zeros((8,8)).astype(complex)
  LS_sp[2:2+LpS.shape[0], 2:2+LpS.shape[1]] = LpS #why adding at the left and top of the matrix?why not right and bottomn?

  def H_nm(kx,ky):
    return (H_nm_2d0(kx,ky) + LS_sp)
    
  def En_nm_2d(kx,ky):
    t1 = np.array((H_nm_2d0(kx,ky) + LS_sp), dtype = 'complex64')
    w1,v1 = LA.eig(t1) #calculate eigen-values and eigen-vectors
    return np.sort(w1.real)

  ### T matrix for nonmagnetic
  t1 = np.array([-tss, 0, 0, -gsp])
  t2 = np.array([0, -tpp, 0, 0])
  t3 = np.array([0, 0, -tpp, 0])
  t4 = np.array([gsp, 0, 0, tps])
  tn = np.array([t1, t2, t3, t4]).T
  Tnm_temp = np.kron(tn, np.eye(2))
  Tnm = np.array(Tnm_temp).astype(complex)

  Ldx = hbar* np.array([[0, 0, -i, -i, -sqrt(3)*i],
                        [0, 0, 0, 0, 0],
                        [i, 0, 0, 0, 0],
                        [i, 0, 0, 0, 0],
                        [sqrt(3)*i, 0, 0, 0, 0]])

  Ldy = hbar* np.array([[0, i, 0, 0, 0],
                        [-i, 0, 0, 0, 0],
                        [0, 0, 0, -i, sqrt(3)*i],
                        [0, 0, i, 0, 0],
                        [0, 0, -sqrt(3)*i, 0, 0]])

  Ldz = hbar* np.array([[0, 0, 0, 2*i, 0],
                        [0, 0, i, 0, 0],
                        [0, -i, 0, 0, 0],
                        [-2*i, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]])

  Sx =(hbar/2)*np.array([[0, 1],
                         [1, 0]])

  Sy =(hbar/(2*i))*np.array([[0, 1],
                             [-1, 0]])

  Sz = hbar*np.array([[1, 0],
                      [0, -1]])

  LdSa = (alpha_fm_so/hbar**2)*(np.kron(Ldx, Sx) + np.kron(Ldy, Sy) + np.kron(Ldz, Sz))
  LdS = np.array(LdSa)

  Mx = np.array([[1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1]])

  My = np.array([[1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1]])

  Mz =  np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])

  Sx =(hbar/2)*np.array([[0, 1],
                         [1, 0]])

  Sy =(hbar/(2*i))*np.array([[0, 1],
                             [-1, 0]])

  Sz =(hbar/2)*np.array([[1, 0],
                         [0, -1]])

  H_fm_xc_n = (J/hbar)*(np.kron(Mz,Sz))
  H_fm_xc = np.array(H_fm_xc_n)

  def Hfm(kx,ky):
    return ((H_fm_2d0(kx,ky) + LdS + H_fm_xc))
  def En_fm(kx,ky):
    t2 = np.array((H_fm_2d0(kx,ky) + LdS + H_fm_xc), dtype = 'complex64')
    w2,v2 = LA.eig(t2)
    return np.sort(w2.real)

  # ## T matrix for the Ferromagent
  t1f = np.array([-tdd, 0, 0, 0, 0])
  t2f = np.array([0, tdp, 0, 0, 0])
  t3f = np.array([0, 0, tdp, 0, 0])
  t4f = np.array([0, 0, 0, -tdd, 0])
  t5f = np.array([0, 0, 0, 0, -tds])
  tfm = np.array([t1f, t2f, t3f, t4f, t5f]).T
  Tfm1 = np.kron(tfm, np.eye(2))
  Tfm = np.array(Tfm1)

  t1i = np.array([0, 0, 0, 0])
  t2i = np.array([0, 0, gpdp, 0])
  t3i = np.array([0, gpdp,  0, 0])
  t4i = np.array([0, 0,  0, 0])
  t5i = np.array([0, 0, 0, -gpds])

  tin = np.array([t1i, t2i, t3i, t4i, t5i])
  Tin1 = np.kron(tin, np.eye(2))
  Tin = np.array(Tin1)


  H = np.zeros((nawf, nawf, nk1, nk2, nk3, nspin), dtype=complex)
  nh = H.shape[0]

  for i,ix in enumerate(kxGM):
    for j,iy in enumerate(kyGM):
      hnm = H_nm(ix, iy)
      nhnm = hnm.shape[0]
      lower = (nh) // 9 - (nhnm // 2)
      upper = (nh // 9) + (nhnm // 2)
      H[lower:upper, lower:upper, i, j, 0, 0] = hnm

      ntnm = np.conj(Tnm).T.shape[0]
      lower = (nh) // 3 - (ntnm // 2)
      upper = (nh // 3) + (ntnm // 2)
      lower1 = (nh // 9) - (ntnm // 2)
      upper1 = (nh // 9) + (ntnm // 2)
      H[lower1:upper1, lower:upper, i, j, 0, 0] = np.conj(Tnm).T

      Ntnm = Tnm.shape[0]
      lower = (nh) // 3 - (Ntnm // 2)
      upper = (nh // 3) + (Ntnm // 2)
      lower1 = (nh // 9) - (Ntnm // 2)
      upper1 = (nh // 9) + (Ntnm // 2)
      H[lower:upper, lower1:upper1, i, j, 0, 0] = Tnm

      lower = (nh // 3) - (nhnm // 2)
      upper = (nh // 3) + (nhnm // 2)
      H[lower:upper, lower:upper, i, j, 0, 0] = hnm

      ntin = np.conj(Tin).T.shape[0]
      lower = (nh) // 3 - (ntin // 2)
      upper = (nh // 3) + (ntin // 2)
      lower1 = (nh // 9) - (ntin // 2)
      upper1 = (nh // 9) + (ntin // 2)
      H[8:16, 16:26, i, j, 0, 0] = np.conj(Tin).T

      nTin = Tin.shape[0]
      lower = (nh) // 3 - (nTin // 2)
      upper = (nh // 3) + (nTin // 2)
      lower1 = (nh // 9) - (nTin // 2)
      upper1 = (nh // 9) + (nTin // 2)
      H[16:26, 8:16, i, j, 0, 0] = Tin

      hfm = Hfm(ix, iy)
      nhfm = hfm.shape[0]
      lower = (nh) // 3 - (nhfm // 2)
      upper = (nh // 3) + (nhfm // 2)
      lower1 = (nh // 9) - (nhfm // 2)
      upper1 = (nh // 9) + (nhfm // 2)
      H[16:26, 16:26, i, j, 0, 0] = hfm

      ntfm = np.conj(Tfm).T.shape[0]
      lower = (nh) // 3 - (ntfm // 2)
      upper = (nh // 3) + (ntfm // 2)
      lower1 = (nh // 9) - (ntfm // 2)
      upper1 = (nh // 9) + (ntfm // 2)
      H[16:26, 26:36, i, j, 0, 0] = np.conj(Tfm).T

      nTfm = Tfm.shape[0]
      lower = (nh) // 3 - (nTfm // 2)
      upper = (nh // 3) + (nTfm // 2)
      lower1 = (nh // 9) - (nTfm // 2)
      upper1 = (nh // 9) + (nTfm // 2)
      H[26:36, 16:26, i, j, 0, 0] = Tfm

      lower = (nh) // 3 - (nhfm // 2)
      upper = (nh // 3) + (nhfm // 2)
      lower1 = (nh // 9) - (nhfm // 2)
      upper1 = (nh // 9) + (nhfm // 2)
      H[26:36, 26:36, i, j, 0, 0] = hfm

#      print(np.array(H))

  arry['HRs'] = H
#  arry['HRs'] = FFT.ifftn(H, axes=(2,3,4))

