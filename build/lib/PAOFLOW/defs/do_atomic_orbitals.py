

def do_atomic_orbitals ( data_controller, cutoff, fprefix ):
  import numpy as np
  from mpi4py import MPI
  from scipy import integrate
  from scipy.special import sph_harm as Y
  from scipy.interpolate import splrep,splev
  from .read_upf import UPF
  from .communication import load_balancing,gather_array

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  arry,attr = data_controller.data_dicts()

  # Read radials from the pseudopotentials
  pseudos = {k:UPF(v) for k,v in arry['species']}

  # Count the number of orbitals and create interpolated radials
  
  chi = {}
  norb = np.zeros(attr['natoms'],dtype=int)
  norm = list(np.zeros(attr['natoms'],dtype=int))
  for i,a in enumerate(arry['atoms']):
    chi[a] = []
    norm[i] = []
    ps = pseudos[a]
    norb[i] = ps.nwfc**2

    for l in range(ps.nwfc):
      chi[a].append(splrep(ps.r,ps.pswfc[l]['wfc']/ps.r,k=3))
#      norm[i].append(np.sqrt(integrate.simps((ps.pswfc[l]['wfc']/ps.r)**2,ps.r)))
  arry['norb'] = norb
  arry['chi'] = chi
  arry['norm'] = norm

  natoms = attr['natoms']
  a_vecs = attr['alat'] * arry['a_vectors']
  nr1,nr2,nr3 = attr['nr1'],attr['nr2'],attr['nr3']

  rdim = np.array([nr1,nr2,nr3])
  nrtot = nr1*nr2*nr3

  # Generate the small grid
  grid = np.zeros((nr1,nr2,nr3,3), dtype=float)
  for i in range(nr1):
    for j in range(nr2):
      for k in range(nr3):
        grid[i,j,k] = np.array([i,j,k]) @ a_vecs / rdim
  arry['grid'] = grid
  grid = grid.reshape((nrtot,3))

  # Compute the supercell atomic positions
  attr['nsc'] = nsc = 10
  Rn = np.zeros((nsc,nsc,nsc,3),dtype=float)
  for i in range(-int(np.floor(nsc/2)), int(np.ceil(nsc/2))):
    for j in range(-int(np.floor(nsc/2)), int(np.ceil(nsc/2))):
      for k in range(-int(np.floor(nsc/2)), int(np.ceil(nsc/2))):
        ijk = np.array([i,j,k])
        Rn[i,j,k,:] = ijk @ a_vecs
  Rn = np.reshape(Rn,(nsc**3,3),order='C')/attr['alat'] 
  Rn2 = np.linalg.norm(Rn,axis=1)
  Rcutoff = np.sort(np.unique(Rn2))[cutoff]
  Rn = Rn[Rn2<=Rcutoff]
  arry['Rn'] = Rn
  nscx = Rn.shape[0]
  sctau = np.zeros((natoms,nscx,3), dtype=float)
  for ia,a in enumerate(arry['tau']):
    for n in range(nscx):
      sctau[ia,n] = a @ arry['a_vectors'] + Rn[n,:]*attr['alat']

  # Helper functions for distance and direction cosines, theta, and phi
  dcos = []
  dist = lambda p0,p1 : np.linalg.norm(p1-p0, axis=1)
  phi = lambda p0,p1,d : np.nan_to_num(np.arccos((p1-p0)[:,2]/d))
  def theta ( p0, p1 ):
    d = p1 - p0
    return np.arctan2(d[:,1],d[:,0])
  lx = lambda p0,p1 : (p1[:,0]-p0[0])/dist(p1,p0)
  ly = lambda p0,p1 : (p1[:,1]-p0[1])/dist(p1,p0)
  lz = lambda p0,p1 : (p1[:,2]-p0[2])/dist(p1,p0)

  start,stop = load_balancing(size, rank, nrtot)
  snrtot = stop - start
  rng = slice(start, stop)
  # Calculate the atomic orbitals on the small grid with periodic images on the large grid
  aux_atorb = np.zeros((snrtot,natoms,np.max(norb),nscx), dtype=float)
  pdist = []
  radius = []
  for ic in range(nscx):
    for ia,a in enumerate(arry['atoms']):
      ps = pseudos[a]
      apos = arry['tau'][ia] @ arry['a_vectors'] 
      pdist.append(np.linalg.norm(sctau[ia,ic]-apos))
#      if np.linalg.norm(sctau[ia,ic]-apos) < cutoff:
      r = dist(sctau[ia,ic], grid[rng])
      radius.append(r)
      x = np.nan_to_num(lx(sctau[ia,ic], grid[rng]))
      y = np.nan_to_num(ly(sctau[ia,ic], grid[rng]))
      z = np.nan_to_num(lz(sctau[ia,ic], grid[rng]))
      dcos.append([x,y,z])
      if norb[ia] == 1:
        R0 = splev(r, chi[a][0])#/norm[ia][0]
        Nc0 = 1/np.sqrt(4*np.pi)
        aux_atorb[:,ia,0,ic] = R * Nc0
        
      elif norb[ia] == 4:
        R0 = splev(r, chi[a][0])#/norm[ia][0]
        R1 = splev(r, chi[a][1])#/norm[ia][1]
        Nc0 = 1/np.sqrt(4*np.pi)
        Nc1 = np.sqrt(3/4/np.pi)
        aux_atorb[:,ia,0,ic] = R0 * Nc0
        aux_atorb[:,ia,1,ic] = R1 * Nc1*z
        aux_atorb[:,ia,2,ic] = R1 * Nc1*x
        aux_atorb[:,ia,3,ic] = R1 * Nc1*y
        
      elif norb[ia] == 9:
        R0 = splev(r, chi[a][0])#/norm[ia][0]
        R1 = splev(r, chi[a][1])#/norm[ia][1]
        R2 = splev(r, chi[a][2])#/norm[ia][2]
        Nc0 = 1/np.sqrt(4*np.pi)
        Nc1 = np.sqrt(3/4/np.pi)
        Nc2 = np.sqrt(15/4/np.pi)
        aux_atorb[:,ia,0,ic] = R0 * Nc0
        aux_atorb[:,ia,1,ic] = R1 * Nc1*z
        aux_atorb[:,ia,2,ic] = R1 * Nc1*x
        aux_atorb[:,ia,3,ic] = R1 * Nc1*y
        aux_atorb[:,ia,4,ic] = R2 * Nc2*(3*z**2 - 1)/(2*np.sqrt(3))
        aux_atorb[:,ia,5,ic] = R2 * Nc2*x*z
        aux_atorb[:,ia,6,ic] = R2 * Nc2*y*z
        aux_atorb[:,ia,7,ic] = R2 * Nc2*(x**2-y**2)/2
        aux_atorb[:,ia,8,ic] = R2 * Nc2*x*y
          
  # safeguard for small r's
  idmax = np.argwhere(aux_atorb > 10)
  idmin = np.argwhere(aux_atorb < -10)
  for n in idmax:
    aux_atorb[n[0]] = 0
  for n in idmin:
    aux_atorb[n[0]] = 0
  
  comm.Barrier()
  atorb = np.empty((nrtot,natoms,np.max(norb),nscx), dtype=float) if rank==0 else None
  gather_array(atorb, aux_atorb)
  
  if rank == 0:
    atorb = np.moveaxis(atorb, 0, -1).reshape((natoms,np.max(norb),nscx,nr1,nr2,nr3))
    atorb = np.nan_to_num(atorb)

  arry['atorb'] = atorb
  arry['sctau'] = sctau