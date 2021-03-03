

def do_atomic_orbitals ( data_controller, cutoff, fprefix ):
  import numpy as np
  from mpi4py import MPI
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
  norb = 0
  for i,a in enumerate(arry['atoms']):
    chi[a] = []
    ps = pseudos[a]
    norb += ps.nwfc**2

    for l in range(ps.nwfc):
      chi[a].append(splrep(ps.r,ps.pswfc[l]['wfc']/ps.r,k=3))

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
  grid = grid.reshape((nrtot,3))

  # Compute the supercell atomic positions
  attr['nsc'] = nsc = 3
  sctau = np.zeros((natoms,nsc,nsc,nsc,3), dtype=float)
  for i in range(-int(np.floor(nsc/2)), int(np.ceil(nsc/2))):
    for j in range(-int(np.floor(nsc/2)), int(np.ceil(nsc/2))):
      for k in range(-int(np.floor(nsc/2)), int(np.ceil(nsc/2))):
        ijk = np.array([i,j,k])
        for ia,a in enumerate(arry['tau']):
          sctau[ia,i,j,k] = a + ijk@a_vecs
  sctau = sctau.reshape((natoms,nsc**3,3))

  # Helper functions for distance, theta, and phi
  dist = lambda p0,p1 : np.linalg.norm(p1-p0, axis=1)
  phi = lambda p0,p1,d : np.nan_to_num(np.arccos((p1-p0)[:,2]/d))
  def theta ( p0, p1 ):
    d = p1 - p0
    return np.arctan2(d[:,1],d[:,0])

  start,stop = load_balancing(size, rank, nrtot)
  snrtot = stop - start
  rng = slice(start, stop)
  # Calculate the atomic orbitals on the small grid with periodic images on the large grid
  aux_atorb = np.zeros((snrtot,norb,nsc**3), dtype=float)
  for ic in range(nsc**3):
    iorb = 0
    for ia,a in enumerate(arry['atoms']):
      ps = pseudos[a]
      apos = arry['tau'][ia]
      if np.linalg.norm(sctau[ia,ic]-apos) < cutoff:
        r = dist(sctau[ia,ic], grid[rng])
        th = theta(sctau[ia,ic], grid[rng])
        ph = phi(sctau[ia,ic], grid[rng], r)
        for l in range(ps.nwfc):
          for m in range(l+1):
            R = splev(r, chi[a][l])
            ylm = Y(np.abs(m), l, th, ph)
            if m == 0:
              aux_atorb[:,iorb,ic] += R * ylm.real
              iorb += 1
            else:
              cnst = (-1)**m / np.sqrt(2)
              aux_atorb[:,iorb,ic] += cnst * (ylm + np.conj(ylm)).real
              aux_atorb[:,iorb+1,ic] += cnst * (ylm - np.conj(ylm)).imag
              iorb += 2

  comm.Barrier()
  atorb = np.empty((nrtot,norb,nsc**3), dtype=float) if rank==0 else None
  gather_array(atorb, aux_atorb)
  
  if rank == 0:
    atorb = np.moveaxis(atorb, 0, -1).reshape((norb,nsc**3,nr1,nr2,nr3))

  if fprefix is not None:
    for i in range(norb):
      fname = fprefix + '_%d.xsf'%i
      orb = None
      if rank == 0:
        orb = np.sum(atorb[i], axis=0)
      data_controller.write_xsf(orb, fname=fname)

  arry['atorb'] = atorb
