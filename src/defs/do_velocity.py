        velocities = True
        if velocities:   # Print velocity dispersion curves
            #velRp = np.zeros((nk1,nk2,nk3,3,nawf,nspin),dtype=float)
            velkp = np.reshape(velkp,(nk1,nk2,nk3,3,nawf,nspin),order='C')
            kq = kpnts_interpolation_mesh(ibrav,alat,a_vectors,dkres)
            nkpi=kq.shape[1]
            for n in range(nkpi):
                kq [:,n]=kq[:,n].dot(b_vectors)

            R,_,_,_,_ = get_R_grid_fft(nk1,nk2,nk3,a_vectors)

            velRp = FFT.ifftn(velkp[:,:,:,:,:,:],axes=[0,1,2])
            velRp = np.reshape(velRp,(nk1*nk2*nk3,3,nawf,nspin),order='C')
            velkn = np.zeros((nkpi,3,nawf,nspin),dtype=float)

            # Load balancing
            ini_ik, end_ik = load_balancing(size,rank,nkpi)

            velknaux = np.zeros((end_ik-ini_ik,3,nawf,nspin),dtype=float)

            for ik in range(ini_ik,end_ik):
                for ispin in range(nspin):
                    for l in range(3):
                        velknaux[ik,l,:,ispin] = np.real(np.sum(velRp[:,l,:,ispin].T*np.exp(2.0*np.pi*kq[:,ik].dot(R[:,:].T)*1j),axis=1))

            comm.Reduce(velknaux,velkn,op=MPI.SUM)

            for ispin in range(nspin):
                for l in range(3):
                    f=open('velocity_'+str(l)+'_'+str(ispin)+'.dat','w')
                    for ik in range(nkpi):
                        s="%d\t"%ik
                        for  j in velkn[ik,l,:bnd,ispin]:s += "%3.5f\t"%j
                        s+="\n"
                        f.write(s)
                    f.close()
            velkp = np.reshape(velkp,(nk1*nk2*nk3,3,nawf,nspin),order='C')

