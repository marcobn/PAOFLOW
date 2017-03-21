from __future__ import print_function
__author__ = 'believe'

import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from   matplotlib.backends.backend_pdf import PdfPages

def PS2AE_print(UPF_fullpath,wfc_ae_fullpath,llabels,ll,rcut,pseudo_e):

    #<PP_CHI.1 type="real" size="1141" columns="4" index="1" label="3S" l="0" occupation="2.000000000000000E+000" n="1"
    mesh_size,rmesh,wfc_ps,wfc_ae = PS2AE_batch(UPF_fullpath,wfc_ae_fullpath,rcut,ll)

    fig_fullpath=os.path.dirname(wfc_ae_fullpath)+"/PP_CHI.pdf"
    print('Printing figures to: ',fig_fullpath)
    with PdfPages(fig_fullpath) as pdf:
         for ichi in range(len(ll)):
             plt.figure(figsize=(3,3))
             plt.plot(rmesh,wfc_ae[:,ichi],':',color='r',linewidth=3,label='AE',alpha=0.8)
             plt.plot(rmesh,wfc_ps[:,ichi],label='PS computed')
             plt.legend(loc=4,prop={'size':6})
             plt.xlim([0,5])
             plt.ylim([-1.5,1.5])
             plt.title('PP_CHI.%s'%(str(ichi+1)))
             pdf.savefig()
             plt.close()

    np.savez(os.path.dirname(wfc_ae_fullpath)+"/AE_and_PS",r=rmesh,AE=wfc_ae,PS=wfc_ps)

    print('Printing formatted PP CHI to: ',os.path.dirname(wfc_ae_fullpath)+"/PP_CHI.txt")
    fid     = open(os.path.dirname(wfc_ae_fullpath)+"/PP_CHI.txt","w")

    for ichi in range(len(ll)):
        #print('<PP_CHI.X type="real" size="{0:d}" label="{1:s}" >'.format(mesh_size,llabels[ichi],ll[ichi]),file=fid)
        occ_str = eformat(0,15,3)
        pseudo_e_str = eformat(pseudo_e[ichi],15,3)
        print('    <PP_CHI.{0:d} type="real" size="{1:d}"'
              ' columns="4" index="x" label="{2:s}"'
              ' l="{3:d}" occupation="{4:s}" n="x"\npseudo_energy="{5:s}">'.format(ichi+1,mesh_size,llabels[ichi],ll[ichi],occ_str,pseudo_e_str),file=fid)
        chi_str= radial2string(wfc_ps[:,ichi])
        print('{0:s}'.format(chi_str),file=fid)
        print('    </PP_CHI.{0:d}>'.format(ichi+1),file=fid)

    fid.close()

def radial2string(chi):
    longstr =""
    for ii,x in enumerate(chi):
        nstr = eformat(x,15,3)
        if (ii+1) == chi.size:
            longstr = longstr+nstr
        elif (ii+1)%4 == 0:
            longstr = longstr+nstr+"\n"
        else:
            longstr = longstr+nstr+" "
    return longstr

def PS2AE_batch(UPF_fullpath,wfc_ae_fullpath,rcut,ll):
    import numpy as np
    from read_UPF import read_UPF

    psp     = read_UPF(UPF_fullpath)
    wfc_ae  = np.loadtxt(wfc_ae_fullpath,skiprows=1)

    nwfcs   = wfc_ae.shape[1]-1
    size    = wfc_ae.shape[0]

    #Check that the meshes and radial parts are the same
    r1 = wfc_ae[:,0]
    r2 = psp['r']
    if len(r2) != size:
        sys.exit('The radial meshes have different size')
    if np.sum(np.abs(r2-r1)) > 1e-5:
        sys.exit('The radial meshes are different')

    wfc_ps  =np.zeros((size,nwfcs))

    for iwfc in range(nwfcs):
       wfc_ps[:,iwfc] = AE2PS_l(psp,wfc_ae[:,iwfc+1],ll[iwfc],rcut)

    return size,r2,wfc_ps,wfc_ae[:,1:]

    #test_plot
    #plt.figure(1,figsize=(3,3))
    #iwfc = 11
    #l = 2
    #plt.clf()
    #ae_computed = PS2AE_l(psp,wfc_ps[:,iwfc],l,rcut)
    #plt.plot(r1,ae_computed,':',color='r',linewidth=3,label='T*PS',alpha=0.8)
    #plt.plot(r1,wfc_ps[:,iwfc],label='PS computed')
    #plt.plot(r1,wfc_ae[:,iwfc+1],label='AE')
    #plt.legend(loc=4,prop={'size':6})
    #plt.xlim([0,5])
    #plt.ylim([-1.5,1.5])
    #plt.draw()
    #plt.show()

def AE2PS_l(data,phi,l,rcut):
    '''
    transform AE into PS function
    '''
    import numpy as np
    lll   = data['lll']
    rab   = data['rab']
    r     = data['r']

    beta_l = np.where(lll==l)[0]
    B = np.zeros((len(beta_l),1))
    A = np.zeros((len(beta_l),len(beta_l)))
    for ii,ibeta in enumerate(beta_l):
        p_i        = data['beta_'+str(ibeta+1)]
        B[ii,0]    = braket(r,rab,p_i,phi,rcut)
        for jj,jbeta in enumerate(beta_l):
            delta_phi_j= data['full_aewfc'][:,jbeta] - data['full_pswfc'][:,jbeta]
            A[ii,jj]   = braket(r,rab,p_i,delta_phi_j,rcut)

    A = A + np.eye(len(beta_l))
    C = np.dot(np.linalg.inv(A),B)


    #finding the PS function
    sphi = phi
    for ii,ibeta in enumerate(beta_l):
        delta_phi_i= data['full_aewfc'][:,ibeta] - data['full_pswfc'][:,ibeta]
        sphi       = sphi - C[ii]*delta_phi_i
    #return C, sphi
    return sphi

def braket(r,rab,f1,f2,rcut):
    import numpy as np
    from scipy import integrate
    #find maximun index
    ii = np.max(np.where(r <= rcut*1.1)[0])
    integrand = np.conj(f1[0:ii])*f2[0:ii]

    #simple sum, trapezoidal integration
    #return r[0:ii],integrand,np.dot(rab[0:ii],integrand),integrate.trapz(integrand,r[0:ii])
    #return np.dot(rab[0:ii],integrand),integrate.trapz(integrand,r[0:ii])
    return np.dot(rab[0:ii],integrand)

    #

def eformat(f, prec, exp_digits):
    s = "%+.*e"%(prec, f)
    mantissa, exp = s.split('e')
    mantissa1=mantissa.replace('+',' ')
    # add 1 to digits as 1 is taken by sign +/-
    return "%sE%+0*d"%(mantissa1, exp_digits+1, int(exp))
