#from __future__ import print_function,division
from __future__ import print_function
__author__ = 'believe'




import numpy as np
import sys 
##################################################
def read_UPF(fullpath):
 import numpy as np
 import re
 import xml.etree.ElementTree as et
 import sys
 import scipy.io as sio
 import cPickle as pickle

 #reads xml file
 #http://www.quantum-espresso.org/pseudopotentials/unified-pseudopotential-format
 Bohr2Angs=0.529177249000000

 #The UPF file may "incorrectly" contain "&" and "<" which is restricted in the xml format
 #temporary fix for "&"
 f1 = open(fullpath,'r')
 f2 = open('tmp.xml','w')
 for line in f1:
     f2.write(line.replace('&','&amp;'))
 f1.close()
 f2.close()

 tree = et.parse('tmp.xml')
 root = tree.getroot()
 psp  = {} 

 #%%
 head = root.find('PP_HEADER')
 mesh    =int(head.attrib["mesh_size"])
 nwfc    =int(head.attrib["number_of_wfc"])
 nbeta   =int(head.attrib["number_of_proj"])

 straux  =head.attrib["is_ultrasoft"]
 if straux.upper()=='.T.' or straux.upper()=='T':
    is_ultrasoft=1
 elif straux.upper()=='.N.' or straux.upper()=='N':
    is_ultrasoft=0
 else:
    sys.exit('is_ultrasoft: String not recognized as boolean %s'%straux)

 straux  =head.attrib["is_paw"]
 if straux.upper()=='.T.' or straux.upper()=='T':
    is_paw=1
 elif straux.upper()=='.F.' or straux.upper()=='F':
    is_paw=0
 else:
    sys.exit('is_paw: String not recognized as boolean %s'%straux)

 straux  =head.attrib["has_wfc"]
 if straux.upper()=='.T.' or straux.upper()=='T':
    has_wfc=1
 elif straux.upper()=='.F.' or straux.upper()=='F':
    has_wfc=0
 else:
    sys.exit('has_wfc: String not recognized as boolean %s'%straux)
 
 psp["is_ultrasoft"]=is_ultrasoft
 psp["is_paw"]      =is_paw
 psp["has_wfc"]     =has_wfc

 #%%
 #-->Reading <PP_R>
 for rootaux in root.iter('PP_R'):
     sizeaux = int(rootaux.attrib['size'])
     if mesh != sizeaux:
        sys.exit('Error: The size of PP_R does not match mesh: %i != %i'%(sizeaux,mesh))
     xxaux = re.split('\n| ',rootaux.text)
     rmesh  =np.array(map(float,filter(None,xxaux))) #In Bohrs
     if mesh != len(rmesh):
       sys.exit('Error: wrong mesh size')
 psp["mesh"]=mesh
 psp["r"]=rmesh

 #%%
 #-->Reading <PP_RAB>
 for rootaux in root.iter('PP_RAB'):
     sizeaux = int(rootaux.attrib['size'])
     if mesh != sizeaux:
        sys.exit('Error: The size of PP_RAB does not match mesh: %i != %i'%(sizeaux,mesh))
     xxaux = re.split('\n| ',rootaux.text)
     rmesh  =np.array(map(float,filter(None,xxaux))) #In Bohrs
     if mesh != len(rmesh):
       sys.exit('Error: wrong mesh size')
 psp["rab"]=rmesh

 #%%
 #-->Reading <PP_BETA.x>
 kkbeta = np.zeros(nbeta)
 lll    = np.zeros(nbeta)
 for ibeta in range(nbeta):
     for rootaux in root.iter('PP_BETA.'+str(ibeta+1)):
         kkbeta[ibeta] = int(rootaux.attrib['size'])
         lll[ibeta]    = int(rootaux.attrib['angular_momentum'])
         xxaux         = re.split('\n| ',rootaux.text)
         rmesh         = np.array(map(float,filter(None,xxaux))) 
         if kkbeta[ibeta] != len(rmesh):
           sys.exit('Error: wrong mesh size')
     psp["beta_"+str(ibeta+1)]=rmesh
 psp["nbeta"] =nbeta
 psp["lll"]   =lll
 psp["kkbeta"]=kkbeta
 #the beta vectors are not necessarily of the same size. That's they are kept separated

 #%%
 #-->Reading <PP_PSWFC>
 chi = np.zeros((0,mesh))
 pswfc = root.find('PP_PSWFC')
 els = []
 lchi= []
 oc  = []
 for node in pswfc:
   print(node.tag, node.attrib['l'],node.attrib['label'])
   els.append(node.attrib['label'])
   lchi.append(int(node.attrib['l']))
   oc.append(float(node.attrib['occupation']))
   xxaux = re.split('\n| ',node.text)
   wfc_aux  =np.array([map(float,filter(None,xxaux))])
   chi = np.concatenate((chi,wfc_aux))
 if nwfc != chi.shape[0]: 
   sys.error('Error: wrong number of PAOs')
 else:
   print('Number of radial wavefunctions found: %i' % chi.shape[0])
 psp["chi"] =np.transpose(chi)
 psp["els"] =els
 psp["lchi"]=lchi
 psp["oc"]  =oc

 #%%
 #-->Reading PAW related data
 if has_wfc:
    rootaux = root.find('PP_FULL_WFC')
    number_of_full_wfc=int(rootaux.attrib['number_of_wfc'])
    psp["number_of_full_wfc"] =number_of_full_wfc
 
    full_aewfc_label=[]
    full_aewfc_l    =[]
    full_aewfc      = np.zeros((0,mesh))
    for ibeta in range(nbeta):
        for rootaux in root.iter('PP_AEWFC.'+str(ibeta+1)):
            sizeaux       = int(rootaux.attrib['size'])
            if sizeaux != mesh: sys.error('Error in mesh size while reading PAW info')
            full_aewfc_l.append(int(rootaux.attrib['l']))
            full_aewfc_label.append(rootaux.attrib['label'])
            xxaux         = re.split('\n| ',rootaux.text)
            rmesh         = np.array([map(float,filter(None,xxaux))]) 
            if sizeaux != rmesh.shape[1]: sys.exit('Error: wrong mesh size')
            full_aewfc    = np.concatenate((full_aewfc,rmesh))
    psp["full_aewfc"]       =np.transpose(full_aewfc)
    psp["full_aewfc_label"] =full_aewfc_label       
    psp["full_aewfc_l"]     =full_aewfc_l       

    full_pswfc_label=[]
    full_pswfc_l    =[]
    full_pswfc      = np.zeros((0,mesh))
    for ibeta in range(nbeta):
        for rootaux in root.iter('PP_PSWFC.'+str(ibeta+1)):
            sizeaux       = int(rootaux.attrib['size'])
            if sizeaux != mesh: sys.error('Error in mesh size while reading PAW info')
            full_pswfc_l.append(int(rootaux.attrib['l']))
            full_pswfc_label.append(rootaux.attrib['label'])
            xxaux         = re.split('\n| ',rootaux.text)
            rmesh         = np.array([map(float,filter(None,xxaux))]) 
            if sizeaux != rmesh.shape[1]: sys.exit('Error: wrong mesh size')
            full_pswfc    = np.concatenate((full_pswfc,rmesh))
    psp["full_pswfc"]       =np.transpose(full_pswfc)
    psp["full_pswfc_label"] =full_pswfc_label       
    psp["full_pswfc_l"]     =full_pswfc_l       
 return psp
 #%%


    

 with open(fullpath + '.p','wb') as fp:
    pickle.dump(psp,fp)

 sio.savemat(fullpath + '.mat',psp)
 

#if __name__ == '__main__':
#  fullpath = str(sys.argv[1])
#  read_UPF(fullpath)

#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from   matplotlib.backends.backend_pdf import PdfPages

def PS2AE_print(UPF_fullpath,wfc_ae_fullpath,llabels,ll,rcut,pseudo_e):

    #<PP_CHI.1 type="real" size="1141" columns="4" index="1" label="3S" l="0" occupation="2.000000000000000E+000" n="1"
    mesh_size,rmesh,wfc_ps,wfc_ae = PS2AE_batch(UPF_fullpath,wfc_ae_fullpath,rcut,ll)


    # try:
    #     fig_fullpath=os.path.dirname(wfc_ae_fullpath)+"/PP_CHI.pdf"
    #     print('Printing figures to: ',fig_fullpath)
    #     with PdfPages(fig_fullpath) as pdf:
    #          for ichi in range(len(ll)):
    #              plt.figure(figsize=(3,3))
    #              plt.plot(rmesh,wfc_ae[:,ichi],':',color='r',linewidth=3,label='AE',alpha=0.8)
    #              plt.plot(rmesh,wfc_ps[:,ichi],label='PS computed')
    #              plt.legend(loc=4,prop={'size':6})
    #              plt.xlim([0,5])
    #              plt.ylim([-1.5,1.5])
    #              plt.title('PP_CHI.%s'%(str(ichi+1)))
    #              pdf.savefig()
    #              plt.close()
    # except:
    #     pass
    print('asdfada')
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
#    from read_UPF import read_UPF

    psp     = read_UPF(UPF_fullpath)
    wfc_ae  = np.loadtxt(wfc_ae_fullpath,skiprows=1)

    nwfcs   = wfc_ae.shape[1]-1
    size    = wfc_ae.shape[0]
    print(wfc_ae.shape,nwfcs,size)
    print(ll)

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



#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################


import re,sys,os

usage = ''' 
                python build_newPP.py <ld1.x input file> <UPF file>
		        
'''


workdir="./"

def pseudizeWFC(ld1FileString, UPF_file):

        import radial_integral as radial

	wfc_ae_fullpath = os.path.join(workdir,"wfc_ae.txt")
	
	if os.path.exists(wfc_ae_fullpath):

		UPF_fullpath = os.path.join(workdir,UPF_file)
		

		nPAO_RE = re.compile(r"/\n(\d+)\s*\n")
		nPAO = int(float(nPAO_RE.findall(ld1FileString)[0]));print("Total No. of PAOs found:", nPAO)

		paoRE = re.compile(r"([0-9][A-Z])\s*\d\s*(\d)\s*(\d+\.\d+)\s*(\d+\.\d+)\s*\d+\.\d+\s*(\d+\.\d+)\s*\d+\.\d+.*\n")
		paoList = paoRE.findall(ld1FileString)

		llabels	= [x[0] for i,x in enumerate(paoList)];print("llabels:", llabels)
		ll	= map(int,[x[1] for i,x in enumerate(paoList)]);print("ll:", ll)
		rcut	= max(map(float,[x[4] for i,x in enumerate(paoList)]));print("rcut:", rcut)
		pseudo_e= map(float, [x[3] for i,x in enumerate(paoList)]);print("pseudo_e:", pseudo_e)
		occups  = map(float,[x[2] for i,x in enumerate(paoList)]);print("occupations:", occups)

		
		PS2AE_print(UPF_fullpath,wfc_ae_fullpath,llabels,ll,rcut,pseudo_e)

                return occups
	else:
		print("Cannot find wfc_ae.txt. CHECK specified workdir")


def build_newPPStr(wfc_combination, occupations, UPF_fullpath):

	PP_CHI_fullpath = os.path.join(workdir,"PP_CHI.txt")

	ppFileLines = file(UPF_fullpath,'r').read()
	paoFileLines = file(PP_CHI_fullpath, 'r').read()
	

	#Make new PP_PSWFC block for PP
	pp_pswfc = "<PP_PSWFC>\n"
	if len(wfc_combination) > 0:
		for i in range(len(wfc_combination)):
			tag = "PP_CHI.%d"%wfc_combination[i]
			index = i+1
			r1 = re.compile(r"<%s.*</%s>\n"%(tag,tag),re.DOTALL)
			chiLines = r1.findall(paoFileLines)[0]
			r2 = re.compile(r"index=\"x\"")
			chiLines = r2.sub("index=\"%d\""%index,chiLines)
                        r5 = re.compile("occupation=\".*\"")
                        chiLines = r5.sub("occupation=\"%.16e\""%occupations[wfc_combination[i]-1],chiLines)
			r3 = re.compile(r"n=\"x\"")
			chiLines = r3.sub("n=\"%d\""%index,chiLines)
			r4 = re.compile(r"%s"%tag)
			newTag = "PP_CHI.%d"%index
			chiLines = r4.sub(r"%s"%newTag,chiLines)
			pp_pswfc += chiLines
		pp_pswfc += "</PP_PSWFC>\n"	
		
		#build new PP file string
		r5 = re.compile(r"<PP_PSWFC>.*</PP_PSWFC>",re.DOTALL)
		newPPStr = r5.sub(pp_pswfc,ppFileLines)
		r6 = re.compile(r"number_of_wfc.*\n")
		newPPStr = r6.sub("number_of_wfc=\"%d\"\n"%len(wfc_combination),newPPStr,count=1)

		return newPPStr
	else:
		print("Zero wfc's specified ")
		raise SystemExit
def main():

	if len(sys.argv) > 2:
		
		ld1_inFile = sys.argv[1]
		UPF_file = sys.argv[2] 

		ld1Str = file(os.path.join(workdir,ld1_inFile),'r').read()

	
		try:
			#Extract wfc_combinations
			r1=re.compile(r"\!cases.*\n!.*",re.DOTALL)
			r1String = r1.findall(ld1Str)[0].split('\n')[1:]

			wfc_combinations = [map(int,x.strip('!').split()) for i, x in enumerate(r1String) if len(x) > 0]
		except:
			print("Error in case specification of ld1.x input file")
			raise SystemExit

		#Pseudize wfc
		print("Pseudizing wfc's in wfc_ae.txt to PP_CHI.txt\n")
		occupations = pseudizeWFC(ld1Str, UPF_file)

		
		for i in range(len(wfc_combinations)):


			postFix = "case" + str(i+1)
			newPPFileName = UPF_file.split("/")[-1].split(".UPF")[0] + "_" + postFix + ".UPF"
			newPPFilePath = os.path.join(workdir,newPPFileName)

			print("Generating case ",i+1, " with wfc's ", str(wfc_combinations[i]).strip('[]'), " to file ", newPPFileName, "\n")
			newPPFileString = build_newPPStr(wfc_combinations[i],occupations,os.path.join(workdir,UPF_file))
			file(newPPFilePath,'w').write(newPPFileString)

	else:
		print(usage)


if __name__ == "__main__":
        main()




