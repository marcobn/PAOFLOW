##################################################
#By: Luis Agapito @
#Marco Buongiorno Nardelli's group @ University of North Texas
#Stefano Curtarolo's group @ Duke University
#July 2014
##################################################

#Reads QE pseudopotential files for UPF v2.0.1


import numpy as np

##################################################
def read_UPF(fullpath):
 import numpy as np
 import re
 import xml.etree.ElementTree as et
 import sys
 import scipy.io as sio
 import pickle as pickle

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
     rmesh  =np.array(list(map(float,[_f for _f in xxaux if _f]))) #In Bohrs
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
     rmesh  =np.array(list(map(float,[_f for _f in xxaux if _f]))) #In Bohrs
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
         rmesh         = np.array(list(map(float,[_f for _f in xxaux if _f]))) 
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
   wfc_aux  =np.array([list(map(float,[_f for _f in xxaux if _f]))])
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
            rmesh         = np.array([list(map(float,[_f for _f in xxaux if _f]))]) 
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
            rmesh         = np.array([list(map(float,[_f for _f in xxaux if _f]))]) 
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
 

if __name__ == '__main__':
  fullpath = str(sys.argv[1])
  read_UPF(fullpath)

