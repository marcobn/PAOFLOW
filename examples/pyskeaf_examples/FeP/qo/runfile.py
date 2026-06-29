import subprocess
import shutil
import os
import re
for i in [1,2,3,4,5,6,7,8]: 
  f = open('config_orig.in', 'r')
  f2 = open('config.in', 'w')
  f2.write('Fermi_surf_band_'+str(i)+'.bxsf'+'\n')
  l = f.readline()
  l = f.readline()
  while l != '':
    f2.write(l)
    l = f.readline()
  f.close()
  f2.close() 
  subprocess.call(['/home/cchenye/codes/SKEAF/skeaf_auto.x'])
  shutil.copyfile('results_freqvsangle.out', 'results_freqvsangle_'+str(i)+'.out')
