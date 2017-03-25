from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np

f= open("full.dat", 'r')
a_k = np.zeros((300,191),dtype=float)
Efermi = 0.0#12.79 

for i in xrange(300):
    a = f.readline()
    aux = a.split()
    a = np.array(aux,dtype="float32")
    a_k[i,:] = a - Efermi

#print (a_k[0,:])
print (a_k.shape[0],a_k.shape[1])


plt.matshow(a_k,fignum=100)
plt.colorbar()
plt.show()
