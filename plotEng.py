#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
#plt.switch_backend('agg')

a = np.loadtxt('EigeBLt_third_12.dat') #('EigeBLt8.dat')
b = np.linspace(0,2,21) #(0,3,31)
for ii in range(np.shape(a)[0]):
  a[ii,:] = sorted(a[ii,:])
#plt.plot(b,a[:,0:4],'-o')
plt.plot(b,a,'-+b')
plt.show()
plt.savefig('EngBLG_third_12.eps')
