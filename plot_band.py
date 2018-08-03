#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
#a = np.loadtxt('EigeH9L4V2_1_Ec_09.txt')
#a = np.loadtxt('EigeH12L4V2_0_Ec_09.txt')
#a = np.loadtxt('EigeH18L4V1_39_V2_39_Ec_1.txt')
#a = np.loadtxt('EigeH18L4V1_10_V2_10_Ec_1.txt')
#a = np.loadtxt('EigeH18L4V1_0_V2_10_Ec_1.txt')
#a = np.loadtxt('EigeH18L4V1_10_V2_0_Ec_1.txt')
#a = np.loadtxt('EigeH15L5V1_39_V2_39_Ec_1.txt')
#a = np.loadtxt('EigeH18L4V1_39_V2_39_V3_39_Ec_1.txt')
a = np.loadtxt('EigeBLt_third_12_t.dat')
print np.shape(a)
fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
print a--1.5323689 #np.amin(np.amin(a))
ax.plot(range(np.shape(a)[0]), a-np.amin(np.amin(a)), 'b_')
#plt.savefig('EigeH18L4V1_39_V2_39_V3_39_Ec_1.eps', format = 'eps')
plt.show()
