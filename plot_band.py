#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt('EigeH9L4V2_1_Ec_09.txt')

fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
ax.plot(range(12), a-np.amin(np.amin(a)), 'b_')
plt.show()
