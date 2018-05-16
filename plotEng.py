#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

a = np.loadtxt('EigeBLt8.dat')
b = np.linspace(0,3,31)

plt.plot(b,a[:,0:4],'-o')
plt.savefig('Eng8.eps')
