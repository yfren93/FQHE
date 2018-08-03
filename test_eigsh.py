import numpy as np
import scipy.sparse as sps
from scipy.sparse import random
from scipy import stats

class CustomRandomState(object):
    def randint(self, k):
        i = np.random.randint(k)
        return i - i % 2

def renorm_a(a):
    ac = np.zeros((4,),dtype=complex)
    ac[0] = a
    ac[1] = -1*a
    ac[2] = 1j*a
    ac[3] = -1j*a
    return ac[np.real(ac).argmax()]

rs = CustomRandomState()
rvs = stats.poisson(25, loc=10).rvs

Sr = random(10,10,density=0.25, random_state=rs, data_rvs=rvs)
Si = random(10,10,density=0.25, random_state=rs, data_rvs=rvs)
S = Sr + 1j*Si
S += np.conjugate(S)

Sr = random(10,10,density=0.25, random_state=rs, data_rvs=rvs)
Si = random(10,10,density=0.25, random_state=rs, data_rvs=rvs)
S1 = Sr + 1j*Si
S1 += np.conjugate(S1)
S1 = S + S1*1e-4

eige, eigv = sps.linalg.eigsh(S, k=3)
eige1, eigv1 = sps.linalg.eigsh(S1, k=3)

gr = eigv[:,eige.argmin()]
gr1 = eigv1[:,eige1.argmin()]

a = np.dot(np.conjugate(gr), gr1)

ac = np.zeros((4,),dtype=complex)
ac[0] = a
ac[1] = -1*a
ac[2] = 1j*a
ac[3] = -1j*a

print ac[np.real(ac).argmax()]

