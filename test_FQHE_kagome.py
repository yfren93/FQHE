#!/usr/bin/env python

from sparse_Ham_fun import *
import numpy as np
import scipy as sp
import scipy.special as sps
import itertools
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from lattice import *
import sys
import time 

global Height, Length, nNNcell, pbcx, pbcy, N_site, n_electron, distNN, distNNN, distN3, Lattice_type

#plt.switch_backend('agg')
# define constants
pi = 3.1415926535897932

nUnit = 3
N = N_site
n = n_electron #N/nUnit

mode1 = 'SA'
eigv_k = 10
" Get single particle states in momentum space "
#UT = GetKagomeUnitaryTransform(phi0=1e-14)

vNN, vNNN, vN3 = 1.0, 1.0, 0.0
Flags={}
Flags['Measure_density'] = False #True

" Define the fourier transform of interaction "
ay0 = np.array([distNN, distNNN]); ax0 = np.array([distN3, 0])
by0 = 2.0*pi/(distNNN)*np.array([0, 1.0])
bx0 = 2.0*pi/(distNNN)*np.array([np.sqrt(3.0)/2.0, -1.0/2.0])
#print 'adb =', np.dot(ax0, bx0), np.dot(ay0, by0), np.dot(ax0, by0), np.dot(ay0, bx0)

tunit, Vunit = unit_kagome(V1 = vNN, V2 = vNNN, t1c=1e-14) # default values are for t1-V1 only model
#print 'Vunit = ', Vunit
UT = GetKagomeUnitaryTransformV1(tunit, Length, Height/nUnit)

#exit()
Vint_q = get_LatticeInteraction_FourierTransform(Vunit, Length, Height/nUnit)

" Define projected basis function "
Ecut = -1.9 #-1.5 #-2.0+1.5 # Energy cut for projection
Ecut1 = -0.0 # Energy cut for projection

print 'Ecut = ', Ecut
print 'vNN = ', vNN, 'vNNN = ', vNNN

SingleParticleBas, OnSite, kBand, NumSPB = get_Basisfun_Ecut(UT, Ecut, Length, Height/nUnit)
print SingleParticleBas, NumSPB, OnSite, kBand
print 'number bas =', sps.comb(NumSPB, n)
#exit()

#kagome_self_eng(UT, Vint_q, SingleParticleBas, (1,2), Length, Height/nUnit)
#exit()

Vjjt = get_Kagome_IntEle(SingleParticleBas, NumSPB, UT, Vint_q, Length, Height/nUnit, nUnit)

print 'keys', Vjjt[(0,1)].keys()

StatMom, num_kbasis = get_ManyParticleBasisfun(SingleParticleBas, NumSPB, n, Length, Height/nUnit)
print num_kbasis
#print 'StatM', StatMom.keys()

print num_kbasis, 'size StatMom =', sys.getsizeof(StatMom)/1024.0/1024.0/1024.0, 'G'

eigekk = np.zeros((Length*Height/nUnit, eigv_k),)
for ii in range(0,Length*Height/nUnit):
  stime = time.time()
  sq2bas, bas2sq = get_newmap(StatMom[ii], num_kbasis[ii], NumSPB)
  #print 'bas old ', ii, '\n', StatMom[ii]
  #print 'bas new ', ii, '\n', sq2bas
  etime = time.time()
  print ''
  print '----------------------------------------'
  print 'ii=', ii, 'time =', etime - stime

  stime = time.time()
  row_s, col_s, data_s = get_Kagome_IntMatEle(bas2sq, sq2bas, Vjjt, OnSite)
  etime = time.time()
  print 'size data_s =', sys.getsizeof(data_s)/1024.0/1024.0/1024.0, 'G'
  print 'time of get matrix element =', etime - stime

  Ham_triple = sp.sparse.coo_matrix((data_s,(row_s,col_s)), shape = (num_kbasis[ii], num_kbasis[ii]))

  stime = time.time()
  Eige0, Eigf0 = eigsh(Ham_triple, k=eigv_k, which = mode1)
  #Eige0 = np.diag(Ham_triple.toarray())
  #Eigf0 = Ham_triple.toarray()

  etime = time.time()
  print 'size Eigf0 =', sys.getsizeof(Eigf0)/1024.0/1024.0/1024.0, 'G'
  print 'time of get eigenvalue =', etime - stime
  print Eige0
  #eigekk[ii,:] = sorted(Eige0)
  eigekk[ii,0:len(Eige0)] = Eige0

  #alp = Ham_triple.toarray() 
  #print alp
  #print 'mx =', np.amax(np.amax(abs(alp-np.conjugate(alp.T))))
  #eff = list(abs(Eigf0[:,1]))
  eff = list(np.conjugate(Eigf0[:,0])*Eigf0[:,0])
  mindex = eff.index(max(eff))
  print 'sum', sum(eff), 'max value', max(eff), 'pos', mindex, 'basis', StatMom[ii][mindex]

#  fig = plt.figure(1)
#  plt.clf()
#  plt.plot(range(Length*Height/nUnit), eigekk-eigekk[0, 0], 'b_')
#  plt.ylim(-0.1, 0.3)
#  plt.pause(1)
  #plt.plot(range(len(eff)), eff, '-ob')
  #plt.show()
np.savetxt('EigeH%dL%dV1_%d_V2_%d_Ec_1.txt'%(Height,Length,int(vNN*10), int(vNNN*10)), eigekk)
#plt.show()
