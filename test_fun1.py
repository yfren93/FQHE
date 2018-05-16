#!/usr/bin/env python

from sparse_Ham_fun import *
import numpy as np
import scipy as sp
import scipy.special as sps
import itertools
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from lattice import *

global Height, Length, nNNcell, pbcx, pbcy, N_site, n_electron, distNN, distNNN, distN3, Lattice_type
# define constants
pi = 3.1415926535897932

nUnit = 3
N = N_site
n = n_electron #N/nUnit

" Get single particle states in momentum space "
#UT = GetKagomeUnitaryTransform(phi0=1e-14)

vNN, vNNN, vN3 = 1.0, 0., 0.0
Flags={}
Flags['Measure_density'] = False #True

" Define the fourier transform of interaction "
ay0 = np.array([distNN, distNNN]); ax0 = np.array([distN3, 0])
by0 = 2.0*pi/(distNNN)*np.array([0, 1.0])
bx0 = 2.0*pi/(distNNN)*np.array([np.sqrt(3.0)/2.0, -1.0/2.0])
#print 'adb =', np.dot(ax0, bx0), np.dot(ay0, by0), np.dot(ax0, by0), np.dot(ay0, bx0)

tunit, Vunit = unit_kagome(V1 = vNN, t1c=1e-14) # default values are for t1-V1 only model
#print 'Vunit = ', Vunit
UT = GetKagomeUnitaryTransformV1(tunit, Length, Height/nUnit)

#exit()
Vint_q = get_LatticeInteraction_FourierTransform(Vunit, Length, Height/nUnit)

" Define projected basis function "
Ecut = 5.0 #-2.0+1.5 # Energy cut for projection
Ecut1 = -0.0 # Energy cut for projection

print 'Ecut = ', Ecut
print 'vNN = ', vNN, 'vNNN = ', vNNN

SingleParticleBas, OnSite, kBand, NumSPB = get_Basisfun_Ecut(UT, Ecut, Length, Height/nUnit)
print SingleParticleBas, NumSPB, OnSite, kBand

Vjjt = get_Kagome_IntEle(SingleParticleBas, NumSPB, UT, Vint_q, Length, Height/nUnit, nUnit)

print 'keys', Vjjt[(0,1)].keys()

StatMom, num_kbasis = get_ManyParticleBasisfun(SingleParticleBas, NumSPB, n, Length, Height/nUnit)
print num_kbasis
print 'StatM', StatMom.keys()
mode1 = 'SA'
eigv_k = 4

for ii in range(0,1): #Length*Height/nUnit):
  sq2bas, bas2sq = get_newmap(StatMom[ii], num_kbasis[ii], NumSPB)
  #print 'bas old ', ii, '\n', StatMom[ii]
  print 'bas new ', ii, '\n', sq2bas

  row_s, col_s, data_s = get_Kagome_IntMatEle(bas2sq, sq2bas, Vjjt, OnSite)
  Ham_triple = sp.sparse.coo_matrix((data_s,(row_s,col_s)), shape = (num_kbasis[ii],num_kbasis[ii]))
  Eige0, Eigf0 = eigsh(Ham_triple, k=eigv_k, which = mode1)
  print sorted(Eige0)
  alp = Ham_triple.toarray() 
  print alp
  print 'mx =', np.amax(np.amax(abs(alp-np.conjugate(alp.T))))
exit()

#for ii in range(0,9):
#  print 'Stat =',ii,' \n', StatMom[ii]
#print 'Stat =',ii,' \n', StatMom[0]

numnonzero = sps.comb(n,2)*sps.comb(NumSPB+2-n,2)
mode1 = 'SA'
eigv_k =10 
Eige = np.zeros((eigv_k*N/nUnit,)) # !!!!!! here define the k number
#print 'shape Eige ', np.shape(Eige)
numss = 0
# Define Hamiltonian of each total momentum
for kk in range(0, Length*Height/nUnit): # Momentum index
  kx = np.mod(kk,Length)
  ky = kk/Length
 
  # define block Hamiltonian
  '''
  Define sparsed Hamiltonian in triple format: row, column and data
  '''
  row_s  = np.zeros(( int(num_kbasis[kk]*(numnonzero)*1.5), ),dtype=int)
  col_s  = np.zeros(( int(num_kbasis[kk]*(numnonzero)*1.5), ),dtype=int)
  data_s = np.zeros(( int(num_kbasis[kk]*(numnonzero)*1.5), ),dtype=complex)
  num_nonzero = -1

  for statk in range(0,num_kbasis[kk]):
    bas = list(StatMom[kk][statk])
    # Diagonal terms from on-site energy
    num_nonzero += 1
    row_s[num_nonzero]  = statk
    col_s[num_nonzero]  = statk
    data_s[num_nonzero] = sum(OnSite[list(bas)])

    # Off diagonalize terms from interactions
    for ii0 in range(0,n):   # ii0, jj0 indicates the position of single particle state in a basis function
      kxi = SingleParticleBas[bas[ii0]][0]
      kyi = SingleParticleBas[bas[ii0]][1]

      for jj0 in range(ii0+1,n): # set(range(0,n))-set([ii0]): #range(ii0+1,n): #set(range(0,n))-set([ii0]): #
        kxj = SingleParticleBas[bas[jj0]][0]
        kyj = SingleParticleBas[bas[jj0]][1]

        for qq in range(0, Length*Height/nUnit):
          qx = np.mod(qq,Length)
          qy = qq/Length
          qqm = np.mod(-qy,Height/nUnit)*Length+np.mod(-qx,Length)

	  kxp, kyp = np.mod(kxi + qx, Length), np.mod(kyi + qy, Height/3)
	  kxm, kym = np.mod(kxj - qx, Length), np.mod(kyj - qy, Height/3)

          for ii1 in kBand[tuple([kxp,kyp])]:   # ii1 and jj1 are band index
            for jj1 in kBand[tuple([kxm,kym])]:
              bas1 = []
              bas1[:] = bas[:]
              bas1[ii0] = SingleParticleBas[tuple([kxp,kyp,ii1])]
              bas1[jj0] = SingleParticleBas[tuple([kxm,kym,jj1])]
              if np.shape(np.array(list(bas1)))[0] > np.shape(np.array(list(set(bas1))))[0]:
                continue
              
              pm_time = permute_time3(tuple(bas1),list(sorted(bas1)))
              iib0 = SingleParticleBas[bas[ii0]][2]
              jjb0 = SingleParticleBas[bas[jj0]][2]
              Walpha = np.array(UT[kxp + kyp * Length]['eigf'])[:,ii1] \
	             * np.conjugate(np.array(UT[kxi + kyi * Length]['eigf'])[:,iib0])
              Wbeta  = np.array(UT[kxm + kym * Length]['eigf'])[:,jj1] \
	             * np.conjugate(np.array(UT[kxj + kyj * Length]['eigf'])[:,jjb0])
              
              statkp = StatMom[kk][tuple(sorted(bas1))]
              num_nonzero += 1
              row_s[num_nonzero]  = statk
              col_s[num_nonzero]  = statkp
              data_s[num_nonzero] = (-1)**pm_time*np.dot(Walpha,np.dot(Vint_q[qqm],Wbeta))/(1.0*Length*Height/nUnit)

  print 'non zero', num_nonzero
  Ham_triple = sp.sparse.coo_matrix((data_s,(row_s,col_s)), shape = (num_kbasis[kk],num_kbasis[kk]))
  Eige0, Eigf0 = eigsh(Ham_triple, k=eigv_k, which = mode1)
  print sorted(Eige0)
  Eige[kk*eigv_k:(kk+1)*eigv_k] = Eige0

  if Flags['Measure_density'] :
    Tstat = 2
    Gstat = np.zeros((Tstat,NumSPB))
    for iiM0 in range(0,Tstat):
      for iiM1 in range(0,num_kbasis[kk]):
        Gstat[iiM0,list(StatMom[kk][iiM1])] += abs(Eigf0[iiM1,iiM0])**2
    print 'Gstat = \n', Gstat
    np.savetxt('Gstatk'+str(kk)+'.txt',Gstat)
    #plt.figure(1)
    #plt.plot(range(0,NumSPB),Gstat[0,:],'-ro')
    #plt.plot(range(0,NumSPB),Gstat[1,:],'-bo')
    #plt.show()

print Eige
#pathd = '/home/yfren/Dropbox/csun/python_codes/kagome_lattice/MomentumSpace/test_time_converg/data'
#np.savetxt(pathd+'/Eige_V12_05_Ec1.txt',np.array(Eige))



