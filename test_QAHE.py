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

vNN, vNNN, vN3 = 1.0, 1e-9, 1e-9
Flags={}
Flags['Measure_density'] = False #True

" Define the fourier transform of interaction "
ay0 = np.array([distNN, distNNN]); ax0 = np.array([distN3, 0])
by0 = 2.0*pi/(distNNN)*np.array([0, 1.0])
bx0 = 2.0*pi/(distNNN)*np.array([np.sqrt(3.0)/2.0, -1.0/2.0])
#print 'adb =', np.dot(ax0, bx0), np.dot(ay0, by0), np.dot(ax0, by0), np.dot(ay0, bx0)

tunit, Vunit = unit_kagome(V1 = vNN, V2 = vNNN, V3 = vN3, t1c=1e-14) # default values are for t1-V1 only model
#print 'Vunit = ', Vunit
UT = GetKagomeUnitaryTransformV1(tunit, Length, Height/nUnit)

#exit()
Vint_q = get_LatticeInteraction_FourierTransform(Vunit, Length, Height/nUnit)

" Define projected basis function "
Ecut = 5 #-1.5 #-2.0+1.5 # Energy cut for projection
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

basG = np.zeros((Length*Height,), dtype = int)
for ii in range(Length*Height/nUnit):
  basG[ii*nUnit] = 1

Eige_ptb = np.zeros((Length*Height/nUnit, eigv_k),)
eigekk = np.zeros((Length*Height/nUnit, eigv_k),)
for ii in range(0,Length*Height/nUnit):
  basG1 = np.zeros((Length*Height,), dtype = int)
  basG1 += basG
  basG1[ii*3] = 0
  basG1[1] = 1
  #print 'basG1 =', basG1, np.dot(basG1, OnSite)

  bas = tuple(basG1)
  #Ham_ptb = PerturbativeCal(bas, Vjjt, OnSite)
  #Eige0, Eigf0 = eigsh(Ham_ptb, which = mode1, k = eigv_k)
  #Eige_ptb[ii, :] = Eige0

  stime = time.time()

  sq2bas, bas2sq, numbas = get_perturb_bas(bas, Vjjt)
  print 'numbas ', ii, numbas
  #row_s, col_s, data_s = get_perturb_Ham(sq2bas, bas2sq, Vjjt, OnSite)
  #Ham_triple1 = sp.sparse.coo_matrix((data_s,(row_s,col_s)), shape = (numbas+1,numbas+1))
  #Hf1 = Ham_triple1.toarray()

  row_s, col_s, data_s = get_perturb_Ham_new(sq2bas, bas2sq, Vjjt, OnSite)
  #Ham_triple2 = sp.sparse.coo_matrix((data_s,(row_s,col_s)), shape = (numbas+1,numbas+1))
  #Hf2 = Ham_triple2.toarray()

  #Hd = abs(Hf2 - Hf1)
  #indx = np.where(Hd == np.amax(np.amax(Hd)))
  #print 'Hdiff \n', np.where(Hd == np.amax(np.amax(Hd)))
  #print 'hf1 =', Hf1[indx[0][0],indx[0][1]] 
  #print 'hf2 =', Hf2[indx[0][0],indx[0][1]] 
  #print sq2bas(indx[0][0])
  #print sq2bas(indx[0][1])
  #exit()

  etime = time.time()
  print 'size data_s =', sys.getsizeof(data_s)/1024.0/1024.0/1024.0, 'G'
  print 'time of get matrix element =', etime - stime

  Ham_triple = sp.sparse.coo_matrix((data_s,(row_s,col_s)), shape = (numbas+1,numbas+1))

  #Hamf = Ham_triple.toarray()
  #aa1 = abs(Hamf[:,844])
  #aa2 = abs(Hamf[:,511])
  #fig33 = plt.figure(33)
  #ax33 = fig33.add_subplot(111)
  #ax33.plot(range(len(aa1)), aa1, '-sr')
  #ax33.plot(range(len(aa2)), aa2, '-ob')
  #plt.show()

  stime = time.time()
  Eige0, Eigf0 = eigsh(Ham_triple, k=eigv_k, which = mode1)
  #Eige0 = np.diag(Ham_triple.toarray())
  #Eigf0 = Ham_triple.toarray()

  etime = time.time()
  print 'size Eigf0 =', sys.getsizeof(Eigf0)/1024.0/1024.0/1024.0, 'G'
  print 'time of get eigenvalue =', etime - stime
  print Eige0
  #eigekk[ii,:] = sorted(Eige0)
  eigekk[ii, 0:len(Eige0)] = Eige0


#Ham_ptb = PerturbativeCal(tuple(basG), Vjjt, OnSite)
#Eige0, Eigf0 = eigsh(Ham_ptb, which = mode1, k = eigv_k)
#Eige_ptb[Length*Height/nUnit, :] = Eige0

np.savetxt('eiekk_L%dH%d_V1_1_V23_0.txt'%(Length, Height), eigekk)

fig = plt.figure(26)
ax = fig.add_subplot(111)
#Eg = np.amin(np.amin(Eige_ptb))
#for ii in range(np.shape(Eige_ptb)[0]):
#  ax.plot([np.mod(ii,Length*Height/nUnit)]*eigv_k, Eige_ptb[ii,:]-Eg, '+b')
Eg = np.amin(eigekk)
plt.title('Eg = '+str(Eg))
plt.plot(range(Length*Height/nUnit), eigekk-Eg, 'b+')
#plt.savefig('PerturbEige_L%dH%d_V12_1.eps'%(Length,Height), format = 'eps')

plt.show()

exit()













#V00, V01, V11, V02 = classify_VjjK(Vjjt, SingleParticleBas) 
#fig = plt.figure(22)
#ax = fig.add_subplot(1,1,1)
#ax.plot(range(len(V00)), abs(np.array(V00)), 'sr',label='00')
#ax.plot(range(len(V01)), abs(np.array(V01)), '^g',label='01')
#ax.plot(range(len(V11)), abs(np.array(V11)), 'ob',label='11')
#ax.plot(range(len(V02)), abs(np.array(V02)), '+k',label='02')
#ax.legend()
#plt.show()
#exit()

StatMom, num_kbasis = get_ManyParticleBasisfun(SingleParticleBas, NumSPB, n, Length, Height/nUnit)
print num_kbasis
print 'StatM', StatMom.keys()

print num_kbasis, 'size StatMom =', sys.getsizeof(StatMom)/1024.0/1024.0/1024.0, 'G'

eigekk = np.zeros((Length*Height/nUnit, eigv_k),)
for ii in range(0,Length*Height/nUnit):
  stime = time.time()
  sq2bas, bas2sq = get_newmap(StatMom[ii], num_kbasis[ii], NumSPB)
  #print 'bas old ', ii, '\n', StatMom[ii]
  #print 'bas new ', ii, '\n', sq2bas
  etime = time.time()
  print 'ii=', ii, 'time =', etime - stime

  stime = time.time()
  row_s, col_s, data_s = get_Kagome_IntMatEle(bas2sq, sq2bas, Vjjt, OnSite)
  etime = time.time()
  print 'size data_s =', sys.getsizeof(data_s)/1024.0/1024.0/1024.0, 'G'
  print 'time of get matrix element =', etime - stime

  Ham_triple = sp.sparse.coo_matrix((data_s,(row_s,col_s)), shape = (num_kbasis[ii], num_kbasis[ii]))

  #Hamf = Ham_triple.toarray()
  #aa1 = abs(Hamf[:,844])
  #aa2 = abs(Hamf[:,511])
  #fig33 = plt.figure(33)
  #ax33 = fig33.add_subplot(111)
  #ax33.plot(range(len(aa1)), aa1, '-sr')
  #ax33.plot(range(len(aa2)), aa2, '-ob')
  #plt.show()

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

  fig = plt.figure(1)
  plt.clf()
  plt.plot(range(Length*Height/nUnit), eigekk-eigekk[0, 0], 'b_')
  plt.ylim(-0.1, 0.3)
  plt.pause(1)
  #plt.plot(range(len(eff)), eff, '-ob')
  #plt.show()
np.savetxt('EigeH%dL%dV1_2_V2_0_Ec_1.txt'%(Height,Length), eigekk)
plt.show()
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



