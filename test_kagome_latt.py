#!/bin/usr/env python

'''
--------------------------------------------------------
               MAIN PROGRAM
--------------------------------------------------------
'''
import numpy as np
import scipy as sp
#from sparse_Ham_fun import *
#from re_basis_fun import *
from ED_basis_fun import *
from lattice import *
import time
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import gc

N1, N2, n_unit = Height/3, Length, 3
N, n = N_site, n_electron

tot_coordinate = 4

#-----------------------

vNN0 = np.linspace(0,1,6) # np.linspace(1.5,5,8)

"Define calculation mode"
mode1 = 'SA'
eigv_k = 2 # 10
Eige = np.zeros((len(vNN0),eigv_k*N1*N2))
print 'shape Eige ', np.shape(Eige)

#-----------------------

tNN = 1.0

"Define momentum"
phase_x = np.linspace(0,2,N2+1)
phase_y = np.linspace(0,2,N1+1)

#-----------------------

ts = time.time()
Neib, Neib1, Neib2 = get_kagome_neighbor_sites(Height,Length,nNNcell)
te = time.time()
print 'get neighbor time ', te - ts

"Define new basis"
ts = time.time()
renormalize_factor, map_dob, dob1, num_stat1 = trans_symm_basis(N, n, N1, N2, n_unit)
num_basis = len(map_dob)
num_basis_block = len(renormalize_factor)
te = time.time()
print 'time to get block diagonalized basis function: ', te - ts

ts = time.time()
DiagInt = get_Diagonal_init(renormalize_factor, dob1, N, map_dob, num_stat1, Neib, Neib1, Neib2)
te = time.time()
print 'time to initialize interaction term: ', te - ts

ts = time.time()
off_hop = get_OffDiag_init(renormalize_factor, dob1, N, n, tNN, Neib, num_stat1, map_dob)
len_offd = len(off_hop)
te = time.time()
print 'time to initialize hopping term: ', te - ts

ts = time.time()
#del renormalize_factor, dob1, map_dob, num_stat1
#del dob1, map_dob, num_stat1
gc.collect()
te = time.time()
print 'time to delete maps: ', te - ts

for iik in range(0, 1): #N1*N2):  # for each momentum iik
  print '--------------------------- %d ------------------'%(iik)

  ts = time.time()

  iix = np.mod(iik,N2)
  iiy = iik/N2
  phase_N = np.zeros((N1*N2,), dtype = complex) # phase for each translation operation
  for ii in range(0,N2):
    for jj in range(0,N1):
      phase_N[ii+jj*N2] = np.exp(1j*pi*(ii*phase_x[iix]+jj*phase_y[iiy]))
 
  "Initialize hopping energy" 
  row_s = [] 
  col_s = [] 
  data_s = [] 

  for iof in range(len(off_hop)):
    offele = off_hop[iof]
    row_s += [offele[0]]
    col_s += [offele[1]]
    data_s += [np.dot(offele[2],phase_N[offele[3]])]
  
  row_s += range(0,num_basis_block) 
  col_s += range(0,num_basis_block)
 
  te = time.time()
  print 'time of initialize hopping term', te - ts
  
  for iivNN in range(0,1): #range(0,len(vNN0)):
    vNN  = 4.0 #vNN0[iivNN]
    vNNN = vNN/2.0 # (3.0*np.sqrt(3.0)) #vNN0[iivNN]
    vN3  = vNN/8.0

    print 'vNN', iivNN, '=', vNN

    ts = time.time()
    for nV123 in DiagInt:
      data_s += [np.dot(nV123, [vNN,vNNN,vN3])/2.0]
    te = time.time()
    print 'time of get diagonal interaction term', te - ts
    print 'max data_s', max(np.imag(data_s))
    ts = time.time()
    Ham_triple = sp.sparse.coo_matrix((np.real(data_s),(row_s, col_s)), shape = (num_basis_block, num_basis_block))
    Eige0, Eigf0 = eigsh(Ham_triple, k=eigv_k, which = mode1)
    print Eige0 
    #print Eigf0 
    Eige[iivNN,iik*eigv_k:(iik+1)*eigv_k] = Eige0
  
    te = time.time()
    print 'eige time', te-ts
    data_s = data_s[:len_offd]

y, z = 1, 0
ts = time.time()
ni, nj = 0, 1
hopij = get_hop_matrixele(renormalize_factor, dob1, N, n, tNN, Neib, num_stat1, map_dob, ni, nj)
te = time.time()
print 'time to get hopping matrix elements: ', te - ts
cur = np.zeros((4,), dtype = complex)
for hel in hopij:
  cur[0] +=     np.conjugate(Eigf0[hel[y],0])*Eigf0[hel[z],0]*hel[2]
  cur[1] +=  1j*np.conjugate(Eigf0[hel[y],0])*Eigf0[hel[z],1]*hel[2]
  cur[2] += -1j*np.conjugate(Eigf0[hel[y],1])*Eigf0[hel[z],0]*hel[2]
  cur[3] +=     np.conjugate(Eigf0[hel[y],1])*Eigf0[hel[z],1]*hel[2]
print 'ni=%d, nj=%d, '%(ni, nj), cur
ts = time.time()
ni, nj = nj, ni
hopij = get_hop_matrixele(renormalize_factor, dob1, N, n, tNN, Neib, num_stat1, map_dob, ni, nj)
te = time.time()
print 'time to get hopping matrix elements: ', te - ts
cur1 = np.zeros((4,), dtype = complex)
for hel in hopij:
  cur1[0] +=     np.conjugate(Eigf0[hel[y],0])*Eigf0[hel[z],0]*hel[2]
  cur1[1] +=  1j*np.conjugate(Eigf0[hel[y],0])*Eigf0[hel[z],1]*hel[2]
  cur1[2] += -1j*np.conjugate(Eigf0[hel[y],1])*Eigf0[hel[z],0]*hel[2]
  cur1[3] +=     np.conjugate(Eigf0[hel[y],1])*Eigf0[hel[z],1]*hel[2]
print 'ni=%d, nj=%d, '%(ni, nj), cur1, '-------', sum(cur-cur1)/2

ts = time.time()
ni, nj = 1, 2
hopij = get_hop_matrixele(renormalize_factor, dob1, N, n, tNN, Neib, num_stat1, map_dob, ni, nj)
te = time.time()
print 'time to get hopping matrix elements: ', te - ts
cur = np.zeros((4,), dtype = complex)
for hel in hopij:
  cur[0] +=     np.conjugate(Eigf0[hel[y],0])*Eigf0[hel[z],0]*hel[2]
  cur[1] +=  1j*np.conjugate(Eigf0[hel[y],0])*Eigf0[hel[z],1]*hel[2]
  cur[2] += -1j*np.conjugate(Eigf0[hel[y],1])*Eigf0[hel[z],0]*hel[2]
  cur[3] +=     np.conjugate(Eigf0[hel[y],1])*Eigf0[hel[z],1]*hel[2]
print 'ni=%d, nj=%d, '%(ni, nj), cur
ts = time.time()
ni, nj = nj, ni
hopij = get_hop_matrixele(renormalize_factor, dob1, N, n, tNN, Neib, num_stat1, map_dob, ni, nj)
te = time.time()
print 'time to get hopping matrix elements: ', te - ts
cur1 = np.zeros((4,), dtype = complex)
for hel in hopij:
  cur1[0] +=     np.conjugate(Eigf0[hel[y],0])*Eigf0[hel[z],0]*hel[2]
  cur1[1] +=  1j*np.conjugate(Eigf0[hel[y],0])*Eigf0[hel[z],1]*hel[2]
  cur1[2] += -1j*np.conjugate(Eigf0[hel[y],1])*Eigf0[hel[z],0]*hel[2]
  cur1[3] +=     np.conjugate(Eigf0[hel[y],1])*Eigf0[hel[z],1]*hel[2]
print 'ni=%d, nj=%d, '%(ni, nj), cur1, '------', sum(cur-cur1)/2

ts = time.time()
ni, nj = 2, 0
hopij = get_hop_matrixele(renormalize_factor, dob1, N, n, tNN, Neib, num_stat1, map_dob, ni, nj)
te = time.time()
print 'time to get hopping matrix elements: ', te - ts
cur = np.zeros((4,), dtype = complex)
for hel in hopij:
  cur[0] +=     np.conjugate(Eigf0[hel[y],0])*Eigf0[hel[z],0]*hel[2]
  cur[1] +=  1j*np.conjugate(Eigf0[hel[y],0])*Eigf0[hel[z],1]*hel[2]
  cur[2] += -1j*np.conjugate(Eigf0[hel[y],1])*Eigf0[hel[z],0]*hel[2]
  cur[3] +=     np.conjugate(Eigf0[hel[y],1])*Eigf0[hel[z],1]*hel[2]
print 'ni=%d, nj=%d, '%(ni, nj), cur
ts = time.time()
ni, nj = nj, ni
hopij = get_hop_matrixele(renormalize_factor, dob1, N, n, tNN, Neib, num_stat1, map_dob, ni, nj)
te = time.time()
print 'time to get hopping matrix elements: ', te - ts
cur1 = np.zeros((4,), dtype = complex)
for hel in hopij:
  cur1[0] +=     np.conjugate(Eigf0[hel[y],0])*Eigf0[hel[z],0]*hel[2]
  cur1[1] +=  1j*np.conjugate(Eigf0[hel[y],0])*Eigf0[hel[z],1]*hel[2]
  cur1[2] += -1j*np.conjugate(Eigf0[hel[y],1])*Eigf0[hel[z],0]*hel[2]
  cur1[3] +=     np.conjugate(Eigf0[hel[y],1])*Eigf0[hel[z],1]*hel[2]
print 'ni=%d, nj=%d, '%(ni, nj), cur1, '------', sum(cur-cur1)/2

#np.savetxt('k_Eige_V2V3_3x.txt',Eige)

#axx0 = np.loadtxt('k_Eige_V2V3.txt')
#axx1 = np.loadtxt('k_Eige_V2V3_x.txt')
#print 'xxdff', np.amax(abs(np.array(sorted(axx0[0,:])) - np.array(sorted(axx1[0,:]))))
#np.savetxt('k_Eige_V2V3_3x.txt',Eige)

#axx0 = np.loadtxt('k_Eige_V2V3.txt')
#axx1 = np.loadtxt('k_Eige_V2V3_x.txt')
#print 'xxdff', np.amax(abs(np.array(sorted(axx0[0,:])) - np.array(sorted(axx1[0,:]))))
#np.savetxt('k_Eige_V2V3_3x.txt',Eige)

#axx0 = np.loadtxt('k_Eige_V2V3.txt')
#axx1 = np.loadtxt('k_Eige_V2V3_x.txt')
#print 'xxdff', np.amax(abs(np.array(sorted(axx0[0,:])) - np.array(sorted(axx1[0,:]))))
