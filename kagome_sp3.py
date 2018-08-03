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

ts = time.time()
global Height, Length, nNNcell, pbcx, pbcy, N_site, n_electron, Lattice_type
global pi, distNN, distNNN, distN3 

N1, N2, n_unit = Height/3, Length, 3

N, n = N_site, n_electron

vNN0 = np.linspace(0,1,6) # np.linspace(1.5,5,8)
"Define calculation mode"
Ham_mod = 'sparse'
#Ham_mod = 'full'
if (Ham_mod == 'sparse') :
  mode1 = 'SA'
  eigv_k = 10
  Eige = np.zeros((len(vNN0),eigv_k*N1*N2))
  print 'shape Eige ', np.shape(Eige)

"Define neighbors"
tot_coordinate = 4
Neib = np.zeros((N,tot_coordinate),dtype = int)
Neib1 = np.zeros((N,4),dtype = int)
Neib2 = np.zeros((N,2),dtype = int)

distNc, pos_x, pos_y = pos_kagome()

for ii in range(0,N):
  num, num1, num2 = -1, -1, -1 
  for jj in set(range(0,N))-set([ii]):
    for kk in range(0,nNNcell):
      if (abs(distNc[ii,jj,kk]-distNN) < 0.01):
        num += 1
        Neib[ii,num] = jj
      elif (abs(distNc[ii,jj,kk]-distNNN) < 0.01):
        num1 += 1
        Neib1[ii,num1] = jj
      elif (abs(distNc[ii,jj,kk]-distN3) < 0.01):
        if ( (np.mod(ii,3)==0) and (ii/Height != jj/Height) and (np.mod(ii,Height)!=np.mod(jj,Height)) ):
          num2 += 1
          print ii,jj,num2
          Neib2[ii,num2] = jj
        elif ( (np.mod(ii,3)==1) and (ii/Height != jj/Height) and (np.mod(ii,Height)==np.mod(jj,Height)) ):
          num2 += 1
          print ii,jj,num2
          Neib2[ii,num2] = jj
        elif ( (np.mod(ii,3)==2) and (ii/Height == jj/Height) ): # and (np.mod(ii,Height)!=np.mod(jj,Height)) ):
          num2 += 1
          print ii,jj,num2
          Neib2[ii,num2] = jj
print 'Neib2 = ', Neib2

#plt.figure(10)
#for iix in range(-1,2):
#  for iiy in range(-1,2):
#    plt.plot(pos_x+iix*6+iiy*3,pos_y+iiy*3*np.sqrt(3),'o')
#for ii in range(len(pos_x)):
#  plt.text(pos_x[ii],pos_y[ii],str(ii))
#  for jj in range(0,2):
#    plt.plot([pos_x[ii],pos_x[Neib2[ii,jj]]],[pos_y[ii],pos_y[Neib2[ii,jj]]],'-')
#
#plt.show()

"Define momentum"
phase_x = np.linspace(0,2,N2+1)
phase_y = np.linspace(0,2,N1+1)

"Define new basis"
renormalize_factor, map_dob, dob1, num_stat1 = trans_symm_basis(N, n, N1, N2, n_unit)
num_basis = len(map_dob)
num_basis_block = len(renormalize_factor)
if (Ham_mod == 'full'):
  Eige = np.zeros((num_basis,))

tNN = 1.0
"Calculate hoppings"

te = time.time()
print 'start time ', te - ts

size_k = {}
size_k0 = np.zeros((N1*N2,),dtype = int)
sizeHam0 = 0
for iivNN in range(0,1): #range(0,len(vNN0)):
  vNN  = 1.0 #vNN0[iivNN]
  vNNN = vNN/(3.0*np.sqrt(3.0)) #vNN0[iivNN]
  vN3  = vNN/8.0

  print 'vNN',vNN
  for iik in range(0,N1*N2):
    ts = time.time()
  
    iix = np.mod(iik,N2)
    iiy = iik/N2
  
    size_k[iik] = [len(renormalize_factor)]
  
    # define phase for each translation operation
    phase_N = np.zeros((N1*N2,), dtype = complex)
    for ii in range(0,N2):
      for jj in range(0,N1):
        phase_N[ii+jj*N2] = np.exp(1j*pi*(ii*phase_x[iix]+jj*phase_y[iiy]))
  
    # define block Hamiltonian
    if (Ham_mod == 'full'):
      Hamk = np.zeros((len(renormalize_factor),len(renormalize_factor)),dtype = complex)
    elif (Ham_mod == 'sparse'):
      '''
      Define sparsed Hamiltonian in triple format: row, column and data
      '''
      row_s  = np.zeros(( num_basis_block*(tot_coordinate*n*2+1), ),dtype=np.int)
      col_s  = np.zeros(( num_basis_block*(tot_coordinate*n*2+1), ),dtype=np.int)
      data_s = np.zeros(( num_basis_block*(tot_coordinate*n*2+1), ),dtype=complex)
      num_nonzero = -1
  
    for mm1 in range(0,len(renormalize_factor)) : # calculate matrix elements for each basis
      bas = list(dob1[mm1][0]) # bas: tuple 
      n_bas = list(set(range(0,N))-set(bas))
  
      ii_norm1 = renormalize_factor[mm1]  
      # pass the line if the basis function has zero normalization
      if (ii_norm1 > 1) :
        num_bas = num_stat1[tuple(sorted(bas))]
        norm = 0j
        for ii in range(0, ii_norm1) :
          norm += phase_N[map_dob[num_bas][2*ii+1]]*(-1)**permute_time3(dob1[mm1][map_dob[num_bas][2*ii+1]],bas)
        if (abs(norm) < 1e-10) :
          size_k[iik][0] += -1 # count the size of Hamk
          size_k[iik] += [mm1] # record the zero-valued basis
          continue 
      
      "# define the diagonal terms"
      num_couple, num_couple1, num_couple2 = 0, 0, 0
      for iin in bas:
        num_couple += len( set(bas) & set(Neib[iin,:]) )
        num_couple1 += len( set(bas) & set(Neib1[iin,:]) )
        num_couple2 += len( set(bas) & set(Neib2[iin,:]) )

      if (Ham_mod == 'full'):
        Hamk[mm1,mm1] += num_couple/2.0*vNN + num_couple1/2.0*vNNN + num_couple2/2.0*vN3
      elif (Ham_mod == 'sparse'):
        num_nonzero += 1
        row_s[num_nonzero]  += mm1
        col_s[num_nonzero]  += mm1
        data_s[num_nonzero] += num_couple/2.0*vNN + num_couple1/2.0*vNNN + num_couple2/2.0*vN3

      #print 'bas[',mm1,']=', bas, 'Eint =', Hamk[mm1,mm1]
      "# define the coupling to basis with neighbors"
      for iin in range(0,n) :
        iin_neib = set() # find the possible replacement at each position of basis
        iin_neib = set(n_bas) & set(Neib[bas[iin],:])
  
        for iinn in iin_neib :
          "define neighbor basis"
          basNN = list([]) # neighbor basis
          basNN[:] = bas[:]
          basNN[iin] = iinn # define new basis with NN site
          
          "find order of this basis in block basis"
          # number of this neighbor basis in old order
          n_basNN = num_stat1[tuple(sorted(basNN))]
          
          # neighbor basis in new order
          mm2 = map_dob[n_basNN][0]
  
          # times that this basis appears in new basis
          pos_mm2 = []
          pos_mm2 = map_dob[n_basNN][1:2*renormalize_factor[mm2]:2]
          for term12 in pos_mm2: # calculate the coupling matrix elements
            if (Ham_mod == 'full'):
              Hamk[mm1,mm2] += tNN*(-1.0)**permute_time3(dob1[mm2][term12],basNN)*phase_N[term12]\
                             *1.0/np.sqrt((renormalize_factor[mm2]+0.0)*renormalize_factor[mm1])
            elif (Ham_mod == 'sparse'):
              num_nonzero += 1
              row_s[num_nonzero]  += mm1 
              col_s[num_nonzero]  += mm2
              data_s[num_nonzero] += tNN*(-1.0)**permute_time3(dob1[mm2][term12],basNN)*phase_N[term12]\
                             *1.0/np.sqrt((renormalize_factor[mm2]+0.0)*renormalize_factor[mm1])
    
    te = time.time()
    print 'Hamk time ', te - ts
  
    ts = time.time()
    # delete extra columns and rows
    if (Ham_mod == 'full'):
      if (size_k[iik][0]<len(renormalize_factor)):
        Hamk = np.delete(Hamk,size_k[iik][1:len(size_k[iik])],0)
        Hamk = np.delete(Hamk,size_k[iik][1:len(size_k[iik])],1)
  
      Eige[sizeHam0:sizeHam0+size_k[iik][0]] = np.real(np.linalg.eigvals(Hamk))
      sizeHam0 += size_k[iik][0]
      size_k0[iik] = size_k[iik][0]
  
    elif (Ham_mod == 'sparse'):
      Ham_triple = sp.sparse.coo_matrix((data_s,(row_s,col_s)), shape = (num_basis_block,num_basis_block))
      Eige0, Eigf0 = eigsh(Ham_triple, k=eigv_k, which = mode1)
      #print Eige0 
      Eige[iivNN,iik*eigv_k:(iik+1)*eigv_k] = Eige0
  
    te = time.time()
    print 'eige time', te-ts

#Eiges = np.zeros((len(Eige),),dtype=complex)
#print np.shape(Eige)
#Eiges = sorted(Eige)
#print Eiges
#np.savetxt('k_Eige_s.txt',Eiges)
np.savetxt('k_Eige_V2V3.txt',Eige)

