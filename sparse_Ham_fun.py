#!/usr/bin/env python

'''
*******************************************************************************
Global parameters
*******************************************************************************
'''
import numpy as np
from globalpara import *
# atomic chains in y and x directions
# Honeycomb lattice: y is zigzag, while x is armchair
# Kagome lattice: Height = 3*ny where ny is number of 
#                 unit cell along [1/2, sqrt(3)/2] direction 
#                 while nx is that along [1, 0] direction
#Height, Length = 3*2, 2 #3, 3

#nNNcell = 9 # number of neighbor unit cells
#pbcx, pbcy = 1, 1 # boundary condition: 1: true. 0: false
#N_site = Height * Length
#n_electron = N_site/3

pi = 3.1415926535897932

#Lattice_type = 'kagome'
#if (Lattice_type == 'honeycomb' or Lattice_type == 'kagome') :
#  distNN, distNNN, distN3 = 1.0, np.sqrt(3.0), 2 # distance between NN/NNN sites



'''
--------------------------------------------------------
 define permutation time /* from Fermion exchange */ for two states in different order
--------------------------------------------------------
'''
def permute_time3(a1,a12) : # a1 tuple, a12 list 
  import numpy as np
  n = len(a1)
  
  da1 = {a1[ii]:ii for ii in range(0,n)} # define a map from site number to order
  ds1 = range(0,n) 
  ds2 = range(0,n) 
  for ii in range(0,n):    # redefine the order 
    ds2[ii] = da1[a12[ii]] 
  
  permut = 0  # calculate permute time
  for ii in range(0,n-1) :  # move all right site to middle
    if ds2[ii] > ds1[ii] :
      permut += ds2[ii]-ds1[ii]
      for jj in range(ii+1,n):
        if (ds2[jj] < ds2[ii] and ds2[jj] >= ds1[ii]):
          ds2[jj] += 1      # when one site is moved to middle, the middle ones increase by one
  
      ds2[ii] = ds1[ii]

  return permut

'''
*******************************************************************************
Given two states bas and bas1 with only one different site 
Calculate the permute time if one make the different sites occupy the same position
in the states.
*******************************************************************************
'''
def permute_time(bas,bas1):
  import numpy as np

  bas1_0 = np.array(bas) # bas: tuple 
  bas1_1 = np.zeros((n_electron,1))
  bas1_1[:,0] = np.array(bas1) # bas1: tuple 
  d_sign0 = ((bas1_0-bas1_1)==0).nonzero()
  d_sign = d_sign0[0]-d_sign0[1]
  permute_time = np.mod( (sum(abs(d_sign))+abs(sum(d_sign)))/2, 2 )

  return permute_time
  
'''
*******************************************************************************
Given a row of numbers "stat" and total sites "N",
Return the order of this state in the basis functions
Suitable for spinless particle in any lattice
There are n numbers in stat, range from 0 to N-1
*******************************************************************************
'''
def num_stat(stat,N):
  import scipy as sp

  n = len(stat) # number of particles

  # Initialize nstate, start from 1
  if (n > 1): # more than one particle
    nstat = stat[n-1] - stat[n-2]
  else:       # single particle case
    nstat = stat[n-1]+1

  # Define order
  for ii in range(1,n): # summation over each site
    if ii == 1:
      lb = 1
    else: 
      lb = stat[ii-2]+1
      lb += 1 # number in stat is start from 0, while that in notes from 1

    ub = stat[ii-1]+1 # number in stat is start from 0, while that in notes from 1
    for jj in range(lb,ub):
      nstat += sp.special.factorial(N-jj)/(sp.special.factorial(n-ii)*sp.special.factorial(N+ii-n-jj))
  
  # Return the order, start from 0
  return int(nstat-1)


'''
*******************************************************************************
Define the position of each site in a cell depending on the lattice type and
boundary conditions                         
The position of sites in nearest 8 cells are also defined for pbc case
*******************************************************************************
'''
def pos_honeycomb(shapeCell = 'square'):
  
# import numpy as np
  import scipy as sp
  import itertools
  import time
  
  global Height, Length, nNNcell, pbcx, pbcy, N_site, n_electron, distNN, distNNN

  '''
  Intialize the lattice and boundary conditions
  '''   
  # define positions of each site
  N = N_site # Length*Height #Length*Height/2 # system size and electron number
  pos_x, pos_y = np.zeros((N,1)), np.zeros((N,1))

  # 'diamond' or 'square' shaped unit cell
  if (shapeCell == 'diamond'):
    for ii in range(0,Height):
      for jj in range(0,Length):
        numij = jj * Height + ii
        pos_x[numij,0] = jj*distNN*3.0/2.0 + np.mod(ii,2)*distNN/2.0
        pos_y[numij,0] = jj*distNNN/2.0    + ii*distNNN/2.0
    # define unit vectors for pbc
    a1 = np.array([0.0, distNNN])*Height/2; a2 = np.array([distNNN*np.sqrt(3.0)/2.0, distNNN/2.0])*Length

  elif (shapeCell == 'square'):
    print 'square unit cell'
    for ii in range(0,Height):
      for jj in range(0,Length):
        numij = jj * Height + ii
        pos_x[numij,0] = jj*distNN*3.0/2.0 - np.mod(ii+jj,2)*distNN/2.0
        pos_y[numij,0] = ii*distNNN/2.0
    # define unit vectors for pbc
    a1 = np.array([0.0, distNNN])*Height/2; a2 = np.array([distNN*3.0/2.0, 0])*Length

  "define distance between every two sites for both intra-cell and inter-cell cases"
  dist = np.sqrt((pos_x-pos_x.T)**2 + (pos_y-pos_y.T)**2) # Inside of unit cell
  distNc = np.zeros((N,N,nNNcell)) # Distance between sites in neighbor cells 
  for kk1 in range(0,3):
    for kk2 in range(0,3):
      pos_xb1 = pos_x + (kk1-1)*a1[0] + (kk2-1)*a2[0]; pos_yb1 = pos_y + (kk1-1)*a1[1] + (kk2-1)*a2[1]
      distNc[:,:,kk1*3+kk2] = np.sqrt( (pos_x-pos_xb1.T)**2 + (pos_y-pos_yb1.T)**2 )

#  # modification PBCs on atomic distances 
#  if ((pbcx==1) and (pbcy==1)) :    
#    for ii0 in range(0,N):
#      for jj0 in range(0,N):
#        dist[ii0,jj0] = min(distNc[ii0,jj0,:])

  return distNc


'''
*******************************************************************************
Define the position of each site in a cell depending on the lattice type and
boundary conditions                         
The position of sites in nearest 8 cells are also defined for pbc case
*******************************************************************************
'''
def pos_kagome(N_site0=N_site, Height0 = Height, Length0=Length):
  
# import numpy as np
  import scipy as sp
  import itertools
  import time
  
  global Height, Length, nNNcell, pbcx, pbcy, N_site, n_electron, distNN, distNNN, distN3

  '''
  Intialize the lattice and boundary conditions
  '''   
  # define positions of each site
  N = N_site0 # Length*Height #Length*Height/2 # system size and electron number
  pos_x, pos_y = np.zeros((N,1)), np.zeros((N,1))
  
  pos_x[0:3,0] = np.array([0,distNN/2,distNN]) # define the first three sites
  pos_y[0:3,0] = np.array([0,distNNN/2,0])

  for ii in range(1,Height0/3): # define the first column
    pos_x[ii*3:ii*3+3,0] = pos_x[0:3,0]+ii*distN3/2.0;
    pos_y[ii*3:ii*3+3,0] = pos_y[0:3,0]+ii*distN3*np.sqrt(3.0)/2.0

  for ii in range(1,Length0): # define the others
    pos_x[ii*Height0:(ii+1)*Height0,0] = pos_x[0:Height0,0] + distN3*ii
    pos_y[ii*Height0:(ii+1)*Height0,0] = pos_y[0:Height0,0] 

  # define unit vectors for pbc
  ay = np.array([distNN, distNNN])*Height0/3; ax = np.array([distN3, 0])*Length0

  "define distance between every two sites for both intra-cell and inter-cell cases"
  dist = np.sqrt((pos_x-pos_x.T)**2 + (pos_y-pos_y.T)**2) # Inside of unit cell
  distNc = np.zeros((N,N,nNNcell)) # Distance between sites in neighbor cells 
  for kky in range(0,3):    # for kky, kkx, ay, ax, xy indicate the two unit vectors
    for kkx in range(0,3):  # for posx, posy, xy indicate the x-y axis
      pos_xb1 = pos_x + (kky-1)*ay[0] + (kkx-1)*ax[0]; pos_yb1 = pos_y + (kky-1)*ay[1] + (kkx-1)*ax[1]
      distNc[:,:,kky*3+kkx] = np.sqrt( (pos_x-pos_xb1.T)**2 + (pos_y-pos_yb1.T)**2 )

  return distNc, pos_x, pos_y

'''
*******************************************************************************
Define the Hamiltonian of single particle kagome lattice for give momentum  
Momentum is defined by the periodic boundary condition of torus used
The eigenvalues and eigenstates of each state is given in UnitaryTransform
*******************************************************************************
'''
def GetKagomeUnitaryTransform(phi0 = 0.0):
  import numpy as np
 
  global Height, Length, nNNcell, pbcx, pbcy, N_site, n_electron, distNN, distNNN, distN3, Lattice_type
  # define constants
  pi = 3.1415926535897932
 
  N=N_site
 
  N0, Height0, Length0 = 3, 3, 1
  # define unit vectors
  ay = np.array([distNN, distNNN])*Height0/3; ax = np.array([distN3, 0])*Length0
  onsite=np.zeros((N0,))
 
  # hopping
  tNN = 1.0 #*np.sqrt(2);
  t2 = -0.0
  phi = phi0*pi;
  distNc, pos_x, pos_y = pos_kagome(N_site0=3, Height0=3, Length0=1)
  hopE = np.zeros([N0,N0,9])*0j
 
  for kky in range(0,3):
    for kkx in range(0,3):
      for ii1 in range(0,N0):
        for ii2 in range(0,N0):
          if (abs(distNc[ii1,ii2,kky*3+kkx]-distNN) < 0.01) :
            if np.mod(ii1-ii2,3) == 2:
              hopE[ii1,ii2,kky*3+kkx] += tNN*np.exp(phi*1j)
            else:
              hopE[ii1,ii2,kky*3+kkx] += tNN*np.exp(-phi*1j)
          elif (abs(distNc[ii1,ii2,kky*3+kkx]-distNNN) < 0.01) :
            hopE[ii1,ii2,kky*3+kkx] += t2

  # define the momentum 
  by = 2.0*pi/(distNNN)*np.array([0, 1.0])
  bx = 2.0*pi/(distNNN)*np.array([np.sqrt(3.0)/2.0, -1.0/2.0])
 
  UnitaryTransform = {}
  for iix in range(0, Length) :
    for iiy in range(0, Height/3) :
      UnitaryTransform[iiy*Length+iix]={} # count momentum from left to right & from bottom to top
      kx = iix*bx[0]/Length + iiy*by[0]/(Height/3)
      ky = iix*bx[1]/Length + iiy*by[1]/(Height/3)
 
      Ham = np.diag(onsite+0j)
      for kky in range(0,3) :
        for kkx in range(0,3) :
          posdx = (kky-1)*ay[0] + (kkx-1)*ax[0]; posdy = (kky-1)*ay[1] + (kkx-1)*ax[1]
          Ham += hopE[:,:,kky*3+kkx]*np.exp(1j*kx*posdx+1j*ky*posdy)
      Ham = (Ham + np.conjugate(Ham.T))/2
#      if (iiy*Length+iix) == 1 :
#        print 'Ham 1 \n', Ham
      eige,eigf = np.linalg.eigh(Ham)
 
      UnitaryTransform[iiy*Length+iix]['eige'] = eige
      UnitaryTransform[iiy*Length+iix]['eigf'] = eigf
#      print 'xx', iix, iiy, iiy*Length+iix 
  return UnitaryTransform


'''
*******************************************************************************
This program is used to find the number of a basis function 
| N1, N2, ..., Nn > with N1 < N2 < ... < Nn
for a spinless Fermionic system with N sites and n particles
*******************************************************************************
'''
def Ham_Operator(which = 'sp', Initialize_stat = 'Normal', vNN = 2.0, vNNN = 0.0, vN3 = 0.0) :   
  # Initialize_stat: 'Normal': calculate all terms 
  #                  'Initialize': only hop, 'N_Initialize': only diagonal term 

# import numpy as np
  import scipy as sp
  import itertools
  import time
  # from scipy.special import factorial
  
  global Height, Length, nNNcell, pbcx, pbcy, N_site, n_electron, distNN, distNNN, distN3, Lattice_type


  ## ---------------------------------  Lattice Dependent ----------------------------------- ##
  "Initialize positions and Hamiltonian parameters"
  if Lattice_type == 'honeycomb' :
    distNc = pos_honeycomb() # Distance between sites in neighbor cells 
    tot_coordinate = 3       # number of coordinate sites with << hopping >>
  elif Lattice_type == 'kagome' :
    distNc,pos_x,pos_y = pos_kagome()
    tot_coordinate = 4 


  ## ---------------------------------  Lattice InDependent ----------------------------------- ##
  N, n = N_site, n_electron # system size and electron number
  dist = np.zeros((N, N))
  # modification PBCs on atomic distances 
  if ((pbcx==1) and (pbcy==1)) :    
    for ii0 in range(0,N):
      for jj0 in range(0,N):
        dist[ii0,jj0] = min(distNc[ii0,jj0,:])
 
  ts = time.time() 
  "Initialize basis functions and their orders"
  num_basis = int(sp.special.comb(N,n)) # total number of basis functions
  order_basis = list( itertools.combinations(range(N),n) ) # basis functions presents in order
  te = time.time()
  print 'Bas num = ', num_basis
  print 'Bas time = ', te - ts

  # Interaction & hopping 
  # vNN, vNNN = 0.0, 0.3 # interactions
  tNN = -1.0          # hopping

  # Define on-site energy that depends on the boundary condition
  onsite = np.zeros((N,)) 
  if (Lattice_type == 'honeycomb') :
    uA1 = -3.0
    uA2, uB = 2.0, 0.
    site_A1 = [1, 10, 13, 22] 
    for ii in range(0,Height):
      for jj in range(0,Length):
        if (np.mod((ii+jj),2) == 0):
          onsite[ii+jj*Height] = uB
        else:
          if ((ii+jj*Height) in site_A1):
            onsite[ii+jj*Height] = uA1
          else:
            onsite[ii+jj*Height] = uA2

  #for ii in range(0,N):
  #  print 'site ',ii+1, ', E =', onsite[ii]

  if (which == 'full'):
    Ham = np.zeros((num_basis,num_basis))

  '''
  Define sparsed Hamiltonian in triple format: row, column and data
  '''
  row_s = np.zeros(( num_basis*(tot_coordinate*n*2+1), ),dtype=np.int)
  col_s = np.zeros(( num_basis*(tot_coordinate*n*2+1), ),dtype=np.int)
  data_s = np.zeros(( num_basis*(tot_coordinate*n*2+1), ))
  if (Initialize_stat == 'Initialize') :
    num_elements = int(num_basis-1) # number of nonzero matrix elements
  else :  
    num_elements = int(-1) # number of nonzero matrix elements

  for ii in range(0,num_basis):
    bas = list(order_basis[ii])  # call basis function
    n_bas = np.delete(np.arange(0,N),bas,None).tolist()

    if (np.mod(ii,10000) == 0):
      print '%-8.5e'%(float(ii)/num_basis)
    
    if (Initialize_stat != 'Initialize') : 
      "# ---------------------- Diagonal terms ---------------------- #"
      # numbers of NN and NNN pairs of particles in each state 
      numNN, numNNN, numN3 = 0, 0, 0
      for kk in range(0,n):
        site1 = bas[kk]
        for kk1 in range(kk+1,n):
          site2 = bas[kk1]
          for kk2 in range(0,1):
            dist12 = dist[site1,site2]
            if (abs(dist12-distNN) < 0.01) :
              numNN += 1
            elif (abs(dist12-distNNN) < 0.01) :
              numNNN += 1
            elif (abs(dist12-distN3) < 0.01) :
              numN3 += 1
#x          for kk2 in range(0,nNNcell):
#x            dist12 = distNc[site1,site2,kk2]
#x            if (abs(dist12-distNN) < 0.01) :
#x              numNN += 1
#x            elif (abs(dist12-distNNN) < 0.01) :
#x              numNNN += 1
#x            elif (abs(dist12-distN3) < 0.01) :
#x              numN3 += 1

      num_elements += 1
      row_s[num_elements] = ii
      col_s[num_elements] = ii
      # on-site energy & Interaction energy
      data_s[num_elements] += sum(onsite[bas]) + numNN*vNN + numNNN*vNNN + numN3*vN3

      if (which == 'full'):
        Ham[ii,ii] += data_s[num_elements]
      #print 'bas[',ii,'] =', bas,'Eint =', Ham[ii,ii]

    if (Initialize_stat != 'N_Initialize') : 
      "# ---------------------- Off-diagonal terms ------------------ #"
      # hopping energy
      for kk in range(0,n): # kk is index
        site1 = bas[kk]     # this the site considered
        # find neighbors
        for kk1 in n_bas:
          if (kk1 > site1): # define upper triangle elements
            if ( abs(dist[site1,kk1]-distNN) < 0.01 ):
              bas1 = np.array(bas)
              bas1[kk] = kk1 # change bas1 to new state by substitute

	      # define permutation time /* from Fermion exchange */
              sign_permute = np.sign(bas1-sorted(bas1))
              permute_time = np.mod( np.sum(sign_permute), 2.0) - 1.0
	      permute_time = permute_time * np.sign(np.sum(np.abs(sign_permute)))


              # upper triangle
              num_elements += 1
              row_s[num_elements] = ii
              col_s[num_elements] = num_stat(sorted(bas1),N)
              data_s[num_elements] = (-1.)**permute_time * tNN

              if (which == 'full'):
                Ham[row_s[num_elements], col_s[num_elements]] += data_s[num_elements]
	    
              # lower triangle
              num_elements += 1
              row_s[num_elements] = col_s[num_elements-1] 
              col_s[num_elements] = row_s[num_elements-1]
              data_s[num_elements] = np.conjugate(data_s[num_elements-1])

              if (which == 'full'):
                Ham[row_s[num_elements], col_s[num_elements]] += data_s[num_elements]
            
  print 'Finish calculation of Ham_sparse'

  if (which == 'full'):
    return Ham
  else:  
    return num_basis, row_s, col_s, data_s
#  # Define the sparse Hamiltonian in triple format
#  if ((which == 'sp') and (Initialize_stat == 'Normal')):
#    Ham_triple = sp.sparse.coo_matrix((data_s,(row_s,col_s)), shape = (num_basis,num_basis))
#    return num_elements, Ham_triple
#  else: 
#    return num_elements, row_s, col_s, data_s

'''
*******************************************************************************
Here is the main program
*******************************************************************************
'''
if __name__ == '__main__':
# import numpy as np
  import scipy as sp
  import matplotlib.pyplot as plt

  from scipy.sparse.linalg import LinearOperator
  from scipy.sparse.linalg import eigsh
  import time

  distNc, pos_x, pos_y = pos_kagome()

  #fig1 = plt.figure(1)
  #ax = fig1.add_subplot(111)
#  plt.plot(pos_x[0:len(pos_x):3,0],pos_y[0:len(pos_x):3,0],'or')
#  plt.plot(pos_x[1:len(pos_x):3,0],pos_y[1:len(pos_x):3,0],'ob')
#  plt.plot(pos_x[2:len(pos_x):3,0],pos_y[2:len(pos_x):3,0],'og')
  #plt.show()
  #np.savetxt('Eige0_full.txt',Eige)  

  Istat = 'Normal'
  
  # Full Hamiltonian
  Ham = Ham_Operator(which = 'full', Initialize_stat = Istat)
  Eige = np.linalg.eigvalsh(Ham)
  print Eige

  k_Eige_s = np.loadtxt('k_Eige_s.txt')
  k_Eige = np.loadtxt('k_Eige.txt')
  size_k0 = np.loadtxt('size_k0.txt')
  print max(Eige-k_Eige_s)
  #np.savetxt('Ediff.txt',Eige-k_Eige_s)
  #plt.plot(Eige-k_Eige_s)
  #plt.plot(np.ones(len(Eige),),Eige,'^r')
  #plt.plot(np.ones(int(size_k0[0]),),k_Eige[0:int(size_k0[0])],'+b')
  #plt.plot(np.ones(int(size_k0[1]),),k_Eige[int(size_k0[0]):int(size_k0[0])+int(size_k0[1])],'+b')
  #plt.savefig('Ediff.eps',format='eps',dpi=1000)

  #fig2 = plt.figure(2)
  #ax = fig2.add_subplot(111)
  #plt.plot(Eige-k_Eige_s)
  #plt.show()
  
#  # Sparse Hamiltonian
#  eigv_k = 9 
#  mode1 = 'SA'
#  time1 = time.time()
#  num_basis, row_s, col_s, data_s = Ham_Operator(which = 'sp', Initialize_stat = Istat)
#  time2 = time.time()
#  print 'time =', time2 - time1
#  Ham_triple = sp.sparse.coo_matrix((data_s,(row_s,col_s)), shape = (num_basis,num_basis))
#  del row_s, col_s, data_s

#  Eige0, Eigf0 = eigsh(Ham_triple, k=eigv_k, which = mode1) 

#  print 'Eige[1:10] = ', '\n', Eige[0:eigv_k]
#  print 'Eige0[1:10] = ', '\n', Eige0[0:eigv_k]
#  plt.figure(2)
#  plt.plot(Eige[0:eigv_k]-Eige0[0:eigv_k])

#  plt.show()
else:
  print 'Just import SparseHam'

