#!/bin/usr/env python
from globalpara import *

def get_basis_U1(N,n):
  """ get the basis function of system with particle conservation
  
  Args: 
      N: total sites
      n: total particles

  Returns:
      sq2bas: map the sequence to basis function
      bas2sq: map the basis function to sequence
  """

  import itertools
  import scipy.special as sps

  num_basis = int(sps.comb(N,n))  # total number of basis functions
  order_basis = list( itertools.combinations(range(N),n) )  # get basis function
  
  num = 0
  bas2sq = {}
  sq2bas = {}
  for ii in order_basis:
    config = [0]*N
    for jj in ii:
      config[jj] = 1
    sq2bas[num] = tuple(config)
    bas2sq[tuple(config)] = num
    num += 1

  return sq2bas, bas2sq


def sortU1basis_k_FQHE(sq2bas0) :
  """ sort the U1 basis functions according to the total momentum
      This code suitable for the FQHE case, or equivalently, 1D 
      chain system.
  
  Args: 
      sq2bas0: a map of sequence to basis function configuration
              e.g., 1: (0101001)

  Returns:
      sq2bas: map the sequence to basis function for each momentum
      bas2sq: map the basis function to sequence for each momentum
  """
  m = len(sq2bas0[0]) # define the total site, or total momentum
  momentum0 = range(0, m) # momentum for each basis, ranging [m, 1 ~ m-1]
  sq2bas = {n:{} for n in range(m)}
  bas2sq = {n:{} for n in range(m)}
  N_bk = [0]*m # number of basis function for each momentum

  for ii in sq2bas0.keys() :
    bas = list(sq2bas0[ii])
    momentum = int(np.mod(np.dot(bas, momentum0), m))
    
    sq2bas[momentum][N_bk[momentum]] = tuple(bas)
    bas2sq[momentum][tuple(bas)] = N_bk[momentum]
    N_bk[momentum] += 1

  return sq2bas, bas2sq, N_bk


def get_IntMatEle(bas2sq, sq2bas, Vjjt): 
  """get interaction matrix element
     Args: 
         bas2sq: map basis function to its sequence
         sq2bas: map sequence to basis function
         Vjjt: interaction table
     Output:
         Hamtri: triple form of Hamiltonian matrix
  """
  import numpy as np
  import itertools

  row = []  # save off diagonal terms
  col = []
  dat = []

  datd = []  # save diagonal terms
  n = sum(sq2bas[0])
 
  for ii in sq2bas.keys():  # for ii-th basis
    bas = list(sq2bas[ii])  # basis function
    occp = np.nonzero(bas)  # occupied positions of electrons
    inits = list(itertools.combinations(list(occp)[0],2))  # find initial pair of electrons 

    print 'Intralayer basis: ', inits, occp

    datd += [0]
    for init_i in inits:  # find the possible scattering states
      for fins in Vjjt[init_i].keys():  # find the possible final states
        bas1 = []
        bas1 = bas1 + bas   # initialize final state list
        bas1[init_i[0]] = 0  # annihilate two electrons of initial state
        bas1[init_i[1]] = 0  
        bas1[fins[0]] = 1  # creat two electrons of final states   
        bas1[fins[1]] = 1
        if sum(bas1) == n:  # if there are two electrons on the same site
          jj = bas2sq[tuple(bas1)]

          ss0 = sorted([init_i[0], fins[0]])
          ss1 = sorted([init_i[1], fins[1]])
          #exchangetime = sum(bas[ss0[0]:ss0[1]]) + sum(bas[ss1[0]:ss1[1]]) 
          exchangetime = sum(bas[0:init_i[0]]) + sum(bas[0:init_i[1]]) - 1
          exchangetime += sum(bas1[0:fins[0]]) + sum(bas1[0:fins[1]]) - 1
           
          if jj == ii :  # if diagonal term
            datd[ii] += (-1)**exchangetime*Vjjt[init_i][fins]
          else : 
            row += [int(ii)]
            col += [int(jj)]
            dat += [(-1)**exchangetime*Vjjt[init_i][fins]]
  
  dat += datd
  row += range(0,len(datd))
  col += range(0,len(datd))
  return row, col, dat

def get_bilayer_bas(sq2basT, sq2basB):
  """get the basis functions of bilayer system based on the basis functions of 
          two layers separately denoted by T & B
     Args:
         sq2basT: map from sequence to basis function configuration, top layer
         sq2basB: map from sequence to basis function configuration, bottom layer
     Outputs:
         mapsb: double map between sequence and bas n <--> (n_T, n_B)
  """
  sq2bas = {}
  bas2sq = {}
  sq = 0
  for ii in sq2basT.keys():
    for jj in sq2basB.keys():
      sq2bas[sq] = tuple([ii,jj])
      bas2sq[tuple([ii,jj])] = sq
      sq += 1

  return sq2bas, bas2sq


def get_FQHE_Interlayer_MatEle(bas2sq, sq2bas, bas2sqT, sq2basT, bas2sqB, sq2basB, VjjtIntL) :
  """get interlayer coupling matrix element of bilayer FQHE system
     ! Electron can only stay in one layer
     Args:
         bas2sq: map basis function of bilayer system to sequence
         sq2bas: map sequence to basis function of bilayer 
         bas2sqT/B: map basis function to sequence for top/bottom layer
         sq2basT/B: map sequence to basis function for top/bottom layer
         VjjtIntL: Interlayer interaction matrix elements
     Output:
         row, col, dat: all nonzero matrix element
     Notes:
         In above, basis function and sequence are for electron in a single layer.
         For a bilayer system, the basis functions are formulated by |m_t, n_b > where
         m and n are basis functions in a single layer, while t and b denote the top 
         and bottom layers, separately.
  """
  import scipy.sparse as sps 
  import numpy as np
  import itertools

  row = []
  col = []
  dat = []
  nT = sum(sq2basT[0])
  nB = sum(sq2basB[0])
 
  for ii in sq2bas.keys():  # for ii-th basis
    bas = sq2bas[ii]
    basT = list(sq2basT[bas[0]])  # basis function of top layer
    basB = list(sq2basB[bas[1]])  
    occpT = tuple(list(np.nonzero(basT))[0])  # occupied positions of electrons
    occpB = tuple(list(np.nonzero(basB))[0])  
    inits = list(itertools.product(occpT,occpB))  # find initial pair of electrons 

    #print 'Interlayer basis: ', inits, occpT, occpB

    for init_i in inits:  # find the possible scattering states
      for fins in VjjtIntL[init_i].keys():  # find the possible final states
        basT1 = []
        basT1 = basT1 + basT   # initialize final state list
        basB1 = []
        basB1 = basB1 + basB   # initialize final state list
        basT1[init_i[0]] = 0  # annihilate two electrons of initial state
        basB1[init_i[1]] = 0  
        basT1[fins[0]] = 1  # creat two electrons of final states   
        basB1[fins[1]] = 1
        if sum(basT1) == nT and sum(basB1) == nB:  # if there are two electrons on the same site
          jjT = bas2sqT[tuple(basT1)]
          jjB = bas2sqB[tuple(basB1)]
          jj = bas2sq[tuple([jjT, jjB])]
          ss0 = sorted([init_i[0], fins[0]])
          ss1 = sorted([init_i[1], fins[1]])
          #exchangetime = sum(basT[ss0[0]:ss0[1]]) + sum(basB[ss1[0]:ss1[1]]) 
          exchangetime = sum(basT[0:init_i[0]]) + sum(basT1[0:fins[0]]) 
          exchangetime += sum(basB[0:init_i[1]]) + sum(basB1[0:fins[1]])
          # sum(basB[0:ss1[0]:ss1[1]]) + sum(basB[ss1[0]:ss1[1]]) 

          row += [int(ii)]
          col += [int(jj)]
          dat += [(-1)**exchangetime*VjjtIntL[init_i][fins]]
  
  return row, col, dat


def get_bilayer_FQHE_Full_MatEle(HamT, HamB, HamTB):
  """ get the nonzero matrix elements of full matrix 
      Args:
          HamT: Top layer Hamiltonian in coo triple format
          HamB: Bottom layer Hamiltonian in coo triple format
          HamTB: Interlayer coupling matrix in coo triple format
      Output:
          Ham: merged full Hamiltonian with coo triple format
  """
  import scipy.sparse as sps

  #Hamkron = sps.kron(HamT, HamB, format='coo')
  #Ham = sps.hstack(Hamkron, HamTB)
  HamTf = HamT.toarray()
  HamBf = HamB.toarray()
  Ham = 0
  return Ham


def permute_time3(a1,a12) : # a1 tuple, a12 list 
  """ define permutation time between electrons during hopping
  Args: 
       a1: tuple, initial state
       a2: list, final state

  Returns: 
       permute: time of permute two electron

  Raises:
  """

  #import numpy as np
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


def num_stat(stat,N):
  """ Get the sequence number of a state among all the basis
  Suitable for spinless particles

  Args:
      stat: [a0, a1, a2, ... an-1] n numbers, occupied sites of n electrons
      N: total sites in the lattice
  
  Returns: 
      a number nstat-1, indicating the sequence of 'stat'
  """

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


def Ham_Operator(which = 'sp', Initialize_stat = 'Normal', vNN = 2.0, vNNN = 0.0, vN3 = 0.0) :
  """ Define Hamiltonian matrix
  Based on the basis function | N1, N2, ..., Nn > with N1 < N2 < ... < Nn of a 
  spinless Fermionic system with N sites and n particles, define the Hamiltonian
  matrix elements and diagonalize

  Args: 
      which: flag for return 
      vNN, vNNN, vN3: 1st, 2nd, and 3rd neighbor interaction strengths
      Initialize_stat: flag for if initialize

  """

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



if __name__ == '__main__':

  import time

  start = time.clock()

  a,b = get_basis_U1(27,9)
  #print a
  #print b
  
  end = time.clock()
  print 'time:', end - start
