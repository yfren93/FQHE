#!/bin/usr/env python
from globalpara import *
from lattice import *
import multiprocessing
import gc
import sys
from kronR1 import *
import time
import numpy as np
import scipy.sparse as sps
from scipy.special import comb
from scipy.sparse.linalg import eigsh

def renorm_a(a):
  ac = np.zeros((4,),dtype=complex)
  ac[0] = a
  ac[1] = -1*a
  ac[2] = 1j*a
  ac[3] = -1j*a
  return ac[np.real(ac).argmax()]

def get_basis_U1(N,n):
  """ get the basis function of system with particle conservation
      U1: particle conservation symmetry  
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



def get_bilayer_bas_kt(sq2baskT, sq2baskB, kt, m, bas2sqT, bas2sqB):
  """get the basis functions of bilayer system of momentum kt based on the basis functions of 
          two layers separately denoted by T & B
     Args:
         sq2basT: map from sequence to basis function configuration, top layer, defined momentum
         sq2basB: map from sequence to basis function configuration, bottom layer, defined momentum
     Outputs:
         mapsb: double map between sequence and bas n <--> (n_T, n_B)
  """
  sq2bas = {}
  bas2sq = {}
  sq = 0

  for m0 in range(m):

    for ii in sq2baskT[m0].keys():
      
      ii0 = bas2sqT[sq2baskT[m0][ii]]

      for jj in sq2baskB[np.mod(kt-m0,m)].keys():
        jj0 = bas2sqB[sq2baskB[np.mod(kt-m0,m)][jj]]

        sq2bas[sq] = tuple([ii0,jj0])
        bas2sq[tuple([ii0,jj0])] = sq
        sq += 1

  return sq2bas, bas2sq


def merge_add(args):
  return get_FQHE_Interlayer_MatEle_sg(*args)

def mpc_get_FQHE_Interlayer_MatEle(Numbas, inpart, npart, bas2sq, sq2bas, bas2sqT, sq2basT, bas2sqB, sq2basB, VjjtIntL):

  cores = multiprocessing.cpu_count()
  pool = multiprocessing.Pool(processes=cores) 

  row, col, dat = [], [], []

  #npart = 20
  #for ii0 in range(npart):
  #  nbas = list( range(ii0*Numbas/npart, min((ii0+1)*Numbas/npart,Numbas)) )
  #  tasks = [(iibas, bas2sq, sq2bas, bas2sqT, sq2basT, bas2sqB, sq2basB, VjjtIntL) for iibas in nbas]
  #  result = pool.map(merge_add,tasks)
  #  del tasks
  #  gc.collect()
    
  #  for ii in result:
  #    col += ii[1]
  #    row += ii[0]*len(ii[1])
  #    dat += ii[2]
  #    #result.remove(ii)
  #    #gc.collect()
  #  
  #  del result
  #  gc.collect()

  #nbas = list(range(Numbas))
  #tasks = [(iibas, bas2sq, sq2bas, bas2sqT, sq2basT, bas2sqB, sq2basB, VjjtIntL) for iibas in nbas]

  #result = pool.map(merge_add,tasks)  

  nbas = list(range(inpart*Numbas/npart, min((inpart+1)*Numbas/npart,Numbas)))
  print 'xxx', inpart, inpart*Numbas/npart, min((inpart+1)*Numbas/npart,Numbas)
  tasks = [(iibas, Numbas, bas2sq, sq2bas, bas2sqT, sq2basT, bas2sqB, sq2basB, VjjtIntL) for iibas in nbas]
  result = pool.map(merge_add, tasks)
  del tasks
  pool.close()
  pool.join()

  gc.collect()

  #for ii in result:
  #  col += ii[1]
  #  row += ii[0]*len(ii[1])
  #  dat += ii[2]
  #  result.remove(ii)
  #  gc.collect()


  for ii in result:
    row += ii[0]
    col += ii[1]
    dat += ii[2]
  #print 'type', type(result)
  #return result #row, col, dat
  return row, col, dat


def get_FQHE_Interlayer_MatEle_sg(iibas, Numbas, bas2sq, sq2bas, bas2sqT, sq2basT, bas2sqB, sq2basB, VjjtIntL) :
  """get interlayer coupling matrix element of bilayer FQHE system for a single basis function iibas
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

  row = [] #[iibas]
  col = []
  dat = []
  #Hf0 = sps.dok_matrix((Numbas, Numbas),dtype=complex)
  nT = sum(sq2basT[0])
  nB = sum(sq2basB[0])
 
  for ii in [iibas]:  # for ii-th basis
    bas = sq2bas[ii]
    basT = list(sq2basT[bas[0]])  # basis function of top layer
    basB = list(sq2basB[bas[1]])  
    occpT = tuple(list(np.nonzero(basT))[0])  # occupied positions of electrons
    occpB = tuple(list(np.nonzero(basB))[0])  
    inits = list(itertools.product(occpT,occpB))  # find initial pair of electrons 

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
          exchangetime = sum(basT[0:init_i[0]]) + sum(basT1[0:fins[0]]) 
          exchangetime += sum(basB[0:init_i[1]]) + sum(basB1[0:fins[1]])
          #Hf0[ii, jj] += (-1)**exchangetime*VjjtIntL[init_i][fins]
          row += [int(ii)]
          col += [int(jj)]
          dat += [(-1)**exchangetime*VjjtIntL[init_i][fins]]
  return row, col, dat
  #return Hf0





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
          exchangetime = sum(basT[0:init_i[0]]) + sum(basT1[0:fins[0]]) 
          exchangetime += sum(basB[0:init_i[1]]) + sum(basB1[0:fins[1]])

          row += [int(ii)]
          col += [int(jj)]
          dat += [(-1)**exchangetime*VjjtIntL[init_i][fins]]
  
  return row, col, dat


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



def get_diag_block(Hamk, N_bk, kt, m):
  """ Get the block-diagonalized Hamiltnoian matrix elements
  Args:
      Hamk: block Hamiltonian in each layer
      N_bk: dimension of each block in single layer
      kt: total momentum of electrons in both layers
      m: momentum number
  Outputs:
      H0: sparse matrix of block-diagonalized Hamiltonian
  """

  import scipy.sparse as sps

  ts = time.time()
  mry0 = psutil.virtual_memory().used/1024.0**3
 
  for ii in [0]:   # Top layer       # Bottom layer
    H0 = sps.kron(np.eye(N_bk[ii]), Hamk[int(np.mod(kt-ii,m))], format='coo')
    H0 += sps.kron(Hamk[ii], np.eye(N_bk[int(np.mod(kt-ii,m))]), format='coo')
  #tnote(ts, mry0, 'get block1 = ')
  #print '  size of H0_%d ='%(ii), sys.getsizeof(H0)/1024.0/1024.0/1024.0, 'G'

  for ii in range(1,m):   # Top layer       # Bottom layer
    ts = time.time()
    HamTBk = sps.kron(np.eye(N_bk[ii]), Hamk[int(np.mod(kt-ii,m))], format='coo')
    HamTBk += sps.kron(Hamk[ii], np.eye(N_bk[int(np.mod(kt-ii,m))]), format='coo')
    H0 = sps.block_diag((H0,HamTBk))
    #tnote(ts, mry0, 'get block%d = '%(ii))
    #print '  size of H0_%d ='%(ii), sys.getsizeof(H0)/1024.0/1024.0/1024.0, 'G'
    
  del HamTBk
  gc.collect()

  return H0


def get_diag_block1(Hamk, N_bk, kt, m):
  """ Get the block-diagonalized Hamiltnoian matrix elements
  Args:
      Hamk: block Hamiltonian in each layer
      N_bk: dimension of each block in single layer
      kt: total momentum of electrons in both layers
      m: momentum number
  Outputs:
      H0: sparse matrix of block-diagonalized Hamiltonian
  """

  import scipy.sparse as sps

  ts = time.time()
  mry0 = psutil.virtual_memory().used/1024.0**3
 
  for ii in [0]:   # Top layer       # Bottom layer
    H0 = kronR(np.eye(N_bk[ii]), Hamk[int(np.mod(kt-ii,m))], format='coo')
    H0 += kronR(Hamk[ii], np.eye(N_bk[int(np.mod(kt-ii,m))]), format='coo')
  tnote(ts, mry0, 'get block1 = ')
  print '  size of H0_%d ='%(ii), sys.getsizeof(H0)/1024.0/1024.0/1024.0, 'G'
  print H0.nnz

  for ii in range(1,m):   # Top layer       # Bottom layer
    ts = time.time()
    HamTBk = kronR(np.eye(N_bk[ii]), Hamk[int(np.mod(kt-ii,m))], format='coo')
    HamTBk += kronR(Hamk[ii], np.eye(N_bk[int(np.mod(kt-ii,m))]), format='coo')
    H0 = sps.block_diag((H0,HamTBk), format='coo')
    del HamTBk
    gc.collect()
    tnote(ts, mry0, 'get block%d = '%(ii))
    print '  size of H0_%d ='%(ii), sys.getsizeof(H0)/1024.0/1024.0/1024.0, 'G'

    exit()
    
  #del HamTBk
  #gc.collect()

  return H0


def get_diag_block2(Hamk, N_bk, kt, m):
  """ As get_diag_block1 costs a lot of memory.
      Here, the code is reprogramed by change the sparse matrix to three lists 
      to avoid the copy of data 
  Args:
      Hamk: block Hamiltonian in each layer
      N_bk: dimension of each block in single layer
      kt: total momentum of electrons in both layers
      m: momentum number
  Outputs:
      H0: sparse matrix of block-diagonalized Hamiltonian
  """

  import scipy.sparse as sps

  ts = time.time()
  mry0 = psutil.virtual_memory().used/1024.0**3
  row = []
  col = []
  data = []
  nnz = 0
  ndim = 0
  for ii in range(0,1):   # Top layer       # Bottom layer
    print 'ii = ', ii
    row0, col0, data0, shape0 = kronR(np.eye(N_bk[ii]), Hamk[int(np.mod(kt-ii,m))], format='coo')
    row0 += ndim
    row += list(row0)
    #del row0
    #gc.collect()
    col0 += ndim
    col += list(col0)
    #del col0 
    #gc.collect()
    #print type(data0), '\n', data0
    data += data0
    print 'len data =', len(data)
    #del data0
    #gc.collect()

    row0, col0, data0, shape0 = kronR(Hamk[ii], np.eye(N_bk[int(np.mod(kt-ii,m))]), format='coo')
    row += list(row0+ndim)
    #del row0
    #gc.collect()
    col += list(col0+ndim)
    #del col0 
    #gc.collect()
    data += data0
    print 'len data =', len(data)
    #del data0
    #gc.collect()
    ndim += shape0

    tnote(ts, mry0, 'get block1 = ')
    print '  size of col_%d ='%(ii), sys.getsizeof(col)/1024.0/1024.0/1024.0, 'G'
  #del H0
  #gc.collect()

    tnote(ts, mry0, 'get block%d = '%(ii))
    print '  size of data_%d ='%(ii), sys.getsizeof(data)/1024.0/1024.0/1024.0, 'G'

  #exit()
  print 'start  gc  '
  del row0, col0, data0
  gc.collect()

  print 'end gc  '
  time.sleep(3)

  return row, col, data, ndim



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


def trans_symm_basis(N, n, N1, N2, n_unit):
  ''' This function sort the basis functions into translational inequivalent configurations 
  denote the equivalent configurations as the same basis.  
  Args: 
      N, n: total site and electron numbers
      N1, N2: unit cell numbers along y and x directions
      n_unit: site number in each unit cell
  Outputs:
      renormalize_factor: define the appearance time of each basis
      map_dob: a dic, map_dob[ii] = [n1,t1,n2,t2 ...]
                                    n1,n2... are sequence numbers of basis functions 
                                    t1,t2... are translation operations that relate them to the first one
      dob1: a dic, dob1[ii] = [(conf1), (conf2), ...] map sequence to all the possible configurations
      num_stat1: give a basis function, find its sequence in old order
  '''
  import scipy as sp
  import numpy as np
  import scipy.special as sps
  import itertools
  import gc

  "Initialize basis functions and their orders: return dob{}, num_stat1{}"

  num_basis = int(sps.comb(N,n)) # total number of basis functions
  order_basis = list( itertools.combinations(range(N),n) ) # basis functions presents in order

  dob = {} # dictionary of ordered basis function. key: value --> number: state tuple
  num_stat1 = {} # same as above dict with key <--> value

  for ii in range(0,num_basis):
    dob[ii] = order_basis[ii] # the value is tuple. Define dob
  num_stat1 = {v:k for k, v in dob.items()} # reverse the key and value

  del order_basis

  "Initialize the translation operations: return tr[]"

  tr = np.zeros((N1,N2,N_site),dtype = int) # define translation operator with torus geometry
  for ii1 in range(0,N1):
    for ii2 in range(0,N2):
      for jj in range(0,n_unit):
        for jj1 in range(0,N1):
          for jj2 in range(0,N2):
            tr[ii1, ii2, jj+jj1*n_unit+jj2*Height] = np.mod(jj+(jj1+ii1)*n_unit,Height) + np.mod((jj2+ii2),Length)*Height

  "Re-organize basis function according to the translation operations: return map_dob{}, dob1{}, renormalize_factor"
  map_dob = {ii:[] for ii in range(0,num_basis)} # new dict that map previous order_basis to new defined basis order
  dob1 = {}   # dict of block basis function
  renormalize_factor = [] # renormalization factor of each block basis.

  num_nb = -1 # number of basis function in reduced block
  bas1 = np.zeros((n,),dtype = int) # basis obtained after operation
  while dob != {}:
    bas = dob[next(iter(dob))]

    num_nb += 1

    dob1[num_nb] = []
    del_ele = []
    for ii1 in range(0,N1):
      for ii2 in range(0,N2):

        bas1[:] = tr[ii1,ii2,bas]
        t_bas = tuple(bas1)

        dob1[num_nb] += [t_bas]

        nbas1 = num_stat1[tuple(sorted(bas1))] # get sequence of basis function
        map_dob[nbas1] += [num_nb]  # map old nbas1-th state to num_nb-th for block basis
        map_dob[nbas1] += [ii2+ii1*N2]     # denote the translations along x and y directions

        del_ele += [nbas1]          # delete the used basis in dob
    for ii in set(del_ele):
      del dob[ii]

    renormalize_factor += [N1*N2/len(set(del_ele))] # define the appearance time of each basis

  gc.collect()

  return renormalize_factor, map_dob, dob1, num_stat1


def get_Diagonal_init(renormalize_factor, dob1, N, map_dob, num_stat1, Neib, Neib1, Neib2):
  '''  Define sparsed Hamiltonian in triple format: row, column and data
  Args:
      renormalize_factor, dob1, N, map_dob, num_stat1, Neib, Neib1, Neib2
  Outputs:
      data_s[ii]=[n1,n2,n3]: number of 1st, 2nd, and 3rd nearest neighbor pairs
  '''
  data_s = [] 
  num_nonzero = -1

  for mm1 in range(0,len(renormalize_factor)) : # calculate matrix elements for each basis
    bas = list(dob1[mm1][0]) # bas: tuple
    n_bas = list(set(range(0,N))-set(bas))

    "# define the diagonal terms"
    num_couple, num_couple1, num_couple2 = 0, 0, 0
    for iin in bas:
      num_couple += len( set(bas) & set(Neib[iin,:]) )
      num_couple1 += len( set(bas) & set(Neib1[iin,:]) )
      num_couple2 += len( set(bas) & set(Neib2[iin,:]) )

    data_s += [[num_couple, num_couple1, num_couple2]]

  return data_s


def get_OffDiag_init(renormalize_factor, dob1, N, n, tNN, Neib, num_stat1, map_dob):

  off_hop = []

  for mm1 in range(0,len(renormalize_factor)) : # calculate matrix elements for each basis
    bas = list(dob1[mm1][0]) # bas: tuple
    n_bas = list(set(range(0,N))-set(bas))

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

        dtv, dtt = [], []
        for term12 in pos_mm2: # calculate the coupling matrix elements
          dtv += [tNN*(-1.0)**permute_time3(dob1[mm2][term12],basNN)\
                         *1.0/np.sqrt((renormalize_factor[mm2]+0.0)*renormalize_factor[mm1])]
          dtt += [term12]
        
        off_hop += [[mm1, mm2, dtv, dtt]]

  return off_hop



def get_hop_matrixele(renormalize_factor, dob1, N, n, tNN, Neib, num_stat1, map_dob, ni, nj):

  hopij = []

  for mm1 in range(0,len(renormalize_factor)) : # calculate matrix elements for each basis

    for bas0x in dob1[mm1]: # bas: tuple
      bas = list(bas0x)
      if nj in bas :
  
        n_bas = list(set(range(0,N))-set(bas))
        #print 'bas index,', bas.index(nj), Neib[bas[bas.index(nj)],:], nj
        #exit()
        "# define the coupling to basis with neighbors"
        for iin in [bas.index(nj)] : #range(0,n) :
          iin_neib = set() # find the possible replacement at each position of basis
          iin_neib = set([ni]) & set(n_bas) & set(Neib[bas[iin],:])

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
              dtv = tNN*(-1.0)**permute_time3(dob1[mm2][term12],basNN)\
                             *1.0/np.sqrt(N*1.0/(3*renormalize_factor[mm2])*N*1.0/(3*renormalize_factor[mm1]))

              hopij += [[mm1, mm2, dtv]]

  return hopij


def getHamiltonian(Hamff, m, asp, d, bas2sqTB, sq2basTB, bas2sq, sq2bas, thetax=0.0, thetay=0.0) :
  ''' This function sort the basis functions into translational inequivalent configurations
  denote the equivalent configurations as the same basis.
  Args:
      N, n: total site and electron numbers
      N1, N2: unit cell numbers along y and x directions
      n_unit: site number in each unit cell
  Outputs:
      renormalize_factor: define the appearance time of each basis
      map_dob: a dic, map_dob[ii] = [n1,t1,n2,t2 ...]
                                    n1,n2... are sequence numbers of basis functions
                                    t1,t2... are translation operations that relate them to the first one
      dob1: a dic, dob1[ii] = [(conf1), (conf2), ...] map sequence to all the possible configurations
      num_stat1: give a basis function, find its sequence in old order
  '''

  numbas = Hamff.shape[0]

  VjjtIntL = FQHE_2DEG_Int_Interlayer(m, asp, d, thetax=thetax, thetay=thetay)

  row1, col1, dat1 = get_FQHE_Interlayer_MatEle(bas2sqTB, sq2basTB, bas2sq, sq2bas, bas2sq, sq2bas, VjjtIntL)

  # get new Hamiltonian and eigenstates
  HamBL = sps.kron(np.eye(numbas), Hamff, format='coo')
  HamBL += sps.kron(Hamff, np.eye(numbas), format='coo') \
        + sp.sparse.coo_matrix((dat1,(row1, col1)), shape=(numbas**2, numbas**2))

  return HamBL


def getBerryCurv(thetax0, thetay0, getHamiltonian, *args):
  ''' This function sort the basis functions into translational inequivalent configurations
  denote the equivalent configurations as the same basis.
  Args:
      N, n: total site and electron numbers
      N1, N2: unit cell numbers along y and x directions
      n_unit: site number in each unit cell
  Outputs:
      renormalize_factor: define the appearance time of each basis
      map_dob: a dic, map_dob[ii] = [n1,t1,n2,t2 ...]
                                    n1,n2... are sequence numbers of basis functions
                                    t1,t2... are translation operations that relate them to the first one
      dob1: a dic, dob1[ii] = [(conf1), (conf2), ...] map sequence to all the possible configurations
      num_stat1: give a basis function, find its sequence in old order
  '''

  eigv_k = 10
  mode1 = 'SA'
  Ntheta = len(thetax0)

  print 'thetax = \n', thetax0, 'thetay = \n', thetay0

  bondflux = np.zeros((Ntheta**2, Ntheta**2), dtype=complex) # iix * Ntheta + iiy
  BerryCur = np.zeros((Ntheta-1, Ntheta-1))

  EigeBLt = np.zeros((Ntheta**2, eigv_k))

  for iix in range(0, Ntheta):
    for iiy in range(0, Ntheta):
      ts = time.time()

      HamBL = getHamiltonian(*args, thetax=thetax0[iix], thetay=thetay0[iiy])

      if iix ==0 and iiy ==0 :
        numbasBL = HamBL.shape[0]
        wavfun = np.zeros((numbasBL, Ntheta), dtype=complex)

      # Calculate eigen-vector with given initial vector from neighbor sites
      if iix == 0 and iiy == 0 :
        EigeBL, EigfBL = eigsh(HamBL, k=eigv_k, which=mode1)
      if iix == 0 and iiy > 0 :
        EigeBL, EigfBL = eigsh(HamBL, k=eigv_k, which=mode1, v0=wavfun[:,iiy-1])
      if iix > 0 :
        EigeBL, EigfBL = eigsh(HamBL, k=eigv_k, which=mode1, v0=wavfun[:,iiy])

      print sorted(EigeBL)
      EigeBL = np.real(EigeBL)

      EigeBLt[iix*Ntheta+iiy, :] = np.real(sorted(EigeBL))  # eigen values

      #plt.clf()
      #plt.plot(range(Ntheta**2), EigeBLt[:, 0:5], 'o')
      #plt.savefig('EigeBL_Chern_'+str(m)+'_d'+str(int(10*d))+'.eps',format='eps')

      # calculate the bondflux
      if iix > 0 :
        point_init, point_final = (iix-1)*Ntheta+iiy, iix*Ntheta+iiy  #
        # < init | final > : overlap between initial state of (theta_x0, theta_y0) and final state of (theta_x0 + d_theta, theta_y0)
        bondflux[point_init, point_final] = np.dot(np.conjugate(wavfun[:,iiy]), \
                                                   EigfBL[:, EigeBL.argmin()])
        bondflux[point_final, point_init] = np.conjugate(bondflux[point_init, point_final])

        print 'bond_flux ', point_init, point_final, bondflux[point_init, point_final]
      wavfun[:, iiy] = EigfBL[:, EigeBL.argmin()]#*np.conjugate(ppv)/abs(ppv)  # save the ground state wavefunction
  
      if iiy > 0 :
        point_init, point_final = iix*Ntheta+iiy-1, iix*Ntheta+iiy  #
        # < init | final > : overlap between initial state of (theta_x0, theta_y0) 
        #                                and final state of (theta_x0, theta_y0 + d_theta)
        bondflux[point_init, point_final] = np.dot(np.conjugate(wavfun[:,iiy-1]), wavfun[:,iiy])
        bondflux[point_final, point_init] = np.conjugate(bondflux[point_init, point_final])

        print 'bond_flux ', point_init, point_final, bondflux[point_init, point_final]

  for iix in range(Ntheta-1) :
    for iiy in range(Ntheta-1) :
      pt00, pt10 = iix*Ntheta+iiy, (iix+1)*Ntheta+iiy
      pt11, pt01 = (iix+1)*Ntheta+(iiy+1), iix*Ntheta+iiy+1 
      BerryCur[iix, iiy] = np.imag(bondflux[pt00,pt10] + bondflux[pt10,pt11] \
                                 + bondflux[pt11,pt01] + bondflux[pt01,pt00])

  return bondflux, BerryCur


if __name__ == '__main__':

  import time

  start = time.clock()

  a,b = get_basis_U1(27,9)
  #print a
  #print b
  
  end = time.clock()
  print 'time:', end - start
