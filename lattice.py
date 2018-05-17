#!/usr/bin/env python

"""
This file define the lattice information with following functions:

fun1: unit cell lattice 
  1.1 : honeycomb lattice
  1.2 : kagome lattice
  1.3 : Lieb lattice

fun2: Column

fun3: Torus/Cylinder

fun4: Central

"""

from globalpara import *
from sparse_Ham_fun import *
import numpy as np

"""
Define the hopping between unit cells of kagome lattice
"""
def unit_kagome(t1 = 1.0,t2 = 0, t3 = 0, t1c = 0, t2c = 0, t3c = 0, V1 = 1, V2 = 0, V3 = 0):
  import numpy as np
  
  tunit = {}
  Vunit = {}
  for ii in range(-1,2):
    for jj in range(-1,2):
      if Flagrc == 'float' : 
        tunit[(ii,jj)] = np.zeros((3,3),dtype=float)
        Vunit[(ii,jj)] = np.zeros((3,3),dtype=float)
      elif Flagrc == 'complex' :
        tunit[(ii,jj)] = np.zeros((3,3),dtype=complex)
        Vunit[(ii,jj)] = np.zeros((3,3),dtype=complex)

  tunit[( 0, 0)] += np.array([[0.,t1,t1],[t1,0,t1],[t1,t1,0]])
  tunit[( 1, 0)] += np.array([[0,0,0],  [t1,0,t2],[t2, 0, t3]])
  tunit[( 0, 1)] += np.array([[0,0,0],  [t2,t3,0],[t1, t2 , 0]])
  tunit[(-1, 1)] += np.array([[t3,t2,0],[0 ,0 ,0],[ t2, t1,0]])

  tunit[(-1, 0)] += np.array([[0,0,0],  [t1,0,t2],[t2, 0, t3]]).T
  tunit[( 0,-1)] += np.array([[0,0,0],  [t2,t3,0],[t1, t2 , 0]]).T 
  tunit[( 1,-1)] += np.array([[t3,t2,0],[0 ,0 ,0],[ t2, t1,0]]).T

  Vunit[( 0, 0)] += np.array([[0,V1,V1],[V1,0,V1],[V1,V1,0]])
  Vunit[( 1, 0)] += np.array([[0,0,0],  [V1,0,V2],[V2, 0, V3]])
  Vunit[( 0, 1)] += np.array([[0,0,0],  [V2,V3,0],[V1, V2 , 0]])
  Vunit[(-1, 1)] += np.array([[V3,V2,0],[0 ,0 ,0],[ V2, V1,0]])

  Vunit[(-1, 0)] += np.array([[0,0,0],  [V1,0,V2],[V2, 0, V3]]).T
  Vunit[( 0,-1)] += np.array([[0,0,0],  [V2,V3,0],[V1, V2 , 0]]).T 
  Vunit[( 1,-1)] += np.array([[V3,V2,0],[0 ,0 ,0],[V2, V1,0]]).T

  if Flagrc == 'complex': # along arrow: minus
    tunit[( 0, 0)] += 1j*np.array([[0,t1c,-t1c],[-t1c,0,t1c],   [t1c,-t1c,0]])
    tunit[( 1, 0)] += 1j*np.array([[0,0,0],     [-t1c, 0,-t2c], [-t2c, 0 ,0]])
    tunit[( 0, 1)] += 1j*np.array([[0,0,0],     [t2c,0,0],      [t1c, t2c , 0]])
    tunit[(-1, 1)] += 1j*np.array([[0,-t2c,0],  [0 ,0 ,0],      [-t2c, -t1c,0]])

    tunit[(-1, 0)] += -1j*np.array([[0,0,0],     [-t1c, 0,-t2c], [-t2c, 0 ,0]]).T
    tunit[( 0,-1)] += -1j*np.array([[0,0,0],     [t2c,0,0],      [t1c, t2c , 0]]).T
    tunit[( 1,-1)] += -1j*np.array([[0,-t2c,0],  [0 ,0 ,0],      [-t2c, -t1c,0]]).T

  return tunit, Vunit


"""
Define the hopping between unit cells of Lieb lattice
"""

def unit_Lieb(t1 = 1.0, t2 = 0, t3 = 0, t1c = 0, t2c = 0, t3c = 0, V1 = 1, V2 = 0, V3 = 0, Ea = 4, Eb = 0, Ec = 0):
  import numpy as np
  
  tunit = {}
  Vunit = {}
  for ii in range(-1,2):
    for jj in range(-1,2):
      if Flagrc == 'float' : 
        tunit[(ii,jj)] = np.zeros((3,3),dtype=float)
        Vunit[(ii,jj)] = np.zeros((3,3),dtype=float)
      elif Flagrc == 'complex' :
        tunit[(ii,jj)] = np.zeros((3,3),dtype=complex)
        Vunit[(ii,jj)] = np.zeros((3,3),dtype=complex)

  tunit[( 0, 0)] += np.array([[Ea,t1,t1],[t1,Eb,t2],[t1,t2,Ec]])
  tunit[( 1, 0)] += np.array([[0,0,0],  [t1,0,t2],[0, 0, t3]])
  tunit[( 0, 1)] += np.array([[0,0,0],  [0,t3,0],[t1, t2 , 0]])
  tunit[(-1, 1)] += np.array([[0,0,0],[0 ,0 ,0],[ 0, 0,0]])

  tunit[(-1, 0)] += np.array([[0,0,0],  [t1,0,t2],[0, 0, t3]]).T
  tunit[( 0,-1)] += np.array([[0,0,0],  [0,t3,0],[t1, t2 , 0]]).T
  tunit[( 1,-1)] += np.array([[0,0,0],[0 ,0 ,0],[ 0, 0,0]]).T

  Vunit[( 0, 0)] += np.array([[0,V1,V1],[V1,0,V2],[V1,V2,0]])
  Vunit[( 1, 0)] += np.array([[0,0,0],  [V1,0,V2],[0, 0, V3]])
  Vunit[( 0, 1)] += np.array([[0,0,0],  [0,V3,0],[V1, V2 , 0]])
  Vunit[(-1, 1)] += np.array([[0,0,0],[0 ,0 ,0], [0, 0,0]])

  Vunit[(-1, 0)] += np.array([[0,0,0],  [V1,0,V2],[0, 0, V3]]).T
  Vunit[( 0,-1)] += np.array([[0,0,0],  [0,V3,0],[V1, V2 , 0]]).T
  Vunit[( 1,-1)] += np.array([[0,0,0],[0 ,0 ,0], [0, 0,0]]).T

#  if Flagrc == 'complex': # along arrow: minus
#    tunit[( 0, 0)] += 1j*np.array([[0,t1c,-t1c],[-t1c,0,t1c],   [t1c,-t1c,0]])
#    tunit[( 1, 0)] += 1j*np.array([[0,0,0],     [-t1c, 0,-t2c], [-t2c, 0 ,0]])
#    tunit[( 0, 1)] += 1j*np.array([[0,0,0],     [t2c,0,0],      [t1c, t2c , 0]])
#    tunit[(-1, 1)] += 1j*np.array([[0,-t2c,0],  [0 ,0 ,0],      [-t2c, -t1c,0]])
#
#    tunit[(-1, 0)] += -1j*np.array([[0,0,0],     [-t1c, 0,-t2c], [-t2c, 0 ,0]]).T
#    tunit[( 0,-1)] += -1j*np.array([[0,0,0],     [t2c,0,0],      [t1c, t2c , 0]]).T
#    tunit[( 1,-1)] += -1j*np.array([[0,-t2c,0],  [0 ,0 ,0],      [-t2c, -t1c,0]]).T

  return tunit, Vunit


"""
Define the hopping and interaction between sites in column
Input : t, V, Ny, and PBC (1/0: PBC/OBC)
"""
def column_tab(tunit,Vunit={},Ny=3,PBC=pbcy, phi=0.0):
  
  nUnit_band = np.shape(tunit[(0,0)])[0]

  h0 = np.zeros((nUnit_band*Ny,nUnit_band*Ny),dtype = Flagrc)
  h1 = np.zeros((nUnit_band*Ny,nUnit_band*Ny),dtype = Flagrc)

  h0 += np.kron(np.diag([1]*(Ny-1),1),tunit[(0,1)]) # upper triangle
  h0 += np.conjugate(h0.T)                          # lower triangle
  h0 += np.kron(np.eye(Ny,),tunit[(0,0)])        # intra-unit cell

  h1 += np.kron(np.eye(Ny,),tunit[(1,0)])        # to the right
  h1 += np.kron(np.diag([1]*(Ny-1), 1),tunit[(1,1)]) # to the upper right
  h1 += np.kron(np.diag([1]*(Ny-1),-1),tunit[(1,-1)]) # to the lower right

  if PBC == 1 : # For PBC condition // One can insert "flux" here //
    h0[(Ny-1)*nUnit_band:Ny*nUnit_band, 0:nUnit_band] += tunit[(0,1)]*np.exp(1j*phi) # hopping Ny --> 1
    h0[0:nUnit_band, (Ny-1)*nUnit_band:Ny*nUnit_band] += tunit[(0,-1)]*np.exp(-1j*phi) #         1 --> Ny 

    h1[(Ny-1)*nUnit_band:Ny*nUnit_band, 0:nUnit_band] += tunit[(1,1)]*np.exp(1j*phi) # upper right cross boundary
    h1[0:nUnit_band, (Ny-1)*nUnit_band:Ny*nUnit_band] += tunit[(1,-1)]*np.exp(-1j*phi)# lower right corss boundary
    
  return h0, h1


"""
Define the position of each site in a cell depending on the lattice type and
boundary conditions                         
The position of sites in nearest 8 cells are also defined for pbc case
"""
def pos_kagome1(nNx=Nx,nNy=Ny):

  pos = np.zeros((nNx*nNy*3,2),)

  # define unit vectors
  a1 = np.array([distNN, distNNN]); a2 = np.array([distN3, 0])

  for ii in range(0,nNx) : 
    for jj in range(0,nNy) :
      pos[(ii*nNy+jj)*3:(ii*nNy+jj+1)*3,0] = ii*a2[0]+jj*a1[0]+np.array([0,distNN,distNN/2.0])
      pos[(ii*nNy+jj)*3:(ii*nNy+jj+1)*3,1] = ii*a2[1]+jj*a1[1]+np.array([0,0,distNN*np.sqrt(3)/2.0])

  return pos

def pos_plot_kag():
  
  nx0,ny0=10,10
  pos = pos_kagome1(nNx=nx0,nNy=ny0)
  fig1=plt.figure(1)
  ax = fig1.add_subplot(111)
  plt.scatter(pos[0::3,0],pos[0::3,1],marker='o',c='r',alpha=1)
  plt.scatter(pos[1::3,0],pos[1::3,1],marker='o',c='g',alpha=1)
  plt.scatter(pos[2::3,0],pos[2::3,1],marker='o',c='b',alpha=1)
  ax.set_aspect('equal')
  plt.xlim(0,2*(nx0+ny0/2))
  plt.ylim(0,ny0*np.sqrt(3))
  plt.show()


"""
Define the Hamiltonian of single particle kagome lattice for give momentum  
Momentum is defined by the periodic boundary condition of torus used
The eigenvalues and eigenstates of each state is given in UnitaryTransform
"""
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


def GetKagomeUnitaryTransformV1(tunit, Length, Heightp):
  """Calculate the Unitary Transform of Kagome lattice based on tunit
     Args:
         tunit: hopping matrix elements between unit cells
         Length/Heightp: length/Height of finite structure, to define momentum
     Outputs:
         UnitaryTransform: eigenvalue and eigenstates at all momentum q
                           eigf'*H*eigf = eige
  """
  UnitaryTransform = {}
  for iix in range(0, Length) :
    for iiy in range(0, Heightp) :
      # count momentum from left to right & from bottom to top
      UnitaryTransform[iiy*Length+iix]={} 
      #kx = iix*bx[0]/Length + iiy*by[0]/(Height/3)
      #ky = iix*bx[1]/Length + iiy*by[1]/(Height/3)
      kk = [float(iix)/Length, float(iiy)/Heightp]

      Ham = tunit[tuple([0, 0])]*0.0 #np.diag(onsite+0j)
      for kky in range(0,3) :
        for kkx in range(0,3) :
          Ham += tunit[tuple([kkx-1, kky-1])]*np.exp(1j*2*np.pi*np.dot(kk, [kkx-1, kky-1]))
          #posdx = (kky-1)*ay[0] + (kkx-1)*ax[0] 
          #posdy = (kky-1)*ay[1] + (kkx-1)*ax[1]
          #Ham += hopE[:,:,kky*3+kkx]*np.exp(1j*kx*posdx+1j*ky*posdy)
      print 'maxx', np.amax(np.amax(abs(Ham - np.conjugate(Ham.T))))
      Ham = (Ham + np.conjugate(Ham.T))/2
#      if (iiy*Length+iix) == 1 :
#        print 'Ham 1 \n', Ham
      eige,eigf = np.linalg.eigh(Ham)

      UnitaryTransform[iiy*Length+iix]['eige'] = eige
      UnitaryTransform[iiy*Length+iix]['eigf'] = eigf
#      print 'xx', iix, iiy, iiy*Length+iix
  return UnitaryTransform


def get_Basisfun_Ecut(UT, Ecut, Length, Heightp):
  """get the single particle basis function in momentum space
     Args:
         UT: single particle eige and eigf
         Ecut: energy cut
         Length/Heightp: system size, to define the momentum
     Outputs:
         SingleParticleBas[ii] = (iikx, iiky, iib) momentum and band index
         OnSite: Onsite energy of each basis function according to ii
         kBand[(iix, iiy)] = band index included for momentum iix iiy
         NumSPB: total number of basis function
  """
  SingleParticleBas = {} # Number of single particle basis, momentum & band index
  OnSite = []            # on-site energy for each state
  NumSPB = 0             # Total number of single particle state
  kBand = {}
  for iix in range(0,Length): #kx
    for iiy in range(0,Height/nUnit): #ky
      ii = iix + iiy * Length
      kBand[tuple([iix,iiy])] = []
      for jj in range(0,nUnit):   # band index
        # momentum in x/b2 and y/b1 directions, and band index
        if UT[ii]['eige'][jj] < Ecut:
          SingleParticleBas[NumSPB] = tuple([iix,iiy,jj])
          SingleParticleBas[tuple([iix,iiy,jj])] = NumSPB
          OnSite += [UT[ii]['eige'][jj]]
          NumSPB += 1
          kBand[tuple([iix, iiy])] += [jj]

  OnSite = np.array(OnSite)
  
  return SingleParticleBas, OnSite, kBand, NumSPB


def get_ManyParticleBasisfun(SingleParticleBas, NumSPB, n, Length, Heightp):
  """get many particle basis function on the basis of momentum space
         sorted by momentum
     Args:
         SingleParticleBasis: a map between sequence ii and (kx,ky,iband)
         NumSPB: total number of single particle basis function
         n: total electron number
         Length/Heightp: system size to determine momentum
     Outputs:
         StatMom: for each momentum, define a map between sequence and basis function
         num_kbasis: for each momentum, number of basis function
  """
  import scipy.special as sps
  import itertools
  import numpy as np
  import time

  # Define many particle basis functions
  num_basis = int(sps.comb(NumSPB,n)) # total number of basis functions
  order_basis = list( itertools.combinations(range(NumSPB),n) ) # basis functions presents in order

  #print 'num_basis', num_basis
  #print 'order_basis', order_basis

  # Reorganize many particle states according to their total momentum
  StatMom = {}
  for ii in range(0,Heightp*Length):
    StatMom[ii] = {0:[0,0]}

  num_kbasis = np.zeros((Length*Height/nUnit,),dtype = int)
  #print 'xx', num_kbasis[0], num_kbasis[1]
  
  stime = time.time()
  #for ii in range(0, num_basis):
  for obii in order_basis: #range(0, num_basis):
    Momtii = np.array([0,0])
    #for jj in order_basis[ii]:
    for jj in obii: #order_basis[ii]:
      Momtii += np.array(SingleParticleBas[jj])[[0,1]]
    num_Momt = np.mod(Momtii[0],Length)+np.mod(Momtii[1],Height/nUnit)*Length
    StatMom[num_Momt][num_kbasis[num_Momt]] = obii #order_basis[ii]
    #StatMom[num_Momt][tuple(order_basis[ii])] = num_kbasis[num_Momt]
    StatMom[num_Momt][tuple(obii)] = num_kbasis[num_Momt]
    num_kbasis[num_Momt] += 1
  etime = time.time()
  print 'numbas =', num_basis, 'time =', etime - stime
  return StatMom, num_kbasis


def Kagome_Intfun(UT, Vint_q, j1b, j2b, j2pb, j1pb, Length, Heightp):
  """ Define the all possible scattering matrix between two basis functions
      Both direct and exchange channels are included
      Args:
          UT: unitary transformation of kagome lattice
          Vint_q: fourier transform of interaction
          j=(kx, ky, nband): state of initial and final states
      Return:
          Vjj: scattering amplitude from (j1p,j2p) --> (j1,j2)
  """
  import numpy as np

  kxi, kyi, iib0 = j1pb[0], j1pb[1], j1pb[2]
  kxj, kyj, jjb0 = j2pb[0], j2pb[1], j2pb[2]
  kxp, kyp, ii1 = j1b[0], j1b[1], j1b[2]
  kxm, kym, jj1 = j2b[0], j2b[1], j2b[2]

  #Walpha = np.array(UT[kxp + kyp * Length]['eigf'])[:,ii1] \
  #       * np.conjugate(np.array(UT[kxi + kyi * Length]['eigf'])[:,iib0])
  #Wbeta  = np.array(UT[kxm + kym * Length]['eigf'])[:,jj1] \
  #       * np.conjugate(np.array(UT[kxj + kyj * Length]['eigf'])[:,jjb0])
  if np.mod(kxi+kxj-kxp-kxm, Length)==0 and np.mod(kyi+kyj-kyp-kym, Heightp)==0:
    Walpha = UT[kxi+kyi*Length]['eigf'][:,iib0] \
           * np.conjugate(UT[kxp+kyp*Length]['eigf'][:,ii1])
    Wbeta = UT[kxj+kyj*Length]['eigf'][:,jjb0] \
           * np.conjugate(UT[kxm+kym*Length]['eigf'][:,jj1]) 
    qx, qy = np.mod(kxp - kxi, Length), np.mod(kyp - kyi, Heightp)
    Vjj = np.dot(Walpha,np.dot(Vint_q[qx + qy*Length],Wbeta))/(1.0*Length*Heightp)
  else:
    Vjj = 0.0

  return Vjj

def get_Kagome_IntEle(SingleParticleBas, m, UT, Vint_q, Length, Heightp, nUnit):
  """ Define the all possible scattering matrix between two basis functions
      Both direct and exchange channels are included
      Args:
          UT: unitary transformation of kagome lattice
          Vint_q: fourier transform of interaction
          Length/Heightp/nUnit: system size parameters
      Return:
          Vjjt: all possible scattering amplitude from (j1p,j2p) --> (j1,j2)
  """
  import numpy as np
  import itertools
  #m = Length*Heightp*nUnit
  twoe_basis = itertools.combinations(range(m), 2)

  Vjjt = {v:{} for v in twoe_basis}

  for inits in Vjjt:
    j1p = inits[0]  # aj1p^+ aj2p^+ | 0 >
    j2p = inits[1]

    j1pb = SingleParticleBas[j1p]
    j2pb = SingleParticleBas[j2p]
    for finls in Vjjt:
      j1 = finls[0]
      j2 = finls[1]

      j1b = SingleParticleBas[j1]
      j2b = SingleParticleBas[j2]

      Vjj1 = Kagome_Intfun(UT, Vint_q, j1b, j2b, j2pb, j1pb, Length, Heightp) # direct scattering
      Vjj2 = Kagome_Intfun(UT, Vint_q, j2b, j1b, j2pb, j1pb, Length, Heightp) # exchange interaction

      if abs(Vjj1) > 1e-14 or abs(Vjj2) > 1e-14:
        Vjjt[inits][finls] = Vjj1 - Vjj2 # [Vjj1, Vjj2]

  return Vjjt


def get_Kagome_IntMatEle(bas2sq, sq2bas, Vjjt, Onsite):
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

    #print 'Intralayer basis: ', inits, occp

    datd += [np.dot(bas, Onsite)]
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
          #exchangetime_d = sum(bas[ss0[0]:ss0[1]])-1
          #exchangetime_e = sum(bas[ss1[0]:ss1[1]])-1
          exchangetime = sum(bas[0:init_i[0]]) + sum(bas[0:init_i[1]]) - 1
          exchangetime += sum(bas1[0:fins[0]]) + sum(bas1[0:fins[1]]) - 1

          if jj == ii :  # if diagonal term
            datd[ii] +=1*((-1)**exchangetime)*Vjjt[init_i][fins]
          else :
            row += [int(ii)]
            col += [int(jj)]
            dat += [(1)*((-1)**exchangetime)*Vjjt[init_i][fins]]

  dat += datd
  row += range(0,len(datd))
  col += range(0,len(datd))

  return row, col, dat


def classify_VjjK(Vjjt, SingleParticleBas): 
  ns = [0, 0, 0, 0]
  V00 = []
  V01 = []
  V11 = []
  V02 = []
  for basi in Vjjt.keys():
    ns[0] = SingleParticleBas[basi[0]][2]
    ns[1] = SingleParticleBas[basi[1]][2]
    for basf in Vjjt[basi].keys():
      ns[2] = SingleParticleBas[basf[0]][2]
      ns[3] = SingleParticleBas[basf[1]][2]
      
      if sum(ns) == 0:
        V00 += [Vjjt[basi][basf]]
      elif sum(ns) == 1:
        V01 += [Vjjt[basi][basf]]
      elif sum(ns) == 2 and max(ns) == 1:
        V11 += [Vjjt[basi][basf]]
      else:
        V02 += [Vjjt[basi][basf]]
  return V00, V01, V11, V02

def get_newmap(oldmap, nbas, m):
  """define new map from old ones
     Args: 
         oldmap: old map between sequence and bas
         nbas: number of basis function
         m: total single electron basis 
     Outputs:
         sq2bas: map sequence to basis function
         bas2sq: map basis function to sequence
  """
  sq2bas={}
  bas2sq={}
  for ii in range(nbas):
    bas0 = np.zeros((m,),dtype=int)
    bas = oldmap[ii]

    for jj in bas:
      bas0[jj] = 1

    sq2bas[ii] = tuple(bas0)
    bas2sq[tuple(bas0)] = ii

  return sq2bas, bas2sq


"""
Define the position of each site in a cell depending on the lattice type and
boundary conditions                         
The position of sites in nearest 8 cells are also defined for pbc case
"""
def pos_honeycomb(shapeCell = 'square'):

# import numpy as np
  import scipy as sp
  import itertools
  import time

  global Height, Length, nNNcell, pbcx, pbcy, N_site, n_electron, distNN, distNNN

  "Intialize the lattice and boundary conditions"
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


def FQHE_2DEG_Intfun(m, asp, j1, j2, j2p, j1p):
  """ define summation serials of a single scattering process, not matrix elements
      Define the Coulomb interaction elements for a single scattering process 
      of j1p --> j1; j2p --> j2
      Args: 
          m: total sites/ total possible momentum
          asp: aspect of width and length of central region
          j1, j1p: final and initial states of electron 1
          j2, j2p: final and initial states of electron 2
      Return:
          Vjj: matrix elements for the specific scattering process
  """

  import numpy as np
 
  sumcut = 20
  iasp = 1.0/asp

  Vjj = 0.0j

  if np.mod(j1+j2-j2p-j1p, m) != 0:
    Vjj = 0
  else:
    for ss in range(-1*sumcut, sumcut) :
      for tt in range(-1*sumcut, sumcut) :
        tt1 = tt*m + j2 - j2p
        if tt1 != 0 or ss != 0 :
          qst = np.sqrt((ss*iasp)**2+(tt1*asp)**2)
          Vjj += np.exp(-np.pi/m*qst**2)*np.exp(-1j*2*pi/m*ss*(j1-j2p))/qst

    #print 'xx, j1/j2 = (%d, %d), j2p/j1p = (%d, %d), Vjj = %6.3f' % (j1, j2, j2p, j1p, Vjj)

  Vjj = Vjj/np.sqrt(2*np.pi*m)

  return Vjj

def FQHE_2DEG_Int(m, asp):
  """ Define the all possible scattering matrix between two basis functions
      Both direct and exchange channels are included
      Args:
          m: total sites
          asp: width/length sqrt(a/b)
      Return:
          Vjjt: all possible scattering amplitude from (j1p,j2p) --> (j1,j2) 
  """
  import numpy as np
  import itertools
  twoe_basis = itertools.combinations(range(m), 2)
  
  Vjjt = {v:{} for v in twoe_basis}
  
  for inits in Vjjt: 
    j1p = inits[0]  # aj1p^+ aj2p^+ | 0 >
    j2p = inits[1]

    for finls in Vjjt:
      j1 = finls[0]
      j2 = finls[1]

      finlsp = tuple([j2,j1])
      Vjj1 = FQHE_2DEG_Intfun(m, asp, j1, j2, j2p, j1p) # direct scattering
      Vjj2 = FQHE_2DEG_Intfun(m, asp, j2, j1, j2p, j1p) # exchange interaction

      if Vjj1 != 0 or Vjj2 != 0:
        Vjjt[inits][finls] = Vjj1 - Vjj2 # [Vjj1, Vjj2]

  return Vjjt


def FQHE_2DEG_Intfun_Interlayer(m, asp, j1, j2, j2p, j1p, d):
  """ define summation serials of a single scattering process, not matrix elements
      Define the Coulomb interaction elements for a single scattering process 
      of j1p --> j1; j2p --> j2
      Args: 
          m: total sites/ total possible momentum
          asp: aspect of width and length of central region
          j1, j1p: final and initial states of electron 1
          j2, j2p: final and initial states of electron 2
          d: interlayer distance, in unit of lB, modify the matrix element by e^(-qd)
            for interlayer interaction, there is no exchange term
      Return:
          Vjj: matrix elements for the specific scattering process
  """

  import numpy as np
 
  sumcut = 20
  iasp = 1.0/asp
  d1 = d*np.sqrt(2.0*np.pi/m) # d in unit of l

  Vjj = 0.0j

  if np.mod(j1+j2-j2p-j1p, m) != 0:
    Vjj = 0
  else:
    for ss in range(-1*sumcut, sumcut) :
      for tt in range(-1*sumcut, sumcut) :
        tt1 = tt*m + j2 - j2p
        if tt1 != 0 or ss != 0 :
          qst = np.sqrt((ss*iasp)**2+(tt1*asp)**2)
          Vjj += np.exp(-d1*qst)*np.exp(-np.pi/m*qst**2)*np.exp(-1j*2*np.pi/m*ss*(j1-j2p))/qst

    #print 'j1/j2 = (%d, %d), j2p/j1p = (%d, %d), Vjj = %6.3f' % (j1, j2, j2p, j1p, Vjj)
  Vjj = Vjj/np.sqrt(2*np.pi*m)

  return Vjj


def FQHE_2DEG_Int_Interlayer(m, asp, d):
  """ Define the all possible scattering matrix between two basis functions
      Both direct and exchange channels are included
      Args:
          m: total sites
          asp: width/length sqrt(a/b)
          d: interlayer distance, in unit of lB, modify the matrix element by e^(-qd)
            for interlayer interaction, there is no exchange term
      Return:
          Vjjt: all possible scattering amplitude of 
                top layer: j1p --> j1
                bottom layer: j2p --> j2 
  """
  import numpy as np
  import itertools
  mT, mB = m, m
  basT = range(mT)
  basB = range(mB)
  twoe_basis = list(itertools.product(basT,basB))
  #print twoe_basis
  Vjjt = {v:{} for v in twoe_basis}
  
  for inits in twoe_basis: 
    j1p = inits[0]  # aj1p^+ aj2p^+ | 0 >
    j2p = inits[1]

    for finls in twoe_basis:
      j1 = finls[0]
      j2 = finls[1]

      Vjj1 = FQHE_2DEG_Intfun_Interlayer(m, asp, j1, j2, j2p, j1p, d) # direct scattering
      #print inits, finls, Vjj1

      if Vjj1 != 0 :
        #print 'finls', finls, Vjj1
        Vjjt[inits][finls] = Vjj1

  return Vjjt


def get_LatticeInteraction_FourierTransform(Vunit, Length, Heightp):
  """get the fourier transformation of Hubbard interaction in lattice model with periodic boundary condition
     Args:
         Vunit: Vuint[ii] = [V]kl is the Hubbard interaction between electron in k and l sites
         Length: number of unit cell along x direction
         Heightp: Height/nUnit, number of unit cell along y direction
     Return:
         Vint_q: fourier transformation of momentum q ranging from 0~Length*Heightp
  """
  Vint_q = {ii:np.zeros((nUnit,nUnit), dtype = complex) for ii in range(0,Length*Heightp)}
  #print 'keys', Vunit.keys(), Vint_q.keys()
  NeibCellx = 3
  NeibCell = NeibCellx**2
  for qq in range(0,Length*Heightp):
    qqx = np.mod(qq,Length)  # momentum along height direction
    qqy = qq/Length  # momentum along length direction
    qqk1 = [2*np.pi*qqx/Length, 2*np.pi*qqy/Heightp]
    #qqk = [2*np.pi*qqy/Heightp, 2*np.pi*qqx/Length]
    #print 'qqk =', qqk1
    for qqv in range(0,NeibCell):
      #qqvr = (np.mod(qqv,NeibCellx)-1) + (qqv/NeibCellx-1)
      qqvr = [np.mod(qqv,NeibCellx)-1, qqv/NeibCellx-1]
      #qqvr = [qqv/NeibCellx-1, np.mod(qqv,NeibCellx)-1]
      #print 'qqvr =', qqvr
      Vint_q[qq] += Vunit[tuple(qqvr)].T*np.exp(-1j*np.dot(qqk1, qqvr))

  return Vint_q

if __name__ == '__main__':

  from scipy.special import comb
  import matplotlib.pyplot as plt

  m = 4
  asp = 1

  numb = int(comb(m, 2))
  Vjjt = FQHE_2DEG_Int(m, asp)

  mat=[[0]]*numb
  print mat
  x = 0
  for ii in Vjjt.keys():
    print 'ii = ', ii, '\n', Vjjt[ii]
    y = 0
    for jj in Vjjt[ii].keys():
      mat[x] += [Vjjt[ii][jj]]
      y += 1
    x += 1

  #print mat
  fig = plt.figure(1)
  plt.plot(range(numb),np.real(mat))
  plt.show()

