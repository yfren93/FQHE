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
  t2 = -0.1
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
  """ define summation serials
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
 
  sumcut = 10
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

  Vjj = Vjj/np.sqrt(2*np.pi*m)

  return Vjj

def FQHE_2DEG_Int(m, asp):

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
      Vjj1 = FQHE_2DEG_Intfun(m, asp, j1, j2, j2p, j1p)
      Vjj2 = FQHE_2DEG_Intfun(m, asp, j2, j1, j2p, j1p)

      if Vjj1 != 0 or Vjj2 != 0:
        Vjjt[inits][finls] = Vjj1 - Vjj2 # [Vjj1, Vjj2]

  return Vjjt


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

