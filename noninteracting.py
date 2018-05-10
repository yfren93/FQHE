#!/usr/bin/env python

"""
This file define the noninteraction properties of lattice model with following functions:

fun1: 2D tight binding Hamiltonian and band structure

fun2: 1D band

fun3: finite size // rotation eigenstates

fun4: Green's function of 2D & 1D

fun5: 2D DOS

fun6: 1D DOS

fun7: Kubo formula ~ disordered superlattice

fun8: Berry curvature

fun9: Surface Green's function

fun10: Transfer matrix
"""

from globalpara import *
import numpy as np
import matplotlib.pyplot as plt

"""
Define the two-dimensional electronic structure
Input: Lattice_type, tunit, superlattice size nx
Output: electronic structure in eps and txt files
"""
def TwoDband(tunit,nx=1, ifshow = True):
  "Input"  
  nUnit_band = np.shape(tunit[(0,0)])[0]

  "Lattice dependent part"
  if Lattice_type == 'kagome' : 
    
   # define unit vectors
    a1 = np.array([distNN, distNNN])*nx; a2 = np.array([distN3, 0])*nUnit_band/nx/3

   # define the momentum
    b1 = 2.0*pi/(distNNN)*np.array([0, 1.0])/nx
    b2 = 2.0*pi/(distNNN)*np.array([np.sqrt(3.0)/2.0, 1.0/2.0])*nx*3/nUnit_band

    posG = b1*0.
    posK1 = (b1+b2)/3.0
    posM = b1/2.0
    plabelc = ['$\Gamma$','K','M','$\Gamma$']

  elif Lattice_type == 'Lieb' :
    
   # define unit vectors
    a1 = np.array([0, distN3])*nx; a2 = np.array([distN3, 0])*nUnit_band/nx/3

   # define the momentum
    b1 = 2.0*pi/(distN3)*np.array([0, 1.0])/nx
    b2 = 2.0*pi/(distN3)*np.array([1.0, 0])*nx*3/nUnit_band

    posG = b1*0.
    posK1 = b2/2.0
    posM = (b1+b2)/2.0
    plabelc = ['$\Gamma$','X','M','$\Gamma$']

  NNk = int(51)
  kkx = np.zeros((3*NNk,))
  kky = np.zeros((3*NNk,))
  kkx[0:NNk]        = np.linspace(posG[0],posK1[0],num=NNk)
  kkx[NNk:2*NNk]    = np.linspace(posK1[0],posM[0],num=NNk)
  kkx[2*NNk:3*NNk]  = np.linspace(posM[0],posG[0],num=NNk)
  kky[0:NNk]        = np.linspace(posG[1],posK1[1],num=NNk)
  kky[NNk:2*NNk]    = np.linspace(posK1[1],posM[1],num=NNk)
  kky[2*NNk:3*NNk]  = np.linspace(posM[1],posG[1],num=NNk)
  
  pkk = np.zeros((3*NNk,))
  kl = 0.0; kr = np.sqrt((posG[0]-posK1[0])**2+(posG[1]-posK1[1])**2)
  pkk[0:NNk] = np.linspace(kl,kr,NNk)
  kl = kr; kr += np.sqrt((posM[0]-posK1[0])**2+(posM[1]-posK1[1])**2)
  pkk[NNk:2*NNk] = np.linspace(kl,kr,NNk)
  kl = kr; kr += np.sqrt((posG[0]-posM[0])**2+(posG[1]-posM[1])**2)
  pkk[2*NNk:3*NNk] = np.linspace(kl,kr,NNk)

  plabelv = pkk[0:3*NNk:NNk-1] 


  "Lattice independent part"

  eige = np.zeros((1+nUnit_band,3*NNk))

  for ii in range(0,3*NNk):
    kx = kkx[ii]
    ky = kky[ii]
    Ham = np.zeros((nUnit_band,nUnit_band),dtype=complex)
    for kk1 in range(-1,2) :
      for kk2 in range(-1,2) :
        posdx = (kk1)*a1[0] + (kk2)*a2[0]; posdy = (kk1)*a1[1] + (kk2)*a2[1]
        Ham += tunit[(kk1,kk2)]*np.exp(1j*kx*posdx+1j*ky*posdy)
    Ham = (Ham + np.conjugate(Ham.T))/2
    eige[1:,ii] = np.linalg.eigvalsh(Ham)
  
  " plot and save "
  if not ifshow :
    plt.switch_backend('agg')

  fig1 = plt.figure(1)
  ax = fig1.add_subplot(111)
  #for ii in range(0,num_vl):
  #plt.scatter(pos_x,pos_y,pos_x*0+20,marker='o',c='b',alpha=0.3)
  eige[0,:] = pkk
  for ii in range(1,nUnit_band+1):
    plt.plot(eige[0,:],eige[ii,:],'-b')
  
  size1 = 13
  plt.xlabel('k',fontsize=size1)
  plt.ylabel('E',fontsize=size1)
  plt.xticks(plabelv,plabelc)
  
  for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #label.set_fontname('Arial')
    label.set_fontsize(13)
  
  if ifshow :
    plt.show()
  else :
    plt.savefig('Band_'+Lattice_type+'.eps',format='eps',dpi=1000,bbox="tight")
    np.savetxt('Band_'+Lattice_type+'.txt',eige)


"""
Define the two-dimensional electronic structure
Input: Lattice_type, tunit, superlattice size nx
Output: electronic structure in eps and txt files
"""
def OneDband(h0,h1):
  
  nUnit_band = np.shape(h0)[0]

  # define the momentum
  NNk = int(31)
  kkx = np.linspace(-1,1,NNk)*pi
  
  # define Ham and calculate eigenvalue
  eige = np.zeros((NNk,nUnit_band+1),)
  eige[:,0] = kkx

  for ii in range(0,NNk) :
    Ham = np.zeros((nUnit_band,nUnit_band),dtype=complex)

    Ham += h1*np.exp(1j*(kkx[ii]+pi))
    Ham += np.conjugate(Ham.T)
    Ham += h0

    eige[ii,1:] = np.linalg.eigvalsh(Ham)

  # plot and save
  np.savetxt('Ribbon_'+Lattice_type+'.txt',eige)

  fig1 = plt.figure(1)
  ax = fig1.add_subplot(111)
  plt.plot(eige[:,0],eige[:,1:],'-b')

  size1 = 13
  plt.xlabel('k',fontsize=size1)
  plt.ylabel('E',fontsize=size1)
  plt.xticks(np.linspace(-1,1,5)*pi,['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'])

  for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #label.set_fontname('Arial')
    label.set_fontsize(13)

  plt.savefig('Ribbon_'+Lattice_type+'.eps',format='eps',dpi=1000,bbox="tight")
  #plt.show()

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

      
