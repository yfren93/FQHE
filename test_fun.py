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
UT = GetKagomeUnitaryTransform(phi0=1e-14)

vNN, vNNN, vN3 = 1, 0., 0.0
Flags={}
Flags['Measure_density'] = False #True


" Define the fourier transform of interaction "
distNc, pos_x, pos_y = pos_kagome(N_site0 = nUnit, Height0 = nUnit, Length0 = 1)
ay0 = np.array([distNN, distNNN]); ax0 = np.array([distN3, 0])
by0 = 2.0*pi/(distNNN)*np.array([0, 1.0])
bx0 = 2.0*pi/(distNNN)*np.array([np.sqrt(3.0)/2.0, -1.0/2.0])
print 'adb =', np.dot(ax0, bx0), np.dot(ay0, by0), np.dot(ax0, by0), np.dot(ay0, bx0)
tunit, Vunit = unit_kagome() # default values are for t1-V1 only model
print 'Vunit = ', Vunit

Vint = {}
for kk in range(0,nNNcell):
  Vint[kk] = np.zeros((nUnit,nUnit))

  for ii in range(0,nUnit):
    for jj in set(range(0,nUnit)):
      if (abs(distNc[ii,jj,kk]-distNN) < 0.01) :
        Vint[kk][ii,jj] = vNN
      elif (abs(distNc[ii,jj,kk]-distNNN) < 0.01) :
        Vint[kk][ii,jj] = vNNN
      elif (abs(distNc[ii,jj,kk]-distN3) < 0.01) :
        #Vint[kk][ii,jj] = vN3

        if ( (np.mod(ii,3)==0) and (ii/nUnit != jj/nUnit ) and (np.mod(ii,nUnit )!=np.mod(jj,nUnit )) ):
          Vint[kk][ii,jj] = vN3
        elif ( (np.mod(ii,3)==1) and (ii/nUnit != jj/nUnit ) and (np.mod(ii,nUnit )==np.mod(jj,nUnit )) ):
          Vint[kk][ii,jj] = vN3
        elif ( (np.mod(ii,3)==2) and (ii/nUnit == jj/nUnit ) ): # and (np.mod(ii,Height)!=np.mod(jj,Height)) ):
          Vint[kk][ii,jj] = vN3

Vint_q = {ii:np.zeros((nUnit,nUnit), dtype = complex) for ii in range(0,Length*Height/nUnit)}
for qq in range(0,Length*Height/nUnit):
  qqx = np.mod(qq,Length)
  qqy = qq/Length
  qqk = qqx*bx0/Length+ qqy*by0/(Height/nUnit) 

  for qqv in range(0,nNNcell):
    print 'Vint', qqv, '\n', Vint[qqv], '\n', Vunit[tuple([qqv/3-1, np.mod(qqv,3)-1])]
    qqvr = (np.mod(qqv,3)-1)*ax0 + (qqv/3-1)*ay0
    Vint_q[qq] += Vint[qqv]*np.exp(1j*np.dot(qqk,qqvr))
    print 'xx', np.exp(1j*np.dot(qqk,qqvr)), np.exp(1j*np.dot([2*np.pi*qqx/Length, 2*np.pi*qqy/(Height/nUnit)], [np.mod(qqv,3)-1, qqv/3-1]))

Vint_q1 = get_LatticeInteraction_FourierTransform(Vunit, Length, Height/nUnit)
for ii in range(9):
  print 'diff %d' %(ii), Vint_q[ii]-Vint_q1[ii]

exit()

" Define projected basis function "
Ecut = -2.0+1.5 # Energy cut for projection
Ecut1 = -0.0 # Energy cut for projection

print 'Ecut = ', Ecut
print 'vNN = ', vNN, 'vNNN = ', vNNN

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
        SingleParticleBas[NumSPB]              = tuple([iix,iiy,jj])
        SingleParticleBas[tuple([iix,iiy,jj])] = NumSPB
        OnSite += [UT[ii]['eige'][jj]]
        NumSPB += 1
        kBand[tuple([iix, iiy])] += [jj]

OnSite = np.array(OnSite)

print 'NumSPB', NumSPB
#print 'SB', SingleParticleBas
#print SingleParticleBas
print 'Onsite', OnSite
#print 'kBand', kBand

# Define many particle basis functions
num_basis = int(sps.comb(NumSPB,n)) # total number of basis functions
order_basis = list( itertools.combinations(range(NumSPB),n) ) # basis functions presents in order

print 'num_basis', num_basis
#print 'order_basis', order_basis

# Reorganize many particle states according to their total momentum
StatMom = {}
for ii in range(0,Height/nUnit*Length): 
  StatMom[ii] = {0:[0,0]} 

num_kbasis = np.zeros((Length*Height/nUnit,),dtype = int)
#print 'xx', num_kbasis[0], num_kbasis[1]

for ii in range(0, num_basis):
  Momtii = np.array([0,0])
  for jj in order_basis[ii]:
    Momtii += np.array(SingleParticleBas[jj])[[0,1]]
  num_Momt = np.mod(Momtii[0],Length)+np.mod(Momtii[1],Height/nUnit)*Length
  StatMom[num_Momt][num_kbasis[num_Momt]] = order_basis[ii]
  StatMom[num_Momt][tuple(order_basis[ii])] = num_kbasis[num_Momt]
  num_kbasis[num_Momt] += 1

#print 'num k bas: ', num_kbasis
#for ii in range(0,9):
#  print 'Stat =',ii,' \n', StatMom[ii]
#print 'Stat =',ii,' \n', StatMom[0]

#Ham_mod = 'full'
Ham_mod = 'sparse'
if (Ham_mod == 'full') :
  Eige=[]
if (Ham_mod == 'sparse') :
  numnonzero = sps.comb(n,2)*sps.comb(NumSPB+2-n,2)
  mode1 = 'SA'
  eigv_k =10 
  Eige = np.zeros((eigv_k*N/nUnit,)) # !!!!!! here define the k number
  #print 'shape Eige ', np.shape(Eige)
numss = 0
# Define Hamiltonian of each total momentum
for kk in range(0,Length*Height/nUnit): # Momentum index
  kx = np.mod(kk,Length)
  ky = kk/Length
 
  # define block Hamiltonian
  if (Ham_mod == 'full'):
    Ham = np.zeros((num_kbasis[kk],num_kbasis[kk]),dtype = complex)
    #Hamk = np.zeros((len(renormalize_factor),len(renormalize_factor)),dtype = complex)
  elif (Ham_mod == 'sparse'):
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
    if (Ham_mod == 'full'):
      Ham[statk,statk] += sum(OnSite[list(bas)])
    elif (Ham_mod == 'sparse'):
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
              if (Ham_mod == 'full'):
                Ham[statk, statkp] += (-1)**pm_time*np.dot(Walpha,np.dot(Vint_q[qqm],Wbeta))/(1.0*Length*Height/nUnit)
                #print 'Walpha =', Walpha, 'Wbeta =', Wbeta, '\n mpl = ',np.dot(Vint_q[qqm],Wbeta), 'xx =', np.dot(Walpha,np.dot(Vint_q[qqm],Wbeta))
                if kk < 10  and statk == statkp and statk == 0 and (bas1[ii0] == bas[ii0] or bas1[jj0] == bas[jj0]) :
                  numss += 1
                  #print 'kk =', kk, 'numss =', numss, np.array_str( ((-1)**pm_time*np.dot(Walpha,np.dot(Vint_q[qqm],Wbeta))/(1.0*Length*Height/nUnit)),precision=4,suppress_small=True)
                  #print 'numss =', numss, np.array_str( ((-1)**pm_time*np.dot(Vint_q[qqm],Wbeta)/(1.0*Length*Height/nUnit)),precision=4,suppress_small=True)
                  #print '--------------------------------------------', (Ham[statk,statkp])
              elif (Ham_mod == 'sparse'):
                num_nonzero += 1
                row_s[num_nonzero]  = statk
                col_s[num_nonzero]  = statkp
                data_s[num_nonzero] = (-1)**pm_time*np.dot(Walpha,np.dot(Vint_q[qqm],Wbeta))/(1.0*Length*Height/nUnit)

  if (Ham_mod == 'full'):
    Eige += list(np.linalg.eigvalsh(Ham))
    #print 'xxx \n', np.array_str(Ham, precision=4, suppress_small=True)#-Eige[0]
    #print list(Eige[kk])
    #print 'xxx \n', np.array_str(Eige[kk], precision=4, suppress_small=True)#-Eige[0]
    #print 'xxx \n', np.array_str(Ham-np.array(Eige[0]), precision=4, suppress_small=True)
  elif (Ham_mod == 'sparse'):
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
pathd = '/home/yfren/Dropbox/csun/python_codes/kagome_lattice/MomentumSpace/test_time_converg/data'
np.savetxt(pathd+'/Eige_V12_05_Ec1.txt',np.array(Eige))



