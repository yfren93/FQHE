#!/bin/usr/env python
import scipy as sp
import scipy.sparse as sps
from scipy.special import comb
from scipy.sparse.linalg import eigsh
from globalpara import *
from lattice import *
from noninteracting import *
from ED_basis_fun import *
import time
import sys
import gc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#plt.switch_backend('agg')

mry0 = psutil.virtual_memory().used/1024.0**3

mode1 = 'SA'
init_date()  # start time

"test tight-binding model and band structure"
testband = 0
if testband :
  if Lattice_type == 'kagome' :
    tunit,vunit = unit_kagome(t2=-0.1,t2c = 0.1,t1c = -0.) 
    TwoDband(tunit)
  elif Lattice_type == 'Lieb' :
    tunit,vunit = unit_Lieb(Ea = 4.0, t3 = 0.2) 
    # QBC appears when Ea*t3>0
    # No QBC when t2 != 0 
    # Without Ea, t1 & t3 only also show QBC

  TwoDband(tunit)

"------------------------------------------------------------"

"test column band"
def CalculateColumnBand :
  h0,h1 = column_tab(tunit, vunit, Ny=30,PBC=1)
  #print np.shape(h0)
  #print h0
  #print h1
  OneDband(h0,h1)

#CalculateColumnBand 

"------------------------------------------------------------"

"test interaction part"
m = m_flux
asp = ab_aspect

eigv_k = 10 
mode1 = 'SA'

# single layer initialize
numbas = int(comb(m,n_electron)) 
print 'numbas', numbas, 'm =', m, 'n =', n_electron

sq2bas, bas2sq = get_basis_U1(m, n_electron)  # get single layer basis 

Vjjt = FQHE_2DEG_Int(m, asp)  # get intralayer interaction

# single layer full Hamiltonian
singlelayer = 1
if singlelayer :
  ts = time.time()
  row, col, dat = get_IntMatEle(bas2sq, sq2bas, Vjjt)
  tnote(ts, mry0, 'time of get Hamiltonian in single layer =')
  print 'len dat =', len(dat), len(dat)*1.0/numbas, 'size dat =', sys.getsizeof(dat)/1024.0/1024.0/1024.0

  ts = time.time()
  Hamff = sp.sparse.coo_matrix((dat,(row,col)), shape=(numbas,numbas))
  Eige0, Eigf0 = eigsh(Hamff, k=eigv_k, which = mode1)
  tnote(ts, mry0, 'get eigenvalue time =') 
  print sorted(Eige0), 'per electron ', sorted(Eige0)[0]/n_electron

"------------------------------------------------------------"

"test bilayer FQHE systems"

#dd = np.linspace(0,2,21)
dd = [0.8]
#dd = [1.4]
bilayerfull = 0
if bilayerfull :
  sq2basTB, bas2sqTB = get_bilayer_bas(sq2bas, sq2bas)
  #dd = [0]
  EigeBLt = np.zeros((len(dd),eigv_k), dtype=float)
  fig = plt.figure(1)
  for ii in range(len(dd)):
    print '# ii = ', ii
    stime = time.time()
    d = dd[ii]

    HamBL = getHamiltonian(Hamff, m, asp, d, bas2sqTB, sq2basTB, bas2sq, sq2bas, thetax=0.0, thetay=0.0)
  
    EigeBL, EigfBL = eigsh(HamBL, k=eigv_k, which=mode1)
    print sorted(EigeBL)
    EigeBLt[ii,:] = np.real(sorted(EigeBL))-np.amin(EigeBL)
  
    etime = time.time()
    print 'get basis time =', etime - stime
  
    plt.clf()
    plt.plot(dd*eigv_k, EigeBLt[:,0:5],'o')
    #plt.savefig('EigeBLt_third_'+str(m)+'_t.eps',format='eps')
    plt.savefig('EigeBLt_'+str(m)+'_t1.eps',format='eps')
  
  #np.savetxt('EigeBLt_third_'+str(m)+'_t1.dat', EigeBLt)
  np.savetxt('EigeBLt_'+str(m)+'_t1.dat', EigeBLt)
  #plt.show()

"------------------------------------------------------------"

"test drag Chern number in bilayer system"
CalculateDragChernNumber = 1
if CalculateDragChernNumber :

  sq2basTB, bas2sqTB = get_bilayer_bas(sq2bas, sq2bas) 
 
  d = dd[0]
  
  Ntheta = 9
  thetax0 = np.linspace(0*np.pi, fillingfrac*2*np.pi, Ntheta)/n_electron
  thetay0 = np.linspace(0*np.pi, fillingfrac*2*np.pi, Ntheta)/n_electron

  bondflux, BerryCur = getBerryCurv(thetax0, thetay0, getHamiltonian, Hamff, m, asp, d, bas2sqTB, sq2basTB, bas2sq, sq2bas)

  np.savetxt('BerryCur_'+str(m)+'_d'+str(int(10*d))+'_i.dat', np.imag(bondflux))

  fig=plt.figure(251)
  ax = fig.gca(projection='3d')
  X,Y = np.meshgrid(thetax0[0:Ntheta-1], thetay0[0:Ntheta-1])
  ax.plot_surface(X,Y,BerryCur,cmap=cm.coolwarm)

end_date()
plt.show()
