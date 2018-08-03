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

plt.switch_backend('agg')

mry0 = psutil.virtual_memory().used/1024.0**3

init_date()
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
columnband = 0
if columnband :
  h0,h1 = column_tab(tunit, vunit, Ny=30,PBC=1)
  #print np.shape(h0)
  #print h0
  #print h1
  OneDband(h0,h1)

"------------------------------------------------------------"

"test interaction part"
m = m_flux
asp = ab_aspect

eigv_k = 7 # 10
mode1 = 'SA'

# single layer initialize
numbas = int(comb(m,n_electron)) 
print 'numbas', numbas, 'm =', m, 'n =', n_electron

ts = time.time()
sq2bas, bas2sq = get_basis_U1(m, n_electron) 
print 'momery of maps =', (sys.getsizeof(sq2bas) + sys.getsizeof(bas2sq))/1024.0/1024.0/1024.0
tnote(ts, mry0, 'get basis time =')

ts = time.time()
Vjjt = FQHE_2DEG_Int(m, asp)
tnote(ts, mry0, 'get interaction matrix elements time =')

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

#fig=plt.figure(20)

#dd = np.linspace(0,2,21)
dd = [2.0]
"test bilayer FQHE systems"
bilayerfull = 1
if bilayerfull :
  sq2basTB, bas2sqTB = get_bilayer_bas(sq2bas, sq2bas)
  #dd = [0]
  EigeBLt = np.zeros((len(dd),eigv_k), dtype=float)
  fig = plt.figure(1)
  for ii in range(len(dd)):
    print '# ii = ', ii
    stime = time.time()
    d = dd[ii]
    VjjtIntL = FQHE_2DEG_Int_Interlayer(m, asp, d)
    #print VjjtIntL
    #break
    row1, col1, dat1 = get_FQHE_Interlayer_MatEle(bas2sqTB, sq2basTB, bas2sq, sq2bas, bas2sq, sq2bas, VjjtIntL)

    #HamBL = sps.kron(Ham, Ham, format='coo') #+ 0.0*sp.sparse.coo_matrix((dat1,(row1, col1)), shape=(numbas**2, numbas**2))
    HamBL = sps.kron(np.eye(numbas), Hamff, format='coo')
    HamBL += sps.kron(Hamff, np.eye(numbas), format='coo') + sp.sparse.coo_matrix((dat1,(row1, col1)), shape=(numbas**2, numbas**2))
  
    EigeBL, EigfBL = eigsh(HamBL, k=eigv_k, which=mode1)
    print EigeBL
    EigeBLt[ii,:] = np.real(sorted(EigeBL))-np.amin(EigeBL)
  
    etime = time.time()
    print 'get basis time =', etime - stime
  
    #plt.clf()
    #plt.plot(dd*eigv_k, EigeBLt[:,0:5],'o')
    #plt.savefig('EigeBLt_third_'+str(m)+'_t.eps',format='eps')
  
np.savetxt('EigeBLt_third_'+str(m)+'_t1.dat', EigeBLt)
#plt.show()

end_date()
