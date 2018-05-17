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

plt.switch_backend('agg')
"test tight-binding model and band structure"
testband = 0
if testband :
  if Lattice_type == 'kagome' :
    tunit,vunit = unit_kagome(t2=-0.0,t2c = -0.0,t1c = 0.0) 
    TwoDband(tunit)
  elif Lattice_type == 'Lieb' :
    tunit,vunit = unit_Lieb(Ea = 4.0, t3 = 0.2) 
    # QBC appears when Ea*t3>0
    # No QBC when t2 != 0 
    # Without Ea, t1 & t3 only also show QBC

  TwoDband(tunit)

#h0,h1 = column_tab(tunit, vunit, Ny=30,PBC=1)
#print np.shape(h0)
#print h0
#print h1
#OneDband(h0,h1)

"------------------------------------------------------------"
"------------------------------------------------------------"
"------------------------------------------------------------"

"test interaction part"
m = m_flux
asp = ab_aspect
stime = time.time()
sq2bas, bas2sq = get_basis_U1(m,n_electron) #m/fillingfrac)
#sq2bas, bas2sq = get_basis_U1(m,3)
etime = time.time()
print 'momery of maps =', (sys.getsizeof(sq2bas) + sys.getsizeof(bas2sq))/1024.0/1024.0/1024.0
print 'get basis time =', etime - stime

print 'sq2bas ', sq2bas

numbas = int(comb(m,n_electron)) #m/fillingfrac))
print 'numbas', numbas, 'm =', m, 'n =', n_electron

stime = time.time()
Vjjt = FQHE_2DEG_Int(m, asp)
etime = time.time()
print 'get int matrix time =', etime - stime
print Vjjt

starttime = time.time()
row, col, dat = get_IntMatEle(bas2sq, sq2bas, Vjjt)
endtime = time.time()
print 'time =', endtime - starttime
print 'len dat =', len(dat), len(dat)*1.0/numbas, 'size dat =', sys.getsizeof(dat)/1024.0/1024.0/1024.0
#numbas = int(comb(m,m/fillingfrac))
#sys.exit()
#print dat
#print row
#print col

stime = time.time()
Ham = sp.sparse.coo_matrix((dat,(row,col)), shape=(numbas,numbas))
Ham1 = Ham.toarray()
print np.amax(np.amax(abs(Ham1-np.conjugate(Ham1.T))))

eigv_k = 10
mode1 = 'SA'
Eige0, Eigf0 = eigsh(Ham, k=eigv_k, which = mode1)
etime = time.time()
print etime-stime

print sorted(Eige0), 'per electron ', sorted(Eige0)[0]/n_electron

eigv_k = 10
"Include translation invariance"
#sq2bask, bas2sqk, N_bk = sortU1basis_k_FQHE(sq2bas)
#print N_bk
#eigek = np.zeros((m,eigv_k))
#for ii in range(m):
#  bas2sq0 = bas2sqk[ii]
#  sq2bas0 = sq2bask[ii]
#  #print 's2b 0 =', sq2bas0.keys()
#  #print 's2b 0 =', sq2bas0.keys()
#  row, col, dat = get_IntMatEle(bas2sq0, sq2bas0, Vjjt)
#  Ham = sp.sparse.coo_matrix((dat,(row,col)), shape=(N_bk[ii], N_bk[ii]))
#  Eige0, Eigf0 = eigsh(Ham, k=eigv_k, which=mode1)
#  eigek[ii,:] = Eige0
#  print 'ii = ', ii, Eige0

#fig=plt.figure(20)
#plt.plot(range(m),eigek,'_')

#stop

"test bilayer FQHE systems"
sq2basTB, bas2sqTB = get_bilayer_bas(sq2bas, sq2bas)
dd = np.linspace(0,2,21)
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
  HamBL = sps.kron(np.eye(numbas), Ham, format='coo')
  HamBL += sps.kron(Ham, np.eye(numbas), format='coo') + sp.sparse.coo_matrix((dat1,(row1, col1)), shape=(numbas**2, numbas**2))

  EigeBL, EigfBL = eigsh(HamBL, k=eigv_k, which=mode1)
  print sorted(EigeBL)
  EigeBLt[ii,:] = np.real(sorted(EigeBL))-np.amin(EigeBL)

  etime = time.time()
  print 'get basis time =', etime - stime

  plt.clf()
  plt.plot(dd, EigeBLt[:,0:5],'-o')
  plt.savefig('EigeBLt'+str(m)+'.eps',format='eps')

np.savetxt('EigeBLt'+str(m)+'.dat', EigeBLt)
#plt.show()
