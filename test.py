#!/bin/usr/env python

import scipy as sp
from scipy.special import comb
from scipy.sparse.linalg import eigsh
from globalpara import *
from lattice import *
from noninteracting import *
from ED_basis_fun import *
import time
import sys

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

"test interaction part"
m = m_flux
asp = ab_aspect
stime = time.time()
sq2bas, bas2sq = get_basis_U1(m,n_electron) #m/fillingfrac)
#sq2bas, bas2sq = get_basis_U1(m,3)
etime = time.time()
print 'momery of maps =', (sys.getsizeof(sq2bas) + sys.getsizeof(bas2sq))/1024.0/1024.0/1024.0
print 'get basis time =', etime - stime

numbas = int(comb(m,n_electron)) #m/fillingfrac))
print 'numbas', numbas, 'm =', m, 'n =', n_electron

stime = time.time()
Vjjt = FQHE_2DEG_Int(m, asp)
etime = time.time()
print 'get int matrix time =', etime - stime
#print Vjjt

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
#print Ham
#Hamf1 = Ham.toarray()
#print Hamf1

#Hamf = np.zeros((numbas,numbas), dtype = complex)
#print 'x', np.amax(np.amax(np.amax(abs(Hamf - Hamf1))))

#for ii in range(0,len(row)): 
#  Hamf[int(row[ii]), int(col[ii])] += dat[ii]
#print np.shape(Hamf)
#print sorted(np.linalg.eigh(Hamf)[0])
#print 'xx', np.amax(np.amax(np.imag(Hamf-np.conjugate(Hamf.T))))

eigv_k = 10
mode1 = 'SA'
Eige0, Eigf0 = eigsh(Ham, k=eigv_k, which = mode1)
etime = time.time()
print etime-stime

print sorted(Eige0), 'per electron ', sorted(Eige0)[0]/n_electron
