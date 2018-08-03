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

# single layer block-diagonalized Hamiltonian
sq2bask, bas2sqk, N_bk = sortU1basis_k_FQHE(sq2bas) # get block-diagonalized H
print 'dimension of each block =', N_bk

ts = time.time()
Hamk = {}
#eigek = np.zeros((m,eigv_k))
for ii in range(m):
  bas2sq0 = bas2sqk[ii]
  sq2bas0 = sq2bask[ii]
  row, col, dat = get_IntMatEle(bas2sq0, sq2bas0, Vjjt)
  Ham = sp.sparse.coo_matrix((dat,(row,col)), shape=(N_bk[ii], N_bk[ii]))
  Eige0, Eigf0 = eigsh(Ham, k=eigv_k, which=mode1)
  #eigek[ii,:] = Eige0
  print 'eigenvalue of %d-th block  = '%(ii), Eige0
  #Hamk[ii] = Ham
  Hamk[ii] = sp.sparse.coo_matrix((dat,(row,col)), shape=(N_bk[ii], N_bk[ii]))

  del row, col, dat, bas2sq0, sq2bas0
  gc.collect()

print '  size of Hamk =', sys.getsizeof(Hamk)/1024.0/1024.0/1024.0, 'G'

del Vjjt
gc.collect()

tnote(ts, mry0, 'get block diagonalized Ham in single layer time =')

print '\n'

'--------------------------------'
"get eigenvalues of bilaye systems"

#fig=plt.figure(20)

#dd = np.linspace(0,2,21)
dd = [2.0]
# get eigenvalues for different d
for ii in range(len(dd)):
  d = dd[ii]
  #ts = time.time()
  VjjtIntL = FQHE_2DEG_Int_Interlayer(m, asp, d)
  #tnote(ts, mry0, 'get Interlayer interaction matrix element time =')

  for kt in range(m):

    H0 = get_diag_block(Hamk, N_bk, kt, m)

    # get basis functions of bilayer systems
    #ts = time.time()
    sq2baskTB, bas2sqkTB = get_bilayer_bas_kt(sq2bask, sq2bask, kt, m, bas2sq, bas2sq)
    #tnote(ts, mry0, 'get block-diagnoalized bilayer basis =')

    "TO DO: define function to calculate sequence based on top-bottom configuration!"

    #print '  size of bas2sq map =', (sys.getsizeof(sq2baskTB)+sys.getsizeof(bas2sqkTB))/1024.0/1024.0/1024.0, 'G'

    #ts = time.time()
    row1, col1, dat1 = get_FQHE_Interlayer_MatEle(bas2sqkTB, sq2baskTB, bas2sq, sq2bas, bas2sq, sq2bas, VjjtIntL)
    #tnote(ts, mry0, 'get off-diagonalized Hamiltonian time =')
    H0 += sps.coo_matrix((dat1,(row1, col1)), shape=(H0.shape[0],H0.shape[0]))
    del row1, col1, dat1
    gc.collect()
    #tnote(ts, mry0, 'del dat ... time =')

    #exit()

#    ts = time.time()
#    npart = 1 #4 #20
#    for inpart in range(npart):
#      ts = time.time()
#      #row1, col1, dat1 = mpc_get_FQHE_Interlayer_MatEle(H0.shape[0], inpart, npart, bas2sqkTB, sq2baskTB, bas2sq, sq2bas, bas2sq, sq2bas, VjjtIntL)
#      #result = mpc_get_FQHE_Interlayer_MatEle(H0.shape[0], inpart, npart, bas2sqkTB, sq2baskTB, bas2sq, sq2bas, bas2sq, sq2bas, VjjtIntL)
#      row, col, dat = mpc_get_FQHE_Interlayer_MatEle(H0.shape[0], inpart, npart, bas2sqkTB, sq2baskTB, bas2sq, sq2bas, bas2sq, sq2bas, VjjtIntL)
#      #tnote(ts, mry0, 'get mpc off-diagonalized Hamiltonian time =')
#      #print 'type', type(result)#, result  
#      #print 'going exit'
#      #exit()
#     
#      #ts = time.time()
#      #H0 += sps.coo_matrix((result[2],(result[0], result[1])), shape=(H0.shape[0],H0.shape[0]))
#      H0 += sps.coo_matrix((dat,(row, col)), shape=(H0.shape[0],H0.shape[0]))
#      #tnote(ts, mry0, 'get full Hamiltonian time =')
#      tnote(ts, mry0, 'get mpc off-diagonal%d time '%(inpart))  
#      del row, col, dat #result
#      gc.collect
#      #tnote(ts, mry0, 'get del result time =')
#
#    del bas2sqkTB, sq2baskTB
#    gc.collect()

#    tnote(ts, mry0, 'get mpc off-diagonal time ')  
#    print '  size of H0 =', sys.getsizeof(H0)/1024.0/1024.0/1024.0, 'G'
#    print '    nonzero elements of H0 =', H0.nnz, 'dimension =', H0.shape
 
    #exit()

    Eige0, Eigf0 = eigsh(H0, k=eigv_k, which=mode1)
    #tnote(ts, mry0, 'get bilayer eigenvalue time =')

    del H0, Eigf0
    gc.collect()

    #tnote(ts, mry0, 'del H0 Eigf0, time =')
    print '        eigenvalue of kt = %d, d = %5.3f, = '%(kt, d), Eige0, '\n'

  del VjjtIntL
  gc.collect()

#t
exit()
    #plt.plot([2]*eigv_k, Eige0, '_')

#stop

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
    print sorted(EigeBL)
    EigeBLt[ii,:] = np.real(sorted(EigeBL))-np.amin(EigeBL)
  
    etime = time.time()
    print 'get basis time =', etime - stime
  
    #plt.clf()
    plt.plot(dd*eigv_k, EigeBLt[:,0:5],'o')
    plt.savefig('EigeBLt_third_'+str(m)+'_t.eps',format='eps')
  
np.savetxt('EigeBLt_third_'+str(m)+'_t.dat', EigeBLt)
plt.show()

end_date()
