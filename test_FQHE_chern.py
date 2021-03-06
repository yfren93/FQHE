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

'''
  Function region starts
'''
#def get_bilayer_Hamiltonian()

'''
  Function region ends
'''

mry0 = psutil.virtual_memory().used/1024.0**3

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

eigv_k = 7 
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
    VjjtIntL = FQHE_2DEG_Int_Interlayer(m, asp, d)
    #print VjjtIntL
    #break
    row1, col1, dat1 = get_FQHE_Interlayer_MatEle(bas2sqTB, sq2basTB, bas2sq, sq2bas, bas2sq, sq2bas, VjjtIntL)

    HamBL = sps.kron(np.eye(numbas), Hamff, format='coo')
    HamBL += sps.kron(Hamff, np.eye(numbas), format='coo') + sp.sparse.coo_matrix((dat1,(row1, col1)), shape=(numbas**2, numbas**2))
  
    EigeBL, EigfBL = eigsh(HamBL, k=eigv_k, which=mode1)
    print EigeBL
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

  Ntheta = 9
  thetax0 = np.linspace(0*np.pi, fillingfrac*2*np.pi, Ntheta)/n_electron
  thetay0 = np.linspace(0*np.pi, fillingfrac*2*np.pi, Ntheta)/n_electron
  print 'thetax = \n', thetax0, 'thetay = \n', thetay0

  sq2basTB, bas2sqTB = get_bilayer_bas(sq2bas, sq2bas)

  d = dd[0]

  bondflux = np.zeros((Ntheta**2, Ntheta**2), dtype=complex) # iix * Ntheta + iiy
  BerryCur = np.zeros((Ntheta-1, Ntheta-1))

  wavfun = np.zeros((numbas**2, Ntheta), dtype=complex)
  EigeBLt = np.zeros((Ntheta**2, eigv_k))

  for iix in range(0, Ntheta):
    for iiy in range(0, Ntheta):
      ts = time.time()

      # get theta dependent interlayer coupling matrix
      VjjtIntL = FQHE_2DEG_Int_Interlayer(m, asp, d, thetax=thetax0[iix], thetay=thetay0[iiy])
      row1, col1, dat1 = get_FQHE_Interlayer_MatEle(bas2sqTB, sq2basTB, bas2sq, sq2bas, bas2sq, sq2bas, VjjtIntL)

      # get new Hamiltonian and eigenstates
      HamBL = sps.kron(np.eye(numbas), Hamff, format='coo')
      HamBL += sps.kron(Hamff, np.eye(numbas), format='coo') + sp.sparse.coo_matrix((dat1,(row1, col1)), shape=(numbas**2, numbas**2))

      # Calculate eigen-vector with given initial vector from neighbor sites
      if iix == 0 and iiy == 0 :
        EigeBL, EigfBL = eigsh(HamBL, k=eigv_k, which=mode1)
      if iix == 0 and iiy > 0 :
        EigeBL, EigfBL = eigsh(HamBL, k=eigv_k, which=mode1, v0=wavfun[:,iiy-1])
      if iix > 0 :
        EigeBL, EigfBL = eigsh(HamBL, k=eigv_k, which=mode1, v0=wavfun[:,iiy])
        
      print EigeBL
      EigeBL = np.real(EigeBL)

      EigeBLt[iix*Ntheta+iiy, :] = np.real(sorted(EigeBL))  # eigen values

      plt.clf()
      plt.plot(range(Ntheta**2), EigeBLt[:, 0:5], 'o')
      plt.savefig('EigeBL_Chern_'+str(m)+'_d'+str(int(10*d))+'.eps',format='eps')

      # calculate the bondflux
      if iix > 0 :
        point_init, point_final = (iix-1)*Ntheta+iiy, iix*Ntheta+iiy  #
        # < init | final > : overlap between initial state of (theta_x0, theta_y0) and final state of (theta_x0 + d_theta, theta_y0)
        #bondflux[point_init, point_final] = renorm_a(np.dot(np.conjugate(wavfun[:,iiy]), EigfBL[:, EigeBL.argmin()]))
        bondflux[point_init, point_final] = np.dot(np.conjugate(wavfun[:,iiy]), EigfBL[:, EigeBL.argmin()])
        bondflux[point_final, point_init] = np.conjugate(bondflux[point_init, point_final])

        print 'bond_flux ', point_init, point_final, bondflux[point_init, point_final]
      wavfun[:, iiy] = EigfBL[:, EigeBL.argmin()]#*np.conjugate(ppv)/abs(ppv)  # save the ground state wavefunction

      if iiy > 0 :
        point_init, point_final = iix*Ntheta+iiy-1, iix*Ntheta+iiy  #
        # < init | final > : overlap between initial state of (theta_x0, theta_y0) and final state of (theta_x0, theta_y0 + d_theta)
        #bondflux[point_init, point_final] = renorm_a(np.dot(np.conjugate(wavfun[:,iiy-1]), wavfun[:,iiy]))
        bondflux[point_init, point_final] = np.dot(np.conjugate(wavfun[:,iiy-1]), wavfun[:,iiy])
        bondflux[point_final, point_init] = np.conjugate(bondflux[point_init, point_final])

        print 'bond_flux ', point_init, point_final, bondflux[point_init, point_final]

  np.savetxt('EigeBL_Chern_'+str(m)+'_d'+str(int(10*d))+'.dat', EigeBLt)
  np.savetxt('BondFluxBL_Chern_'+str(m)+'_d'+str(int(10*d))+'_r.dat', np.real(bondflux))
  np.savetxt('BondFluxBL_Chern_'+str(m)+'_d'+str(int(10*d))+'_i.dat', np.imag(bondflux))
  #plt.show()

  for iix in range(Ntheta-1) :
    for iiy in range(Ntheta-1) :
      pt00, pt10, pt11, pt01 = iix*Ntheta+iiy, (iix+1)*Ntheta+iiy, (iix+1)*Ntheta+(iiy+1), iix*Ntheta+iiy+1
      BerryCur[iix, iiy] = np.imag(bondflux[pt00,pt10]+bondflux[pt10,pt11]+bondflux[pt11,pt01]+bondflux[pt01,pt00])
      print 'Berry Cur ', iix, iiy, BerryCur[iix, iiy]

  np.savetxt('BerryCur_'+str(m)+'_d'+str(int(10*d))+'_i.dat', np.imag(bondflux))

  fig=plt.figure(251)
  ax = fig.gca(projection='3d')
  X,Y = np.meshgrid(thetax0[0:Ntheta-1], thetay0[0:Ntheta-1])
  ax.plot_surface(X,Y,BerryCur,cmap=cm.coolwarm)

end_date()
plt.show()
