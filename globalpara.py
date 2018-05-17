#!/bin/usr/env python

"""
Global parameters
--
Remember to modify both global global parameters 
& parameters for specific calculation
"""

import numpy as np
import scipy as sp

"Physical Constant in SI unit"
me_emass = 9.10938356*1e-31  # kg
h_planck = 6.626070040*1e-34  # J*s
hbar_planck = h_planck/np.pi/2  # J*s
c_lightspeed = 2.99792458*1e8  # m/s   
ke_coulombconstant = 8.9875517873681764*1e9  # J*m/C^2 # 1/(4pi*epsilon)
e_charge = 1.6021766208*1e-19  # C
muB_bohr = e_charge*hbar_planck/2/me_emass  # 9.274*1e-24 J/T
phi0_fluxquant = h_planck/2/e_charge  # h/2e

pi = 3.1415926535897932

# atomic chains in y and x directions
# Honeycomb lattice: y is zigzag, while x is armchair
# Kagome lattice: Height = 3*ny where ny is number of 
#                 unit cell along [1/2, sqrt(3)/2] direction 
#                 while nx is that along [1, 0] direction

"Global parameters for ED calculation in different lattices"
Lattice_type = 'kagome'

if Lattice_type == 'kagome' or Lattice_type == 'Lieb' :
  nUnit = 3
elif Lattice_type == 'honeycomb' or Lattice_type == 'checkerboard' :
  nUnit = 2

if (Lattice_type == 'honeycomb' or Lattice_type == 'kagome') :
  distNN, distNNN, distN3 = 1.0, np.sqrt(3.0), 2  # distance between NN/NNN sites
elif (Lattice_type == 'Lieb') :
  distNN, distNNN, distN3 = 1.0, np.sqrt(2.0), 2  # distance between NN/NNN sites  

Ny,Nx = 3, 3

Height, Length = nUnit*Ny, Nx  #3, 3

Flagrc = 'complex'  # 'float' # 'complex' // define if there are complex hopping parameters

nNNcell = 9  # number of neighbor unit cells
pbcx, pbcy = 1, 1  # boundary condition: 1: true. 0: false

"Global parameters in FQHE of 2DEG"
V0_interaction = 1  # ke*e^2/epsilon/2l # Energy unit 
m_flux = 12  # number of flux quantum in area of a*b: a*b/(2*pi*l^2) = a*b*B/(h/2e)
             # corresponds to the total sites
ab_aspect = 1  # aspect of the square region defined as sqrt(a/b)

"global Global parameters"
fillingfrac = 3  # Filling fraction of total site

"All system need parameters: "
N_site = Height * Length  # for lattice model
#N_site = m_flux  # for FQHE system
n_electron = N_site/fillingfrac


if __name__ == '__main__':
  """
  Here is the main program
  """
  print('Run global parameters') 
else:
  print('\n Import global parameters \n ')

