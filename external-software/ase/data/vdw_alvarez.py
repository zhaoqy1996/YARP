# encoding: utf-8
""" Van der Waals radii in [A] taken from:
A cartography of the van der Waals territories
S. Alvarez, Dalton Trans., 2013, 42, 8617-8636
DOI: 10.1039/C3DT50599E
"""

import numpy as np

vdw_radii = np.array([
    np.nan, # X  
       1.2, # H  
      1.43, # He [larger uncertainty]
      2.12, # Li 
      1.98, # Be 
      1.91, # B  
      1.77, # C  
      1.66, # N  
       1.5, # O  
      1.46, # F  
      1.58, # Ne [larger uncertainty]
       2.5, # Na 
      2.51, # Mg 
      2.25, # Al 
      2.19, # Si 
       1.9, # P  
      1.89, # S  
      1.82, # Cl 
      1.83, # Ar 
      2.73, # K  
      2.62, # Ca 
      2.58, # Sc 
      2.46, # Ti 
      2.42, # V  
      2.45, # Cr 
      2.45, # Mn 
      2.44, # Fe 
       2.4, # Co 
       2.4, # Ni 
      2.38, # Cu 
      2.39, # Zn 
      2.32, # Ga 
      2.29, # Ge 
      1.88, # As 
      1.82, # Se 
      1.86, # Br 
      2.25, # Kr 
      3.21, # Rb 
      2.84, # Sr 
      2.75, # Y  
      2.52, # Zr 
      2.56, # Nb 
      2.45, # Mo 
      2.44, # Tc 
      2.46, # Ru 
      2.44, # Rh 
      2.15, # Pd 
      2.53, # Ag 
      2.49, # Cd 
      2.43, # In 
      2.42, # Sn 
      2.47, # Sb 
      1.99, # Te 
      2.04, # I  
      2.06, # Xe 
      3.48, # Cs 
      3.03, # Ba 
      2.98, # La 
      2.88, # Ce 
      2.92, # Pr 
      2.95, # Nd 
    np.nan, # Pm 
       2.9, # Sm 
      2.87, # Eu 
      2.83, # Gd 
      2.79, # Tb 
      2.87, # Dy 
      2.81, # Ho 
      2.83, # Er 
      2.79, # Tm 
       2.8, # Yb 
      2.74, # Lu 
      2.63, # Hf 
      2.53, # Ta 
      2.57, # W  
      2.49, # Re 
      2.48, # Os 
      2.41, # Ir 
      2.29, # Pt 
      2.32, # Au 
      2.45, # Hg 
      2.47, # Tl 
       2.6, # Pb 
      2.54, # Bi 
    np.nan, # Po 
    np.nan, # At 
    np.nan, # Rn 
    np.nan, # Fr 
    np.nan, # Ra 
       2.8, # Ac [larger uncertainty]
      2.93, # Th 
      2.88, # Pa [larger uncertainty]
      2.71, # U  
      2.82, # Np 
      2.81, # Pu 
      2.83, # Am 
      3.05, # Cm [larger uncertainty]
       3.4, # Bk [larger uncertainty]
      3.05, # Cf [larger uncertainty]
       2.7, # Es [larger uncertainty]
    np.nan, # Fm 
    np.nan, # Md 
    np.nan, # No 
    np.nan, # Lr
])
vdw_radii.flags.writeable = False
