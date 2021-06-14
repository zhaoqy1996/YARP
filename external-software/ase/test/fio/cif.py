import io
import numpy as np
import warnings

from ase.io import read
from ase.io import write

content = u"""
data_1


_chemical_name_common                  'Mysterious something'
_cell_length_a                         5.50000
_cell_length_b                         5.50000
_cell_length_c                         5.50000
_cell_angle_alpha                      90
_cell_angle_beta                       90
_cell_angle_gamma                      90
_space_group_name_H-M_alt              'F m -3 m'
_space_group_IT_number                 225

loop_
_space_group_symop_operation_xyz
   'x, y, z'
   '-x, -y, -z'
   '-x, -y, z'
   'x, y, -z'
   '-x, y, -z'
   'x, -y, z'
   'x, -y, -z'
   '-x, y, z'
   'z, x, y'
   '-z, -x, -y'
   'z, -x, -y'
   '-z, x, y'
   '-z, -x, y'
   'z, x, -y'
   '-z, x, -y'
   'z, -x, y'
   'y, z, x'
   '-y, -z, -x'
   '-y, z, -x'
   'y, -z, x'
   'y, -z, -x'
   '-y, z, x'
   '-y, -z, x'
   'y, z, -x'
   'y, x, -z'
   '-y, -x, z'
   '-y, -x, -z'
   'y, x, z'
   'y, -x, z'
   '-y, x, -z'
   '-y, x, z'
   'y, -x, -z'
   'x, z, -y'
   '-x, -z, y'
   '-x, z, y'
   'x, -z, -y'
   '-x, -z, -y'
   'x, z, y'
   'x, -z, y'
   '-x, z, -y'
   'z, y, -x'
   '-z, -y, x'
   'z, -y, x'
   '-z, y, -x'
   '-z, y, x'
   'z, -y, -x'
   '-z, -y, -x'
   'z, y, x'
   'x, y+1/2, z+1/2'
   '-x, -y+1/2, -z+1/2'
   '-x, -y+1/2, z+1/2'
   'x, y+1/2, -z+1/2'
   '-x, y+1/2, -z+1/2'
   'x, -y+1/2, z+1/2'
   'x, -y+1/2, -z+1/2'
   '-x, y+1/2, z+1/2'
   'z, x+1/2, y+1/2'
   '-z, -x+1/2, -y+1/2'
   'z, -x+1/2, -y+1/2'
   '-z, x+1/2, y+1/2'
   '-z, -x+1/2, y+1/2'
   'z, x+1/2, -y+1/2'
   '-z, x+1/2, -y+1/2'
   'z, -x+1/2, y+1/2'
   'y, z+1/2, x+1/2'
   '-y, -z+1/2, -x+1/2'
   '-y, z+1/2, -x+1/2'
   'y, -z+1/2, x+1/2'
   'y, -z+1/2, -x+1/2'
   '-y, z+1/2, x+1/2'
   '-y, -z+1/2, x+1/2'
   'y, z+1/2, -x+1/2'
   'y, x+1/2, -z+1/2'
   '-y, -x+1/2, z+1/2'
   '-y, -x+1/2, -z+1/2'
   'y, x+1/2, z+1/2'
   'y, -x+1/2, z+1/2'
   '-y, x+1/2, -z+1/2'
   '-y, x+1/2, z+1/2'
   'y, -x+1/2, -z+1/2'
   'x, z+1/2, -y+1/2'
   '-x, -z+1/2, y+1/2'
   '-x, z+1/2, y+1/2'
   'x, -z+1/2, -y+1/2'
   '-x, -z+1/2, -y+1/2'
   'x, z+1/2, y+1/2'
   'x, -z+1/2, y+1/2'
   '-x, z+1/2, -y+1/2'
   'z, y+1/2, -x+1/2'
   '-z, -y+1/2, x+1/2'
   'z, -y+1/2, x+1/2'
   '-z, y+1/2, -x+1/2'
   '-z, y+1/2, x+1/2'
   'z, -y+1/2, -x+1/2'
   '-z, -y+1/2, -x+1/2'
   'z, y+1/2, x+1/2'
   'x+1/2, y, z+1/2'
   '-x+1/2, -y, -z+1/2'
   '-x+1/2, -y, z+1/2'
   'x+1/2, y, -z+1/2'
   '-x+1/2, y, -z+1/2'
   'x+1/2, -y, z+1/2'
   'x+1/2, -y, -z+1/2'
   '-x+1/2, y, z+1/2'
   'z+1/2, x, y+1/2'
   '-z+1/2, -x, -y+1/2'
   'z+1/2, -x, -y+1/2'
   '-z+1/2, x, y+1/2'
   '-z+1/2, -x, y+1/2'
   'z+1/2, x, -y+1/2'
   '-z+1/2, x, -y+1/2'
   'z+1/2, -x, y+1/2'
   'y+1/2, z, x+1/2'
   '-y+1/2, -z, -x+1/2'
   '-y+1/2, z, -x+1/2'
   'y+1/2, -z, x+1/2'
   'y+1/2, -z, -x+1/2'
   '-y+1/2, z, x+1/2'
   '-y+1/2, -z, x+1/2'
   'y+1/2, z, -x+1/2'
   'y+1/2, x, -z+1/2'
   '-y+1/2, -x, z+1/2'
   '-y+1/2, -x, -z+1/2'
   'y+1/2, x, z+1/2'
   'y+1/2, -x, z+1/2'
   '-y+1/2, x, -z+1/2'
   '-y+1/2, x, z+1/2'
   'y+1/2, -x, -z+1/2'
   'x+1/2, z, -y+1/2'
   '-x+1/2, -z, y+1/2'
   '-x+1/2, z, y+1/2'
   'x+1/2, -z, -y+1/2'
   '-x+1/2, -z, -y+1/2'
   'x+1/2, z, y+1/2'
   'x+1/2, -z, y+1/2'
   '-x+1/2, z, -y+1/2'
   'z+1/2, y, -x+1/2'
   '-z+1/2, -y, x+1/2'
   'z+1/2, -y, x+1/2'
   '-z+1/2, y, -x+1/2'
   '-z+1/2, y, x+1/2'
   'z+1/2, -y, -x+1/2'
   '-z+1/2, -y, -x+1/2'
   'z+1/2, y, x+1/2'
   'x+1/2, y+1/2, z'
   '-x+1/2, -y+1/2, -z'
   '-x+1/2, -y+1/2, z'
   'x+1/2, y+1/2, -z'
   '-x+1/2, y+1/2, -z'
   'x+1/2, -y+1/2, z'
   'x+1/2, -y+1/2, -z'
   '-x+1/2, y+1/2, z'
   'z+1/2, x+1/2, y'
   '-z+1/2, -x+1/2, -y'
   'z+1/2, -x+1/2, -y'
   '-z+1/2, x+1/2, y'
   '-z+1/2, -x+1/2, y'
   'z+1/2, x+1/2, -y'
   '-z+1/2, x+1/2, -y'
   'z+1/2, -x+1/2, y'
   'y+1/2, z+1/2, x'
   '-y+1/2, -z+1/2, -x'
   '-y+1/2, z+1/2, -x'
   'y+1/2, -z+1/2, x'
   'y+1/2, -z+1/2, -x'
   '-y+1/2, z+1/2, x'
   '-y+1/2, -z+1/2, x'
   'y+1/2, z+1/2, -x'
   'y+1/2, x+1/2, -z'
   '-y+1/2, -x+1/2, z'
   '-y+1/2, -x+1/2, -z'
   'y+1/2, x+1/2, z'
   'y+1/2, -x+1/2, z'
   '-y+1/2, x+1/2, -z'
   '-y+1/2, x+1/2, z'
   'y+1/2, -x+1/2, -z'
   'x+1/2, z+1/2, -y'
   '-x+1/2, -z+1/2, y'
   '-x+1/2, z+1/2, y'
   'x+1/2, -z+1/2, -y'
   '-x+1/2, -z+1/2, -y'
   'x+1/2, z+1/2, y'
   'x+1/2, -z+1/2, y'
   '-x+1/2, z+1/2, -y'
   'z+1/2, y+1/2, -x'
   '-z+1/2, -y+1/2, x'
   'z+1/2, -y+1/2, x'
   '-z+1/2, y+1/2, -x'
   '-z+1/2, y+1/2, x'
   'z+1/2, -y+1/2, -x'
   '-z+1/2, -y+1/2, -x'
   'z+1/2, y+1/2, x'

loop_
   _atom_site_label
   _atom_site_occupancy
   _atom_site_fract_x
   _atom_site_fract_y
   _atom_site_fract_z
   _atom_site_adp_type
   _atom_site_B_iso_or_equiv
   _atom_site_type_symbol
   Na         0.7500  0.000000      0.000000      0.000000     Biso  1.000000 Na
   K          0.2500  0.000000      0.000000      0.000000     Biso  1.000000 K
   Cl         0.3000  0.500000      0.500000      0.500000     Biso  1.000000 Cl
   I          0.5000  0.250000      0.250000      0.250000     Biso  1.000000 I
"""

cif_file = io.StringIO(content)

# legacy behavior is to not read the K atoms
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    atoms = read(cif_file, format='cif', fractional_occupancies=False)
elements = np.unique(atoms.get_atomic_numbers())
for n in (11, 17, 53):
    assert n in elements
try:
    atoms.info['occupancy']
    raise AssertionError
except KeyError:
    pass

cif_file = io.StringIO(content)
# new behavior is to still not read the K atoms, but build tags and info
natoms = read(cif_file, format='cif', fractional_occupancies=True)

assert len(atoms) == len(natoms)
assert np.all(atoms.get_atomic_numbers() == natoms.get_atomic_numbers())
# yield the same old atoms...
assert atoms == natoms

elements = np.unique(atoms.get_atomic_numbers())
for n in (11, 17, 53):
    assert n in elements

assert natoms.info['occupancy']
for a in natoms:
    if a.symbol == 'Na':
        assert len(natoms.info['occupancy'][a.tag]) == 2
        assert natoms.info['occupancy'][a.tag]['K'] == 0.25
        assert natoms.info['occupancy'][a.tag]['Na'] == 0.75
    else:
        assert len(natoms.info['occupancy'][a.tag]) == 1

# read/write
fname = 'testfile.cif'
with open(fname, 'w') as fd:
    write(fd, natoms, format='cif')

with open(fname) as fd:
    natoms = read(fd, format='cif', fractional_occupancies=True)

assert natoms.info['occupancy']
for a in natoms:
    if a.symbol == 'Na':
        assert len(natoms.info['occupancy'][a.tag]) == 2
        assert natoms.info['occupancy'][a.tag]['K'] == 0.25
        assert natoms.info['occupancy'][a.tag]['Na'] == 0.75
    else:
        assert len(natoms.info['occupancy'][a.tag]) == 1

# ICSD-like file from issue #293
content = u"""
data_global
_cell_length_a 9.378(5)
_cell_length_b 7.488(5)
_cell_length_c 6.513(5)
_cell_angle_alpha 90.
_cell_angle_beta 91.15(5)
_cell_angle_gamma 90.
_cell_volume 457.27
_cell_formula_units_Z 2
_symmetry_space_group_name_H-M 'P 1 n 1'
_symmetry_Int_Tables_number 7
_refine_ls_R_factor_all 0.071
loop_
_symmetry_equiv_pos_site_id
_symmetry_equiv_pos_as_xyz
1 'x+1/2, -y, z+1/2'
2 'x, y, z'
loop_
_atom_type_symbol
_atom_type_oxidation_number
Sn2+ 2
As4+ 4
Se2- -2
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_symmetry_multiplicity
_atom_site_Wyckoff_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_B_iso_or_equiv
_atom_site_occupancy
_atom_site_attached_hydrogens
Sn1 Sn2+ 2 a 0.5270(2) 0.3856(2) 0.7224(3) 0.0266(4) 1. 0
Sn2 Sn2+ 2 a 0.0279(2) 0.1245(2) 0.7870(2) 0.0209(4) 1. 0
As1 As4+ 2 a 0.6836(4) 0.1608(5) 0.8108(6) 0.0067(7) 1. 0
As2 As4+ 2 a 0.8174(4) 0.6447(6) 0.1908(6) 0.0057(6) 1. 0
Se1 Se2- 2 a 0.4898(4) 0.7511(6) 0.8491(6) 0.0110(6) 1. 0
Se2 Se2- 2 a 0.7788(4) 0.6462(6) 0.2750(6) 0.0097(6) 1. 0
Se3 Se2- 2 a 0.6942(4) 0.0517(5) 0.5921(6) 0.2095(6) 1. 0
Se4 Se2- 2 a 0.0149(4) 0.3437(6) 0.5497(7) 0.1123(7) 1. 0
Se5 Se2- 2 a 0.1147(4) 0.5633(4) 0.3288(6) 0.1078(6) 1. 0
Se6 Se2- 2 a 0.0050(4) 0.4480(6) 0.9025(6) 0.9102(6) 1. 0
"""

cif_file = io.StringIO(content)
atoms = read(cif_file, format='cif')
