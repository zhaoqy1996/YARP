from ase.build import fcc111
from ase.build import bcc111
from ase.build import hcp0001
from ase.build import fcc111_root
from ase.build import root_surface
from ase.build import root_surface_analysis

# Make samples of primitive cell
prim_fcc111 = fcc111("H", (1, 1, 2), a=1)
prim_bcc111 = bcc111("H", (1, 1, 2), a=1)
prim_hcp0001 = hcp0001("H", (1, 1, 2), a=1)

# Check valid roots up to root 21 (the 10th root cell)
valid_fcc111 = root_surface_analysis(prim_fcc111, 21)
valid_bcc111 = root_surface_analysis(prim_bcc111, 21)
valid_hcp0001 = root_surface_analysis(prim_hcp0001, 21)

# These should have different positions, but the same
# cell geometry.
assert valid_fcc111 == valid_bcc111 == valid_hcp0001

# Make an easy sample to check code errors
atoms1 = root_surface(prim_fcc111, 7)

# Ensure the valid roots are the roots are valid against
# a set of manually checked roots for this system
assert valid_fcc111 == [1.0, 3.0, 4.0, 7.0, 9.0,
                        12.0, 13.0, 16.0, 19.0, 21.0]

# Remake easy sample using surface function
atoms2 = fcc111_root("H", 7, (1, 1, 2), a=1)

# Right number of atoms
assert len(atoms1) == len(atoms2) == 14

# Same positions
assert (atoms1.positions == atoms2.positions).all()

# Same cell
assert (atoms1.cell == atoms2.cell).all()
