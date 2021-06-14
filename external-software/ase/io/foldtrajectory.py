"""foldtrajectory - folds atoms into the periodic computational box.

Usage:
    python -m ase.io.foldtrajectory infile.traj outfile.traj
    
In molecular dynamics simulations with periodic boundary conditions,
atoms sometimes move out of one side of the computational box and in
through the other.  Such atoms have coordinates outside the box.
This facilitates analysis of e.g. diffusion, but can be problematic
when plotting.  This script reads through a trajectory file, and
write a new one where all atoms are mapped into the computational box.
If there are axes with free boundary conditions, the corresponding
coordinate is left unchanged.

SIDE EFFECT: All energies, forces and stresses are removed (yes, this
can be considered as a bug!)
"""

from __future__ import print_function
import sys
from ase.io.trajectory import Trajectory

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)
    
infile = Trajectory(sys.argv[1])
outfile = None

for atoms in infile:
    atoms.set_scaled_positions(atoms.get_scaled_positions())
    atoms.set_calculator(None)  # or the singlepointcalculator fails!
    if outfile is None:
        outfile = Trajectory(sys.argv[2], 'w')
    outfile.write(atoms)
        
outfile.close()
