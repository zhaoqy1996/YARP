"""Get energy from a LAMMPS calculation"""

from __future__ import print_function

import os
import numpy as np
from ase import Atom
from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
import ase.io
from ase import units
from ase.md.verlet import VelocityVerlet

# potential_path must be set as an environment variable
potential_path = os.environ.get('LAMMPS_POTENTIALS_PATH', '.')

cmds = ["pair_style eam/alloy",
        "pair_coeff * * {path}/NiAlH_jea.eam.alloy Ni H"
        "".format(path=potential_path)]

nickel = bulk('Ni', cubic=True)
nickel += Atom('H', position=nickel.cell.diagonal()/2)
# Bit of distortion
nickel.set_cell(nickel.cell + [[0.1, 0.2, 0.4],
                               [0.3, 0.2, 0.0],
                               [0.1, 0.1, 0.1]], scale_atoms=True)

lammps = LAMMPSlib(lmpcmds=cmds,
                   atom_types={'Ni': 1, 'H': 2},
                   log_file='test.log', keep_alive=True)

nickel.set_calculator(lammps)

E = nickel.get_potential_energy()
F = nickel.get_forces()
S = nickel.get_stress()

print('Energy: ', E)
print('Forces:', F)
print('Stress: ', S)
print()

E = nickel.get_potential_energy()
F = nickel.get_forces()
S = nickel.get_stress()


lammps = LAMMPSlib(lmpcmds=cmds,
                   log_file='test.log', keep_alive=True)
nickel.set_calculator(lammps)

E2 = nickel.get_potential_energy()
F2 = nickel.get_forces()
S2 = nickel.get_stress()

assert np.allclose(E, E2)
assert np.allclose(F, F2)
assert np.allclose(S, S2)

nickel.rattle(stdev=0.2)
E3 = nickel.get_potential_energy()
F3 = nickel.get_forces()
S3 = nickel.get_stress()

print('rattled atoms')
print('Energy: ', E3)
print('Forces:', F3)
print('Stress: ', S3)
print()

assert not np.allclose(E, E3)
assert not np.allclose(F, F3)
assert not np.allclose(S, S3)

nickel += Atom('H', position=nickel.cell.diagonal()/4)
E4 = nickel.get_potential_energy()
F4 = nickel.get_forces()
S4 = nickel.get_stress()

assert not np.allclose(E4, E3)
assert not np.allclose(F4[:-1,:], F3)
assert not np.allclose(S4, S3)


# the example from the docstring

cmds = ["pair_style eam/alloy",
        "pair_coeff * * {path}/NiAlH_jea.eam.alloy Al H".format(path=potential_path)]

Ni = bulk('Ni', cubic=True)
H = Atom('H', position=Ni.cell.diagonal()/2)
NiH = Ni + H

lammps = LAMMPSlib(lmpcmds=cmds, log_file='test.log')

NiH.set_calculator(lammps)
print("Energy ", NiH.get_potential_energy())


# a more complicated example, reading in a LAMMPS data file

# first, we generate the LAMMPS data file
lammps_data_file = """
8 atoms
1 atom types
6 bonds
1 bond types
4 angles
1 angle types

-5.1188800000000001e+01 5.1188800000000001e+01 xlo xhi
-5.1188800000000001e+01 5.1188800000000001e+01 ylo yhi
-5.1188800000000001e+01 5.1188800000000001e+01 zlo zhi
0.000000 0.000000 0.000000 xy xz yz

Masses

1 56

Bond Coeffs

1 646.680887 1.311940

Angle Coeffs

1 300.0 107.0

Pair Coeffs

1 0.105000 3.430851

Atoms

1 1 1 0.0 -7.0654012878945753e+00 -4.7737244253442213e-01 -5.1102452666801824e+01 2 -1 6
2 1 1 0.0 -8.1237844371679362e+00 -1.3340695922796841e+00 4.7658302278206179e+01 2 -1 5
3 1 1 0.0 -1.2090525219882498e+01 -3.2315354021627760e+00 4.7363437099502839e+01 2 -1 5
4 1 1 0.0 -8.3272244953257601e+00 -4.8413162043515321e+00 4.5609055410298623e+01 2 -1 5
5 2 1 0.0 -5.3879618209198750e+00 4.9524635221072280e+01 3.0054862714858366e+01 6 -7 -2
6 2 1 0.0 -8.4950075933508273e+00 -4.9363297129348325e+01 3.2588925816534982e+01 6 -6 -2
7 2 1 0.0 -9.7544282093133940e+00 4.9869755980935565e+01 3.6362287886934432e+01 6 -7 -2
8 2 1 0.0 -5.5712437770663756e+00 4.7660225526197003e+01 3.8847235874270240e+01 6 -7 -2

Velocities

1 -1.2812627962466232e-02 -1.8102422526771818e-03 8.8697845357364469e-03
2 7.7087896348612683e-03 -5.6149199730983867e-04 1.3646724560472424e-02
3 -3.5128553734623657e-03 1.2368758037550581e-03 9.7460093657088121e-03
4 1.1626059392751346e-02 -1.1942908859710665e-05 8.7505240354339674e-03
5 1.0953500823880464e-02 -1.6710422557096375e-02 2.2322216388444985e-03
6 3.7515599452757294e-03 1.4091708517087744e-02 7.2963916249300454e-03
7 5.3953961772651359e-03 -8.2013715102925017e-03 2.0159609509813853e-02
8 7.5074008407567160e-03 5.9398495239242483e-03 7.3144909044607909e-03

Bonds

1 1 1 2
2 1 2 3
3 1 3 4
4 1 5 6
5 1 6 7
6 1 7 8

Angles

1 1 1 2 3
2 1 2 3 4
3 1 5 6 7
4 1 6 7 8
"""
with open('lammps.data', 'w') as fd:
    fd.write(lammps_data_file)

# then we run the actual test

Z_of_type = {1:26}
atom_types = {'Fe':1,}

at = ase.io.read('lammps.data', format='lammps-data', Z_of_type=Z_of_type, units='real')

header = ["units           real",
          "atom_style      full",
          "boundary        p p p",
          "box tilt        large",
          "pair_style      lj/cut/coul/long 12.500",
          "bond_style      harmonic",
          "angle_style     harmonic",
          "kspace_style    ewald 0.0001",
          "read_data       lammps.data"]
cmds = [] 

lammps = LAMMPSlib(lammps_header=header, lmpcmds=cmds, atom_types=atom_types, create_atoms=False, create_box=False, boundary=False, keep_alive=True, log_file='test.log')
at.set_calculator(lammps)
dyn = VelocityVerlet(at, 1 * units.fs)

energy = at.get_potential_energy()
energy_ref = 2041.41198295
diff = abs((energy - energy_ref) / energy_ref)
assert diff < 1e-10

dyn.run(10)
energy = at.get_potential_energy()
energy_ref = 312.431585607
diff = abs((energy - energy_ref) / energy_ref)
assert diff < 1e-10, "%d" % energy
