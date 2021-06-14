import os

from ase.test import require
from ase.build import bulk
from ase.calculators.calculator import kpts2mp
from ase.calculators.elk import ELK

require('elk')

atoms = bulk('Al', 'bcc', a=4.0)

# save ELK_SPECIES_PATH
ELK_SPECIES_PATH = os.environ.get('ELK_SPECIES_PATH', None)
assert ELK_SPECIES_PATH is not None

# find rmt of the default species
sfile = os.path.join(os.environ['ELK_SPECIES_PATH'], 'elk.in')
assert os.path.exists(sfile)
slines = open(sfile, 'r').readlines()
rmt_orig = {}
for name in ['Al']:
    found = False
    for n, line in enumerate(slines):
        if line.find("'" + name + "'") > -1:
            begline = n - 1
    for n, line in enumerate(slines[begline:]):
        if not line.strip():  # first empty line
            endline = n
            found = True
            break
    assert found
    # split needed because H is defined with comments
    rmt_orig[name] = float(slines[begline + 3].split()[0].strip())

assert rmt_orig['Al'] == 2.2  # 2.2 Bohr default

# test1

# generate species with custom rmt 2.1
rmt = {'Al': 2.1}
label = 'rmt2.1'

atomsrmt = atoms.copy()
os.environ['ELK_SPECIES_PATH'] = ELK_SPECIES_PATH
atomsrmt.calc = ELK(tasks=0, label=label, rmt=rmt)  # minimal calc
atomsrmt.get_potential_energy()
del atomsrmt.calc
del atomsrmt

# hack ELK_SPECIES_PATH to use custom species
os.environ['ELK_SPECIES_PATH'] = os.path.abspath(label) + '/'
# run calculation
calc = ELK(tasks=0, label=label,
           rgkmax=4.0, kpts=tuple(kpts2mp(atoms, 2.0, even=True)))
atoms.set_calculator(calc)
e1 = atoms.get_potential_energy()

# test2

# generate species with custom rmt 2.1
rmt = {'Al': -0.1}
label = 'rmt0.1m'

atomsrmt = atoms.copy()
os.environ['ELK_SPECIES_PATH'] = ELK_SPECIES_PATH
atomsrmt.calc = ELK(tasks=0, label=label, rmt=rmt)  # minimal calc
atomsrmt.get_potential_energy()
del atomsrmt.calc
del atomsrmt

# hack ELK_SPECIES_PATH to use custom species
os.environ['ELK_SPECIES_PATH'] = os.path.abspath(label) + '/'
# run calculation
calc = ELK(tasks=0, label=label,
           rgkmax=4.0, kpts=tuple(kpts2mp(atoms, 2.0, even=True)))
atoms.set_calculator(calc)
e2 = atoms.get_potential_energy()

# restore ELK_SPECIES_PATH
os.environ['ELK_SPECIES_PATH'] = ELK_SPECIES_PATH

assert abs(e1 - e2) < 1.0e-4
