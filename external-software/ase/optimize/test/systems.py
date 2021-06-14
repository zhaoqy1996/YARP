from math import sin, cos, pi

from ase import Atoms
from ase.build import fcc111, fcc100, add_adsorbate
from ase.db import connect
from ase.constraints import FixAtoms
from ase.lattice.cubic import FaceCenteredCubic

systems = []

cell = (5, 5, 5)
atoms = Atoms('H2', [(0, 0, 0), (0, 0, 1.4)], cell=cell)
atoms.center()
systems.append((atoms, 'Hydrogen molecule'))

#
atoms = FaceCenteredCubic(
    directions=[[1, -1, 0], [1, 1, 0], [0, 0, 1]],
    size=(2, 2, 2),
    symbol='Cu',
    pbc=(1, 1, 1))
atoms.rattle(stdev=0.1, seed=42)
systems.append((atoms, 'Shaken bulk copper'))

#
a = 2.70
c = 1.59 * a

slab = Atoms('2Cu', [(0., 0., 0.), (1 / 3., 1 / 3., -0.5 * c)],
             tags=(0, 1),
             pbc=(1, 1, 0))
slab.set_cell([(a, 0, 0),
               (a / 2, 3**0.5 * a / 2, 0),
               (0, 0, 1)])
slab.center(vacuum=3, axis=2)
mask = [a.tag == 1 for a in slab]
slab.set_constraint(FixAtoms(mask=mask))
systems.append((slab, 'Distorted Cu(111) surface'))

#
zpos = cos(134.3 / 2.0 * pi / 180.0) * 1.197
xpos = sin(134.3 / 2.0 * pi / 180.0) * 1.19
co = Atoms('CO', positions=[(-xpos + 1.2, 0, -zpos),
                            (-xpos + 1.2, -1.1, -zpos)])
slab = fcc111('Au', size=(2, 2, 2), orthogonal=True)
add_adsorbate(slab, co, 1.5, 'bridge')
slab.center(vacuum=6, axis=2)
slab.set_pbc((True, True, False))
constraint = FixAtoms(mask=[a.tag == 2 for a in slab])
slab.set_constraint(constraint)
systems.append((slab, 'CO on Au(111) surface'))

#
atoms = Atoms(symbols='C5H12',
              cell=[16.83752497, 12.18645905, 11.83462179],
              positions=[[5.90380523, 5.65545388, 5.91569796],
                         [7.15617518, 6.52907738, 5.91569796],
                         [8.41815022, 5.66384716, 5.92196554],
                         [9.68108996, 6.52891016, 5.91022362],
                         [10.93006206, 5.65545388, 5.91569796],
                         [5.00000011, 6.30002353, 5.9163716],
                         [5.88571848, 5.0122839, 6.82246859],
                         [5.88625613, 5.01308931, 5.01214155],
                         [7.14329342, 7.18115393, 6.81640316],
                         [7.14551332, 7.17200869, 5.00879027],
                         [8.41609966, 5.00661165, 5.02355167],
                         [8.41971183, 5.0251482, 6.83462168],
                         [9.69568096, 7.18645894, 6.8078633],
                         [9.68914668, 7.16663649, 5.00000011],
                         [10.95518898, 5.02163182, 6.8289018],
                         [11.83752486, 6.29836826, 5.90274952],
                         [10.94464142, 5.00000011, 5.01802495]])
systems.append((atoms, 'Pentane molecule'))

#
slab = fcc100('Cu', size=(2, 2, 2), vacuum=3.5)
add_adsorbate(slab, 'C', 1.5, 'hollow')
mask = [a.tag > 1 for a in slab]
constraint = FixAtoms(mask=mask)
slab.set_constraint(constraint)
systems.append((slab, 'C/Cu(100)'))


def create_database():
    db = connect('systems.db', append=False)
    for atoms, description in systems:
        name = atoms.get_chemical_formula()
        db.write(atoms, description=description, name=name)

    if False:
        for atoms, description in systems:
            for seed in range(5):
                a = atoms.copy()
                a.rattle(0.1, seed=seed)
                name = a.get_chemical_formula() + '-' + str(seed)
                db.write(a, description=description, seed=seed, name=name)


if __name__ == '__main__':
    create_database()
