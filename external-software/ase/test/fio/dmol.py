from ase.build import bulk, molecule
from ase.io import read, write
from ase.calculators.dmol import find_transformation
import numpy as np


def check(atoms, ref_atoms, dist_tol=1e-6):

    # check pbc conditions
    assert all(atoms.pbc == ref_atoms.pbc), (atoms.pbc, ref_atoms.pbc)

    # check cell
    if all(atoms.pbc):
        assert abs(atoms.cell - ref_atoms.cell).max() < dist_tol, \
            (atoms.cell - ref_atoms.cell)

    # check positions
    assert abs(atoms.positions - ref_atoms.positions).max() < dist_tol, \
        (atoms.positions - ref_atoms.positions)

    # check symbols
    assert atoms.get_chemical_symbols() == ref_atoms.get_chemical_symbols()


ref_molecule = molecule('H2O')
ref_bulk = bulk('Si', 'diamond')
ref_molecule_images = [ref_molecule, ref_molecule]
ref_bulk_images = [ref_bulk, ref_bulk]


# .car format
fname = 'dmol_tmp.car'
write(fname, ref_molecule, format='dmol-car')
for atoms in [read(fname, format='dmol-car'), read(fname)]:
    check(atoms, ref_molecule)

fname = 'dmol_tmp.car'
write(fname, ref_bulk, format='dmol-car')
for atoms in [read(fname, format='dmol-car'), read(fname)]:
    R, _ = find_transformation(atoms, ref_bulk)
    atoms.cell = np.dot(atoms.cell, R)
    atoms.positions = np.dot(atoms.positions, R)
    check(atoms, ref_bulk)


# .incoor format
fname = 'dmol_tmp.incoor'
write(fname, ref_bulk, format='dmol-incoor')
atoms = read(fname, format='dmol-incoor')
check(atoms, ref_bulk)


# .arc format
fname = 'dmol_tmp.arc'
write(fname, ref_molecule_images, format='dmol-arc')
images = read(fname + '@:', format='dmol-arc')
for image, ref_image in zip(images, ref_molecule_images):
    check(image, ref_image)

fname = 'dmol_tmp.arc'
write(fname, ref_bulk_images, format='dmol-arc')
images = read(fname + '@:', format='dmol-arc')
for image, ref_image in zip(images, ref_bulk_images):
    R, _ = find_transformation(image, ref_image)
    image.cell = np.dot(image.cell, R)
    image.positions = np.dot(image.positions, R)
    check(image, ref_image)
