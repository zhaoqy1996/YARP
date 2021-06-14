#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check writing and reading a xtl mustem file."""

from ase import Atoms
from ase.io import read
from ase.test import must_raise

# Reproduce the sto xtl file distributed with muSTEM
atoms = Atoms(['Sr', 'Ti', 'O', 'O', 'O'],
              scaled_positions=[[0, 0, 0],
                                [0.5, 0.5, 0.5],
                                [0.5, 0.5, 0],
                                [0.5, 0, 0.5],
                                [0, 0.5, 0.5]],
              cell=[3.905, 3.905, 3.905],
              pbc=True)

filename = 'sto_mustem.xtl'

with must_raise(TypeError):
    atoms.write(filename)

with must_raise(TypeError):
    atoms.write(filename, keV=300)

with must_raise(TypeError):
    atoms.write(filename,
                DW={'Sr': 0.78700E-02, 'O': 0.92750E-02, 'Ti': 0.55700E-02})

atoms.write(filename, keV=300,
            DW={'Sr': 0.78700E-02, 'O': 0.92750E-02, 'Ti': 0.55700E-02})

atoms2 = read(filename, format='mustem')

tol = 1E-6
assert sum(abs((atoms.positions - atoms2.positions).ravel())) < tol
assert sum(abs((atoms.cell - atoms2.cell).ravel())) < tol

atoms3 = read(filename)
assert sum(abs((atoms.positions - atoms3.positions).ravel())) < tol
assert sum(abs((atoms.cell - atoms3.cell).ravel())) < tol

with must_raise(ValueError):
    # Raise an error if there is a missing key.
    atoms.write(filename, keV=300, DW={'Sr': 0.78700E-02, 'O': 0.92750E-02})

atoms.write(filename, keV=300,
            DW={'Sr': 0.78700E-02, 'O': 0.92750E-02, 'Ti': 0.55700E-02},
            occupancy={'Sr': 1.0, 'O': 0.5, 'Ti': 0.9})

with must_raise(ValueError):
    # Raise an error if there is a missing key.
    atoms.write(filename, keV=300,
                DW={'Sr': 0.78700E-02, 'O': 0.92750E-02, 'Ti': 0.55700E-02},
                occupancy={'O': 0.5, 'Ti': 0.9})

with must_raise(ValueError):
    # Raise an error if the unit cell is not defined.
    atoms4 = Atoms(['Sr', 'Ti', 'O', 'O', 'O'],
                   positions=[[0, 0, 0],
                              [0.5, 0.5, 0.5],
                              [0.5, 0.5, 0],
                              [0.5, 0, 0.5],
                              [0, 0.5, 0.5]])
    atoms4.write(filename, keV=300,
                DW={'Sr': 0.78700E-02, 'O': 0.92750E-02, 'Ti': 0.55700E-02})
