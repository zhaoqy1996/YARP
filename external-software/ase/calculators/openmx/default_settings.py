"""
The ASE Calculator for OpenMX <http://www.openmx-square.org>: Python interface
to the software package for nano-scale material simulations based on density
functional theories.
    Copyright (C) 2017 Charles Thomas Johnson, JaeHwan Shim and JaeJun Yu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 2.1 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with ASE.  If not, see <http://www.gnu.org/licenses/>.
"""

# Dictionary containing information about the LCAO datasets
# The keys are the chemical symbol of the atom. The value of 'orbitals used'
# is list of integers which correspond to the number of 's', 'p', 'd' and 'f'
# orbitals used respectively.
# Information was brought from the http://www.jaist.ac.jp/~t-ozaki/vps_pao2013/

default_dictionary = {
    'H': {
           'cutoff radius': 6,
           'orbitals used': [3, 2],
           'pseudo-potential suffix': ''
          },
    'He': {
        'cutoff radius': 8,
        'orbitals used': [2, 2, 1],
        'pseudo-potential suffix': ''
    },
    'Li': {
        'cutoff radius': 10,
        'orbitals used': [3, 3, 2],
        'pseudo-potential suffix': ''
    },
    'Be': {
        'cutoff radius': 8,
        'orbitals used': [3, 2],
        'pseudo-potential suffix': ''
    },
    'B': {
        'cutoff radius': 8,
        'orbitals used': [2, 2, 1],
        'pseudo-potential suffix': ''
    },
    'C': {
        'cutoff radius': 6,
        'orbitals used': [2, 2, 1],
        'pseudo-potential suffix': ''
    },
    'N': {
        'cutoff radius': 6,
        'orbitals used': [3, 3, 2, 1],
        'pseudo-potential suffix': ''
    },
    'O': {
        'cutoff radius': 6,
        'orbitals used': [3, 3, 2],
        'pseudo-potential suffix': ''
    },
    'F': {
        'cutoff radius': 6,
        'orbitals used': [2, 2, 1],
        'pseudo-potential suffix': ''
    },
    'Ne': {
        'cutoff radius': 9,
        'orbitals used': [3, 2, 2],
        'pseudo-potential suffix': ''
    },
    'Na': {
        'cutoff radius': 11,
        'orbitals used': [3, 3, 2],
        'pseudo-potential suffix': ''
    },
    'Mg': {
        'cutoff radius': 9,
        'orbitals used': [3, 3, 2],
        'pseudo-potential suffix': ''
    },
    'Al': {
        'cutoff radius': 8,
        'orbitals used': [4, 4, 2],
        'pseudo-potential suffix': ''
    },
    'Si': {
        'cutoff radius': 8,
        'orbitals used': [2, 2, 1],
        'pseudo-potential suffix': ''
    },
    'P': {
        'cutoff radius': 8,
        'orbitals used': [4, 3, 3, 2],
        'pseudo-potential suffix': ''
    },
    'S': {
        'cutoff radius': 8,
        'orbitals used': [4, 3, 3, 2],
        'pseudo-potential suffix': ''
    },
    'Cl': {
        'cutoff radius': 8,
        'orbitals used': [2, 2, 1],
        'pseudo-potential suffix': ''
    },
    'Ar': {
        'cutoff radius': 9,
        'orbitals used': [3, 2, 2, 1],
        'pseudo-potential suffix': ''
    },
    'K': {
        'cutoff radius': 12,
        'orbitals used': [4, 3, 3, 1],
        'pseudo-potential suffix': ''
    },
    'Ca': {
        'cutoff radius': 11,
        'orbitals used': [4, 3, 2],
        'pseudo-potential suffix': ''
    },
    'Sc': {
        'cutoff radius': 9,
        'orbitals used': [4, 3, 2],
        'pseudo-potential suffix': ''
    },
    'Ti': {
        'cutoff radius': 9,
        'orbitals used': [3, 3, 3, 1],
        'pseudo-potential suffix': ''
    },
    'V': {
        'cutoff radius': 8,
        'orbitals used': [3, 3, 3, 1],
        'pseudo-potential suffix': ''
    },
    'Cr': {
        'cutoff radius': 8,
        'orbitals used': [3, 3, 2],
        'pseudo-potential suffix': ''
    },
    'Mn': {
        'cutoff radius': 8,
        'orbitals used': [3, 3, 3, 1],
        'pseudo-potential suffix': ''
    },
    'Fe': {
        'cutoff radius': 8,
        'orbitals used': [3, 3, 2],
        'pseudo-potential suffix': 'S'
    },
    'Co': {
        'cutoff radius': 8,
        'orbitals used': [3, 4, 3, 2],
        'pseudo-potential suffix': 'S'
    },
    'Ni': {
        'cutoff radius': 8,
        'orbitals used': [4, 4, 3, 2],
        'pseudo-potential suffix': 'S'
    },
    'Cu': {
        'cutoff radius': 8,
        'orbitals used': [2, 2, 2],
        'pseudo-potential suffix': 'S'
    },
    'Zn': {
        'cutoff radius': 6,
        'orbitals used': [3, 2],
        'pseudo-potential suffix': ''
    },
    'Ga': {
        'cutoff radius': 6,
        'orbitals used': [3, 2],
        'pseudo-potential suffix': ''
    },
    'Ge': {
        'cutoff radius': 6,
        'orbitals used': [3, 2],
        'pseudo-potential suffix': ''
    },
    'As': {
        'cutoff radius': 9,
        'orbitals used': [3, 3, 3, 2],
        'pseudo-potential suffix': ''
    },
    'I': {
        'cutoff radius': 9,
        'orbitals used': [3, 3, 2, 1],
        'pseudo-potential suffix': ''
    }

}

default_kpath = [
    {
        'kpts': 20,
        'start_point': (0., 0., 0.),
        'end_point': (1., 0., 0.),
        'path_symbols': ('g', 'X')
    },
    {
        'kpts': 20,
        'start_point': (1., 0., 0.),
        'end_point': (1., 0.5, 0.),
        'path_symbols': ('X', 'W')
    },
    {
        'kpts': 20,
        'start_point': (1., 0.5, 0.),
        'end_point': (0.5, 0.5, 0.),
        'path_symbols': ('W', 'L')
    },
    {
        'kpts': 20,
        'start_point': (0.5, 0.5, 0.),
        'end_point': (0., 0., 0.),
        'path_symbols': ('L', 'g')
    },
    {
        'kpts': 20,
        'start_point': (0., 0., 0.),
        'end_point': (1., 1., 0.),
        'path_symbols': ('g', 'X')
    },
]
