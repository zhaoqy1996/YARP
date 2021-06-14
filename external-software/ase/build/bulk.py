from __future__ import division
from math import sqrt

from ase.atoms import Atoms
from ase.symbols import string2symbols
from ase.data import reference_states, atomic_numbers, chemical_symbols
from ase.utils import plural


def bulk(name, crystalstructure=None, a=None, c=None, covera=None, u=None,
         orthorhombic=False, cubic=False):
    """Creating bulk systems.

    Crystal structure and lattice constant(s) will be guessed if not
    provided.

    name: str
        Chemical symbol or symbols as in 'MgO' or 'NaCl'.
    crystalstructure: str
        Must be one of sc, fcc, bcc, hcp, diamond, zincblende,
        rocksalt, cesiumchloride, fluorite or wurtzite.
    a: float
        Lattice constant.
    c: float
        Lattice constant.
    covera: float
        c/a ratio used for hcp.  Default is ideal ratio: sqrt(8/3).
    u: float
        Internal coordinate for Wurtzite structure.
    orthorhombic: bool
        Construct orthorhombic unit cell instead of primitive cell
        which is the default.
    cubic: bool
        Construct cubic unit cell if possible.
    """

    if covera is not None and c is not None:
        raise ValueError("Don't specify both c and c/a!")

    xref = None
    ref = {}

    if name in chemical_symbols:
        Z = atomic_numbers[name]
        ref = reference_states[Z]
        if ref is not None:
            xref = ref['symmetry']

    structures = {'sc': 1, 'fcc': 1, 'bcc': 1, 'hcp': 1, 'diamond': 1,
                  'zincblende': 2, 'rocksalt':2, 'cesiumchloride':2,
                  'fluorite': 3, 'wurtzite': 2}

    if crystalstructure is None:
        crystalstructure = xref
        if crystalstructure not in structures:
            raise ValueError('No suitable reference data for bulk {}.'
                             '  Reference data: {}'
                             .format(name, ref))

    if crystalstructure not in structures:
        raise ValueError('Unknown structure: {}.'
                         .format(crystalstructure))

    # Check name:
    n = len(string2symbols(name))
    n0 = structures[crystalstructure]
    if n != n0:
        raise ValueError('Please specify {} for {} and not {}'
                         .format(plural(n0, 'atom'), crystalstructure, n))

    if a is None:
        if xref != crystalstructure:
            raise ValueError('You need to specify the lattice constant.')
        try:
            a = ref['a']
        except KeyError:
            raise KeyError('No reference lattice parameter "a" for "{}"'
                           .format(name))

    if crystalstructure in ['hcp', 'wurtzite']:
        cubic = False
        if c is not None:
            covera = c / a
        elif covera is None:
            if xref == crystalstructure:
                covera = ref['c/a']
            else:
                covera = sqrt(8 / 3)

    if orthorhombic and crystalstructure != 'sc':
        return _orthorhombic_bulk(name, crystalstructure, a, covera, u)

    if cubic and crystalstructure in ['bcc', 'cesiumchloride']:
        return _orthorhombic_bulk(name, crystalstructure, a, covera)

    if cubic and crystalstructure != 'sc':
        return _cubic_bulk(name, crystalstructure, a)

    if crystalstructure == 'sc':
        atoms = Atoms(name, cell=(a, a, a), pbc=True)
    elif crystalstructure == 'fcc':
        b = a / 2
        atoms = Atoms(name, cell=[(0, b, b), (b, 0, b), (b, b, 0)], pbc=True)
    elif crystalstructure == 'bcc':
        b = a / 2
        atoms = Atoms(name, cell=[(-b, b, b), (b, -b, b), (b, b, -b)],
                      pbc=True)
    elif crystalstructure == 'hcp':
        atoms = Atoms(2 * name,
                      scaled_positions=[(0, 0, 0),
                                        (1 / 3, 2 / 3, 0.5)],
                      cell=[(a, 0, 0),
                            (-a / 2, a * sqrt(3) / 2, 0),
                            (0, 0, covera * a)],
                      pbc=True)
    elif crystalstructure == 'diamond':
        atoms = bulk(2 * name, 'zincblende', a)
    elif crystalstructure == 'zincblende':
        s1, s2 = string2symbols(name)
        atoms = bulk(s1, 'fcc', a) + bulk(s2, 'fcc', a)
        atoms.positions[1] += a / 4
    elif crystalstructure == 'rocksalt':
        s1, s2 = string2symbols(name)
        atoms = bulk(s1, 'fcc', a) + bulk(s2, 'fcc', a)
        atoms.positions[1, 0] += a / 2
    elif crystalstructure == 'cesiumchloride':
        s1, s2 = string2symbols(name)
        atoms = bulk(s1, 'sc', a) + bulk(s2, 'sc', a)
        atoms.positions[1, :] += a / 2
    elif crystalstructure == 'fluorite':
        s1, s2, s3 = string2symbols(name)
        atoms = bulk(s1, 'fcc', a) + bulk(s2, 'fcc', a) + bulk(s3, 'fcc', a)
        atoms.positions[1, :] += a / 4
        atoms.positions[2, :] += a * 3 / 4
    elif crystalstructure == 'wurtzite':
        u = u or 0.25 + 1 / 3 / covera**2
        atoms = Atoms(2 * name,
                      scaled_positions=[(0, 0, 0),
                                        (1 / 3, 2 / 3, 0.5 - u),
                                        (1 / 3, 2 / 3, 0.5),
                                        (0, 0, 1 - u)],
                      cell=[(a, 0, 0),
                            (-a / 2, a * sqrt(3) / 2, 0),
                            (0, 0, a * covera)],
                      pbc=True)
    else:
        raise ValueError('Unknown crystal structure: ' + crystalstructure)

    return atoms


def _orthorhombic_bulk(name, crystalstructure, a, covera=None, u=None):
    if crystalstructure == 'fcc':
        b = a / sqrt(2)
        atoms = Atoms(2 * name, cell=(b, b, a), pbc=True,
                      scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)])
    elif crystalstructure == 'bcc':
        atoms = Atoms(2 * name, cell=(a, a, a), pbc=True,
                      scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)])
    elif crystalstructure == 'hcp':
        atoms = Atoms(4 * name,
                      cell=(a, a * sqrt(3), covera * a),
                      scaled_positions=[(0, 0, 0),
                                        (0.5, 0.5, 0),
                                        (0.5, 1 / 6, 0.5),
                                        (0, 2 / 3, 0.5)],
                      pbc=True)
    elif crystalstructure == 'diamond':
        atoms = _orthorhombic_bulk(2 * name, 'zincblende', a)
    elif crystalstructure == 'zincblende':
        s1, s2 = string2symbols(name)
        b = a / sqrt(2)
        atoms = Atoms(2 * name, cell=(b, b, a), pbc=True,
                      scaled_positions=[(0, 0, 0), (0.5, 0, 0.25),
                                        (0.5, 0.5, 0.5), (0, 0.5, 0.75)])
    elif crystalstructure == 'rocksalt':
        s1, s2 = string2symbols(name)
        b = a / sqrt(2)
        atoms = Atoms(2 * name, cell=(b, b, a), pbc=True,
                      scaled_positions=[(0, 0, 0), (0.5, 0.5, 0),
                                        (0.5, 0.5, 0.5), (0, 0, 0.5)])
    elif crystalstructure == 'cesiumchloride':
        atoms = Atoms(name, cell=(a, a, a), pbc=True,
                      scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)])
    elif crystalstructure == 'wurtzite':
        u = u or 0.25 + 1 / 3 / covera**2
        atoms = Atoms(4 * name,
                      cell=(a, a * 3**0.5, covera * a),
                      scaled_positions=[(0, 0, 0),
                                        (0, 1 / 3, 0.5 - u),
                                        (0, 1 / 3, 0.5),
                                        (0, 0, 1 - u),
                                        (0.5, 0.5, 0),
                                        (0.5, 5 / 6, 0.5 - u),
                                        (0.5, 5 / 6, 0.5),
                                        (0.5, 0.5, 1 - u)],
                      pbc=True)
    else:
        raise RuntimeError

    return atoms


def _cubic_bulk(name, crystalstructure, a):
    if crystalstructure == 'fcc':
        atoms = Atoms(4 * name, cell=(a, a, a), pbc=True,
                      scaled_positions=[(0, 0, 0), (0, 0.5, 0.5),
                                        (0.5, 0, 0.5), (0.5, 0.5, 0)])
    elif crystalstructure == 'diamond':
        atoms = _cubic_bulk(2 * name, 'zincblende', a)
    elif crystalstructure == 'zincblende':
        atoms = Atoms(4 * name, cell=(a, a, a), pbc=True,
                      scaled_positions=[(0, 0, 0), (0.25, 0.25, 0.25),
                                        (0, 0.5, 0.5), (0.25, 0.75, 0.75),
                                        (0.5, 0, 0.5), (0.75, 0.25, 0.75),
                                        (0.5, 0.5, 0), (0.75, 0.75, 0.25)])
    elif crystalstructure == 'rocksalt':
        atoms = Atoms(4 * name, cell=(a, a, a), pbc=True,
                      scaled_positions=[(0, 0, 0), (0.5, 0, 0),
                                        (0, 0.5, 0.5), (0.5, 0.5, 0.5),
                                        (0.5, 0, 0.5), (0, 0, 0.5),
                                        (0.5, 0.5, 0), (0, 0.5, 0)])
    else:
        raise RuntimeError

    return atoms
