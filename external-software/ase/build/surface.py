"""Helper functions for creating the most common surfaces and related tasks.

The helper functions can create the most common low-index surfaces,
add vacuum layers and add adsorbates.

"""

from __future__ import division
from math import sqrt
from operator import itemgetter

import numpy as np

from ase.atom import Atom
from ase.atoms import Atoms
from ase.data import reference_states, atomic_numbers
from ase.lattice.cubic import FaceCenteredCubic
from ase.utils import basestring


def fcc100(symbol, size, a=None, vacuum=None, orthogonal=True,
           periodic=False):
    """FCC(100) surface.

    Supported special adsorption sites: 'ontop', 'bridge', 'hollow'."""
    if not orthogonal:
        raise NotImplementedError("Can't do non-orthogonal cell yet!")

    return _surface(symbol, 'fcc', '100', size, a, None, vacuum,
                    periodic=periodic,
                    orthogonal=orthogonal)


def fcc110(symbol, size, a=None, vacuum=None, orthogonal=True,
           periodic=False):
    """FCC(110) surface.

    Supported special adsorption sites: 'ontop', 'longbridge',
    'shortbridge', 'hollow'."""
    if not orthogonal:
        raise NotImplementedError("Can't do non-orthogonal cell yet!")

    return _surface(symbol, 'fcc', '110', size, a, None, vacuum,
                    periodic=periodic,
                    orthogonal=orthogonal)



def bcc100(symbol, size, a=None, vacuum=None, orthogonal=True,
           periodic=False):
    """BCC(100) surface.

    Supported special adsorption sites: 'ontop', 'bridge', 'hollow'."""
    if not orthogonal:
        raise NotImplementedError("Can't do non-orthogonal cell yet!")

    return _surface(symbol, 'bcc', '100', size, a, None, vacuum,
                    periodic=periodic,
                    orthogonal=orthogonal)


def bcc110(symbol, size, a=None, vacuum=None, orthogonal=False,
           periodic=False):
    """BCC(110) surface.

    Supported special adsorption sites: 'ontop', 'longbridge',
    'shortbridge', 'hollow'.

    Use *orthogonal=True* to get an orthogonal unit cell - works only
    for size=(i,j,k) with j even."""
    return _surface(symbol, 'bcc', '110', size, a, None, vacuum,
                    periodic=periodic,
                    orthogonal=orthogonal)


def bcc111(symbol, size, a=None, vacuum=None, orthogonal=False,
           periodic=False):
    """BCC(111) surface.

    Supported special adsorption sites: 'ontop'.

    Use *orthogonal=True* to get an orthogonal unit cell - works only
    for size=(i,j,k) with j even."""
    return _surface(symbol, 'bcc', '111', size, a, None, vacuum,
                    periodic=periodic,
                    orthogonal=orthogonal)


def fcc111(symbol, size, a=None, vacuum=None, orthogonal=False,
           periodic=False):
    """FCC(111) surface.

    Supported special adsorption sites: 'ontop', 'bridge', 'fcc' and 'hcp'.

    Use *orthogonal=True* to get an orthogonal unit cell - works only
    for size=(i,j,k) with j even."""
    return _surface(symbol, 'fcc', '111', size, a, None, vacuum,
                    periodic=periodic,
                    orthogonal=orthogonal)


def hcp0001(symbol, size, a=None, c=None, vacuum=None, orthogonal=False,
            periodic=False):
    """HCP(0001) surface.

    Supported special adsorption sites: 'ontop', 'bridge', 'fcc' and 'hcp'.

    Use *orthogonal=True* to get an orthogonal unit cell - works only
    for size=(i,j,k) with j even."""
    return _surface(symbol, 'hcp', '0001', size, a, c, vacuum,
                    periodic=periodic,
                    orthogonal=orthogonal)


def hcp10m10(symbol, size, a=None, c=None, vacuum=None, orthogonal=True,
             periodic=False):
    """HCP(10m10) surface.

    Supported special adsorption sites: 'ontop'.

    Works only for size=(i,j,k) with j even."""
    if not orthogonal:
        raise NotImplementedError("Can't do non-orthogonal cell yet!")

    return _surface(symbol, 'hcp', '10m10', size, a, c, vacuum,
                    periodic=periodic,
                    orthogonal=orthogonal)


def diamond100(symbol, size, a=None, vacuum=None, orthogonal=True,
               periodic=False):
    """DIAMOND(100) surface.

    Supported special adsorption sites: 'ontop'."""
    if not orthogonal:
        raise NotImplementedError("Can't do non-orthogonal cell yet!")

    return _surface(symbol, 'diamond', '100', size, a, None, vacuum,
                    periodic=periodic,
                    orthogonal=orthogonal)


def diamond111(symbol, size, a=None, vacuum=None, orthogonal=False,
               periodic=False):
    """DIAMOND(111) surface.

    Supported special adsorption sites: 'ontop'."""

    if orthogonal:
        raise NotImplementedError("Can't do orthogonal cell yet!")
    return _surface(symbol, 'diamond', '111', size, a, None, vacuum,
                    periodic=periodic,
                    orthogonal=orthogonal)


def add_adsorbate(slab, adsorbate, height, position=(0, 0), offset=None,
                  mol_index=0):
    """Add an adsorbate to a surface.

    This function adds an adsorbate to a slab.  If the slab is
    produced by one of the utility functions in ase.build, it
    is possible to specify the position of the adsorbate by a keyword
    (the supported keywords depend on which function was used to
    create the slab).

    If the adsorbate is a molecule, the atom indexed by the mol_index
    optional argument is positioned on top of the adsorption position
    on the surface, and it is the responsibility of the user to orient
    the adsorbate in a sensible way.

    This function can be called multiple times to add more than one
    adsorbate.

    Parameters:

    slab: The surface onto which the adsorbate should be added.

    adsorbate:  The adsorbate. Must be one of the following three types:
        A string containing the chemical symbol for a single atom.
        An atom object.
        An atoms object (for a molecular adsorbate).

    height: Height above the surface.

    position: The x-y position of the adsorbate, either as a tuple of
        two numbers or as a keyword (if the surface is produced by one
        of the functions in ase.build).

    offset (default: None): Offsets the adsorbate by a number of unit
        cells. Mostly useful when adding more than one adsorbate.

    mol_index (default: 0): If the adsorbate is a molecule, index of
        the atom to be positioned above the location specified by the
        position argument.

    Note *position* is given in absolute xy coordinates (or as
    a keyword), whereas offset is specified in unit cells.  This
    can be used to give the positions in units of the unit cell by
    using *offset* instead.

    """
    info = slab.info.get('adsorbate_info', {})

    pos = np.array([0.0, 0.0])  # (x, y) part
    spos = np.array([0.0, 0.0])  # part relative to unit cell
    if offset is not None:
        spos += np.asarray(offset, float)

    if isinstance(position, basestring):
        # A site-name:
        if 'sites' not in info:
            raise TypeError('If the atoms are not made by an ' +
                            'ase.build function, ' +
                            'position cannot be a name.')
        if position not in info['sites']:
            raise TypeError('Adsorption site %s not supported.' % position)
        spos += info['sites'][position]
    else:
        pos += position

    if 'cell' in info:
        cell = info['cell']
    else:
        cell = slab.get_cell()[:2, :2]

    pos += np.dot(spos, cell)

    # Convert the adsorbate to an Atoms object
    if isinstance(adsorbate, Atoms):
        ads = adsorbate
    elif isinstance(adsorbate, Atom):
        ads = Atoms([adsorbate])
    else:
        # Assume it is a string representing a single Atom
        ads = Atoms([Atom(adsorbate)])

    # Get the z-coordinate:
    if 'top layer atom index' in info:
        a = info['top layer atom index']
    else:
        a = slab.positions[:, 2].argmax()
        if 'adsorbate_info' not in slab.info:
            slab.info['adsorbate_info'] = {}
        slab.info['adsorbate_info']['top layer atom index'] = a
    z = slab.positions[a, 2] + height

    # Move adsorbate into position
    ads.translate([pos[0], pos[1], z] - ads.positions[mol_index])

    # Attach the adsorbate
    slab.extend(ads)


def add_vacuum(atoms, vacuum):
    """Add vacuum layer to the atoms.

    Parameters:

    atoms: Atoms object
        Most likely created by one of the surface functions.
    vacuum: float
        The thickness of the vacuum layer (in Angstrom).
    """
    uc = atoms.get_cell()
    normal = np.cross(uc[0], uc[1])
    costheta = np.dot(normal, uc[2]) / np.sqrt(np.dot(normal, normal) *
                                               np.dot(uc[2], uc[2]))
    length = np.sqrt(np.dot(uc[2], uc[2]))
    newlength = length + vacuum / costheta
    uc[2] *= newlength / length
    atoms.set_cell(uc)


def _surface(symbol, structure, face, size, a, c, vacuum, periodic,
             orthogonal=True):
    """Function to build often used surfaces.

    Don't call this function directly - use fcc100, fcc110, bcc111, ..."""

    Z = atomic_numbers[symbol]

    if a is None:
        sym = reference_states[Z]['symmetry']
        if sym != structure:
            raise ValueError("Can't guess lattice constant for %s-%s!" %
                             (structure, symbol))
        a = reference_states[Z]['a']

    if structure == 'hcp' and c is None:
        if reference_states[Z]['symmetry'] == 'hcp':
            c = reference_states[Z]['c/a'] * a
        else:
            c = sqrt(8 / 3.0) * a

    positions = np.empty((size[2], size[1], size[0], 3))
    positions[..., 0] = np.arange(size[0]).reshape((1, 1, -1))
    positions[..., 1] = np.arange(size[1]).reshape((1, -1, 1))
    positions[..., 2] = np.arange(size[2]).reshape((-1, 1, 1))

    numbers = np.ones(size[0] * size[1] * size[2], int) * Z

    tags = np.empty((size[2], size[1], size[0]), int)
    tags[:] = np.arange(size[2], 0, -1).reshape((-1, 1, 1))

    slab = Atoms(numbers,
                 tags=tags.ravel(),
                 pbc=(True, True, periodic),
                 cell=size)

    surface_cell = None
    sites = {'ontop': (0, 0)}
    surf = structure + face
    if surf == 'fcc100':
        cell = (sqrt(0.5), sqrt(0.5), 0.5)
        positions[-2::-2, ..., :2] += 0.5
        sites.update({'hollow': (0.5, 0.5), 'bridge': (0.5, 0)})
    elif surf == 'diamond100':
        cell = (sqrt(0.5), sqrt(0.5), 0.5 / 2)
        positions[-4::-4, ..., :2] += (0.5, 0.5)
        positions[-3::-4, ..., :2] += (0.0, 0.5)
        positions[-2::-4, ..., :2] += (0.0, 0.0)
        positions[-1::-4, ..., :2] += (0.5, 0.0)
    elif surf == 'fcc110':
        cell = (1.0, sqrt(0.5), sqrt(0.125))
        positions[-2::-2, ..., :2] += 0.5
        sites.update({'hollow': (0.5, 0.5), 'longbridge': (0.5, 0),
                      'shortbridge': (0, 0.5)})
    elif surf == 'bcc100':
        cell = (1.0, 1.0, 0.5)
        positions[-2::-2, ..., :2] += 0.5
        sites.update({'hollow': (0.5, 0.5), 'bridge': (0.5, 0)})
    else:
        if orthogonal and size[1] % 2 == 1:
            raise ValueError(("Can't make orthorhombic cell with size=%r.  " %
                              (tuple(size),)) +
                             'Second number in size must be even.')
        if surf == 'fcc111':
            cell = (sqrt(0.5), sqrt(0.375), 1 / sqrt(3))
            if orthogonal:
                positions[-1::-3, 1::2, :, 0] += 0.5
                positions[-2::-3, 1::2, :, 0] += 0.5
                positions[-3::-3, 1::2, :, 0] -= 0.5
                positions[-2::-3, ..., :2] += (0.0, 2.0 / 3)
                positions[-3::-3, ..., :2] += (0.5, 1.0 / 3)
            else:
                positions[-2::-3, ..., :2] += (-1.0 / 3, 2.0 / 3)
                positions[-3::-3, ..., :2] += (1.0 / 3, 1.0 / 3)
            sites.update({'bridge': (0.5, 0), 'fcc': (1.0 / 3, 1.0 / 3),
                          'hcp': (2.0 / 3, 2.0 / 3)})
        elif surf == 'diamond111':
            cell = (sqrt(0.5), sqrt(0.375), 1 / sqrt(3) / 2)
            assert not orthogonal
            positions[-1::-6, ..., :3] += (0.0, 0.0, 0.5)
            positions[-2::-6, ..., :2] += (0.0, 0.0)
            positions[-3::-6, ..., :3] += (-1.0 / 3, 2.0 / 3, 0.5)
            positions[-4::-6, ..., :2] += (-1.0 / 3, 2.0 / 3)
            positions[-5::-6, ..., :3] += (1.0 / 3, 1.0 / 3, 0.5)
            positions[-6::-6, ..., :2] += (1.0 / 3, 1.0 / 3)
        elif surf == 'hcp0001':
            cell = (1.0, sqrt(0.75), 0.5 * c / a)
            if orthogonal:
                positions[:, 1::2, :, 0] += 0.5
                positions[-2::-2, ..., :2] += (0.0, 2.0 / 3)
            else:
                positions[-2::-2, ..., :2] += (-1.0 / 3, 2.0 / 3)
            sites.update({'bridge': (0.5, 0), 'fcc': (1.0 / 3, 1.0 / 3),
                          'hcp': (2.0 / 3, 2.0 / 3)})
        elif surf == 'hcp10m10':
            cell = (1.0, 0.5 * c / a, sqrt(0.75))
            assert orthogonal
            positions[-2::-2, ..., 0] += 0.5
            positions[:, ::2, :, 2] += 2.0 / 3
        elif surf == 'bcc110':
            cell = (1.0, sqrt(0.5), sqrt(0.5))
            if orthogonal:
                positions[:, 1::2, :, 0] += 0.5
                positions[-2::-2, ..., :2] += (0.0, 1.0)
            else:
                positions[-2::-2, ..., :2] += (-0.5, 1.0)
            sites.update({'shortbridge': (0, 0.5),
                          'longbridge': (0.5, 0),
                          'hollow': (0.375, 0.25)})
        elif surf == 'bcc111':
            cell = (sqrt(2), sqrt(1.5), sqrt(3) / 6)
            if orthogonal:
                positions[-1::-3, 1::2, :, 0] += 0.5
                positions[-2::-3, 1::2, :, 0] += 0.5
                positions[-3::-3, 1::2, :, 0] -= 0.5
                positions[-2::-3, ..., :2] += (0.0, 2.0 / 3)
                positions[-3::-3, ..., :2] += (0.5, 1.0 / 3)
            else:
                positions[-2::-3, ..., :2] += (-1.0 / 3, 2.0 / 3)
                positions[-3::-3, ..., :2] += (1.0 / 3, 1.0 / 3)
            sites.update({'hollow': (1.0 / 3, 1.0 / 3)})
        else:
            2 / 0

        surface_cell = a * np.array([(cell[0], 0),
                                     (cell[0] / 2, cell[1])])
        if not orthogonal:
            cell = np.array([(cell[0], 0, 0),
                             (cell[0] / 2, cell[1], 0),
                             (0, 0, cell[2])])

    if surface_cell is None:
        surface_cell = a * np.diag(cell[:2])

    if isinstance(cell, tuple):
        cell = np.diag(cell)

    slab.set_positions(positions.reshape((-1, 3)))
    slab.set_cell([a * v * n for v, n in zip(cell, size)], scale_atoms=True)

    if not periodic:
        slab.cell[2] = 0.0

    if vacuum is not None:
        slab.center(vacuum, axis=2)

    if 'adsorbate_info' not in slab.info:
        slab.info.update({'adsorbate_info': {}})

    slab.info['adsorbate_info']['cell'] = surface_cell
    slab.info['adsorbate_info']['sites'] = sites
    return slab


def fcc211(symbol, size, a=None, vacuum=None, orthogonal=True):
    """FCC(211) surface.

    Does not currently support special adsorption sites.

    Currently only implemented for *orthogonal=True* with size specified
    as (i, j, k), where i, j, and k are number of atoms in each direction.
    i must be divisible by 3 to accommodate the step width.
    """
    if not orthogonal:
        raise NotImplementedError('Only implemented for orthogonal '
                                  'unit cells.')
    if size[0] % 3 != 0:
        raise NotImplementedError('First dimension of size must be '
                                  'divisible by 3.')
    atoms = FaceCenteredCubic(symbol,
                              directions=[[1, -1, -1],
                                          [0, 2, -2],
                                          [2, 1, 1]],
                              miller=(None, None, (2, 1, 1)),
                              latticeconstant=a,
                              size=(1, 1, 1),
                              pbc=True)
    z = (size[2] + 1) // 2
    atoms = atoms.repeat((size[0] // 3, size[1], z))
    if size[2] % 2:  # Odd: remove bottom layer and shrink cell.
        remove_list = [atom.index for atom in atoms
                       if atom.z < atoms[1].z]
        del atoms[remove_list]
        dz = atoms[0].z
        atoms.translate((0., 0., -dz))
        atoms.cell[2][2] -= dz

    atoms.cell[2] = 0.0
    atoms.pbc[2] = False
    if vacuum:
        atoms.center(vacuum, axis=2)

    # Renumber systematically from top down.
    orders = [(atom.index, round(atom.x, 3), round(atom.y, 3),
               -round(atom.z, 3), atom.index) for atom in atoms]
    orders.sort(key=itemgetter(3, 1, 2))
    newatoms = atoms.copy()
    for index, order in enumerate(orders):
        newatoms[index].position = atoms[order[0]].position.copy()

    # Add empty 'sites' dictionary for consistency with other functions
    newatoms.info['adsorbate_info'] = {'sites': {}}
    return newatoms


def mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19,
        size=(1, 1, 1), vacuum=None):
    """Create three-layer 2D materials with hexagonal structure.

    For metal dichalcogenites, etc.

    The kind argument accepts '2H', which gives a mirror plane symmetry
    and '1T', which gives an inversion symmetry."""

    if kind == '2H':
        basis = [(0, 0, 0),
                 (2 / 3, 1 / 3, 0.5 * thickness),
                 (2 / 3, 1 / 3, -0.5 * thickness)]
    elif kind == '1T':
        basis = [(0, 0, 0),
                 (2 / 3, 1 / 3, 0.5 * thickness),
                 (1 / 3, 2 / 3, -0.5 * thickness)]
    else:
        raise ValueError('Structure not recognized:', kind)

    cell = [[a, 0, 0], [-a / 2, a * 3**0.5 / 2, 0], [0, 0, 0]]

    atoms = Atoms(formula, cell=cell, pbc=(1, 1, 0))
    atoms.set_scaled_positions(basis)
    if vacuum is not None:
        atoms.center(vacuum, axis=2)
    atoms = atoms.repeat(size)

    return atoms


def _all_surface_functions():
    # Convenient for debugging.
    d = {}
    for func in [fcc100, fcc110, bcc100, bcc110, bcc111, fcc111, hcp0001,
                 hcp10m10, diamond100, diamond111, fcc111, mx2]:
        d[func.__name__] = func
    return d
