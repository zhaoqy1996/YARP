from math import atan2, ceil, cos, sin, log10

import numpy as np

from ase.build import hcp0001, fcc111, bcc111


def hcp0001_root(symbol, root, size, a=None, c=None,
                 vacuum=None, orthogonal=False):
    """HCP(0001) surface maniupulated to have a x unit side length
    of *root* before repeating.  This also results in *root* number
    of repetitions of the cell.


    The first 20 valid roots for nonorthogonal are...
    1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 25,
    27, 28, 31, 36, 37, 39, 43, 48, 49"""
    atoms = hcp0001(symbol=symbol, size=(1, 1, size[2]),
                    a=a, c=c, vacuum=vacuum, orthogonal=orthogonal)
    atoms = root_surface(atoms, root)
    atoms *= (size[0], size[1], 1)
    return atoms


def fcc111_root(symbol, root, size, a=None,
                vacuum=None, orthogonal=False):
    """FCC(111) surface maniupulated to have a x unit side length
    of *root* before repeating. This also results in *root* number
    of repetitions of the cell.

    The first 20 valid roots for nonorthogonal are...
    1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 25, 27,
    28, 31, 36, 37, 39, 43, 48, 49"""
    atoms = fcc111(symbol=symbol, size=(1, 1, size[2]),
                   a=a, vacuum=vacuum, orthogonal=orthogonal)
    atoms = root_surface(atoms, root)
    atoms *= (size[0], size[1], 1)
    return atoms


def bcc111_root(symbol, root, size, a=None,
                vacuum=None, orthogonal=False):
    """BCC(111) surface maniupulated to have a x unit side length
    of *root* before repeating. This also results in *root* number
    of repetitions of the cell.


    The first 20 valid roots for nonorthogonal are...
    1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 25,
    27, 28, 31, 36, 37, 39, 43, 48, 49"""
    atoms = bcc111(symbol=symbol, size=(1, 1, size[2]),
                   a=a, vacuum=vacuum, orthogonal=orthogonal)
    atoms = root_surface(atoms, root)
    atoms *= (size[0], size[1], 1)
    return atoms


def root_surface(primitive_slab, root, swap_alpha=False, eps=1e-8):
    """Creates a cell from a primitive cell that repeats along the x and y
    axis in a way consisent with the primitive cell, that has been cut
    to have a side length of *root*.

    *primitive cell* should be a primitive 2d cell of your slab, repeated
    as needed in the z direction.

    *root* should be determined using an analysis tool such as the
    root_surface_analysis function, or prior knowledge. It should always
    be a whole number as it represents the number of repetitions.

    *swap_alpha* swaps the alpha angle of the cell."""

    logeps = int(-log10(eps))
    atoms = primitive_slab.copy()

    xscale = np.linalg.norm(atoms.cell[0][0:2])
    xx, xy = atoms.cell[0][0:2] / xscale
    yx, yy = atoms.cell[1][0:2] / xscale
    cell_vectors = [[xx, xy], [yx, yy]]

    # Make (0, 0) corner's angle flip from acute to obtuse or
    # obtuse to acute with a small trick
    if swap_alpha:
        cell_vectors[1][0] *= -1

    # Manipulate the cell vectors to find the best search zone and
    # cast to numpy array.
    cell_vectors = np.array(cell_vectors)
    cell_vectors_mag = [np.linalg.norm(x) for x in cell_vectors]
    cell_search = [int(ceil(float(root * 1.2) / float(x)))
                   for x in cell_vectors_mag]

    # Make these variables in function scope
    # x,  y  = Raw grid point
    # tx, ty = Transformed grid point
    x, y, tx, ty = 0, 0, 0, 0

    # Calculate square distances and break when appropriate
    for x in range(cell_search[0]):
        for y in range(cell_search[1]):
            if x == 0 or y == 0:
                continue
            vect = (cell_vectors[0] * x) + (cell_vectors[1] * y)
            dist = round((vect ** 2).sum(), logeps)
            if dist == root:
                tx, ty = vect
                break
        else:
            continue
        break
    else:
        # A root cell could not be found for this combination
        raise RuntimeError("Can't find a root cell of {0} in [{1}, {2}]".
                           format(root, cell_vectors[0], cell_vectors[1]))

    tmag = np.linalg.norm((tx, ty))
    root_angle = -atan2(ty, tx)
    cell_scale = tmag / cell_vectors_mag[0]

    root_rotation = [[cos(root_angle), -sin(root_angle)],
                     [sin(root_angle), cos(root_angle)]]
    cell = [np.dot(x, root_rotation) * cell_scale for x in cell_vectors]

    def pretrim(atoms):
        cell = atoms.cell
        pos = atoms.positions

        vertices = np.array([[0, 0],
                             cell[0][0:2],
                             cell[1][0:2],
                             cell[0][0:2] + cell[1][0:2]])

        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)

        out = np.where(np.logical_not((pos[:, 0] >= mins[0] - eps * 10) &
                                      (pos[:, 0] <= maxs[0] + eps * 10) &
                                      (pos[:, 1] >= mins[1] - eps * 10) &
                                      (pos[:, 1] <= maxs[1] + eps * 10)))

        del atoms[out]

    def remove_doubles(atoms, shift=True):
        shift_vector = np.array([eps * 100, eps * 200, eps * 300])
        if shift:
            atoms.translate(shift_vector)
        atoms.set_scaled_positions(atoms.get_scaled_positions())
        valid = [0]
        for x in range(len(atoms)):
            for ypos, y in enumerate(valid):
                xa = atoms[x].position
                ya = atoms[y].position
                if np.linalg.norm(xa - ya) < eps:
                    break
            else:
                valid.append(x)
        del atoms[[i for i in range(len(atoms)) if i not in valid]]
        if shift:
            atoms.translate(shift_vector * -1)

    atoms_cell_mag = [np.linalg.norm(x)
                      for x in np.array(atoms.cell[0:2, 0:2])]
    cell_vect_mag = [np.linalg.norm(x) for x in np.array(cell_vectors)]
    cell_scale = np.divide(atoms_cell_mag, cell_vect_mag)
    atoms *= (cell_search[0], cell_search[1], 1)
    atoms.cell[0:2, 0:2] = cell * cell_scale
    atoms.center()
    pretrim(atoms)
    remove_doubles(atoms, shift=False)
    remove_doubles(atoms, shift=True)

    def rot(vector, angle):
        return [(vector[0] * cos(angle)) - (vector[1] * sin(angle)),
                (vector[0] * sin(angle)) + (vector[1] * cos(angle))]
    angle = -atan2(atoms.cell[0][1], atoms.cell[0][0])
    atoms.cell[0][0:2] = rot(atoms.cell[0][0:2], angle)
    atoms.cell[1][0:2] = rot(atoms.cell[1][0:2], angle)
    for atom in atoms:
        atom.position[0:2] = rot(atom.position[0:2], angle)
    atoms.center()

    atoms.positions = np.around(atoms.positions, decimals=logeps)
    ind = np.lexsort(
        (atoms.positions[:, 0], atoms.positions[:, 1], atoms.positions[:, 2],))
    return atoms[ind]


def root_surface_analysis(primitive_slab, root, allow_above=False, eps=1e-8):
    """A tool to analyze a slab and look for valid roots that exist, up to
       the given root. This is useful for generating all possible cells
       without prior knowledge.

       *primitive slab* is the primitive cell to analyze.

       *root* is the desired root to find, and all below.

       *allow_above* allows you to also include cells above
       the given *root* if found in the process.  Otherwise these
       are trimmed off."""

    logeps = int(-log10(eps))
    atoms = primitive_slab
    # Normalize the x axis to a distance of 1, and use the cell
    # We ignore the z axis because this code cannot handle it
    xscale = np.linalg.norm(atoms.cell[0][0:2])
    xx, xy = atoms.cell[0][0:2] / xscale
    yx, yy = atoms.cell[1][0:2] / xscale
    cell_vectors = [[xx, xy], [yx, yy]]

    # Manipulate the cell vectors to find the best search zone and
    # cast to numpy array.
    cell_vectors = np.array(cell_vectors)
    cell_vectors_mag = [np.linalg.norm(x) for x in cell_vectors]
    cell_search = [int(ceil(float(root * 1.2) / float(x)))
                   for x in cell_vectors_mag]

    # Returns valid roots that are found in the given search
    # space.  To find more, use a higher root.
    valid = set()
    for x in range(cell_search[0]):
        for y in range(cell_search[1]):
            if x == y == 0:
                continue
            vect = (cell_vectors[0] * x) + (cell_vectors[1] * y)
            dist = round((vect ** 2).sum(), logeps)
            # Only integer roots make sense logically
            if dist.is_integer():
                if dist <= root or allow_above:
                    valid.add(int(dist))
    return sorted(list(valid))
