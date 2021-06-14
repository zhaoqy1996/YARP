from __future__ import print_function
# Copyright (C) 2010, Jesper Friis
# (see accompanying license files for details).

"""Utility tools for atoms/geometry manipulations.
   - convenient creation of slabs and interfaces of
different orientations.
   - detection of duplicate atoms / atoms within cutoff radius
"""

from math import pi

import numpy as np

from ase.geometry import complete_cell


def wrap_positions(positions, cell, pbc=True, center=(0.5, 0.5, 0.5),
                   eps=1e-7):
    """Wrap positions to unit cell.

    Returns positions changed by a multiple of the unit cell vectors to
    fit inside the space spanned by these vectors.  See also the
    :meth:`ase.Atoms.wrap` method.

    Parameters:

    positions: float ndarray of shape (n, 3)
        Positions of the atoms
    cell: float ndarray of shape (3, 3)
        Unit cell vectors.
    pbc: one or 3 bool
        For each axis in the unit cell decides whether the positions
        will be moved along this axis.
    center: three float
        The positons in fractional coordinates that the new positions
        will be nearest possible to.
    eps: float
        Small number to prevent slightly negative coordinates from being
        wrapped.

    Example:

    >>> from ase.geometry import wrap_positions
    >>> wrap_positions([[-0.1, 1.01, -0.5]],
    ...                [[1, 0, 0], [0, 1, 0], [0, 0, 4]],
    ...                pbc=[1, 1, 0])
    array([[ 0.9 ,  0.01, -0.5 ]])
    """

    if not hasattr(pbc, '__len__'):
        pbc = (pbc,) * 3

    if not hasattr(center, '__len__'):
        center = (center,) * 3

    shift = np.asarray(center) - 0.5 - eps

    # Don't change coordinates when pbc is False
    shift[np.logical_not(pbc)] = 0.0

    assert np.asarray(cell)[np.asarray(pbc)].any(axis=1).all(), (cell, pbc)

    cell = complete_cell(cell)
    fractional = np.linalg.solve(cell.T,
                                 np.asarray(positions).T).T - shift

    for i, periodic in enumerate(pbc):
        if periodic:
            fractional[:, i] %= 1.0
            fractional[:, i] += shift[i]

    return np.dot(fractional, cell)


def get_layers(atoms, miller, tolerance=0.001):
    """Returns two arrays describing which layer each atom belongs
    to and the distance between the layers and origo.

    Parameters:

    miller: 3 integers
        The Miller indices of the planes. Actually, any direction
        in reciprocal space works, so if a and b are two float
        vectors spanning an atomic plane, you can get all layers
        parallel to this with miller=np.cross(a,b).
    tolerance: float
        The maximum distance in Angstrom along the plane normal for
        counting two atoms as belonging to the same plane.

    Returns:

    tags: array of integres
        Array of layer indices for each atom.
    levels: array of floats
        Array of distances in Angstrom from each layer to origo.

    Example:

    >>> import numpy as np
    >>> from ase.spacegroup import crystal
    >>> atoms = crystal('Al', [(0,0,0)], spacegroup=225, cellpar=4.05)
    >>> np.round(atoms.positions, decimals=5)
    array([[ 0.   ,  0.   ,  0.   ],
           [ 0.   ,  2.025,  2.025],
           [ 2.025,  0.   ,  2.025],
           [ 2.025,  2.025,  0.   ]])
    >>> get_layers(atoms, (0,0,1))  # doctest: +ELLIPSIS
    (array([0, 1, 1, 0]...), array([ 0.   ,  2.025]))
    """
    miller = np.asarray(miller)

    metric = np.dot(atoms.cell, atoms.cell.T)
    c = np.linalg.solve(metric.T, miller.T).T
    miller_norm = np.sqrt(np.dot(c, miller))
    d = np.dot(atoms.get_scaled_positions(), miller) / miller_norm

    keys = np.argsort(d)
    ikeys = np.argsort(keys)
    mask = np.concatenate(([True], np.diff(d[keys]) > tolerance))
    tags = np.cumsum(mask)[ikeys]
    if tags.min() == 1:
        tags -= 1

    levels = d[keys][mask]
    return tags, levels


def find_mic(D, cell, pbc=True):
    """Finds the minimum-image representation of vector(s) D"""

    cell = complete_cell(cell)
    # Calculate the 4 unique unit cell diagonal lengths
    diags = np.sqrt((np.dot([[1, 1, 1],
                             [-1, 1, 1],
                             [1, -1, 1],
                             [-1, -1, 1],
                             ], cell)**2).sum(1))

    # calculate 'mic' vectors (D) and lengths (D_len) using simple method
    Dr = np.dot(D, np.linalg.inv(cell))
    D = np.dot(Dr - np.round(Dr) * pbc, cell)
    D_len = np.sqrt((D**2).sum(1))
    # return mic vectors and lengths for only orthorhombic cells,
    # as the results may be wrong for non-orthorhombic cells
    if (max(diags) - min(diags)) / max(diags) < 1e-9:
        return D, D_len

    # The cutoff radius is the longest direct distance between atoms
    # or half the longest lattice diagonal, whichever is smaller
    cutoff = min(max(D_len), max(diags) / 2.)

    # The number of neighboring images to search in each direction is
    # equal to the ceiling of the cutoff distance (defined above) divided
    # by the length of the projection of the lattice vector onto its
    # corresponding surface normal. a's surface normal vector is e.g.
    # b x c / (|b| |c|), so this projection is (a . (b x c)) / (|b| |c|).
    # The numerator is just the lattice volume, so this can be simplified
    # to V / (|b| |c|). This is rewritten as V |a| / (|a| |b| |c|)
    # for vectorization purposes.
    latt_len = np.sqrt((cell**2).sum(1))
    V = abs(np.linalg.det(cell))
    n = pbc * np.array(np.ceil(cutoff * np.prod(latt_len) /
                               (V * latt_len)), dtype=int)

    # Construct a list of translation vectors. For example, if we are
    # searching only the nearest images (27 total), tvecs will be a
    # 27x3 array of translation vectors. This is the only nested loop
    # in the routine, and it takes a very small fraction of the total
    # execution time, so it is not worth optimizing further.
    tvecs = []
    for i in range(-n[0], n[0] + 1):
        latt_a = i * cell[0]
        for j in range(-n[1], n[1] + 1):
            latt_ab = latt_a + j * cell[1]
            for k in range(-n[2], n[2] + 1):
                tvecs.append(latt_ab + k * cell[2])
    tvecs = np.array(tvecs)

    # Translate the direct displacement vectors by each translation
    # vector, and calculate the corresponding lengths.
    D_trans = tvecs[np.newaxis] + D[:, np.newaxis]
    D_trans_len = np.sqrt((D_trans**2).sum(2))

    # Find mic distances and corresponding vector(s) for each given pair
    # of atoms. For symmetrical systems, there may be more than one
    # translation vector corresponding to the MIC distance; this finds the
    # first one in D_trans_len.
    D_min_len = np.min(D_trans_len, axis=1)
    D_min_ind = D_trans_len.argmin(axis=1)
    D_min = D_trans[list(range(len(D_min_ind))), D_min_ind]

    return D_min, D_min_len


def get_angles(v1, v2, cell=None, pbc=None):
    """Get angles formed by two lists of vectors.

    calculate angle in degrees between vectors v1 and v2

    Set a cell and pbc to enable minimum image
    convention, otherwise angles are taken as-is.
    """

    f = 180 / pi

    # Check if using mic
    if cell is not None or pbc is not None:
        if cell is None or pbc is None:
            raise ValueError("cell or pbc must be both set or both be None")

        v1 = find_mic(v1, cell, pbc)[0]
        v2= find_mic(v2, cell, pbc)[0]


    v1 /= np.linalg.norm(v1, axis=1)[:, np.newaxis]
    v2 /= np.linalg.norm(v2, axis=1)[:, np.newaxis]

    angles = np.arccos(np.einsum('ij,ij->i', v1, v2))

    return angles * f


def get_distances(p1, p2=None, cell=None, pbc=None):
    """Return distance matrix of every position in p1 with every position in p2

    if p2 is not set, it is assumed that distances between all positions in p1
    are desired. p2 will be set to p1 in this case.

    Use set cell and pbc to use the minimum image convention.
    """
    if p2 is None:
        p2 = p1

    p1, p2 = np.array(p1), np.array(p2)

    # Allocate matrix for vectors as [p1, p2, 3]
    D = np.zeros((len(p1), len(p2), 3))

    for offset, pos1 in enumerate(p1):
        D[offset, :, :] = p2 - pos1 

    # Collapse to linear indexing
    D.shape = (-1, 3)

    # Check if using mic
    if cell is not None or pbc is not None:
        if cell is None or pbc is None:
            raise ValueError("cell or pbc must be both set or both be None")

        D, D_len = find_mic(D, cell, pbc)
    else:
        D_len = np.sqrt((D**2).sum(1))

    # Expand back to matrix indexing
    D.shape = (-1, len(p2), 3)
    D_len.shape = (-1, len(p2))

    return D, D_len


def get_duplicate_atoms(atoms, cutoff=0.1, delete=False):
    """Get list of duplicate atoms and delete them if requested.

    Identify all atoms which lie within the cutoff radius of each other.
    Delete one set of them if delete == True.
    """
    from scipy.spatial.distance import pdist
    dists = pdist(atoms.get_positions(), 'sqeuclidean')
    dup = np.nonzero(dists < cutoff**2)
    rem = np.array(_row_col_from_pdist(len(atoms), dup[0]))
    if delete:
        if rem.size != 0:
            del atoms[rem[:, 0]]
    else:
        return rem


def _row_col_from_pdist(dim, i):
    """Calculate the i,j index in the square matrix for an index in a
    condensed (triangular) matrix.
    """
    i = np.array(i)
    b = 1 - 2 * dim
    x = (np.floor((-b - np.sqrt(b**2 - 8 * i)) / 2)).astype(int)
    y = (i + x * (b + x + 2) / 2 + 1).astype(int)
    if i.shape:
        return list(zip(x, y))
    else:
        return [(x, y)]
