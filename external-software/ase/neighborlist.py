from math import sqrt

import numpy as np
from scipy import sparse as sp

from ase.data import atomic_numbers
from ase.geometry import complete_cell


def mic(dr, cell, pbc=None):
    """
    Apply minimum image convention to an array of distance vectors.

    Parameters
    ----------
    dr : array_like
        Array of distance vectors.
    cell : array_like
        Simulation cell.
    pbc : array_like, optional
        Periodic boundary conditions in x-, y- and z-direction. Default is to
        assume periodic boundaries in all directions.

    Returns
    -------
    dr : array
        Array of distance vectors, wrapped according to the minimum image
        convention.
    """
    # Check where distance larger than 1/2 cell. Particles have crossed
    # periodic boundaries then and need to be unwrapped.
    icell = np.linalg.pinv(cell)
    if pbc is not None:
        icell *= np.array(pbc, dtype=int).reshape(3, 1)
    cell_shift_vectors = np.round(np.dot(dr, icell))

    # Unwrap
    return dr - np.dot(cell_shift_vectors, cell)


def primitive_neighbor_list(quantities, pbc, cell, positions, cutoff,
                            numbers=None, self_interaction=False,
                            use_scaled_positions=False, max_nbins=1e6):
    """Compute a neighbor list for an atomic configuration.

    Atoms outside periodic boundaries are mapped into the box. Atoms
    outside nonperiodic boundaries are included in the neighbor list
    but complexity of neighbor list search for those can become n^2.

    The neighbor list is sorted by first atom index 'i', but not by second
    atom index 'j'.

    Parameters:

    quantities: str
        Quantities to compute by the neighbor list algorithm. Each character
        in this string defines a quantity. They are returned in a tuple of
        the same order. Possible quantities are

            * 'i' : first atom index
            * 'j' : second atom index
            * 'd' : absolute distance
            * 'D' : distance vector
            * 'S' : shift vector (number of cell boundaries crossed by the bond
              between atom i and j). With the shift vector S, the
              distances D between atoms can be computed from:
              D = positions[j]-positions[i]+S.dot(cell)
    pbc: array_like
        3-tuple indicating giving periodic boundaries in the three Cartesian
        directions.
    cell: 3x3 matrix
        Unit cell vectors.
    positions: list of xyz-positions
        Atomic positions.  Anything that can be converted to an ndarray of
        shape (n, 3) will do: [(x1,y1,z1), (x2,y2,z2), ...]. If
        use_scaled_positions is set to true, this must be scaled positions.
    cutoff: float or dict
        Cutoff for neighbor search. It can be:

            * A single float: This is a global cutoff for all elements.
            * A dictionary: This specifies cutoff values for element
              pairs. Specification accepts element numbers of symbols.
              Example: {(1, 6): 1.1, (1, 1): 1.0, ('C', 'C'): 1.85}
            * A list/array with a per atom value: This specifies the radius of
              an atomic sphere for each atoms. If spheres overlap, atoms are
              within each others neighborhood. See :func:`~ase.utils.natural_cutoffs`
              for an example on how to get such a list.
    self_interaction: bool
        Return the atom itself as its own neighbor if set to true.
        Default: False
    use_scaled_positions: bool
        If set to true, positions are expected to be scaled positions.
    max_nbins: int
        Maximum number of bins used in neighbor search. This is used to limit
        the maximum amount of memory required by the neighbor list.

    Returns:

    i, j, ... : array
        Tuple with arrays for each quantity specified above. Indices in `i`
        are returned in ascending order 0..len(a)-1, but the order of (i,j)
        pairs is not guaranteed.

    """

    # Naming conventions: Suffixes indicate the dimension of an array. The
    # following convention is used here:
    #     c: Cartesian index, can have values 0, 1, 2
    #     i: Global atom index, can have values 0..len(a)-1
    #     xyz: Bin index, three values identifying x-, y- and z-component of a
    #          spatial bin that is used to make neighbor search O(n)
    #     b: Linearized version of the 'xyz' bin index
    #     a: Bin-local atom index, i.e. index identifying an atom *within* a
    #        bin
    #     p: Pair index, can have value 0 or 1
    #     n: (Linear) neighbor index

    # Return empty neighbor list if no atoms are passed here
    if len(positions) == 0:
        empty_types = dict(i=(np.int, (0, )),
                           j=(np.int, (0, )),
                           D=(np.float, (0, 3)),
                           d=(np.float, (0, )),
                           S=(np.int, (0, 3)))
        retvals = []
        for i in quantities:
            dtype, shape = empty_types[i]
            retvals += [np.array([], dtype=dtype).reshape(shape)]
        if len(retvals) == 1:
            return retvals[0]
        else:
            return tuple(retvals)

    # Compute reciprocal lattice vectors.
    b1_c, b2_c, b3_c = np.linalg.pinv(cell).T

    # Compute distances of cell faces.
    l1 = np.linalg.norm(b1_c)
    l2 = np.linalg.norm(b2_c)
    l3 = np.linalg.norm(b3_c)
    face_dist_c = np.array([1 / l1 if l1 > 0 else 1,
                            1 / l2 if l2 > 0 else 1,
                            1 / l3 if l3 > 0 else 1])

    if isinstance(cutoff, dict):
        max_cutoff = max(cutoff.values())
    else:
        if np.isscalar(cutoff):
            max_cutoff = cutoff
        else:
            cutoff = np.asarray(cutoff)
            max_cutoff = 2*np.max(cutoff)

    # We use a minimum bin size of 3 A
    bin_size = max(max_cutoff, 3)
    # Compute number of bins such that a sphere of radius cutoff fit into eight
    # neighboring bins.
    nbins_c = np.maximum((face_dist_c / bin_size).astype(int), [1, 1, 1])
    nbins = np.prod(nbins_c)
    # Make sure we limit the amount of memory used by the explicit bins.
    while nbins > max_nbins:
        nbins_c = np.maximum(nbins_c // 2, [1, 1, 1])
        nbins = np.prod(nbins_c)

    # Compute over how many bins we need to loop in the neighbor list search.
    neigh_search_x, neigh_search_y, neigh_search_z = \
        np.ceil(bin_size * nbins_c / face_dist_c).astype(int)

    # Sort atoms into bins.
    if use_scaled_positions:
        scaled_positions_ic = positions
        positions = np.dot(scaled_positions_ic, cell)
    else:
        scaled_positions_ic = np.linalg.solve(complete_cell(cell).T,
                                              positions.T).T
    bin_index_ic = np.floor(scaled_positions_ic*nbins_c).astype(int)
    cell_shift_ic = np.zeros_like(bin_index_ic)

    for c in range(3):
        if pbc[c]:
            # (Note: np.divmod does not exist in older numpies)
            cell_shift_ic[:, c], bin_index_ic[:, c] = \
                divmod(bin_index_ic[:, c], nbins_c[c])
        else:
            bin_index_ic[:, c] = np.clip(bin_index_ic[:, c], 0, nbins_c[c]-1)

    # Convert Cartesian bin index to unique scalar bin index.
    bin_index_i = (bin_index_ic[:, 0] +
                   nbins_c[0] * (bin_index_ic[:, 1] +
                                 nbins_c[1] * bin_index_ic[:, 2]))

    # atom_i contains atom index in new sort order.
    atom_i = np.argsort(bin_index_i)
    bin_index_i = bin_index_i[atom_i]

    # Find max number of atoms per bin
    max_natoms_per_bin = np.bincount(bin_index_i).max()

    # Sort atoms into bins: atoms_in_bin_ba contains for each bin (identified
    # by its scalar bin index) a list of atoms inside that bin. This list is
    # homogeneous, i.e. has the same size *max_natoms_per_bin* for all bins.
    # The list is padded with -1 values.
    atoms_in_bin_ba = -np.ones([nbins, max_natoms_per_bin], dtype=int)
    for i in range(max_natoms_per_bin):
        # Create a mask array that identifies the first atom of each bin.
        mask = np.append([True], bin_index_i[:-1] != bin_index_i[1:])
        # Assign all first atoms.
        atoms_in_bin_ba[bin_index_i[mask], i] = atom_i[mask]

        # Remove atoms that we just sorted into atoms_in_bin_ba. The next
        # "first" atom will be the second and so on.
        mask = np.logical_not(mask)
        atom_i = atom_i[mask]
        bin_index_i = bin_index_i[mask]

    # Make sure that all atoms have been sorted into bins.
    assert len(atom_i) == 0
    assert len(bin_index_i) == 0

    # Now we construct neighbor pairs by pairing up all atoms within a bin or
    # between bin and neighboring bin. atom_pairs_pn is a helper buffer that
    # contains all potential pairs of atoms between two bins, i.e. it is a list
    # of length max_natoms_per_bin**2.
    atom_pairs_pn = np.indices((max_natoms_per_bin, max_natoms_per_bin),
                               dtype=int)
    atom_pairs_pn = atom_pairs_pn.reshape(2, -1)

    # Initialized empty neighbor list buffers.
    first_at_neightuple_nn = []
    secnd_at_neightuple_nn = []
    cell_shift_vector_x_n = []
    cell_shift_vector_y_n = []
    cell_shift_vector_z_n = []

    # This is the main neighbor list search. We loop over neighboring bins and
    # then construct all possible pairs of atoms between two bins, assuming
    # that each bin contains exactly max_natoms_per_bin atoms. We then throw
    # out pairs involving pad atoms with atom index -1 below.
    binz_xyz, biny_xyz, binx_xyz = np.meshgrid(np.arange(nbins_c[2]),
                                               np.arange(nbins_c[1]),
                                               np.arange(nbins_c[0]),
                                               indexing='ij')
    # The memory layout of binx_xyz, biny_xyz, binz_xyz is such that computing
    # the respective bin index leads to a linearly increasing consecutive list.
    # The following assert statement succeeds:
    #     b_b = (binx_xyz + nbins_c[0] * (biny_xyz + nbins_c[1] *
    #                                     binz_xyz)).ravel()
    #     assert (b_b == np.arange(np.prod(nbins_c))).all()

    # First atoms in pair.
    _first_at_neightuple_n = atoms_in_bin_ba[:, atom_pairs_pn[0]]
    for dz in range(-neigh_search_z, neigh_search_z+1):
        for dy in range(-neigh_search_y, neigh_search_y+1):
            for dx in range(-neigh_search_x, neigh_search_x+1):
                # Bin index of neighboring bin and shift vector.
                shiftx_xyz, neighbinx_xyz = divmod(binx_xyz + dx, nbins_c[0])
                shifty_xyz, neighbiny_xyz = divmod(biny_xyz + dy, nbins_c[1])
                shiftz_xyz, neighbinz_xyz = divmod(binz_xyz + dz, nbins_c[2])
                neighbin_b = (neighbinx_xyz + nbins_c[0] *
                              (neighbiny_xyz + nbins_c[1] * neighbinz_xyz)
                              ).ravel()

                # Second atom in pair.
                _secnd_at_neightuple_n = \
                    atoms_in_bin_ba[neighbin_b][:, atom_pairs_pn[1]]

                # Shift vectors.
                _cell_shift_vector_x_n = \
                    np.resize(shiftx_xyz.reshape(-1, 1),
                              (max_natoms_per_bin**2, shiftx_xyz.size)).T
                _cell_shift_vector_y_n = \
                    np.resize(shifty_xyz.reshape(-1, 1),
                              (max_natoms_per_bin**2, shifty_xyz.size)).T
                _cell_shift_vector_z_n = \
                    np.resize(shiftz_xyz.reshape(-1, 1),
                              (max_natoms_per_bin**2, shiftz_xyz.size)).T

                # We have created too many pairs because we assumed each bin
                # has exactly max_natoms_per_bin atoms. Remove all surperfluous
                # pairs. Those are pairs that involve an atom with index -1.
                mask = np.logical_and(_first_at_neightuple_n != -1,
                                      _secnd_at_neightuple_n != -1)
                if mask.sum() > 0:
                    first_at_neightuple_nn += [_first_at_neightuple_n[mask]]
                    secnd_at_neightuple_nn += [_secnd_at_neightuple_n[mask]]
                    cell_shift_vector_x_n += [_cell_shift_vector_x_n[mask]]
                    cell_shift_vector_y_n += [_cell_shift_vector_y_n[mask]]
                    cell_shift_vector_z_n += [_cell_shift_vector_z_n[mask]]

    # Flatten overall neighbor list.
    first_at_neightuple_n = np.concatenate(first_at_neightuple_nn)
    secnd_at_neightuple_n = np.concatenate(secnd_at_neightuple_nn)
    cell_shift_vector_n = np.transpose([np.concatenate(cell_shift_vector_x_n),
                                        np.concatenate(cell_shift_vector_y_n),
                                        np.concatenate(cell_shift_vector_z_n)])

    # Add global cell shift to shift vectors
    cell_shift_vector_n += cell_shift_ic[first_at_neightuple_n] - \
        cell_shift_ic[secnd_at_neightuple_n]

    # Remove all self-pairs that do not cross the cell boundary.
    if not self_interaction:
        m = np.logical_not(np.logical_and(
            first_at_neightuple_n == secnd_at_neightuple_n,
            (cell_shift_vector_n == 0).all(axis=1)))
        first_at_neightuple_n = first_at_neightuple_n[m]
        secnd_at_neightuple_n = secnd_at_neightuple_n[m]
        cell_shift_vector_n = cell_shift_vector_n[m]

    # For nonperiodic directions, remove any bonds that cross the domain
    # boundary.
    for c in range(3):
        if not pbc[c]:
            m = cell_shift_vector_n[:, c] == 0
            first_at_neightuple_n = first_at_neightuple_n[m]
            secnd_at_neightuple_n = secnd_at_neightuple_n[m]
            cell_shift_vector_n = cell_shift_vector_n[m]

    # Sort neighbor list.
    i = np.argsort(first_at_neightuple_n)
    first_at_neightuple_n = first_at_neightuple_n[i]
    secnd_at_neightuple_n = secnd_at_neightuple_n[i]
    cell_shift_vector_n = cell_shift_vector_n[i]

    # Compute distance vectors.
    distance_vector_nc = positions[secnd_at_neightuple_n] - \
        positions[first_at_neightuple_n] + \
        cell_shift_vector_n.dot(cell)
    abs_distance_vector_n = \
        np.sqrt(np.sum(distance_vector_nc*distance_vector_nc, axis=1))

    # We have still created too many pairs. Only keep those with distance
    # smaller than max_cutoff.
    mask = abs_distance_vector_n < max_cutoff
    first_at_neightuple_n = first_at_neightuple_n[mask]
    secnd_at_neightuple_n = secnd_at_neightuple_n[mask]
    cell_shift_vector_n = cell_shift_vector_n[mask]
    distance_vector_nc = distance_vector_nc[mask]
    abs_distance_vector_n = abs_distance_vector_n[mask]

    if isinstance(cutoff, dict) and numbers is not None:
        # If cutoff is a dictionary, then the cutoff radii are specified per
        # element pair. We now have a list up to maximum cutoff.
        per_pair_cutoff_n = np.zeros_like(abs_distance_vector_n)
        for (atomic_number1, atomic_number2), c in cutoff.items():
            try:
                atomic_number1 = atomic_numbers[atomic_number1]
            except KeyError:
                pass
            try:
                atomic_number2 = atomic_numbers[atomic_number2]
            except KeyError:
                pass
            if atomic_number1 == atomic_number2:
                mask = np.logical_and(
                    numbers[first_at_neightuple_n] == atomic_number1,
                    numbers[secnd_at_neightuple_n] == atomic_number2)
            else:
                mask = np.logical_or(
                    np.logical_and(
                        numbers[first_at_neightuple_n] == atomic_number1,
                        numbers[secnd_at_neightuple_n] == atomic_number2),
                    np.logical_and(
                        numbers[first_at_neightuple_n] == atomic_number2,
                        numbers[secnd_at_neightuple_n] == atomic_number1))
            per_pair_cutoff_n[mask] = c
        mask = abs_distance_vector_n < per_pair_cutoff_n
        first_at_neightuple_n = first_at_neightuple_n[mask]
        secnd_at_neightuple_n = secnd_at_neightuple_n[mask]
        cell_shift_vector_n = cell_shift_vector_n[mask]
        distance_vector_nc = distance_vector_nc[mask]
        abs_distance_vector_n = abs_distance_vector_n[mask]
    elif not np.isscalar(cutoff):
        # If cutoff is neither a dictionary nor a scalar, then we assume it is
        # a list or numpy array that contains atomic radii. Atoms are neighbors
        # if their radii overlap.
        mask = abs_distance_vector_n < \
            cutoff[first_at_neightuple_n] + cutoff[secnd_at_neightuple_n]
        first_at_neightuple_n = first_at_neightuple_n[mask]
        secnd_at_neightuple_n = secnd_at_neightuple_n[mask]
        cell_shift_vector_n = cell_shift_vector_n[mask]
        distance_vector_nc = distance_vector_nc[mask]
        abs_distance_vector_n = abs_distance_vector_n[mask]

    # Assemble return tuple.
    retvals = []
    for q in quantities:
        if q == 'i':
            retvals += [first_at_neightuple_n]
        elif q == 'j':
            retvals += [secnd_at_neightuple_n]
        elif q == 'D':
            retvals += [distance_vector_nc]
        elif q == 'd':
            retvals += [abs_distance_vector_n]
        elif q == 'S':
            retvals += [cell_shift_vector_n]
        else:
            raise ValueError('Unsupported quantity specified.')
    if len(retvals) == 1:
        return retvals[0]
    else:
        return tuple(retvals)


def neighbor_list(quantities, a, cutoff, self_interaction=False,
                  max_nbins=1e6):
    """Compute a neighbor list for an atomic configuration.

    Atoms outside periodic boundaries are mapped into the box. Atoms
    outside nonperiodic boundaries are included in the neighbor list
    but complexity of neighbor list search for those can become n^2.

    The neighbor list is sorted by first atom index 'i', but not by second
    atom index 'j'.

    Parameters:

    quantities: str
        Quantities to compute by the neighbor list algorithm. Each character
        in this string defines a quantity. They are returned in a tuple of
        the same order. Possible quantities are:

           * 'i' : first atom index
           * 'j' : second atom index
           * 'd' : absolute distance
           * 'D' : distance vector
           * 'S' : shift vector (number of cell boundaries crossed by the bond
             between atom i and j). With the shift vector S, the
             distances D between atoms can be computed from:
             D = a.positions[j]-a.positions[i]+S.dot(a.cell)
    a: :class:`ase.Atoms`
        Atomic configuration.
    cutoff: float or dict
        Cutoff for neighbor search. It can be:

            * A single float: This is a global cutoff for all elements.
            * A dictionary: This specifies cutoff values for element
              pairs. Specification accepts element numbers of symbols.
              Example: {(1, 6): 1.1, (1, 1): 1.0, ('C', 'C'): 1.85}
            * A list/array with a per atom value: This specifies the radius of
              an atomic sphere for each atoms. If spheres overlap, atoms are
              within each others neighborhood. See :func:`~ase.utils.natural_cutoffs`
              for an example on how to get such a list.

    self_interaction: bool
        Return the atom itself as its own neighbor if set to true.
        Default: False
    max_nbins: int
        Maximum number of bins used in neighbor search. This is used to limit
        the maximum amount of memory required by the neighbor list.

    Returns:

    i, j, ...: array
        Tuple with arrays for each quantity specified above. Indices in `i`
        are returned in ascending order 0..len(a), but the order of (i,j)
        pairs is not guaranteed.

    Examples:

    Examples assume Atoms object *a* and numpy imported as *np*.

    1. Coordination counting::

        i = neighbor_list('i', a, 1.85)
        coord = np.bincount(i)

    2. Coordination counting with different cutoffs for each pair of species::

        i = neighbor_list('i', a,
                          {('H', 'H'): 1.1, ('C', 'H'): 1.3, ('C', 'C'): 1.85})
        coord = np.bincount(i)

    3. Pair distribution function::

        d = neighbor_list('d', a, 10.00)
        h, bin_edges = np.histogram(d, bins=100)
        pdf = h/(4*np.pi/3*(bin_edges[1:]**3 - bin_edges[:-1]**3)) * a.get_volume()/len(a)

    4. Pair potential::

        i, j, d, D = neighbor_list('ijdD', a, 5.0)
        energy = (-C/d**6).sum()
        pair_forces = (6*C/d**5  * (D/d).T).T
        forces_x = np.bincount(j, weights=pair_forces[:, 0], minlength=len(a)) - \
                   np.bincount(i, weights=pair_forces[:, 0], minlength=len(a))
        forces_y = np.bincount(j, weights=pair_forces[:, 1], minlength=len(a)) - \
                   np.bincount(i, weights=pair_forces[:, 1], minlength=len(a))
        forces_z = np.bincount(j, weights=pair_forces[:, 2], minlength=len(a)) - \
                   np.bincount(i, weights=pair_forces[:, 2], minlength=len(a))

    5. Dynamical matrix for a pair potential stored in a block sparse format::

        from scipy.sparse import bsr_matrix
        i, j, dr, abs_dr = neighbor_list('ijDd', atoms)
        energy = (dr.T / abs_dr).T
        dynmat = -(dde * (energy.reshape(-1, 3, 1) * energy.reshape(-1, 1, 3)).T).T \
                 -(de / abs_dr * (np.eye(3, dtype=energy.dtype) - \
                   (energy.reshape(-1, 3, 1) * energy.reshape(-1, 1, 3))).T).T
        dynmat_bsr = bsr_matrix((dynmat, j, first_i), shape=(3*len(a), 3*len(a)))

        dynmat_diag = np.empty((len(a), 3, 3))
        for x in range(3):
            for y in range(3):
                dynmat_diag[:, x, y] = -np.bincount(i, weights=dynmat[:, x, y])

        dynmat_bsr += bsr_matrix((dynmat_diag, np.arange(len(a)),
                                  np.arange(len(a) + 1)),
                                 shape=(3 * len(a), 3 * len(a)))

    """
    return primitive_neighbor_list(quantities, a.pbc,
                                   a.get_cell(complete=True),
                                   a.positions, cutoff, numbers=a.numbers,
                                   self_interaction=self_interaction,
                                   max_nbins=max_nbins)


def first_neighbors(natoms, first_atom):
    """
    Compute an index array pointing to the ranges within the neighbor list that
    contain the neighbors for a certain atom.

    Parameters
    ----------
    natoms : int
        Total number of atom.
    first_atom : array_like
        Array containing the first atom 'i' of the neighbor tuple returned
        by the neighbor list.

    Returns
    -------
    seed : array
        Array containing pointers to the start and end location of the
        neighbors of a certain atom. Neighbors of atom k have indices from s[k]
        to s[k+1]-1.
    """
    if len(first_atom) == 0:
        return np.zeros(natoms+1, dtype=int)
    # Create a seed array (which is returned by this function) populated with
    # -1.
    seed = -np.ones(natoms+1, dtype=int)

    first_atom = np.asarray(first_atom)

    # Mask array contains all position where the number in the (sorted) array
    # with first atoms (in the neighbor pair) changes.
    mask = first_atom[:-1] != first_atom[1:]

    # Seed array needs to start at 0
    seed[first_atom[0]] = 0
    # Seed array needs to stop at the length of the neighbor list
    seed[-1] = len(first_atom)
    # Populate all intermediate seed with the index of where the mask array is
    # true, i.e. the index where the first_atom array changes.
    seed[first_atom[1:][mask]] = (np.arange(len(mask))+1)[mask]

    # Now fill all remaining -1 value with the value in the seed array right
    # behind them. (There are no neighbor so seed[i] and seed[i+1] must point)
    # to the same index.
    mask = seed == -1
    while mask.any():
        seed[mask] = seed[np.arange(natoms+1)[mask]+1]
        mask = seed == -1
    return seed

def get_connectivity_matrix(nl, sparse=True):
    """Return connectivity matrix for a given NeighborList (dtype=numpy.int8).

    A matrix of shape (nAtoms, nAtoms) will be returned.
    Connected atoms i and j will have matrix[i,j] == 1, unconnected
    matrix[i,j] == 0. If bothways=True the matrix will be symmetric,
    otherwise not!

    If *sparse* is True, a scipy csr matrix is returned.
    If *sparse* is False, a numpy matrix is returned.

    Note that the old and new neighborlists might give different results
    for periodic systems if bothways=False.

    Example:

    Determine which molecule in a system atom 1 belongs to.

    >>> from ase import neighborlist
    >>> from ase.build import molecule
    >>> from ase.utils import natural_cutoffs
    >>> from scipy import sparse
    >>> mol = molecule('CH3CH2OH')
    >>> cutOff = natural_cutoffs(mol)
    >>> neighborList = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
    >>> neighborList.update(mol)
    >>> matrix = neighborList.get_connectivity_matrix()
    >>> #or: matrix = neighborlist.get_connectivity_matrix(neighborList.nl)
    >>> n_components, component_list = sparse.csgraph.connected_components(matrix)
    >>> idx = 1
    >>> molIdx = component_list[idx]
    >>> print("There are {} molecules in the system".format(n_components))
    >>> print("Atom {} is part of molecule {}".format(idx, molIdx))
    >>> molIdxs = [ i for i in range(len(component_list)) if component_list[i] == molIdx ]
    >>> print("The following atoms are part of molecule {}: {}".format(molIdx, molIdxs))
    """

    nAtoms = len(nl.cutoffs)

    if nl.nupdates <= 0:
        raise RuntimeError('Must call update(atoms) on your neighborlist first!')

    if sparse:
        matrix = sp.dok_matrix((nAtoms, nAtoms), dtype=np.int8)
    else:
        matrix = np.zeros((nAtoms, nAtoms), dtype=np.int8)

    for i in range(nAtoms):
        for idx in nl.get_neighbors(i)[0]:
            matrix[i, idx] = 1

    return matrix


class NewPrimitiveNeighborList:
    """Neighbor list object. Wrapper around neighbor_list and first_neighbors.

    cutoffs: list of float
        List of cutoff radii - one for each atom. If the spheres (defined by
        their cutoff radii) of two atoms overlap, they will be counted as
        neighbors.
    skin: float
        If no atom has moved more than the skin-distance since the
        last call to the :meth:`~ase.neighborlist.NewPrimitiveNeighborList.update()`
        method, then the neighbor list can be reused. This will save
        some expensive rebuilds of the list, but extra neighbors outside
        the cutoff will be returned.
    sorted: bool
        Sort neighbor list.
    self_interaction: bool
        Should an atom return itself as a neighbor?
    bothways: bool
        Return all neighbors.  Default is to return only "half" of
        the neighbors.

    Example::

      nl = NeighborList([2.3, 1.7])
      nl.update(atoms)
      indices, offsets = nl.get_neighbors(0)
    """

    def __init__(self, cutoffs, skin=0.3, sorted=False, self_interaction=True,
                 bothways=False, use_scaled_positions=False):
        self.cutoffs = np.asarray(cutoffs) + skin
        self.skin = skin
        self.sorted = sorted
        self.self_interaction = self_interaction
        self.bothways = bothways
        self.nupdates = 0
        self.use_scaled_positions = use_scaled_positions
        self.nneighbors = 0
        self.npbcneighbors = 0

    def update(self,  pbc, cell, positions, numbers=None):
        """Make sure the list is up to date."""

        if self.nupdates == 0:
            self.build(pbc, cell, positions, numbers=numbers)
            return True

        if ((self.pbc != pbc).any() or (self.cell != cell).any() or
            ((self.positions - positions)**2).sum(1).max() > self.skin**2):
            self.build(pbc, cell, positions, numbers=numbers)
            return True

        return False

    def build(self, pbc, cell, positions, numbers=None):
        """Build the list.
        """
        self.pbc = np.array(pbc, copy=True)
        self.cell = np.array(cell, copy=True)
        self.positions = np.array(positions, copy=True)

        self.pair_first, self.pair_second, self.offset_vec = \
            primitive_neighbor_list(
                'ijS', pbc, cell, positions, self.cutoffs, numbers=numbers,
                self_interaction=self.self_interaction,
                use_scaled_positions=self.use_scaled_positions)

        if len(positions) > 0 and not self.bothways:
            mask = np.logical_or(
                np.logical_and(
                    self.pair_first <= self.pair_second,
                    (self.offset_vec == 0).all(axis=1)
                    ),
                np.logical_or(
                    self.offset_vec[:, 0] > 0,
                    np.logical_and(
                        self.offset_vec[:, 0] == 0,
                        np.logical_or(
                            self.offset_vec[:, 1] > 0,
                            np.logical_and(
                                self.offset_vec[:, 1] == 0,
                                self.offset_vec[:, 2] > 0)
                            )
                        )
                    )
                )
            self.pair_first = self.pair_first[mask]
            self.pair_second = self.pair_second[mask]
            self.offset_vec = self.offset_vec[mask]

        if len(positions) > 0 and self.sorted:
            mask = np.argsort(self.pair_first * len(self.pair_first) +
                              self.pair_second)
            self.pair_first = self.pair_first[mask]
            self.pair_second = self.pair_second[mask]
            self.offset_vec = self.offset_vec[mask]

        # Compute the index array point to the first neighbor
        self.first_neigh = first_neighbors(len(positions), self.pair_first)

        self.nupdates += 1

    def get_neighbors(self, a):
        """Return neighbors of atom number a.

        A list of indices and offsets to neighboring atoms is
        returned.  The positions of the neighbor atoms can be
        calculated like this:

        >>>  indices, offsets = nl.get_neighbors(42)
        >>>  for i, offset in zip(indices, offsets):
        >>>      print(atoms.positions[i] + dot(offset, atoms.get_cell()))

        Notice that if get_neighbors(a) gives atom b as a neighbor,
        then get_neighbors(b) will not return a as a neighbor - unless
        bothways=True was used."""

        return (self.pair_second[self.first_neigh[a]:self.first_neigh[a+1]],
                self.offset_vec[self.first_neigh[a]:self.first_neigh[a+1]])



class PrimitiveNeighborList:
    """Neighbor list that works without Atoms objects.

    This is less fancy, but can be used to avoid conversions between
    scaled and non-scaled coordinates which may affect cell offsets
    through rounding errors.
    """
    def __init__(self, cutoffs, skin=0.3, sorted=False, self_interaction=True,
                 bothways=False, use_scaled_positions=False):
        self.cutoffs = np.asarray(cutoffs) + skin
        self.skin = skin
        self.sorted = sorted
        self.self_interaction = self_interaction
        self.bothways = bothways
        self.nupdates = 0
        self.use_scaled_positions = use_scaled_positions
        self.nneighbors = 0
        self.npbcneighbors = 0

    def update(self, pbc, cell, coordinates):
        """Make sure the list is up to date."""

        if self.nupdates == 0:
            self.build(pbc, cell, coordinates)
            return True

        if ((self.pbc != pbc).any() or (self.cell != cell).any() or
            ((self.coordinates - coordinates)**2).sum(1).max() > self.skin**2):
            self.build(pbc, cell, coordinates)
            return True

        return False

    def build(self, pbc, cell, coordinates):
        """Build the list.

        Coordinates are taken to be scaled or not according
        to self.use_scaled_positions.
        """
        self.pbc = pbc = np.array(pbc, copy=True)
        self.cell = cell = np.array(cell, copy=True)
        self.coordinates = coordinates = np.array(coordinates, copy=True)

        if len(self.cutoffs) != len(coordinates):
            raise ValueError('Wrong number of cutoff radii: {0} != {1}'
                             .format(len(self.cutoffs), len(coordinates)))

        if len(self.cutoffs) > 0:
            rcmax = self.cutoffs.max()
        else:
            rcmax = 0.0

        icell = np.linalg.pinv(cell)

        if self.use_scaled_positions:
            scaled = coordinates
            positions = np.dot(scaled, cell)
        else:
            positions = coordinates
            scaled = np.dot(positions, icell)

        scaled0 = scaled.copy()

        N = []
        for i in range(3):
            if self.pbc[i]:
                scaled0[:, i] %= 1.0
                v = icell[:, i]
                h = 1 / sqrt(np.dot(v, v))
                n = int(2 * rcmax / h) + 1
            else:
                n = 0
            N.append(n)

        offsets = (scaled0 - scaled).round().astype(int)
        positions0 = positions + np.dot(offsets, self.cell)
        natoms = len(positions)
        indices = np.arange(natoms)

        self.nneighbors = 0
        self.npbcneighbors = 0
        self.neighbors = [np.empty(0, int) for a in range(natoms)]
        self.displacements = [np.empty((0, 3), int) for a in range(natoms)]
        for n1 in range(0, N[0] + 1):
            for n2 in range(-N[1], N[1] + 1):
                for n3 in range(-N[2], N[2] + 1):
                    if n1 == 0 and (n2 < 0 or n2 == 0 and n3 < 0):
                        continue
                    displacement = np.dot((n1, n2, n3), self.cell)
                    for a in range(natoms):
                        d = positions0 + displacement - positions0[a]
                        i = indices[(d**2).sum(1) <
                                    (self.cutoffs + self.cutoffs[a])**2]
                        if n1 == 0 and n2 == 0 and n3 == 0:
                            if self.self_interaction:
                                i = i[i >= a]
                            else:
                                i = i[i > a]
                        self.nneighbors += len(i)
                        self.neighbors[a] = np.concatenate(
                            (self.neighbors[a], i))
                        disp = np.empty((len(i), 3), int)
                        disp[:] = (n1, n2, n3)
                        disp += offsets[i] - offsets[a]
                        self.npbcneighbors += disp.any(1).sum()
                        self.displacements[a] = np.concatenate(
                            (self.displacements[a], disp))

        if self.bothways:
            neighbors2 = [[] for a in range(natoms)]
            displacements2 = [[] for a in range(natoms)]
            for a in range(natoms):
                for b, disp in zip(self.neighbors[a], self.displacements[a]):
                    neighbors2[b].append(a)
                    displacements2[b].append(-disp)
            for a in range(natoms):
                nbs = np.concatenate((self.neighbors[a], neighbors2[a]))
                disp = np.array(list(self.displacements[a]) +
                                displacements2[a])
                # Force correct type and shape for case of no neighbors:
                self.neighbors[a] = nbs.astype(int)
                self.displacements[a] = disp.astype(int).reshape((-1, 3))

        if self.sorted:
            for a, i in enumerate(self.neighbors):
                mask = (i < a)
                if mask.any():
                    j = i[mask]
                    offsets = self.displacements[a][mask]
                    for b, offset in zip(j, offsets):
                        self.neighbors[b] = np.concatenate(
                            (self.neighbors[b], [a]))
                        self.displacements[b] = np.concatenate(
                            (self.displacements[b], [-offset]))
                    mask = np.logical_not(mask)
                    self.neighbors[a] = self.neighbors[a][mask]
                    self.displacements[a] = self.displacements[a][mask]

        self.nupdates += 1

    def get_neighbors(self, a):
        """Return neighbors of atom number a.

        A list of indices and offsets to neighboring atoms is
        returned.  The positions of the neighbor atoms can be
        calculated like this::

          indices, offsets = nl.get_neighbors(42)
          for i, offset in zip(indices, offsets):
              print(atoms.positions[i] + dot(offset, atoms.get_cell()))

        Notice that if get_neighbors(a) gives atom b as a neighbor,
        then get_neighbors(b) will not return a as a neighbor - unless
        bothways=True was used."""

        return self.neighbors[a], self.displacements[a]


class NeighborList:
    """Neighbor list object.

    cutoffs: list of float
        List of cutoff radii - one for each atom. If the spheres (defined by
        their cutoff radii) of two atoms overlap, they will be counted as
        neighbors. See :func:`~ase.utils.natural_cutoffs` for an example on how to
        get such a list.

    skin: float
        If no atom has moved more than the skin-distance since the
        last call to the :meth:`~ase.neighborlist.NeighborList.update()` method,
        then the neighbor list can be reused.  This will save some expensive rebuilds
        of the list, but extra neighbors outside the cutoff will be returned.
    self_interaction: bool
        Should an atom return itself as a neighbor?
    bothways: bool
        Return all neighbors.  Default is to return only "half" of
        the neighbors.
    primitive: :class:`~ase.neighborlist.PrimitiveNeighborList` or :class:`~ase.neighborlist.NewPrimitiveNeighborList` class
        Define which implementation to use. Older and quadratically-scaling
        :class:`~ase.neighborlist.PrimitiveNeighborList` or newer and
        linearly-scaling :class:`~ase.neighborlist.NewPrimitiveNeighborList`.

    Example::

      nl = NeighborList([2.3, 1.7])
      nl.update(atoms)
      indices, offsets = nl.get_neighbors(0)
    """

    def __init__(self, cutoffs, skin=0.3, sorted=False, self_interaction=True,
                 bothways=False, primitive=PrimitiveNeighborList):
        self.nl = primitive(cutoffs, skin, sorted,
                            self_interaction=self_interaction,
                            bothways=bothways)

    def update(self, atoms):
        """
        See :meth:`ase.neighborlist.PrimitiveNeighborList.update` or
        :meth:`ase.neighborlist.PrimitiveNeighborList.update`.
        """
        return self.nl.update(atoms.pbc, atoms.get_cell(complete=True),
                              atoms.positions)

    def get_neighbors(self, a):
        """
        See :meth:`ase.neighborlist.PrimitiveNeighborList.get_neighbors` or
        :meth:`ase.neighborlist.PrimitiveNeighborList.get_neighbors`.
        """
        return self.nl.get_neighbors(a)

    def get_connectivity_matrix(self, sparse=True):
        """
        See :func:`~ase.neighborlist.get_connectivity_matrix`.
        """
        return get_connectivity_matrix(self.nl, sparse)

    @property
    def nupdates(self):
        """Get number of updates."""
        return self.nl.nupdates

    @property
    def nneighbors(self):
        """Get number of neighbors."""
        return self.nl.nneighbors

    @property
    def npbcneighbors(self):
        """Get number of pbc neighbors."""
        return self.nl.npbcneighbors
