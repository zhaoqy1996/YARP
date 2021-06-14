""" Various utility methods used troughout the GA. """
from ase.data import covalent_radii
import itertools
import numpy as np
from ase.io import write, read
import os
import time
import math
from ase.ga import get_neighbor_list


def closest_distances_generator(atom_numbers, ratio_of_covalent_radii):
    """ Generates the blmin dict used across the GA.
        The distances are based on the covalent radii of the atoms.
    """
    cr = covalent_radii
    ratio = ratio_of_covalent_radii

    blmin = dict()
    for i in atom_numbers:
        blmin[(i, i)] = cr[i] * 2 * ratio
        for j in atom_numbers:
            if i == j:
                continue
            if (i, j) in blmin.keys():
                continue
            blmin[(i, j)] = blmin[(j, i)] = ratio * (cr[i] + cr[j])
    return blmin


def get_mic_distance(p1, p2, cell, pbc):
    """ This method calculates the shortest distance between p1 and p2
         through the cell boundaries defined by cell and pbc.
         This method works for reasonable unit cells, but not for extremely
         elongated ones.
    """
    ct = cell.T
    pos = np.mat((p1, p2))
    scaled = np.linalg.solve(ct, pos.T).T
    for i in range(3):
        if pbc[i]:
            scaled[:, i] %= 1.0
            scaled[:, i] %= 1.0
    P = np.dot(scaled, cell)

    pbc_directions = [[-1, 1] * int(direction) + [0] for direction in pbc]
    translations = np.mat(list(itertools.product(*pbc_directions))).T
    p0r = np.tile(np.reshape(P[0, :], (3, 1)), (1, translations.shape[1]))
    p1r = np.tile(np.reshape(P[1, :], (3, 1)), (1, translations.shape[1]))
    dp_vec = p0r + ct * translations
    d = np.min(np.power(p1r - dp_vec, 2).sum(axis=0))**0.5
    return d


def db_call_with_error_tol(db_cursor, expression, args=[]):
    """ In case the GA is used on older versions of networking
         filesystems there might be some delays. For this reason
         some extra error tolerance when calling the SQLite db is
         employed.
    """
    import sqlite3
    i = 0
    while i < 10:
        try:
            db_cursor.execute(expression, args)
            return
        except sqlite3.OperationalError as e:
            print(e)
            time.sleep(2.)
        i += 1
    raise sqlite3.OperationalError(
        'Database still locked after 10 attempts (20 s)')


def save_trajectory(confid, trajectory, folder):
    """ Saves traj files to the database folder.
         This method should never be used directly,
         but only through the DataConnection object.
    """
    fname = os.path.join(folder, 'traj%05d.traj' % confid)
    write(fname, trajectory)
    return fname


def get_trajectory(fname):
    """ Extra error tolerance when loading traj files. """
    fname = str(fname)
    try:
        t = read(fname)
    except IOError as e:
        print('get_trajectory error ' + e)
    return t


def atoms_too_close(a, bl):
    """ Checks if any atoms in a are too close, as defined by
        the distances in the bl dictionary. """
    num = a.numbers
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if a.get_distance(i, j, True) < bl[(num[i], num[j])]:
                return True
    return False


def atoms_too_close_two_sets(a, b, bl):
    """ Checks if any atoms in a are too close to an atom in b,
        as defined by the bl dictionary. """
    tot = a + b
    num = tot.numbers
    for i in range(len(a)):
        for j in range(len(a), len(tot)):
            if tot.get_distance(i, j, True) < bl[(num[i], num[j])]:
                return True
    return False


def get_all_atom_types(slab, atom_numbers_to_optimize):
    """ Utility method used to extract all unique atom types
        from the atoms object slab and the list of atomic numbers
        atom_numbers_to_optimize. """
    from_slab = list(set(slab.numbers))
    from_top = list(set(atom_numbers_to_optimize))
    from_slab.extend(from_top)
    return list(set(from_slab))


def get_distance_matrix(atoms, self_distance=1000):
    """ NB: This function is way slower than atoms.get_all_distances()
        Returns a numpy matrix with the distances between the atoms
        in the supplied atoms object, with the indices of the matrix
        corresponding to the indices in the atoms object.
        The parameter self_distance will be put in the diagonal
        elements ([i][i])
    """
    dm = np.zeros([len(atoms), len(atoms)])
    for i in range(len(atoms)):
        dm[i][i] = self_distance
        for j in range(i + 1, len(atoms)):
            rij = atoms.get_distance(i, j)
            dm[i][j] = rij
            dm[j][i] = rij
    return dm


def get_rdf(atoms, rmax, nbins, distance_matrix=None,
            elements=None, no_dists=False):
    """
    Returns two numpy arrays; the radial distribution function
    and the corresponding distances of the supplied atoms object.
    If no_dists = True then only the first array is returned.

    Parameters:

    rmax : float
        The maximum distance that will contribute to the rdf.

    nbins : int
        Number of bins to divide the rdf into.

    distance_matrix : numpy.array
        An array of distances between atoms, typically
        obtained by atoms.get_all_distances().
        Default None meaning that it will be calculated.

    elements : list or tuple
        List of two atomic numbers. If elements is not None the partial
        rdf for the supplied elements will be returned.

    no_dists : bool
        If True then the second array with rdf distances will not be returned
    """
    dm = distance_matrix
    if dm is None:
        dm = atoms.get_all_distances()
    rdf = np.zeros(nbins + 1)
    dr = float(rmax / nbins)

    if elements is None:
        # Coefficients to use for normalization
        phi = len(atoms) / atoms.get_volume()
        norm = 2.0 * math.pi * dr * phi * len(atoms)

        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                rij = dm[i][j]
                index = int(math.ceil(rij / dr))
                if index <= nbins:
                    rdf[index] += 1
    else:
        i_indices = np.where(atoms.numbers == elements[0])[0]
        phi = len(i_indices) / atoms.get_volume()
        norm = 4.0 * math.pi * dr * phi * len(atoms)

        for i in i_indices:
            for j in np.where(atoms.numbers == elements[1])[0]:
                rij = dm[i][j]
                index = int(math.ceil(rij / dr))
                if index <= nbins:
                    rdf[index] += 1

    dists = []
    for i in range(1, nbins + 1):
        rrr = (i - 0.5) * dr
        dists.append(rrr)
        # Normalize
        rdf[i] /= (norm * ((rrr**2) + (dr**2) / 12.))

    if no_dists:
        return rdf[1:]
    return rdf[1:], np.array(dists)


def get_nndist(atoms, distance_matrix):
    """
    Returns an estimate of the nearest neighbor bond distance
    in the supplied atoms object given the supplied distance_matrix.
    The estimate comes from the first peak in the radial distribution
    function.
    """
    rmax = 10.  # No bonds longer than 10 angstrom expected
    nbins = 200
    rdf, dists = get_rdf(atoms, rmax, nbins, distance_matrix)
    return dists[np.argmax(rdf)]


def get_nnmat(atoms, mic=False):
    """
    Calculate the nearest neighbor matrix as specified in
    S. Lysgaard et al., Top. Catal., 2014, 57 (1-4), pp 33-39

    Returns an array of average numbers of nearest neighbors
    the order is determined by self.elements.
    Example: self.elements = ["Cu", "Ni"]
    get_nnmat returns a single list [Cu-Cu bonds/N(Cu),
    Cu-Ni bonds/N(Cu), Ni-Cu bonds/N(Ni), Ni-Ni bonds/N(Ni)]
    where N(element) is the number of atoms of the type element
    in the atoms object.

    The distance matrix can be quite costly to calculate every
    time nnmat is required (and disk intensive if saved), thus
    it makes sense to calculate nnmat along with e.g. the
    potential energy and save it in atoms.info['data']['nnmat'].
    """
    if 'data' in atoms.info and 'nnmat' in atoms.info['data']:
        return atoms.info['data']['nnmat']
    elements = sorted(set(atoms.get_chemical_symbols()))
    nnmat = np.zeros((len(elements), len(elements)))
    # dm = get_distance_matrix(atoms)
    dm = atoms.get_all_distances(mic=mic)
    nndist = get_nndist(atoms, dm) + 0.2
    for i in range(len(atoms)):
        row = [j for j in range(len(elements))
               if atoms[i].symbol == elements[j]][0]
        neighbors = [j for j in range(len(dm[i])) if dm[i][j] < nndist]
        for n in neighbors:
            column = [j for j in range(len(elements))
                      if atoms[n].symbol == elements[j]][0]
            nnmat[row][column] += 1
    # divide by the number of that type of atoms in the structure
    for i, el in enumerate(elements):
        nnmat[i] /= len([j for j in range(len(atoms))
                         if atoms[int(j)].symbol == el])
    # makes a single list out of a list of lists
    nnlist = np.reshape(nnmat, (len(nnmat)**2))
    return nnlist


def get_nnmat_string(atoms, decimals=2, mic=False):
    nnmat = get_nnmat(atoms, mic=mic)
    s = '-'.join(['{1:2.{0}f}'.format(decimals, i)
                  for i in nnmat])
    if len(nnmat) == 1:
        return s + '-'
    return s


def get_connections_index(atoms, max_conn=5, no_count_types=None):
    """
    This method returns a dictionary where each key value are a
    specific number of neighbors and list of atoms indices with
    that amount of neighbors respectively. The method utilizes the
    neighbor list and hence inherit the restrictions for
    neighbors. Option added to remove connections between
    defined atom types.

    Parameters
    ----------

    atoms : Atoms object
        The connections will be counted using this supplied Atoms object

    max_conn : int
        Any atom with more connections than this will be counted as
        having max_conn connections.
        Default 5

    no_count_types : list or None
        List of atomic numbers that should be excluded in the count.
        Default None (meaning all atoms count).
    """
    conn = get_neighbor_list(atoms)

    if conn is None:
        conn = get_neighborlist(atoms)

    if no_count_types is None:
        no_count_types = []

    conn_index = {}
    for i in range(len(atoms)):
        if atoms[i].number not in no_count_types:
            cconn = min(len(conn[i]), max_conn - 1)
            if cconn not in conn_index:
                conn_index[cconn] = []
            conn_index[cconn].append(i)

    return conn_index


def get_atoms_connections(atoms, max_conn=5, no_count_types=None):
    """
    This method returns a list of the numbers of atoms
    with X number of neighbors. The method utilizes the
    neighbor list and hence inherit the restrictions for
    neighbors. Option added to remove connections between
    defined atom types.
    """
    conn_index = get_connections_index(atoms, max_conn=max_conn,
                                       no_count_types=no_count_types)

    no_of_conn = [0] * max_conn
    for i in conn_index:
        no_of_conn[i] += len(conn_index[i])

    return no_of_conn


def get_angles_distribution(atoms, ang_grid=9):
    """
    Method to get the distribution of bond angles
    in bins (default 9) with bonds defined from
    the get_neighbor_list().
    """
    conn = get_neighbor_list(atoms)

    if conn is None:
        conn = get_neighborlist(atoms)

    bins = [0] * ang_grid

    for atom in atoms:
        for i in conn[atom.index]:
            for j in conn[atom.index]:
                if j != i:
                    a = atoms.get_angle(i, atom.index, j)
                    for k in range(ang_grid):
                        if (k + 1) * 180. / ang_grid > a > k * 180. / ang_grid:
                            bins[k] += 1
    # Removing dobbelt counting
    for i in range(ang_grid):
        bins[i] /= 2.
    return bins


def get_neighborlist(atoms, dx=0.2, no_count_types=None):
    """
    Method to get the a dict with list of neighboring
    atoms defined as the two covalent radii + fixed distance.
    Option added to remove neighbors between defined atom types.
    """
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    if no_count_types is None:
        no_count_types = []

    conn = {}
    for atomi in atoms:
        conn_this_atom = []
        for atomj in atoms:
            if atomi.index != atomj.index:
                if atomi.number not in no_count_types:
                    if atomj.number not in no_count_types:
                        d = get_mic_distance(atomi.position,
                                             atomj.position,
                                             cell,
                                             pbc)
                        cri = covalent_radii[atomi.number]
                        crj = covalent_radii[atomj.number]
                        d_max = crj + cri + dx
                        if d < d_max:
                            conn_this_atom.append(atomj.index)
        conn[atomi.index] = conn_this_atom
    return conn


def get_atoms_distribution(atoms, number_of_bins=5, max_distance=8,
                           center=None, no_count_types=None):
    """
    Method to get the distribution of atoms in the
    structure in bins of distances from a defined
    center. Option added to remove counting of
    certain atom types.
    """
    pbc = atoms.get_pbc()
    cell = atoms.get_cell()
    if center is None:
        # Center used for the atom distribution if None is supplied!
        cx = sum(cell[:, 0]) / 2.
        cy = sum(cell[:, 1]) / 2.
        cz = sum(cell[:, 2]) / 2.
        center = (cx, cy, cz)
    bins = [0] * number_of_bins

    if no_count_types is None:
        no_count_types = []

    for atom in atoms:
        if atom.number not in no_count_types:
            d = get_mic_distance(atom.position, center, cell, pbc)
            for k in range(number_of_bins - 1):
                min_dis_cur_bin = k * max_distance / (number_of_bins - 1.)
                max_dis_cur_bin = ((k + 1) * max_distance /
                                   (number_of_bins - 1.))
                if min_dis_cur_bin < d < max_dis_cur_bin:
                    bins[k] += 1
            if d > max_distance:
                bins[number_of_bins - 1] += 1
    return bins


def get_rings(atoms, rings=[5, 6, 7]):
    """
    This method return a list of the number of atoms involved
    in rings in the structures. It uses the neighbor
    list hence inherit the restriction used for neighbors.
    """
    conn = get_neighbor_list(atoms)

    if conn is None:
        conn = get_neighborlist(atoms)

    no_of_loops = [0] * 8
    for s1 in range(len(atoms)):
        for s2 in conn[s1]:
            v12 = [s1] + [s2]
            for s3 in [s for s in conn[s2] if s not in v12]:
                v13 = v12 + [s3]
                if s1 in conn[s3]:
                    no_of_loops[3] += 1
                for s4 in [s for s in conn[s3] if s not in v13]:
                    v14 = v13 + [s4]
                    if s1 in conn[s4]:
                        no_of_loops[4] += 1
                    for s5 in [s for s in conn[s4] if s not in v14]:
                        v15 = v14 + [s5]
                        if s1 in conn[s5]:
                            no_of_loops[5] += 1
                        for s6 in [s for s in conn[s5] if s not in v15]:
                            v16 = v15 + [s6]
                            if s1 in conn[s6]:
                                no_of_loops[6] += 1
                            for s7 in [s for s in conn[s6] if s not in v16]:
                                # v17 = v16 + [s7]
                                if s1 in conn[s7]:
                                    no_of_loops[7] += 1

    to_return = []
    for ring in rings:
        to_return.append(no_of_loops[ring])

    return to_return
