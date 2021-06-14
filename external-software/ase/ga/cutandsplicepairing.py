""" Implementaiton of the cut and splice paring operator described by
Deaven and Ho.

"""
import numpy as np
from ase import Atoms
from random import random, randrange
from ase.ga.utilities import atoms_too_close
from ase.ga.utilities import atoms_too_close_two_sets
from ase.ga.offspring_creator import OffspringCreator
from math import pi, cos, sin


class Position(object):
    """Position helper object.

    This class is just a simple representation used by the pairing
    operator.

    Parameters:

    position: [x, y, z] coordinate
    number: Atomic species at the position
    distance: Signed distance to the cutting plane
    origin: Either 0 or 1 and determines which side of the plane
    the position should be at.

    """
    def __init__(self, position, number, distance, origin):
        self.number = number
        self.position = position
        self.distance = distance
        self.origin = origin

    def to_use(self):
        """ Method which tells if this position is at the right side.
        """
        if self.distance > 0. and self.origin == 0:
            return True
        elif self.distance < 0. and self.origin == 1:
            return True
        else:
            return False


def get_distance(point, cutting_plane, cutting_point):
    """
    Utility method for calculating the distance from a plane to a point
    """
    cm = cutting_point
    n = cutting_plane
    d = np.dot(point - cm, n)
    return d


class CutAndSplicePairing(OffspringCreator):

    """ The Cut and splice operator implemented as described in
    L.B. Vilhelmsen and B. Hammer, PRL, 108, 126101 (2012)

    Parameters:

    slab: Atoms object with supercell to optimize the structure in
    n_top: The number of atoms to optimize
    blmin: Dictionary with pairs of atom numbers and the closest
    distance these can have to each other.
       """
    def __init__(self, slab, n_top, blmin, verbose=False):
        OffspringCreator.__init__(self, verbose)
        self.blmin = blmin
        self.slab = slab
        self.n_top = n_top
        self.descriptor = 'CutAndSplicePairing'
        self.min_inputs = 2

    def _get_pairing_(self, a1, a2, cutting_plane, cutting_point):
        """ Pairs configuration a1 and a2 through the defined plane.
            This method does not check if atoms are too close.
        """
        N = len(a1)
        num = a1.numbers[:]
        unique_types = list(set(num))

        types = dict()
        for u in unique_types:
            types[u] = sum(num == u)

        # Generate list of all atoms
        p1 = [Position(a.position, a.number,
                       get_distance(a.position,
                                    cutting_plane,
                                    cutting_point), origin=0) for a in a1]
        p2 = [Position(a.position, a.number,
                       get_distance(a.position,
                                    cutting_plane,
                                    cutting_point), origin=1) for a in a2]
        all_points = p1
        all_points.extend(p2)

        # Sort these by their atomic number
        all_points.sort(key=lambda x: x.number, reverse=True)

        # For each atom type make the pairing
        unique_types.sort()
        use_total = dict()
        for u in unique_types:
            used = []
            not_used = []
            # The list is looked trough in
            # reverse order so atoms can be removed
            # from the list along the way.
            for i in reversed(range(len(all_points))):
                # If there are no more atoms of this type
                if all_points[i].number != u:
                    break
                # Check if the atom should be included
                if all_points[i].to_use():
                    used.append(all_points.pop(i))
                else:
                    not_used.append(all_points.pop(i))

            assert len(used) + len(not_used) == types[u] * 2

            # While we have too few of the given atom type
            while len(used) < types[u]:
                used.append(not_used.pop(randrange(0, len(not_used))))

            # While we have too many of the given atom type
            while len(used) > types[u]:
                index = used.index(max(used, key=lambda x: abs(x.distance)))
                not_used.append(used.pop(index))

            use_total[u] = used

        n_tot = sum([len(ll) for ll in use_total.values()])
        assert n_tot == N

        # Reorder the atoms to follow the atom types in the original order
        pos_new = [use_total[n].pop().position for n in num]
        return Atoms(numbers=num,
                     positions=pos_new, pbc=a1.get_pbc(), cell=a1.get_cell())

    def get_new_individual(self, parents):
        """ The method called by the user that
        returns the paired structure. """
        f, m = parents

        indi = self.cross(f, m)
        desc = 'pairing: {0} {1}'.format(f.info['confid'],
                                         m.info['confid'])
        # It is ok for an operator to return None
        # It means that it could not make a legal offspring
        # within a reasonable amount of time
        if indi is None:
            return indi, desc
        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid'],
                                        m.info['confid']]
        
        return self.finalize_individual(indi), desc

    def cross(self, a1, a2, test_dist_to_slab=True):
        """Crosses the two atoms objects and returns one"""
        
        if len(a1) != len(self.slab) + self.n_top:
            raise ValueError('Wrong size of structure to optimize')
        if len(a1) != len(a2):
            raise ValueError('The two structures do not have the same length')
        
        N = self.n_top

        # Only consider the atoms to optimize
        a1 = a1[len(a1) - N: len(a1)]
        a2 = a2[len(a2) - N: len(a2)]
        
        # if not np.array_equal(a1.numbers, a2.numbers):
        #     a1.numbers.sort()
        #     a2.numbers.sort()

        if not np.array_equal(a1.numbers, a2.numbers):
            err = 'Trying to pair two structures with different stoichiometry'
            raise ValueError(err)

        # Find the common center of the two clusters
        c1cm = np.average(a1.get_positions(), axis=0)
        c2cm = np.average(a2.get_positions(), axis=0)
        cutting_point = (c1cm + c2cm) / 2.

        counter = 0
        too_close = True
        n_max = 1000
        # Run until a valid pairing is made or 1000 pairings are tested.
        while too_close and counter < n_max:

            # Generate the cutting plane
            theta = pi * random()
            phi = 2. * pi * random()
            n = (cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta))
            n = np.array(n)

            # Get the pairing
            top = self._get_pairing_(a1, a2,
                                     cutting_plane=n,
                                     cutting_point=cutting_point)

            # Check if the candidate is valid
            too_close = atoms_too_close(top, self.blmin)
            if not too_close and test_dist_to_slab:
                too_close = atoms_too_close_two_sets(self.slab,
                                                     top, self.blmin)

            # Verify that the generated structure contains atoms from
            # both parents
            n1 = -1 * np.ones((N, ))
            n2 = -1 * np.ones((N, ))
            for i in range(N):
                for j in range(N):
                    if np.all(a1.positions[j, :] == top.positions[i, :]):
                        n1[i] = j
                        break
                    elif np.all(a2.positions[j, :] == top.positions[i, :]):
                        n2[i] = j
                        break
                assert (n1[i] > -1 and n2[i] == -1) or (n1[i] == -1 and
                                                        n2[i] > -1)

            if not (len(n1[n1 > -1]) > 0 and len(n2[n2 > -1]) > 0):
                too_close = True

            counter += 1

        if counter == n_max:
            return None
        
        return self.slab + top
