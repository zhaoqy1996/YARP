import numpy as np
from ase.ga import get_raw_score


def get_sorted_dist_list(atoms, mic=False):
    """ Utility method used to calculate the sorted distance list
        describing the cluster in atoms. """
    numbers = atoms.numbers
    unique_types = set(numbers)
    pair_cor = dict()
    for n in unique_types:
        i_un = [i for i in range(len(atoms)) if atoms[i].number == n]
        d = []
        for i, n1 in enumerate(i_un):
            for n2 in i_un[i + 1:]:
                d.append(atoms.get_distance(n1, n2, mic))
        d.sort()
        pair_cor[n] = np.array(d)
    return pair_cor


class InteratomicDistanceComparator(object):

    """ An implementation of the comparison criteria described in
          L.B. Vilhelmsen and B. Hammer, PRL, 108, 126101 (2012)

        Parameters:

        n_top: The number of atoms being optimized by the GA.
            Default 0 - meaning all atoms.

        pair_cor_cum_diff: The limit in eq. 2 of the letter.
        pair_cor_max: The limit in eq. 3 of the letter
        dE: The limit of eq. 1 of the letter
        mic: Determines if distances are calculated
        using the minimum image convention
    """
    def __init__(self, n_top=None, pair_cor_cum_diff=0.015,
                 pair_cor_max=0.7, dE=0.02, mic=False):
        self.pair_cor_cum_diff = pair_cor_cum_diff
        self.pair_cor_max = pair_cor_max
        self.dE = dE
        self.n_top = n_top or 0
        self.mic = mic

    def looks_like(self, a1, a2):
        """ Return if structure a1 or a2 are similar or not. """
        if len(a1) != len(a2):
            raise Exception('The two configurations are not the same size')

        # first we check the energy criteria
        dE = abs(a1.get_potential_energy() - a2.get_potential_energy())
        if dE >= self.dE:
            return False

        # then we check the structure
        a1top = a1[-self.n_top:]
        a2top = a2[-self.n_top:]
        cum_diff, max_diff = self.__compare_structure__(a1top, a2top)

        return (cum_diff < self.pair_cor_cum_diff
                and max_diff < self.pair_cor_max)

    def __compare_structure__(self, a1, a2):
        """ Private method for calculating the structural difference. """
        p1 = get_sorted_dist_list(a1, mic=self.mic)
        p2 = get_sorted_dist_list(a2, mic=self.mic)
        numbers = a1.numbers
        total_cum_diff = 0.
        max_diff = 0
        for n in p1.keys():
            cum_diff = 0.
            c1 = p1[n]
            c2 = p2[n]
            assert len(c1) == len(c2)
            if len(c1) == 0:
                continue
            t_size = np.sum(c1)
            d = np.abs(c1 - c2)
            cum_diff = np.sum(d)
            max_diff = np.max(d)
            ntype = float(sum([i == n for i in numbers]))
            total_cum_diff += cum_diff / t_size * ntype / float(len(numbers))
        return (total_cum_diff, max_diff)


class SequentialComparator(object):
    """Use more than one comparison class and test them all in sequence.

    Supply a list of integers if for example two comparison tests both
    need to be positive if two atoms objects are truly equal.
    Ex:
    methods = [a, b, c, d], logics = [0, 1, 1, 2]
    if a or d is positive -> return True
    if b and c are positive -> return True
    if b and not c are positive (or vice versa) -> return False
    """
    def __init__(self, methods, logics=None):
        if not isinstance(methods, list):
            methods = [methods]
        if logics is None:
            logics = [i for i in range(len(methods))]
        if not isinstance(logics, list):
            logics = [logics]
        assert len(logics) == len(methods)

        self.methods = []
        self.logics = []
        for m, l in zip(methods, logics):
            if hasattr(m, 'looks_like'):
                self.methods.append(m)
                self.logics.append(l)

    def looks_like(self, a1, a2):
        mdct = dict((l, []) for l in self.logics)
        for m, l in zip(self.methods, self.logics):
            mdct[l].append(m)

        for methods in mdct.values():
            for m in methods:
                if not m.looks_like(a1, a2):
                    break
            else:
                return True
        return False


class StringComparator(object):
    """Compares the calculated hash strings. These strings should be stored
       in atoms.info['key_value_pairs'][key1] and
       atoms.info['key_value_pairs'][key2] ...
       where the keys should be supplied as parameters i.e.
       StringComparator(key1, key2, ...)
    """
    def __init__(self, *keys):
        self.keys = keys

    def looks_like(self, a1, a2):
        for k in self.keys:
            if a1.info['key_value_pairs'][k] == a2.info['key_value_pairs'][k]:
                return True
        return False


class EnergyComparator(object):
    """Compares the energy of the supplied atoms objects using
       get_potential_energy().

       Parameters:

       dE: the difference in energy below which two energies are
       deemed equal.
    """
    def __init__(self, dE=0.02):
        self.dE = dE

    def looks_like(self, a1, a2):
        dE = abs(a1.get_potential_energy() - a2.get_potential_energy())
        if dE >= self.dE:
            return False
        else:
            return True


class RawScoreComparator(object):
    """Compares the raw_score of the supplied individuals
       objects using a1.info['key_value_pairs']['raw_score'].

       Parameters:

       dist: the difference in raw_score below which two
       scores are deemed equal.
    """
    def __init__(self, dist=0.02):
        self.dist = dist

    def looks_like(self, a1, a2):
        d = abs(get_raw_score(a1) - get_raw_score(a2))
        if d >= self.dist:
            return False
        else:
            return True


class NoComparator(object):
    """Returns False always. If you don't want any comparator."""
    def looks_like(self, *args):
        return False


class AtomsComparator(object):
    """Compares the Atoms objects directly."""
    def looks_like(self, a1, a2):
        return a1 == a2


class CompositionComparator(object):
    """Compares the composition of the Atoms objects."""
    def looks_like(self, a1, a2):
        return a1.get_chemical_formula() == a2.get_chemical_formula()
