"""Comparators originally meant to be used with particles"""
import numpy as np
from ase.ga.utilities import get_nnmat


class NNMatComparator(object):
    """Use the nearest neighbor matrix to determine differences
    in the distribution (and to a slighter degree structure)
    of atoms. As specified in
    S. Lysgaard et al., Top. Catal., 57 (1-4), pp 33-39, (2014)"""
    def __init__(self, d=0.2, elements=None, mic=False):
        self.d = d
        if elements is None:
            elements = []
        self.elements = elements
        self.mic = mic

    def looks_like(self, a1, a2):
        """ Return if structure a1 or a2 are similar or not. """
        elements = self.elements
        if elements == []:
            elements = sorted(set(a1.get_chemical_symbols()))
        a1, a2 = a1.copy(), a2.copy()
        a1.set_constraint()
        a2.set_constraint()
        del a1[[a.index for a in a1 if a.symbol not in elements]]
        del a2[[a.index for a in a2 if a.symbol not in elements]]

        nnmat_a1 = get_nnmat(a1, mic=self.mic)
        nnmat_a2 = get_nnmat(a2, mic=self.mic)

        diff = np.linalg.norm(nnmat_a1 - nnmat_a2)

        if diff < self.d:
            return True
        else:
            return False
