"""Crossover classes, that cross the elements in the supplied
atoms objects.

"""
import random
import numpy as np

from ase.ga.offspring_creator import OffspringCreator


class ElementCrossover(OffspringCreator):
    """Base class for all operators where the elements of
    the atoms objects cross.

    """
    def __init__(self, element_pool, max_diff_elements,
                 min_percentage_elements, verbose):
        OffspringCreator.__init__(self, verbose)
        if not isinstance(element_pool[0], (list, np.ndarray)):
            self.element_pools = [element_pool]
        else:
            self.element_pools = element_pool

        if max_diff_elements is None:
            self.max_diff_elements = [None for _ in self.element_pools]
        elif isinstance(max_diff_elements, int):
            self.max_diff_elements = [max_diff_elements]
        else:
            self.max_diff_elements = max_diff_elements
        assert len(self.max_diff_elements) == len(self.element_pools)

        if min_percentage_elements is None:
            self.min_percentage_elements = [0 for _ in self.element_pools]
        elif isinstance(min_percentage_elements, (int, float)):
            self.min_percentage_elements = [min_percentage_elements]
        else:
            self.min_percentage_elements = min_percentage_elements
        assert len(self.min_percentage_elements) == len(self.element_pools)
        
        self.min_inputs = 2

    def get_new_individual(self, parents):
        raise NotImplementedError


class OnePointElementCrossover(ElementCrossover):
    """Crossover of the elements in the atoms objects. Point of cross
    is chosen randomly.

    Parameters:

    element_pool: List of elements in the phase space. The elements can be
        grouped if the individual consist of different types of elements.
        The list should then be a list of lists e.g. [[list1], [list2]]

    max_diff_elements: The maximum number of different elements in the
        individual. Default is infinite. If the elements are grouped
        max_diff_elements should be supplied as a list with each input
        corresponding to the elements specified in the same input in
        element_pool.

    min_percentage_elements: The minimum percentage of any element in
        the individual. Default is any number is allowed. If the elements
        are grouped min_percentage_elements should be supplied as a list
        with each input corresponding to the elements specified in the
        same input in element_pool.

    Example: element_pool=[[A,B,C,D],[x,y,z]], max_diff_elements=[3,2],
        min_percentage_elements=[.25, .5]
        An individual could be "D,B,B,C,x,x,x,x,z,z,z,z"
    """
    def __init__(self, element_pool, max_diff_elements=None,
                 min_percentage_elements=None, verbose=False):
        ElementCrossover.__init__(self, element_pool,
                                  max_diff_elements,
                                  min_percentage_elements, verbose)
        self.descriptor = 'OnePointElementCrossover'

    def get_new_individual(self, parents):
        f, m = parents

        indi = self.initialize_individual(f)
        indi.info['data']['parents'] = [i.info['confid'] for i in parents]

        cut_choices = [i for i in range(1, len(f) - 1)]
        random.shuffle(cut_choices)
        for cut in cut_choices:
            fsyms = f.get_chemical_symbols()
            msyms = m.get_chemical_symbols()
            syms = fsyms[:cut] + msyms[cut:]
            ok = True
            for i, e in enumerate(self.element_pools):
                elems = e[:]
                elems_in, indices_in = zip(*[(a.symbol, a.index) for a in f
                                             if a.symbol in elems])
                max_diff_elem = self.max_diff_elements[i]
                min_percent_elem = self.min_percentage_elements[i]
                if min_percent_elem == 0:
                    min_percent_elem = 1. / len(elems_in)
                if max_diff_elem is None:
                    max_diff_elem = len(elems_in)

                syms_in = [syms[i] for i in indices_in]
                for s in set(syms_in):
                    percentage = syms_in.count(s) / float(len(syms_in))
                    if percentage < min_percent_elem:
                        ok = False
                        break
                    num_diff = len(set(syms_in))
                    if num_diff > max_diff_elem:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                break

        # Sufficient or does some individuals appear
        # below min_percentage_elements

        for a in f[:cut] + m[cut:]:
            indi.append(a)

        parent_message = ':Parents {0} {1}'.format(f.info['confid'],
                                                   m.info['confid'])
        return (self.finalize_individual(indi),
                self.descriptor + parent_message)


class TwoPointElementCrossover(ElementCrossover):
    """Crosses two individuals by choosing two cross points
    at random"""
    def __init__(self, element_pool, max_diff_elements=None,
                 min_percentage_elements=None, verbose=False):
        ElementCrossover.__init__(self, element_pool,
                                  max_diff_elements,
                                  min_percentage_elements, verbose)
        self.descriptor = 'TwoPointElementCrossover'

    def get_new_individual(self, parents):
        raise NotImplementedError
