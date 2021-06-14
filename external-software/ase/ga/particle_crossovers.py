"""Crossover operations originally intended for medium sized particles"""
import random
import numpy as np
from itertools import chain

from ase import Atoms
from ase.ga.offspring_creator import OffspringCreator


class Crossover(OffspringCreator):
    """Base class for all particle crossovers.
    Do not call this class directly."""
    def __init__(self):
        OffspringCreator.__init__(self)
        self.descriptor = 'Crossover'
        self.min_inputs = 2


class CutSpliceCrossover(Crossover):
    """Crossover that cuts two particles through a plane in space and
    merges two halfes from different particles together.

    Implementation of the method presented in:
    D. M. Deaven and K. M. Ho, Phys. Rev. Lett., 75, 2, 288-291 (1995)

    It keeps the correct composition by randomly assigning elements in
    the new particle. If some of the atoms in the two particle halves
    are too close, the halves are moved away from each other perpendicular
    to the cutting plane.

    Parameters:

    blmin: dictionary of minimum distance between atomic numbers.
        e.g. {(28,29): 1.5}
    
    keep_composition: boolean that signifies if the composition should
        be the same as in the parents.
    """
    def __init__(self, blmin, keep_composition=True):
        Crossover.__init__(self)
        self.blmin = blmin
        self.keep_composition = keep_composition
        self.descriptor = 'CutSpliceCrossover'
        
    def get_new_individual(self, parents):
        f, m = parents
        
        indi = self.initialize_individual(f)
        indi.info['data']['parents'] = [i.info['confid'] for i in parents]
        
        theta = random.random() * 2 * np.pi  # 0,2pi
        phi = random.random() * np.pi  # 0,pi
        e = np.array((np.sin(phi) * np.cos(theta),
                      np.sin(theta) * np.sin(phi),
                      np.cos(phi)))
        eps = 0.0001
        
        f.translate(-f.get_center_of_mass())
        m.translate(-m.get_center_of_mass())
        
        # Get the signed distance to the cutting plane
        # We want one side from f and the other side from m
        fmap = [np.dot(x, e) for x in f.get_positions()]
        mmap = [-np.dot(x, e) for x in m.get_positions()]
        ain = sorted([i for i in chain(fmap, mmap) if i > 0],
                     reverse=True)
        aout = sorted([i for i in chain(fmap, mmap) if i < 0],
                      reverse=True)

        off = len(ain) - len(f)

        # Translating f and m to get the correct number of atoms
        # in the offspring
        if off < 0:
            # too few
            # move f and m away from the plane
            dist = (abs(aout[abs(off) - 1]) + abs(aout[abs(off)])) * .5
            f.translate(e * dist)
            m.translate(-e * dist)
        elif off > 0:
            # too many
            # move f and m towards the plane
            dist = (abs(ain[-off - 1]) + abs(ain[-off])) * .5
            f.translate(-e * dist)
            m.translate(e * dist)
        if off != 0 and dist == 0:
            # Exactly same position => we continue with the wrong number
            # of atoms. What should be done? Fail or return None or
            # remove one of the two atoms with exactly the same position.
            pass

        # Determine the contributing parts from f and m
        tmpf, tmpm = Atoms(), Atoms()
        for atom in f:
            if np.dot(atom.position, e) > 0:
                atom.tag = 1
                tmpf.append(atom)
        for atom in m:
            if np.dot(atom.position, e) < 0:
                atom.tag = 2
                tmpm.append(atom)

        # Check that the correct composition is employed
        if self.keep_composition:
            opt_sm = sorted(f.numbers)
            tmpf_numbers = list(tmpf.numbers)
            tmpm_numbers = list(tmpm.numbers)
            cur_sm = sorted(tmpf_numbers + tmpm_numbers)
            # correct_by: dictionary that specifies how many
            # of the atom_numbers should be removed (a negative number)
            # or added (a positive number)
            correct_by = dict([(j, opt_sm.count(j)) for j in set(opt_sm)])
            for n in cur_sm:
                correct_by[n] -= 1
            correct_in = random.choice([tmpf, tmpm])
            to_add, to_rem = [], []
            for num, amount in correct_by.items():
                if amount > 0:
                    to_add.extend([num] * amount)
                elif amount < 0:
                    to_rem.extend([num] * abs(amount))
            for add, rem in zip(to_add, to_rem):
                tbc = [a.index for a in correct_in if a.number == rem]
                if len(tbc) == 0:
                    pass
                ai = random.choice(tbc)
                correct_in[ai].number = add

        # Move the contributing apart if any distance is below blmin
        maxl = 0.
        for sv, min_dist in self.get_vectors_below_min_dist(tmpf + tmpm):
            lsv = np.linalg.norm(sv)  # length of shortest vector
            d = [-np.dot(e, sv)] * 2
            d[0] += np.sqrt(np.dot(e, sv)**2 - lsv**2 + min_dist**2)
            d[1] -= np.sqrt(np.dot(e, sv)**2 - lsv**2 + min_dist**2)
            l = sorted([abs(i) for i in d])[0] / 2. + eps
            if l > maxl:
                maxl = l
        tmpf.translate(e * maxl)
        tmpm.translate(-e * maxl)

        # Put the two parts together
        for atom in chain(tmpf, tmpm):
            indi.append(atom)

        parent_message = ':Parents {0} {1}'.format(f.info['confid'],
                                                   m.info['confid'])
        return (self.finalize_individual(indi),
                self.descriptor + parent_message)

    def get_numbers(self, atoms):
        """Returns the atomic numbers of the atoms object using only
        the elements defined in self.elements"""
        ac = atoms.copy()
        if self.elements is not None:
            del ac[[a.index for a in ac
                    if a.symbol in self.elements]]
        return ac.numbers
        
    def get_vectors_below_min_dist(self, atoms):
        """Generator function that returns each vector (between atoms)
        that is shorter than the minimum distance for those atom types
        (set during the initialization in blmin)."""
        norm = np.linalg.norm
        ap = atoms.get_positions()
        an = atoms.numbers
        for i in range(len(atoms)):
            pos = atoms[i].position
            for j, d in enumerate([norm(k - pos) for k in ap[i:]]):
                if d == 0:
                    continue
                min_dist = self.blmin[tuple(sorted((an[i], an[j + i])))]
                if d < min_dist:
                    yield atoms[i].position - atoms[j + i].position, min_dist

    def get_shortest_dist_vector(self, atoms):
        norm = np.linalg.norm
        mind = 10000.
        ap = atoms.get_positions()
        for i in range(len(atoms)):
            pos = atoms[i].position
            for j, d in enumerate([norm(k - pos) for k in ap[i:]]):
                if d == 0:
                    continue
                if d < mind:
                    mind = d
                    lowpair = (i, j + i)
        return atoms[lowpair[0]].position - atoms[lowpair[1]].position
