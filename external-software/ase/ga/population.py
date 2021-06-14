""" Implementaiton of a population for maintaining a GA population and
proposing structures to pair. """
from random import randrange, random
from math import tanh, sqrt, exp
from operator import itemgetter
import numpy as np

from ase.db.core import now
from ase.ga import get_raw_score


def count_looks_like(a, all_cand, comp):
    """Utility method for counting occurrences."""
    n = 0
    for b in all_cand:
        if a.info['confid'] == b.info['confid']:
            continue
        if comp.looks_like(a, b):
            n += 1
    return n


class Population(object):
    """Population class which maintains the current population
    and proposes which candidates to pair together.

    Parameters:

    data_connection: DataConnection object
        Bla bla bla.

    population_size: int
        The number of candidates in the population.

    comparator: Comparator object
        this will tell if two configurations are equal.
        Default compare atoms objects directly.

    logfile: str
        Text file that contains information about the population
        The format is::

            timestamp: generation(if available): id1,id2,id3...

        Using this file greatly speeds up convergence checks.
        Default None meaning that no file is written.

    use_extinct: boolean
        Set this to True if mass extinction and the extinct key
        are going to be used. Default is False.
    """
    def __init__(self, data_connection, population_size,
                 comparator=None, logfile=None, use_extinct=False):
        self.dc = data_connection
        self.pop_size = population_size
        if comparator is None:
            from ase.ga.standard_comparators import AtomsComparator
            comparator = AtomsComparator()
        self.comparator = comparator
        self.logfile = logfile
        self.use_extinct = use_extinct
        self.pop = []
        self.pairs = None
        self.all_cand = None
        self.__initialize_pop__()

    def __initialize_pop__(self):
        """ Private method that initalizes the population when
            the population is created. """

        # Get all relaxed candidates from the database
        ue = self.use_extinct
        all_cand = self.dc.get_all_relaxed_candidates(use_extinct=ue)
        all_cand.sort(key=lambda x: x.info['key_value_pairs']['raw_score'],
                      reverse=True)
        # all_cand.sort(key=lambda x: x.get_potential_energy())

        # Fill up the population with the self.pop_size most stable
        # unique candidates.
        i = 0
        while i < len(all_cand) and len(self.pop) < self.pop_size:
            c = all_cand[i]
            i += 1
            eq = False
            for a in self.pop:
                if self.comparator.looks_like(a, c):
                    eq = True
                    break
            if not eq:
                self.pop.append(c)

        for a in self.pop:
            a.info['looks_like'] = count_looks_like(a, all_cand,
                                                    self.comparator)

        self.all_cand = all_cand
        self.__calc_participation__()

    def __calc_participation__(self):
        """ Determines, from the database, how many times each
            candidate has been used to generate new candidates. """
        (participation, pairs) = self.dc.get_participation_in_pairing()
        for a in self.pop:
            if a.info['confid'] in participation.keys():
                a.info['n_paired'] = participation[a.info['confid']]
            else:
                a.info['n_paired'] = 0
        self.pairs = pairs

    def update(self, new_cand=None):
        """ New candidates can be added to the database
            after the population object has been created.
            This method extracts these new candidates from the
            database and includes them in the population. """

        if len(self.pop) == 0:
            self.__initialize_pop__()

        if new_cand is None:
            ue = self.use_extinct
            new_cand = self.dc.get_all_relaxed_candidates(only_new=True,
                                                          use_extinct=ue)

        for a in new_cand:
            self.__add_candidate__(a)
            self.all_cand.append(a)
        self.__calc_participation__()
        self._write_log()

    def get_current_population(self):
        """ Returns a copy of the current population. """
        self.update()
        return [a.copy() for a in self.pop]

    def get_population_after_generation(self, gen):
        """ Returns a copy of the population as it where
        after generation gen"""
        if self.logfile is not None:
            f = open(self.logfile, 'r')
            gens = {}
            for l in f:
                _, no, popul = l.split(':')
                gens[int(no)] = [int(i) for i in popul.split(',')]
            f.close()
            return [c.copy() for c in self.all_cand[::-1]
                    if c.info['relax_id'] in gens[gen]]

        all_candidates = [c for c in self.all_cand
                          if c.info['key_value_pairs']['generation'] <= gen]
        cands = [all_candidates[0]]
        for b in all_candidates:
            if b not in cands:
                for a in cands:
                    if self.comparator.looks_like(a, b):
                        break
                else:
                    cands.append(b)
        pop = cands[:self.pop_size]
        return [a.copy() for a in pop]

    def __add_candidate__(self, a):
        """ Adds a single candidate to the population. """

        # check if the structure is too low in raw score
        raw_score_a = get_raw_score(a)
        raw_score_worst = get_raw_score(self.pop[-1])
        if raw_score_a < raw_score_worst \
                and len(self.pop) == self.pop_size:
            return

        # check if the new candidate should
        # replace a similar structure in the population
        for (i, b) in enumerate(self.pop):
            if self.comparator.looks_like(a, b):
                if get_raw_score(b) < raw_score_a:
                    del self.pop[i]
                    a.info['looks_like'] = count_looks_like(a,
                                                            self.all_cand,
                                                            self.comparator)
                    self.pop.append(a)
                    self.pop.sort(key=lambda x: get_raw_score(x),
                                  reverse=True)
                return

        # the new candidate needs to be added, so remove the highest
        # energy one
        if len(self.pop) == self.pop_size:
            del self.pop[-1]

        # add the new candidate
        a.info['looks_like'] = count_looks_like(a,
                                                self.all_cand,
                                                self.comparator)
        self.pop.append(a)
        self.pop.sort(key=lambda x: get_raw_score(x), reverse=True)

    def __get_fitness__(self, indecies, with_history=True):
        """Calculates the fitness using the formula from
            L.B. Vilhelmsen et al., JACS, 2012, 134 (30), pp 12807-12816

        Sign change on the fitness compared to the formulation in the
        abovementioned paper due to maximizing raw_score instead of
        minimizing energy. (Set raw_score=-energy to optimize the energy)
        """

        scores = [get_raw_score(x) for x in self.pop]
        min_s = min(scores)
        max_s = max(scores)
        T = min_s - max_s
        if isinstance(indecies, int):
            indecies = [indecies]

        f = [0.5 * (1. - tanh(2. * (scores[i] - max_s) / T - 1.))
             for i in indecies]
        if with_history:
            M = [float(self.pop[i].info['n_paired']) for i in indecies]
            L = [float(self.pop[i].info['looks_like']) for i in indecies]
            f = [f[i] * 1. / sqrt(1. + M[i]) * 1. / sqrt(1. + L[i])
                 for i in range(len(f))]
        return f

    def get_two_candidates(self, with_history=True):
        """ Returns two candidates for pairing employing the
            fitness criteria from
            L.B. Vilhelmsen et al., JACS, 2012, 134 (30), pp 12807-12816
            and the roulete wheel selection scheme described in
            R.L. Johnston Dalton Transactions,
            Vol. 22, No. 22. (2003), pp. 4193-4207
        """

        if len(self.pop) < 2:
            self.update()

        if len(self.pop) < 2:
            return None

        fit = self.__get_fitness__(range(len(self.pop)), with_history)
        fmax = max(fit)
        c1 = self.pop[0]
        c2 = self.pop[0]
        used_before = False
        while c1.info['confid'] == c2.info['confid'] and not used_before:
            nnf = True
            while nnf:
                t = randrange(0, len(self.pop), 1)
                if fit[t] > random() * fmax:
                    c1 = self.pop[t]
                    nnf = False
            nnf = True
            while nnf:
                t = randrange(0, len(self.pop), 1)
                if fit[t] > random() * fmax:
                    c2 = self.pop[t]
                    nnf = False

            c1id = c1.info['confid']
            c2id = c2.info['confid']
            used_before = (min([c1id, c2id]), max([c1id, c2id])) in self.pairs
        return (c1.copy(), c2.copy())

    def get_one_candidate(self, with_history=True):
        """Returns one candidate for mutation employing the
        fitness criteria from
        L.B. Vilhelmsen et al., JACS, 2012, 134 (30), pp 12807-12816
        and the roulete wheel selection scheme described in
        R.L. Johnston Dalton Transactions,
        Vol. 22, No. 22. (2003), pp. 4193-4207
        """
        if len(self.pop) < 1:
            self.update()

        if len(self.pop) < 1:
            return None

        fit = self.__get_fitness__(range(len(self.pop)), with_history)
        fmax = max(fit)
        nnf = True
        while nnf:
            t = randrange(0, len(self.pop), 1)
            if fit[t] > random() * fmax:
                c1 = self.pop[t]
                nnf = False

        return c1.copy()

    def _write_log(self):
        """Writes the population to a logfile.

        The format is::

            timestamp: generation(if available): id1,id2,id3..."""
        if self.logfile is not None:
            ids = [str(a.info['relax_id']) for a in self.pop]
            if ids != []:
                try:
                    gen_nums = [c.info['key_value_pairs']['generation']
                                for c in self.all_cand]
                    max_gen = max(gen_nums)
                except KeyError:
                    max_gen = ' '
                f = open(self.logfile, 'a')
                f.write('{time}: {gen}: {pop}\n'.format(time=now(),
                                                        pop=','.join(ids),
                                                        gen=max_gen))
                f.close()

    def is_uniform(self, func, min_std, pop=None):
        """Tests whether the current population is uniform or diverse.
        Returns True if uniform, False otherwise.

        Parameters:

        func: function
            that takes one argument an atoms object and returns a value that
            will be used for testing against the rest of the population.

        min_std: int or float
            The minimum standard deviation, if the population has a lower
            std dev it is uniform.

        pop: list, optional
            use this list of Atoms objects instead of the current population.
        """
        if pop is None:
            pop = self.pop
        vals = [func(a) for a in pop]
        stddev = np.std(vals)
        if stddev < min_std:
            return True
        return False

    def mass_extinction(self, ids):
        """Kills every candidate in the database with gaid in the
        supplied list of ids. Typically used on the main part of the current
        population if the diversity is to small.

        Parameters:

        ids: list
            list of ids of candidates to be killed.

        """
        for confid in ids:
            self.dc.kill_candidate(confid)
        self.pop = []


class RandomPopulation(Population):
    def __init__(self, data_connection, population_size,
                 comparator=None, logfile=None, exclude_used_pairs=False,
                 bad_candidates=0, use_extinct=False):
        self.exclude_used_pairs = exclude_used_pairs
        self.bad_candidates = bad_candidates
        Population.__init__(self, data_connection, population_size,
                            comparator, logfile, use_extinct)

    def __initialize_pop__(self):
        """ Private method that initalizes the population when
            the population is created. """

        # Get all relaxed candidates from the database
        ue = self.use_extinct
        all_cand = self.dc.get_all_relaxed_candidates(use_extinct=ue)
        all_cand.sort(key=lambda x: get_raw_score(x), reverse=True)
        # all_cand.sort(key=lambda x: x.get_potential_energy())

        if len(all_cand) > 0:
            # Fill up the population with the self.pop_size most stable
            # unique candidates.
            ratings = []
            best_raw = get_raw_score(all_cand[0])
            i = 0
            while i < len(all_cand):
                c = all_cand[i]
                i += 1
                eq = False
                for a in self.pop:
                    if self.comparator.looks_like(a, c):
                        eq = True
                        break
                if not eq:
                    if len(self.pop) < self.pop_size - self.bad_candidates:
                        self.pop.append(c)
                    else:
                        exp_fact = exp(get_raw_score(c) / best_raw)
                        ratings.append([c, (exp_fact - 1) * random()])
            ratings.sort(key=itemgetter(1), reverse=True)

            for i in range(self.bad_candidates):
                self.pop.append(ratings[i][0])

        for a in self.pop:
            a.info['looks_like'] = count_looks_like(a, all_cand,
                                                    self.comparator)

        self.all_cand = all_cand
        self.__calc_participation__()

    def update(self):
        """ The update method in Population will add to the end of
        the population, that can't be used here since we might have
        bad candidates that need to stay in the population, therefore
        just recalc the population every time. """

        self.pop = []
        self.__initialize_pop__()

        self._write_log()

    def get_one_candidate(self):
        """Returns one candidates at random."""
        if len(self.pop) < 1:
            self.update()

        if len(self.pop) < 1:
            return None

        t = randrange(0, len(self.pop), 1)
        c = self.pop[t]

        return c.copy()

    def get_two_candidates(self):
        """Returns two candidates at random."""
        if len(self.pop) < 2:
            self.update()

        if len(self.pop) < 2:
            return None

        c1 = self.pop[0]
        c2 = self.pop[0]
        used_before = False
        while c1.info['confid'] == c2.info['confid'] and not used_before:
            t = randrange(0, len(self.pop), 1)
            c1 = self.pop[t]
            t = randrange(0, len(self.pop), 1)
            c2 = self.pop[t]

            c1id = c1.info['confid']
            c2id = c2.info['confid']
            used_before = (tuple(sorted([c1id, c2id])) in self.pairs and
                           self.exclude_used_pairs)
        return (c1.copy(), c2.copy())


class FitnessSharingPopulation(Population):
    """ Fitness sharing population that penalizes structures if they are
    too similar. This is determined by a distance measure

    Parameters:

    comp_key: string
        Key where the distance measure can be found in the
        atoms.info['key_value_pairs'] dictionary.

    threshold: float or int
        Value above which no penalization of the fitness takes place

    alpha_sh: float or int
        Determines the shape of the sharing function.
        Default is 1, which gives a linear sharing function.

    """
    def __init__(self, data_connection, population_size,
                 comp_key, threshold, alpha_sh=1.,
                 comparator=None, logfile=None, use_extinct=False):
        self.comp_key = comp_key
        self.dt = threshold  # dissimilarity threshold
        self.alpha_sh = alpha_sh
        self.fit_scaling = 1.

        self.sh_cache = dict()

        Population.__init__(self, data_connection, population_size,
                            comparator, logfile, use_extinct)

    def __get_fitness__(self, candidates):
        """Input should be sorted according to raw_score."""
        max_s = get_raw_score(candidates[0])
        min_s = get_raw_score(candidates[-1])
        T = min_s - max_s

        shared_fit = []
        for c in candidates:
            sc = get_raw_score(c)
            obj_fit = 0.5 * (1. - tanh(2. * (sc - max_s) / T - 1.))
            m = 1.
            ck = c.info['key_value_pairs'][self.comp_key]
            for other in candidates:
                if other != c:
                    name = tuple(sorted([c.info['confid'],
                                         other.info['confid']]))
                    if name not in self.sh_cache:
                        ok = other.info['key_value_pairs'][self.comp_key]
                        d = abs(ck - ok)
                        if d < self.dt:
                            v = 1 - (d / self.dt)**self.alpha_sh
                            self.sh_cache[name] = v
                        else:
                            self.sh_cache[name] = 0
                    m += self.sh_cache[name]

            shf = (obj_fit ** self.fit_scaling) / m
            shared_fit.append(shf)
        return shared_fit

    def update(self):
        """ The update method in Population will add to the end of
        the population, that can't be used here since the shared fitness
        will change for all candidates when new are added, therefore
        just recalc the population every time. """

        self.pop = []
        self.__initialize_pop__()

        self._write_log()

    def __initialize_pop__(self):
        # Get all relaxed candidates from the database
        ue = self.use_extinct
        all_cand = self.dc.get_all_relaxed_candidates(use_extinct=ue)
        all_cand.sort(key=lambda x: get_raw_score(x), reverse=True)

        if len(all_cand) > 0:
            shared_fit = self.__get_fitness__(all_cand)
            all_sorted = list(zip(*sorted(zip(shared_fit, all_cand),
                                          reverse=True)))[1]

            # Fill up the population with the self.pop_size most stable
            # unique candidates.
            i = 0
            while i < len(all_sorted) and len(self.pop) < self.pop_size:
                c = all_sorted[i]
                i += 1
                eq = False
                for a in self.pop:
                    if self.comparator.looks_like(a, c):
                        eq = True
                        break
                if not eq:
                    self.pop.append(c)

            for a in self.pop:
                a.info['looks_like'] = count_looks_like(a, all_cand,
                                                        self.comparator)
        self.all_cand = all_cand

    def get_two_candidates(self):
        """ Returns two candidates for pairing employing the
            fitness criteria from
            L.B. Vilhelmsen et al., JACS, 2012, 134 (30), pp 12807-12816
            and the roulete wheel selection scheme described in
            R.L. Johnston Dalton Transactions,
            Vol. 22, No. 22. (2003), pp. 4193-4207
        """

        if len(self.pop) < 2:
            self.update()

        if len(self.pop) < 2:
            return None

        fit = self.__get_fitness__(self.pop)
        fmax = max(fit)
        c1 = self.pop[0]
        c2 = self.pop[0]
        while c1.info['confid'] == c2.info['confid']:
            nnf = True
            while nnf:
                t = randrange(0, len(self.pop), 1)
                if fit[t] > random() * fmax:
                    c1 = self.pop[t]
                    nnf = False
            nnf = True
            while nnf:
                t = randrange(0, len(self.pop), 1)
                if fit[t] > random() * fmax:
                    c2 = self.pop[t]
                    nnf = False

        return (c1.copy(), c2.copy())


class RankFitnessPopulation(Population):
    """ Ranks the fitness relative to set variable to flatten the surface
        in a certain direction such that mating across variable is equally
        likely irrespective of raw_score.

        Parameters:

        variable_function: function
            A function that takes as input an Atoms object and returns
            the variable that differentiates the ranks.

        exp_function: boolean
            If True use an exponential function for ranking the fitness.
            If False use the same as in Population. Default True.

        exp_prefactor: float
            The prefactor used in the exponential fitness scaling function.
            Default 0.5
    """
    def __init__(self, data_connection, population_size, variable_function,
                 comparator=None, logfile=None, use_extinct=False,
                 exp_function=True, exp_prefactor=0.5):
        self.exp_function = exp_function
        self.exp_prefactor = exp_prefactor
        self.vf = variable_function
        # The current fitness is set at each update of the population
        self.current_fitness = None

        Population.__init__(self, data_connection, population_size,
                            comparator, logfile, use_extinct)

    def get_rank(self, rcand, key=None):
        # Set the initial order of the candidates, will need to
        # be returned in this order at the end of ranking.
        ordered = list(zip(range(len(rcand)), rcand))

        # Niche and rank candidates.
        rec_nic = []
        rank_fit = []
        for o, c in ordered:
            if o not in rec_nic:
                ntr = []
                ce1 = self.vf(c)
                rec_nic.append(o)
                ntr.append([o, c])
                for oother, cother in ordered:
                    if oother not in rec_nic:
                        ce2 = self.vf(cother)
                        if ce1 == ce2:
                            # put the now processed in oother
                            # in rec_nic as well
                            rec_nic.append(oother)
                            ntr.append([oother, cother])
                # Each niche is sorted according to raw_score and
                # assigned a fitness according to the ranking of
                # the candidates
                ntr.sort(key=lambda x: x[1].info['key_value_pairs'][key],
                         reverse=True)
                start_rank = -1
                cor = 0
                for on, cn in ntr:
                    rank = start_rank - cor
                    rank_fit.append([on, cn, rank])
                    cor += 1
        # The original order is reformed
        rank_fit.sort(key=itemgetter(0), reverse=False)
        return np.array(list(zip(*rank_fit))[2])

    def __get_fitness__(self, candidates):
        expf = self.exp_function
        rfit = self.get_rank(candidates, key='raw_score')

        if not expf:
            rmax = max(rfit)
            rmin = min(rfit)
            T = rmin - rmax
            # If using obj_rank probability, must have non-zero T val.
            # pop_size must be greater than number of permutations.
            # We test for this here
            msg = "Equal fitness for best and worst candidate in the "
            msg += "population! Fitness scaling is impossible! "
            msg += "Try with a larger population."
            assert T != 0., msg
            return 0.5 * (1. - np.tanh(2. * (rfit - rmax) / T - 1.))
        else:
            return self.exp_prefactor ** (-rfit - 1)

    def update(self):
        """ The update method in Population will add to the end of
        the population, that can't be used here since the fitness
        will potentially change for all candidates when new are added,
        therefore just recalc the population every time. """

        self.pop = []
        self.__initialize_pop__()
        self.current_fitness = self.__get_fitness__(self.pop)

        self._write_log()

    def __initialize_pop__(self):
        # Get all relaxed candidates from the database
        ue = self.use_extinct
        all_cand = self.dc.get_all_relaxed_candidates(use_extinct=ue)
        all_cand.sort(key=lambda x: get_raw_score(x), reverse=True)

        if len(all_cand) > 0:
            fitf = self.__get_fitness__(all_cand)
            all_sorted = list(zip(fitf, all_cand))
            all_sorted.sort(key=itemgetter(0), reverse=True)
            sort_cand = []
            for _, t2 in all_sorted:
                sort_cand.append(t2)
            all_sorted = sort_cand

            # Fill up the population with the self.pop_size most stable
            # unique candidates.
            i = 0
            while i < len(all_sorted) and len(self.pop) < self.pop_size:
                c = all_sorted[i]
                c_vf = self.vf(c)
                i += 1
                eq = False
                for a in self.pop:
                    a_vf = self.vf(a)
                    # Only run comparator if the variable_function (self.vf)
                    # returns the same. If it returns something different the
                    # candidates are inherently different.
                    # This is done to speed up.
                    if a_vf == c_vf:
                        if self.comparator.looks_like(a, c):
                            eq = True
                            break
                if not eq:
                    self.pop.append(c)
        self.all_cand = all_cand

    def get_two_candidates(self):
        """ Returns two candidates for pairing employing the
            roulete wheel selection scheme described in
            R.L. Johnston Dalton Transactions,
            Vol. 22, No. 22. (2003), pp. 4193-4207
        """

        if len(self.pop) < 2:
            self.update()

        if len(self.pop) < 2:
            return None

        # Use saved fitness
        fit = self.current_fitness
        fmax = max(fit)
        c1 = self.pop[0]
        c2 = self.pop[0]
        while c1.info['confid'] == c2.info['confid']:
            nnf = True
            while nnf:
                t = randrange(0, len(self.pop), 1)
                if fit[t] > random() * fmax:
                    c1 = self.pop[t]
                    nnf = False
            nnf = True
            while nnf:
                t = randrange(0, len(self.pop), 1)
                if fit[t] > random() * fmax:
                    c2 = self.pop[t]
                    nnf = False

        return (c1.copy(), c2.copy())


class MultiObjectivePopulation(RankFitnessPopulation):
    """ Allows for assignment of fitness based on a set of two variables
        such that fitness is ranked according to a Pareto-front of
        non-dominated candidates.

    Parameters
    ----------
        abs_data: list
            Set of key_value_pairs in atoms object for which fitness should
            be assigned based on absolute value.

        rank_data: list
            Set of key_value_pairs in atoms object for which data should
            be ranked in order to ascribe fitness.

        variable_function: function
            A function that takes as input an Atoms object and returns
            the variable that differentiates the ranks. Only use if
            data is ranked.

        exp_function: boolean
            If True use an exponential function for ranking the fitness.
            If False use the same as in Population. Default True.

        exp_prefactor: float
            The prefactor used in the exponential fitness scaling function.
            Default 0.5

    """

    def __init__(self, data_connection, population_size,
                 variable_function=None, comparator=None, logfile=None,
                 use_extinct=False, abs_data=None, rank_data=None,
                 exp_function=True, exp_prefactor=0.5):
        # The current fitness is set at each update of the population
        self.current_fitness = None

        if rank_data is None:
            rank_data = []
        self.rank_data = rank_data

        if abs_data is None:
            abs_data = []
        self.abs_data = abs_data

        RankFitnessPopulation.__init__(self, data_connection, population_size,
                                       variable_function, comparator, logfile,
                                       use_extinct, exp_function,
                                       exp_prefactor)

    def get_nonrank(self, nrcand, key=None):
        """"Returns a list of fitness values."""
        nrc_list = []
        for nrc in nrcand:
            nrc_list.append(nrc.info['key_value_pairs'][key])
        return nrc_list

    def __get_fitness__(self, candidates):
        # There are no defaults set for the datasets to be
        # used in this function, as such we test that the
        # user has specified at least two here.
        msg = "This is a multi-objective fitness function"
        msg += " so there must be at least two datasets"
        msg += " stated in the rank_data and abs_data variables"
        assert len(self.rank_data) + len(self.abs_data) >= 2, msg

        expf = self.exp_function

        all_fitnesses = []
        used = set()
        for rd in self.rank_data:
            used.add(rd)
            # Build ranked fitness based on rd
            all_fitnesses.append(self.get_rank(candidates, key=rd))

        for d in self.abs_data:
            if d not in used:
                used.add(d)
                # Build fitness based on d
                all_fitnesses.append(self.get_nonrank(candidates, key=d))

        # Set the initial order of the ranks, will need to
        # be returned in this order at the end.
        fordered = list(zip(range(len(all_fitnesses[0])), *all_fitnesses))
        mvf_rank = -1  # Start multi variable rank at -1.
        rec_vrc = []  # A record of already ranked candidates.
        mvf_list = []  # A list for all candidate ranks.
        # Sort by raw_score_1 in case this is different from
        # the stored raw_score() variable that all_cands are
        # sorted by.
        fordered.sort(key=itemgetter(1), reverse=True)
        # Niche candidates with equal or better raw_score to
        # current candidate.
        for a in fordered:
            order, rest = a[0], a[1:]
            if order not in rec_vrc:
                pff = []
                pff2 = []
                rec_vrc.append(order)
                pff.append((order, rest))
                for b in fordered:
                    border, brest = b[0], b[1:]
                    if border not in rec_vrc:
                        if np.any(np.array(brest) >= np.array(rest)):
                            pff.append((border, brest))
                # Remove any candidate from pff list that is dominated
                # by another in the list.
                for na in pff:
                    norder, nrest = na[0], na[1:]
                    dom = False
                    for nb in pff:
                        nborder, nbrest = nb[0], nb[1:]
                        if norder != nborder:
                            if np.all(np.array(nbrest) > np.array(nrest)):
                                dom = True
                    if not dom:
                        pff2.append((norder, nrest))
                # Assign pareto rank from -1 to -N niches.
                for ffa in pff2:
                    fforder, ffrest = ffa[0], ffa[1:]
                    rec_vrc.append(fforder)
                    mvf_list.append((fforder, mvf_rank, ffrest))
                mvf_rank = mvf_rank - 1
        # The original order is reformed
        mvf_list.sort(key=itemgetter(0), reverse=False)
        rfro = np.array(list(zip(*mvf_list))[1])

        if not expf:
            rmax = max(rfro)
            rmin = min(rfro)
            T = rmin - rmax
            # If using obj_rank probability, must have non-zero T val.
            # pop_size must be greater than number of permutations.
            # We test for this here
            msg = "Equal fitness for best and worst candidate in the "
            msg += "population! Fitness scaling is impossible! "
            msg += "Try with a larger population."
            assert T != 0., msg
            return 0.5 * (1. - np.tanh(2. * (rfro - rmax) / T - 1.))
        else:
            return self.exp_prefactor ** (-rfro - 1)

    def __initialize_pop__(self):
        # Get all relaxed candidates from the database
        ue = self.use_extinct
        all_cand = self.dc.get_all_relaxed_candidates(use_extinct=ue)
        all_cand.sort(key=lambda x: get_raw_score(x), reverse=True)

        if len(all_cand) > 0:
            fitf = self.__get_fitness__(all_cand)
            all_sorted = list(zip(fitf, all_cand))
            all_sorted.sort(key=itemgetter(0), reverse=True)
            sort_cand = []
            for _, t2 in all_sorted:
                sort_cand.append(t2)
            all_sorted = sort_cand

            # Fill up the population with the self.pop_size most stable
            # unique candidates.
            i = 0
            while i < len(all_sorted) and len(self.pop) < self.pop_size:
                c = all_sorted[i]
                # Use variable_function to decide whether to run comparator
                # if the function has been defined by the user. This does not
                # need to be dependent on using the rank_data function.
                if self.vf is not None:
                    c_vf = self.vf(c)
                i += 1
                eq = False
                for a in self.pop:
                    if self.vf is not None:
                        a_vf = self.vf(a)
                        # Only run comparator if the variable_function
                        # (self.vf) returns the same. If it returns something
                        # different the candidates are inherently different.
                        # This is done to speed up.
                        if a_vf == c_vf:
                            if self.comparator.looks_like(a, c):
                                eq = True
                                break
                    else:
                        if self.comparator.looks_like(a, c):
                            eq = True
                            break
                if not eq:
                    self.pop.append(c)
        self.all_cand = all_cand
