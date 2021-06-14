"""Classes that determine convergence of an algorithm run
based on population stagnation or max raw score reached"""
from ase.ga import get_raw_score


class Convergence(object):
    """
    Base class for all convergence object to be based on.
    It is necessary to supply the population instance, to be
    able to obtain current and former populations.
    """
    def __init__(self, population_instance):
        self.pop = population_instance
        self.pops = {}

    def converged(self):
        """This function is called to find out if the algorithm
        run has converged, it should return True or False.
        Overwrite this in the inherited class."""
        raise NotImplementedError

    def populate_pops(self, to_gen):
        """Populate the pops dictionary with how the population
        looked after i number of generations."""
        for i in range(to_gen):
            if i not in self.pops.keys():
                self.pops[i] = self.pop.get_population_after_generation(i)


class GenerationRepetitionConvergence(Convergence):
    """Returns True if the latest finished population is stagnated for
       number_of_generations.

    Parameters:

    number_of_generations: int
        How many generations need to be equal before convergence.

    number_of_individuals: int
        How many of the fittest individuals should be included in the
        convergence test. Default is -1 meaning all in the population.

    max_generations: int
        The maximum number of generations the GA is allowed to run.
        Default is indefinite.
    """
    def __init__(self, population_instance, number_of_generations,
                 number_of_individuals=-1, max_generations=100000000):
        Convergence.__init__(self, population_instance)
        self.numgens = number_of_generations
        self.numindis = number_of_individuals
        self.maxgen = max_generations

    def converged(self):
        size = self.pop.pop_size
        cur_gen_num = self.pop.dc.get_generation_number(size)

        if cur_gen_num >= self.maxgen:
            return True

        if cur_gen_num <= 1:
            return False

        cur_pop = self.pop.get_current_population()
        newest = max([i.info['key_value_pairs']['generation']
                      for i in cur_pop[:self.numindis]])
        if newest + self.numgens > cur_gen_num:
            return False

        self.populate_pops(cur_gen_num)

        duplicate_gens = 1
        latest_pop = self.pops[cur_gen_num - 1]
        for i in range(cur_gen_num - 2, -1, -1):
            test_pop = self.pops[i]
            if test_pop[:self.numindis] == latest_pop[:self.numindis]:
                duplicate_gens += 1
            if duplicate_gens >= self.numgens:
                return True
        return False


class RawScoreConvergence(Convergence):
    """Returns True if the supplied max_raw_score has been reached"""
    def __init__(self, population_instance, max_raw_score, eps=1e-3):
        Convergence.__init__(self, population_instance)
        self.max_raw_score = max_raw_score
        self.eps = eps

    def converged(self):
        cur_pop = self.pop.get_current_population()
        if abs(get_raw_score(cur_pop[0]) - self.max_raw_score) <= self.eps:
            return True
        return False

        
class NeverConvergence(object):
    """Test class that never converges."""
    def __init__(self):
        pass
        
    def converged(self):
        return False
