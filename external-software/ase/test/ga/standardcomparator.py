from ase.ga.standard_comparators import (InteratomicDistanceComparator,
                                         EnergyComparator, RawScoreComparator,
                                         SequentialComparator)
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.ga import set_raw_score


a1 = Atoms('AgAgAg', positions=[[0, 0, 0], [1.5, 0, 0], [1.5, 1.5, 0]])
a2 = Atoms('AgAgAg', positions=[[0, 0, 0], [1.4, 0, 0], [1.5, 1.5, 0]])

e1 = 1.0
e2 = 0.8

a1.set_calculator(SinglePointCalculator(a1, energy=e1))
a2.set_calculator(SinglePointCalculator(a2, energy=e2))

comp1 = InteratomicDistanceComparator(n_top=3,
                                      pair_cor_cum_diff=0.03,
                                      pair_cor_max=0.7,
                                      dE=0.3)
assert comp1.looks_like(a1, a2)


comp2 = InteratomicDistanceComparator(n_top=3,
                                      pair_cor_cum_diff=0.03,
                                      pair_cor_max=0.7,
                                      dE=0.15)
assert not comp2.looks_like(a1, a2)


comp3 = InteratomicDistanceComparator(n_top=3,
                                      pair_cor_cum_diff=0.02,
                                      pair_cor_max=0.7,
                                      dE=0.3)
assert not comp3.looks_like(a1, a2)


hard_E_comp = EnergyComparator(dE=1.0)
assert hard_E_comp.looks_like(a1, a2)

soft_E_comp = EnergyComparator(dE=.01)
assert not soft_E_comp.looks_like(a1, a2)


set_raw_score(a1, .1)
set_raw_score(a2, .27)

rs_comp = RawScoreComparator(0.15)
assert not rs_comp.looks_like(a1, a2)


comp1 = SequentialComparator([hard_E_comp, rs_comp], [0, 0])
assert not comp1.looks_like(a1, a2)

comp2 = SequentialComparator([hard_E_comp, rs_comp], [0, 1])
assert comp2.looks_like(a1, a2)
