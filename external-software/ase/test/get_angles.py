from ase.build import graphene_nanoribbon
import numpy as np

g = graphene_nanoribbon(3, 2, type="zigzag", vacuum=5)

test_set = [[0, 1, x] for x in range(2, len(g))]

manual_results = [g.get_angle(a1, a2, a3, mic=True) 
                  for a1, a2, a3 in test_set]

set_results = g.get_angles(test_set, mic=True)

assert(np.allclose(manual_results, set_results))
