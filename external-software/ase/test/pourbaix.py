from __future__ import print_function
from unittest import SkipTest

raise SkipTest('WIP')

import numpy as np

import ase.db
from ase.phasediagram import bisect, Pourbaix, solvated

if 0:
    N = 80
    A = np.zeros((N, N), int)
    A[:] = -1

    def f(x, y):
        dmin = 100
        for i, (a, b) in enumerate([(0, 0), (0, 2), (1, 1)]):
            d = (x - a)**2 + (y - b)**2
            if d < dmin:
                dmin = d
                imin = i
        return imin

    bisect(A, np.linspace(0, 2, N), np.linspace(0, 2, N), f)
    print(A)

    import matplotlib.pyplot as plt
    plt.imshow(A)
    plt.show()


if 0:
    con = ase.db.connect('cubic_perovskites.db')
    references = [(row.count_atoms(), row.energy)
                  for row in con.select('reference')]
    std = {}
    for count, energy in references:
        if len(count) == 1:
            symbol, n = list(count.items())[0]
            assert symbol not in std
            std[symbol] = energy / n

    std['O'] += 2.46

    refs = []
    for refcount, energy in references:
        for symbol, n in refcount.items():
            energy -= n * std[symbol]
        if list(refcount) == ['O']:
            energy = 0.0
        refs.append((refcount, energy))
if 1:
    refs = [#({'O': 1}, 0.0),
            ('O4Ti2', -17.511826939900217),
            ('Sr4O4', -20.474907588620653),
            ('Sr4', 0.0),
            ('Ti2', 0.0)]
else:
    refs = [({'O': 1}, 0.0),
            ({'Zn': 1}, 0.0),
            ({'Zn': 2, 'O': 2}, -5.33991412178575),
            ({'Zn': 4, 'O': 8}, -7.594)]


pb = Pourbaix(refs + solvated('SrTi'), Sr=1, Ti=1, O=3)
#pb = Pourbaix(refs, Zn=1, O=1)
print(pb.decompose(0, 9))
pH = np.linspace(-1, 15, 17)
if 0:
    d, names = pb.diagram([0], pH)
    print(d)
    print('\n'.join(names))
U = np.linspace(-2, 2, 5)
if 0:
    d, names = pb.diagram(U, [0])
    for i, u in zip(d, U):
        print(u, names[i])

if 1:
    U = np.linspace(-3, 3, 200)
    pH = np.linspace(-1, 15, 300)
    d, names = pb.diagram(U, pH, plot=True)
