from __future__ import print_function

from collections import defaultdict
from numpy import inf

import ase.db


def analyze(filename, tag='results'):
    energies = defaultdict(list)
    mintimes = defaultdict(lambda: 999999)
    formulas = []
    db = ase.db.connect(filename)
    for row in db.select(sort='formula'):
        if row.formula not in formulas:
            formulas.append(row.formula)
        energies[row.formula].append(row.get('energy', inf))
    emin = {formula: min(energies[formula]) for formula in energies}

    data = defaultdict(list)
    for row in db.select(sort='formula'):
        if row.get('energy', inf) - emin[row.formula] < 0.01:
            t = row.t
            if row.n < 100:
                nsteps = row.n
                mintimes[row.formula] = min(mintimes[row.formula], t)
            else:
                nsteps = 9999
                t = inf
        else:
            nsteps = 9999
            t = inf
        data[row.optimizer].append((nsteps, t))

    print(formulas)

    D = sorted(data.items(), key=lambda x: sum(y[0] for y in x[1]))
    with open(tag + '-iterations.csv', 'w') as f:
        print('optimizer,' + ','.join(formulas), file=f)
        for o, d in D:
            print('{:18},{}'
                  .format(o, ','.join('{:3}'.format(x[0])
                                      if x[0] < 100 else '   '
                                      for x in d)),
                  file=f)

    data = {opt: [(n, t / mintimes[f]) for (n, t), f in zip(x, formulas)]
            for opt, x in data.items()}
    D = sorted(data.items(), key=lambda x: sum(min(y[1], 999) for y in x[1]))
    with open(tag + '-time.csv', 'w') as f:
        print('optimizer,' + ','.join(formulas), file=f)
        for o, d in D:
            print('{:18},{}'
                  .format(o, ','.join('{:8.1f}'.format(x[1])
                                      if x[0] < 100 else '        '
                                      for x in d)),
                  file=f)


if __name__ == '__main__':
    analyze('results.db')
