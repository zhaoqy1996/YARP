import argparse
import traceback
from time import time

import ase.db
import ase.optimize
from ase.calculators.emt import EMT
from ase.io import Trajectory


all_optimizers = ase.optimize.__all__ + ['PreconLBFGS', 'PreconFIRE',
                                         'SciPyFminCG', 'SciPyFminBFGS']


def get_optimizer(name):
    if name.startswith('Precon'):
        import ase.optimize.precon as precon
        return getattr(precon, name)
    if name.startswith('SciPy'):
        import ase.optimize.sciopt as sciopt
        return getattr(sciopt, name)
    return getattr(ase.optimize, name)


class Wrapper:
    def __init__(self, atoms):
        self.t0 = time()
        self.texcl = 0.0
        self.nsteps = 0
        self.atoms = atoms
        self.ready = False
        self.pos = None

    def get_potential_energy(self, force_consistent=False):
        t1 = time()
        e = self.atoms.get_potential_energy(force_consistent)
        t2 = time()
        self.texcl += t2 - t1
        if not self.ready:
            self.nsteps += 1
        self.ready = True
        return e

    def get_forces(self):
        t1 = time()
        f = self.atoms.get_forces()
        t2 = time()
        self.texcl += t2 - t1
        if not self.ready:
            self.nsteps += 1
        self.ready = True
        return f

    def set_positions(self, pos):
        if self.pos is not None and abs(pos - self.pos).max() > 1e-15:
            self.ready = False
            if self.nsteps == 200:
                raise RuntimeError('Did not converge!')

        self.pos = pos
        self.atoms.set_positions(pos)

    def get_positions(self):
        return self.atoms.get_positions()

    @property
    def cell(self):
        return self.atoms.cell

    def get_cell(self, complete=False):
        return self.atoms.get_cell(complete)

    @property
    def pbc(self):
        return self.atoms.pbc

    @property
    def positions(self):
        return self.atoms.positions

    @property
    def constraints(self):
        return self.atoms.constraints

    def copy(self):
        return self.atoms.copy()

    def get_calculator(self):
        return self.atoms.calc

    def __len__(self):
        return len(self.atoms)


def run_test(atoms, optimizer, tag, fmax=0.02):
    wrapper = Wrapper(atoms)
    relax = optimizer(wrapper, logfile=tag + '.log')
    relax.attach(Trajectory(tag + '.traj', 'w', atoms=atoms))

    tincl = -time()
    error = ''

    try:
        relax.run(fmax=fmax, steps=10000000)
    except Exception as x:
        wrapper.nsteps = float('inf')
        error = '{}: {}'.format(x.__class__.__name__, x)
        tb = traceback.format_exc()

        with open(tag + '.err', 'w') as fd:
            fd.write('{}\n{}\n'.format(error, tb))

    tincl += time()

    return error, wrapper.nsteps, wrapper.texcl, tincl


def test_optimizer(systems, optimizer, calculator, prefix='', db=None):
    for name, atoms in systems:
        if db is not None:
            optname = optimizer.__name__
            id = db.reserve(optimizer=optname, name=name)
            if id is None:
                continue
        atoms = atoms.copy()
        tag = '{}{}-{}'.format(prefix, optname, name)
        atoms.calc = calculator(txt=tag + '.txt')
        error, nsteps, texcl, tincl = run_test(atoms, optimizer, tag)

        if db is not None:
            db.write(atoms,
                     id=id,
                     optimizer=optname,
                     name=name,
                     error=error,
                     n=nsteps,
                     t=texcl,
                     T=tincl)


def main():
    parser = argparse.ArgumentParser(
        description='Test ASE optimizers')

    parser.add_argument('systems')
    parser.add_argument('optimizer', nargs='*',
                        help='Optimizer name.')

    args = parser.parse_args()

    systems = [(row.name, row.toatoms())
               for row in ase.db.connect(args.systems).select()]

    db = ase.db.connect('results.db')

    if not args.optimizer:
        args.optimizer = all_optimizers

    for opt in args.optimizer:
        print(opt)
        optimizer = get_optimizer(opt)
        test_optimizer(systems, optimizer, EMT, db=db)


if __name__ == '__main__':
    main()
