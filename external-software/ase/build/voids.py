from __future__ import print_function
import numpy as np
from ase import Atom, Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import FIRE
from ase.data import atomic_numbers
from ase.data.vdw import vdw_radii


class RepulsivePotential(Calculator):
    """Purely repulsive potential (Gaussian)"""
    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms, properties, changes):
        radii_a = np.array([
                vdw_radii[atomic_numbers[a.symbol]] for a in atoms])
        self.radii_a = radii_a

        # last atom is the moving one
        energy = 0.0
        forces = np.zeros((len(atoms), 3))
        for a in range(len(atoms) - 1):
            d_c = atoms.get_distance(a, -1, mic=True, vector=True)
            d = np.linalg.norm(d_c)

            sigma2 = radii_a[a]**2 / (2 * np.log(2))
            pre = np.exp(- d**2 / (2 * sigma2))
            energy += pre
            forces[-1] += pre * d_c / sigma2
            
        self.results['energy'] = energy
        self.results['forces'] = forces


def voids(atoms_in):
    """Find location and size of voids in a given structure.

    Returns the voids as 'X' atoms. The atoms' charge is misused
    to contain the voids' radius.
    """
    
    trials = 6  # XXX do not hardwire

    atoms = atoms_in.copy()
    # append moving atom
    atoms.append(Atom('X'))
    atoms.set_calculator(RepulsivePotential())

    voids_a = Atoms()
    voids_a.set_cell(atoms.get_cell())
    voids_a.set_pbc(atoms.get_pbc())

    positions = atoms.get_positions()
    for pos in positions[:-1]:
        for c in range(trials):
            positions[-1] = pos + 0.1 * np.random.uniform(-1, 1, size=3)
            atoms.set_positions(positions)
            
            # XXX do not hardwire
            relax = FIRE(atoms,
                         logfile=None
                         )
            # XXX do not hardwire
            relax.run(fmax=0.001, steps=100)
                
            # get minimal distance
            Rmin = 100000
            for b in range(len(atoms) - 1):
                R = atoms.get_distance(b, -1, mic=True)
                if R < Rmin:
                    Rmin = R

            # check if new or better
            voids_a.append(Atom('X',
                                atoms.get_positions()[-1],
                                charge=Rmin))
            voids_a.set_positions(voids_a.get_positions(wrap=True))

            remove = []
            last = len(voids_a) - 1
            for ia, a in enumerate(voids_a[:-1]):
                d = voids_a.get_distance(ia, -1, mic=True)
                if d < a.charge or d < Rmin:
                    if a.charge > Rmin:
                        remove.append(last)
                    else:
                        remove.append(ia)
            remove.sort()
            if last not in remove:
                p = voids_a.get_positions()[-1]
                print('found new void at [%g,%g,%g], R=%g' %
                      (p[0], p[1], p[2], Rmin))
            for a in remove[::-1]:
                if a != last:
                    p = voids_a.get_positions()[a]
                    print('removing void at [%g,%g,%g], R=%g' %
                          (p[0], p[1], p[2], voids_a[a].charge))
                voids_a.pop(a)

    return voids_a
