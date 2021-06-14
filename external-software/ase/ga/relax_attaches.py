""" An object which can be associated with a local relaxation in order
to make the relaxations run more smoothly."""
from math import sqrt


class VariansBreak(object):

    """ Helper class which can be attached to a structure optimization,
        in order to terminale stalling calculations.

        Parameters:

        atoms: Atoms object being optimized
        dyn: The relaxation object being used
        min_stdev: The limiting std. deviation in forces to terminate at
        N: The number of steps used to calculate the st. dev.
    """
    def __init__(self, atoms, dyn, min_stdev=0.005, N=15):
        self.atoms = atoms
        self.dyn = dyn
        self.N = N
        self.forces = []
        self.min_stdev = min_stdev

    def write(self):
        """ The method called by the optimizer in each step. """
        if len(self.forces) >= self.N:
            self.forces.pop(0)
        fmax = (self.atoms.get_forces()**2).sum(axis=1).max()**0.5
        self.forces.append(fmax)

        m = sum(self.forces) / float(len(self.forces))

        stdev = sqrt(sum([(c - m)**2 for c in self.forces]) /
                     float(len(self.forces)))

        if len(self.forces) >= self.N and stdev < self.min_stdev:
            self.dyn.converged = lambda x: True
