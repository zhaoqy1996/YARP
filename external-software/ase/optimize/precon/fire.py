import numpy as np

from ase.optimize.optimize import Optimizer
from ase.constraints import UnitCellFilter
import time

class PreconFIRE(Optimizer):

    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 dt=0.1, maxmove=0.2, dtmax=1.0, Nmin=5, finc=1.1, fdec=0.5,
                 astart=0.1, fa=0.99, a=0.1, theta=0.1, master=None,
                 precon=None, use_armijo=True, variable_cell=False):
        """
        Preconditioned version of the FIRE optimizer

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        master: bool
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        variable_cell: bool
            If True, wrap atoms in UnitCellFilter to relax cell and positions.

        In time this implementation is expected to replace
        ase.optimize.fire.FIRE.
        """
        if variable_cell:
            atoms = UnitCellFilter(atoms)
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)

        self.dt = dt
        self.Nsteps = 0
        self.maxmove = maxmove
        self.dtmax = dtmax
        self.Nmin = Nmin
        self.finc = finc
        self.fdec = fdec
        self.astart = astart
        self.fa = fa
        self.a = a
        self.theta = theta
        self.precon = precon
        self.use_armijo = use_armijo

    def initialize(self):
        self.v = None
        self.skip_flag = False
        self.e1 = None

    def read(self):
        self.v, self.dt = self.load()

    def step(self, f):
        atoms = self.atoms
        r = atoms.get_positions()

        if self.precon is not None:
            # Can this be moved out of the step method?
            self.precon.make_precon(atoms)
            invP_f = self.precon.solve(f.reshape(-1)).reshape(len(atoms), -1)

        if self.v is None:
            self.v = np.zeros((len(self.atoms), 3))
        else:
            if self.use_armijo:

                if self.precon is None:
                    v_test = self.v + self.dt * f
                else:
                    v_test = self.v + self.dt * invP_f

                r_test = r + self.dt * v_test

                self.skip_flag = False
                func_val = self.func(r_test)
                self.e1 = func_val
                if (func_val > self.func(r) -
                      self.theta * self.dt * np.vdot(v_test, f)):
                    self.v[:] *= 0.0
                    self.a = self.astart
                    self.dt *= self.fdec
                    self.Nsteps = 0
                    self.skip_flag = True

            if not self.skip_flag:

                v_f = np.vdot(self.v, f)
                if v_f > 0.0:
                    if self.precon is None:
                        self.v = (1.0 - self.a) * self.v + self.a * f / \
                            np.sqrt(np.vdot(f, f)) * \
                            np.sqrt(np.vdot(self.v, self.v))
                    else:
                        self.v = (
                            (1.0 - self.a) * self.v +
                            self.a *
                            (np.sqrt(self.precon.dot(self.v.reshape(-1),
                                                     self.v.reshape(-1))) /
                             np.sqrt(np.dot(f.reshape(-1),
                                            invP_f.reshape(-1))) * invP_f))
                    if self.Nsteps > self.Nmin:
                        self.dt = min(self.dt * self.finc, self.dtmax)
                        self.a *= self.fa
                    self.Nsteps += 1
                else:
                    self.v[:] *= 0.0
                    self.a = self.astart
                    self.dt *= self.fdec
                    self.Nsteps = 0

        if self.precon is None:
            self.v += self.dt * f
        else:
            self.v += self.dt * invP_f
        dr = self.dt * self.v
        normdr = np.sqrt(np.vdot(dr, dr))
        if normdr > self.maxmove:
            dr = self.maxmove * dr / normdr
        atoms.set_positions(r + dr)
        self.dump((self.v, self.dt))

    def func(self, x):
        """Objective function for use of the optimizers"""
        self.atoms.set_positions(x.reshape(-1, 3))
        potl = self.atoms.get_potential_energy()
        return potl

    def run(self, fmax=0.05, steps=100000000, smax=None):
        if smax is None:
            smax = fmax
        self.smax = smax
        return Optimizer.run(self, fmax, steps)

    def converged(self, forces=None):
        """Did the optimization converge?"""
        if forces is None:
            forces = self.atoms.get_forces()
        if isinstance(self.atoms, UnitCellFilter):
            natoms = len(self.atoms.atoms)
            forces, stress = forces[:natoms], self.atoms.stress
            fmax_sq = (forces**2).sum(axis=1).max()
            smax_sq = (stress**2).max()
            return (fmax_sq < self.fmax**2 and smax_sq < self.smax**2)
        else:
            fmax_sq = (forces**2).sum(axis=1).max()
            return fmax_sq < self.fmax**2

    def log(self, forces):
        if isinstance(self.atoms, UnitCellFilter):
            natoms = len(self.atoms.atoms)
            forces, stress = forces[:natoms], self.atoms.stress
            fmax = np.sqrt((forces**2).sum(axis=1).max())
            smax = np.sqrt((stress**2).max())
        else:
            fmax = np.sqrt((forces**2).sum(axis=1).max())
        if self.e1 is not None:
            # reuse energy at end of line search to avoid extra call
            e = self.e1
        else:
            e = self.atoms.get_potential_energy()
        T = time.localtime()
        if self.logfile is not None:
            name = self.__class__.__name__
            if isinstance(self.atoms, UnitCellFilter):
                self.logfile.write(
                    '%s: %3d  %02d:%02d:%02d %15.6f %12.4f %12.4f\n' %
                    (name, self.nsteps, T[3], T[4], T[5], e, fmax, smax))

            else:
                self.logfile.write(
                    '%s: %3d  %02d:%02d:%02d %15.6f %12.4f\n' %
                    (name, self.nsteps, T[3], T[4], T[5], e, fmax))
            self.logfile.flush()
