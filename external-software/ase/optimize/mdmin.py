import numpy as np

from ase.optimize.optimize import Optimizer


class MDMin(Optimizer):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 dt=None, master=None):
        """Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.2 Angstroms).

        logfile: string
            Text file used to write summary information.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.
        """
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)

        if dt is not None:
            self.dt = dt

    def initialize(self):
        self.v = None
        self.dt = 0.2

    def read(self):
        self.v, self.dt = self.load()
        
    def step(self, f):
        atoms = self.atoms

        if self.v is None:
            self.v = np.zeros((len(atoms), 3))
        else:
            self.v += 0.5 * self.dt * f
            # Correct velocities:
            vf = np.vdot(self.v, f)
            if vf < 0.0:
                self.v[:] = 0.0
            else:
                self.v[:] = f * vf / np.vdot(f, f)

        self.v += 0.5 * self.dt * f
        r = atoms.get_positions()
        atoms.set_positions(r + self.dt * self.v)
        self.dump((self.v, self.dt))
