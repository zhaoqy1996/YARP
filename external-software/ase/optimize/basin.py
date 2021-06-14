import numpy as np

from ase.optimize.optimize import Dynamics
from ase.optimize.fire import FIRE
from ase.units import kB
from ase.parallel import world
from ase.io.trajectory import Trajectory
from ase.utils import basestring


class BasinHopping(Dynamics):
    """Basin hopping algorithm.

    After Wales and Doye, J. Phys. Chem. A, vol 101 (1997) 5111-5116

    and

    David J. Wales and Harold A. Scheraga, Science, Vol. 285, 1368 (1999)
    """

    def __init__(self, atoms,
                 temperature=100 * kB,
                 optimizer=FIRE,
                 fmax=0.1,
                 dr=0.1,
                 logfile='-',
                 trajectory='lowest.traj',
                 optimizer_logfile='-',
                 local_minima_trajectory='local_minima.traj',
                 adjust_cm=True):
        """Parameters:

        atoms: Atoms object
            The Atoms object to operate on.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.
        """
        self.kT = temperature
        self.optimizer = optimizer
        self.fmax = fmax
        self.dr = dr
        if adjust_cm:
            self.cm = atoms.get_center_of_mass()
        else:
            self.cm = None

        self.optimizer_logfile = optimizer_logfile
        self.lm_trajectory = local_minima_trajectory
        if isinstance(local_minima_trajectory, basestring):
            self.lm_trajectory = Trajectory(local_minima_trajectory,
                                            'w', atoms)

        Dynamics.__init__(self, atoms, logfile, trajectory)
        self.initialize()

    def todict(self):
        d = {'type': 'optimization',
             'optimizer': self.__class__.__name__,
             'local-minima-optimizer': self.optimizer.__name__,
             'temperature': self.kT,
             'max-force': self.fmax,
             'maximal-step-width': self.dr}
        return d

    def initialize(self):
        self.positions = 0.0 * self.atoms.get_positions()
        self.Emin = self.get_energy(self.atoms.get_positions()) or 1.e32
        self.rmin = self.atoms.get_positions()
        self.positions = self.atoms.get_positions()
        self.call_observers()
        self.log(-1, self.Emin, self.Emin)

    def run(self, steps):
        """Hop the basins for defined number of steps."""

        ro = self.positions
        Eo = self.get_energy(ro)

        for step in range(steps):
            En = None
            while En is None:
                rn = self.move(ro)
                En = self.get_energy(rn)

            if En < self.Emin:
                # new minimum found
                self.Emin = En
                self.rmin = self.atoms.get_positions()
                self.call_observers()
            self.log(step, En, self.Emin)

            accept = np.exp((Eo - En) / self.kT) > np.random.uniform()
            if accept:
                ro = rn.copy()
                Eo = En

    def log(self, step, En, Emin):
        if self.logfile is None:
            return
        name = self.__class__.__name__
        self.logfile.write('%s: step %d, energy %15.6f, emin %15.6f\n'
                           % (name, step, En, Emin))
        self.logfile.flush()

    def move(self, ro):
        """Move atoms by a random step."""
        atoms = self.atoms
        # displace coordinates
        disp = np.random.uniform(-1., 1., (len(atoms), 3))
        rn = ro + self.dr * disp
        atoms.set_positions(rn)
        if self.cm is not None:
            cm = atoms.get_center_of_mass()
            atoms.translate(self.cm - cm)
        rn = atoms.get_positions()
        world.broadcast(rn, 0)
        atoms.set_positions(rn)
        return atoms.get_positions()

    def get_minimum(self):
        """Return minimal energy and configuration."""
        atoms = self.atoms.copy()
        atoms.set_positions(self.rmin)
        return self.Emin, atoms

    def get_energy(self, positions):
        """Return the energy of the nearest local minimum."""
        if np.sometrue(self.positions != positions):
            self.positions = positions
            self.atoms.set_positions(positions)

            opt = self.optimizer(self.atoms,
                                 logfile=self.optimizer_logfile)
            opt.run(fmax=self.fmax)
            if self.lm_trajectory is not None:
                self.lm_trajectory.write(self.atoms)

            self.energy = self.atoms.get_potential_energy()

        return self.energy
