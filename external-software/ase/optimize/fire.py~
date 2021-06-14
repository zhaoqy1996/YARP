import numpy as np

from ase.optimize.optimize import Optimizer


class FIRE(Optimizer):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 dt=0.1, maxmove=0.2, dtmax=1.0, Nmin=5, finc=1.1, fdec=0.5,
                 astart=0.1, fa=0.99, a=0.1, master=None, downhill_check=False,
                 position_reset_callback=None, force_consistent=None):
        """Parameters:

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

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        downhill_check: boolean
            Downhill check directly compares potential energies of subsequent
            steps of the FIRE algorithm rather than relying on the current
            product v*f that is positive if the FIRE dynamics moves downhill.
            This can detect numerical issues where at large time steps the step
            is uphill in energy even though locally v*f is positive, i.e. the
            algorithm jumps over a valley because of a too large time step.

        position_reset_callback: function(atoms, r, e, e_last)
            Function that takes current *atoms* object, an array of position
            *r* that the optimizer will revert to, current energy *e* and
            energy of last step *e_last*. This is only called if e > e_last.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K).  By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.  Only meaningful
            when downhill_check is True.
        """
        Optimizer.__init__(self, atoms, restart, logfile, trajectory,
                           master, force_consistent=force_consistent)

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
        self.downhill_check = downhill_check
        self.position_reset_callback = position_reset_callback

    def initialize(self):
        self.v = None

    def read(self):
        self.v, self.dt = self.load()

    def step(self, f):
        atoms = self.atoms
        if self.v is None:
            self.v = np.zeros((len(atoms), 3))
            if self.downhill_check:
                self.e_last = atoms.get_potential_energy(
                    force_consistent=self.force_consistent)
                self.r_last = atoms.get_positions().copy()
                self.v_last = self.v.copy()
        else:
            is_uphill = False
            if self.downhill_check:
                e = atoms.get_potential_energy(
                    force_consistent=self.force_consistent)
                # Check if the energy actually decreased
                if e > self.e_last:
                    # If not, reset to old positions...
                    if self.position_reset_callback is not None:
                        self.position_reset_callback(atoms, self.r_last, e,
                                                     self.e_last)
                    atoms.set_positions(self.r_last)
                    is_uphill = True
                self.e_last = atoms.get_potential_energy(
                    force_consistent=self.force_consistent)
                self.r_last = atoms.get_positions().copy()
                self.v_last = self.v.copy()

            vf = np.vdot(f, self.v)
            if vf > 0.0 and not is_uphill:
                self.v = (1.0 - self.a) * self.v + self.a * f / np.sqrt(
                    np.vdot(f, f)) * np.sqrt(np.vdot(self.v, self.v))
                if self.Nsteps > self.Nmin:
                    self.dt = min(self.dt * self.finc, self.dtmax)
                    self.a *= self.fa
                self.Nsteps += 1
            else:
                self.v[:] *= 0.0
                self.a = self.astart
                self.dt *= self.fdec
                self.Nsteps = 0

        self.v += self.dt * f
        dr = self.dt * self.v
        normdr = np.sqrt(np.vdot(dr, dr))
        if normdr > self.maxmove:
            dr = self.maxmove * dr / normdr
        r = atoms.get_positions()
        atoms.set_positions(r + dr)
        self.dump((self.v, self.dt))

