"""Langevin dynamics class."""

import numpy as np

from ase.md.md import MolecularDynamics
from ase.parallel import world


class Langevin(MolecularDynamics):
    """Langevin (constant N, V, T) molecular dynamics.

    Usage: Langevin(atoms, dt, temperature, friction)

    atoms
        The list of atoms.

    dt
        The time step.

    temperature
        The desired temperature, in energy units.

    friction
        A friction coefficient, typically 1e-4 to 1e-2.

    fixcm
        If True, the position and momentum of the center of mass is
        kept unperturbed.  Default: True.

    rng
        Random number generator, by default numpy.random.  Must have a
        standard_normal method matching the signature of
        numpy.random.standard_normal.

    The temperature and friction are normally scalars, but in principle one
    quantity per atom could be specified by giving an array.

    RATTLE constraints can be used with these propagators, see:
    E. V.-Eijnden, and G. Ciccotti, Chem. Phys. Lett. 429, 310 (2006)

    The propagator is Equation 23 (Eq. 39 if RATTLE constraints are used)
    of the above reference.  That reference also contains another
    propagator in Eq. 21/34; but that propagator is not quasi-symplectic
    and gives a systematic offset in the temperature at large time steps.

    This dynamics accesses the atoms using Cartesian coordinates."""

    # Helps Asap doing the right thing.  Increment when changing stuff:
    _lgv_version = 3

    def __init__(self, atoms, timestep, temperature, friction, fixcm=True,
                 trajectory=None, logfile=None, loginterval=1,
                 communicator=world, rng=np.random):
        self.temp = temperature
        self.fr = friction
        self.fixcm = fixcm  # will the center of mass be held fixed?
        self.communicator = communicator
        self.rng = rng
        MolecularDynamics.__init__(self, atoms, timestep, trajectory,
                                   logfile, loginterval)
        self.updatevars()

    def todict(self):
        d = MolecularDynamics.todict(self)
        d.update({'temperature': self.temp,
                  'friction': self.fr,
                  'fix-cm': self.fixcm})
        return d

    def set_temperature(self, temperature):
        self.temp = temperature
        self.updatevars()

    def set_friction(self, friction):
        self.fr = friction
        self.updatevars()

    def set_timestep(self, timestep):
        self.dt = timestep
        self.updatevars()

    def updatevars(self):
        dt = self.dt
        T = self.temp
        fr = self.fr
        masses = self.masses
        sigma = np.sqrt(2 * T * fr / masses)

        self.c1 = dt / 2. - dt * dt * fr / 8.
        self.c2 = dt * fr / 2 - dt * dt * fr * fr / 8.
        self.c3 = np.sqrt(dt) * sigma / 2. - dt**1.5 * fr * sigma / 8.
        self.c5 = dt**1.5 * sigma / (2 * np.sqrt(3))
        self.c4 = fr / 2. * self.c5

        # Works in parallel Asap, #GLOBAL number of atoms:
        self.natoms = self.atoms.get_number_of_atoms()

    def step(self, f):
        atoms = self.atoms
        natoms = len(atoms)

        # This velocity as well as xi, eta and a few other variables are stored
        # as attributes, so Asap can do its magic when atoms migrate between
        # processors.
        self.v = atoms.get_velocities()

        self.xi = self.rng.standard_normal(size=(natoms, 3))
        self.eta = self.rng.standard_normal(size=(natoms, 3))

        if self.communicator is not None:
            self.communicator.broadcast(self.xi, 0)
            self.communicator.broadcast(self.eta, 0)

        # First halfstep in the velocity.
        self.v += (self.c1 * f / self.masses - self.c2 * self.v +
                   self.c3 * self.xi - self.c4 * self.eta)

        # Full step in positions
        x = atoms.get_positions()
        if self.fixcm:
            old_cm = atoms.get_center_of_mass()
        # Step: x^n -> x^(n+1) - this applies constraints if any.
        atoms.set_positions(x + self.dt * self.v + self.c5 * self.eta)
        if self.fixcm:
            new_cm = atoms.get_center_of_mass()
            d = old_cm - new_cm
            # atoms.translate(d)  # Does not respect constraints
            atoms.set_positions(atoms.get_positions() + d)

        # recalc velocities after RATTLE constraints are applied
        self.v = (self.atoms.get_positions() - x -
                  self.c5 * self.eta) / self.dt
        f = atoms.get_forces(md=True)

        # Update the velocities
        self.v += (self.c1 * f / self.masses - self.c2 * self.v +
                   self.c3 * self.xi - self.c4 * self.eta)

        if self.fixcm:  # subtract center of mass vel
            v_cm = self._get_com_velocity()
            self.v -= v_cm

        # Second part of RATTLE taken care of here
        atoms.set_momenta(self.v * self.masses)

        return f

    def _get_com_velocity(self):
        """Return the center of mass velocity.

        Internal use only.  This function can be reimplemented by Asap.
        """
        return np.dot(self.masses.flatten(), self.v) / self.masses.sum()
