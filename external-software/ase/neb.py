# -*- coding: utf-8 -*-
import pickle
import sys
import threading
from math import sqrt

import numpy as np

import ase.parallel as mpi
from ase.build import minimize_rotation_and_translation
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from ase.optimize import MDMin
from ase.geometry import find_mic
from ase.utils import basestring


class NEB:
    def __init__(self, images, k=0.1, fmax=0.05, climb=False, parallel=False,
                 remove_rotation_and_translation=False, world=None,
                 method='aseneb', dynamic_relaxation=False):
        """Nudged elastic band.

        Paper I:

            G. Henkelman and H. Jonsson, Chem. Phys, 113, 9978 (2000).

        Paper II:

            G. Henkelman, B. P. Uberuaga, and H. Jonsson, Chem. Phys,
            113, 9901 (2000).

        Paper III:

            E. L. Kolsbjerg, M. N. Groves, and B. Hammer, J. Chem. Phys,
            submitted (2016)

        images: list of Atoms objects
            Images defining path from initial to final state.
        k: float or list of floats
            Spring constant(s) in eV/Ang.  One number or one for each spring.
        climb: bool
            Use a climbing image (default is no climbing image).
        parallel: bool
            Distribute images over processors.
        remove_rotation_and_translation: bool
            TRUE actives NEB-TR for removing translation and
            rotation during NEB. By default applied non-periodic
            systems
        dynamic_relaxation: bool
            TRUE calculates the norm of the forces acting on each image
            in the band. An image is optimized only if its norm is above
            the convergence criterion. The list fmax_images is updated
            every force call; if a previously converged image goes out
            of tolerance (due to spring adjustments between the image
            and its neighbors), it will be optimized again. This routine
            can speed up calculations if convergence is non-uniform.
            Convergence criterion should be the same as that given to
            the optimizer. Not efficient when parallelizing over images.
        method: string of method
            Choice betweeen three method:

            * aseneb: standard ase NEB implementation
            * improvedtangent: Paper I NEB implementation
            * eb: Paper III full spring force implementation
        """
        self.images = images
        self.climb = climb
        self.parallel = parallel
        self.natoms = len(images[0])
        pbc = images[0].pbc
        for img in images:
            if len(img) != self.natoms:
                raise ValueError('Images have different numbers of atoms')
            if (pbc != img.pbc).any():
                raise ValueError('Images have different boundary conditions')
        self.nimages = len(images)
        self.emax = np.nan

        self.remove_rotation_and_translation = remove_rotation_and_translation
        self.dynamic_relaxation = dynamic_relaxation
        self.fmax = fmax

        if method in ['aseneb', 'eb', 'improvedtangent']:
            self.method = method
        else:
            raise NotImplementedError(method)

        if isinstance(k, (float, int)):
            k = [k] * (self.nimages - 1)
        self.k = list(k)

        if world is None:
            world = mpi.world
        self.world = world

        if parallel:
            assert world.size == 1 or world.size % (self.nimages - 2) == 0

        self.real_forces = None  # ndarray of shape (nimages, natom, 3)
        self.energies = None  # ndarray of shape (nimages,)

    def interpolate(self, method='linear', mic=False):
        if self.remove_rotation_and_translation:
            minimize_rotation_and_translation(self.images[0], self.images[-1])

        interpolate(self.images, mic)
        
        if method == 'idpp':
            self.idpp_interpolate(traj=None, log=None, mic=mic)

    def idpp_interpolate(self, traj='idpp.traj', log='idpp.log', fmax=0.1,
                         optimizer=MDMin, mic=False, steps=100):
        d1 = self.images[0].get_all_distances(mic=mic)
        d2 = self.images[-1].get_all_distances(mic=mic)
        d = (d2 - d1) / (self.nimages - 1)
        old = []
        for i, image in enumerate(self.images):
            old.append(image.calc)
            image.calc = IDPP(d1 + i * d, mic=mic)
        opt = optimizer(self, trajectory=traj, logfile=log)
        # BFGS was originally used by the paper, but testing shows that
        # MDMin results in nearly the same results in 3-4 orders of magnitude
        # less time. Known working optimizers = BFGS, MDMin, FIRE, HessLBFGS
        # Internal testing shows BFGS is only needed in situations where MDMIN
        # cannot converge easily and tends to be obvious on inspection.
        #
        # askhl: 3-4 orders of magnitude difference cannot possibly be
        # true unless something is actually broken.  Should it not be
        # "3-4 times"?
        opt.run(fmax=fmax, steps=steps)
        for image, calc in zip(self.images, old):
            image.calc = calc

    def get_positions(self):
        positions = np.empty(((self.nimages - 2) * self.natoms, 3))
        n1 = 0
        for image in self.images[1:-1]:
            n2 = n1 + self.natoms
            positions[n1:n2] = image.get_positions()
            n1 = n2
        return positions

    def set_positions(self, positions):
        n1 = 0
        for i, image in enumerate(self.images[1:-1]):
            if self.dynamic_relaxation:
                if self.parallel:
                    msg = ('Dynamic relaxation does not work efficiently '
                           'when parallelizing over images. Try AutoNEB '
                           'routine for freezing images in parallel.')
                    raise ValueError(msg)
                else:
                    forces_dyn = self.get_fmax_all(self.images)
                    if forces_dyn[i] < self.fmax:
                        n1 += self.natoms
                    else:
                        n2 = n1 + self.natoms
                        image.set_positions(positions[n1:n2])
                        n1 = n2
            else:
                n2 = n1 + self.natoms
                image.set_positions(positions[n1:n2])
                n1 = n2

    def get_fmax_all(self, images):
        n = self.natoms
        f_i = self.get_forces()
        fmax_images = []
        for i in range(self.nimages-2):
            n1 = n * i
            n2 = n + n * i
            fmax_images.append(np.sqrt((f_i[n1:n2]**2).sum(axis=1)).max())
        return fmax_images

    def get_forces(self):
        """Evaluate and return the forces."""
        images = self.images
        calculators = [image.calc for image in images
                       if image.calc is not None]
        if len(set(calculators)) != len(calculators):
            msg = ('One or more NEB images share the same calculator.  '
                   'Each image must have its own calculator.  '
                   'You may wish to use the ase.neb.SingleCalculatorNEB '
                   'class instead, although using separate calculators '
                   'is recommended.')
            raise ValueError(msg)

        forces = np.empty(((self.nimages - 2), self.natoms, 3))
        energies = np.empty(self.nimages)

        if self.remove_rotation_and_translation:
            # Remove translation and rotation between
            # images before computing forces:
            for i in range(1, self.nimages):
                minimize_rotation_and_translation(images[i - 1], images[i])

        if self.method != 'aseneb':
            energies[0] = images[0].get_potential_energy()
            energies[-1] = images[-1].get_potential_energy()

        if not self.parallel:
            # Do all images - one at a time:
            for i in range(1, self.nimages - 1):
                energies[i] = images[i].get_potential_energy()
                forces[i - 1] = images[i].get_forces()
        elif self.world.size == 1:
            def run(image, energies, forces):
                energies[:] = image.get_potential_energy()
                forces[:] = image.get_forces()
            threads = [threading.Thread(target=run,
                                        args=(images[i],
                                              energies[i:i + 1],
                                              forces[i - 1:i]))
                       for i in range(1, self.nimages - 1)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            # Parallelize over images:
            i = self.world.rank * (self.nimages - 2) // self.world.size + 1
            try:
                energies[i] = images[i].get_potential_energy()
                forces[i - 1] = images[i].get_forces()
            except Exception:
                # Make sure other images also fail:
                error = self.world.sum(1.0)
                raise
            else:
                error = self.world.sum(0.0)
                if error:
                    raise RuntimeError('Parallel NEB failed!')

            for i in range(1, self.nimages - 1):
                root = (i - 1) * self.world.size // (self.nimages - 2)
                self.world.broadcast(energies[i:i + 1], root)
                self.world.broadcast(forces[i - 1], root)

        # Save for later use in iterimages:
        self.energies = energies
        self.real_forces = np.zeros((self.nimages, self.natoms, 3))
        self.real_forces[1:-1] = forces

        imax = 1 + np.argsort(energies[1:-1])[-1]
        self.emax = energies[imax]

        t1 = find_mic(images[1].get_positions() -
                      images[0].get_positions(),
                      images[0].get_cell(), images[0].pbc)[0]

        if self.method == 'eb':
            beeline = (images[self.nimages - 1].get_positions() -
                       images[0].get_positions())
            beelinelength = np.linalg.norm(beeline)
            eqlength = beelinelength / (self.nimages - 1)

        nt1 = np.linalg.norm(t1)

        for i in range(1, self.nimages - 1):
            t2 = find_mic(images[i + 1].get_positions() -
                          images[i].get_positions(),
                          images[i].get_cell(), images[i].pbc)[0]
            nt2 = np.linalg.norm(t2)

            if self.method == 'eb':
                # Tangents are bisections of spring-directions
                # (formula C8 of paper III)
                tangent = t1 / nt1 + t2 / nt2
                # Normalize the tangent vector
                tangent /= np.linalg.norm(tangent)
            elif self.method == 'improvedtangent':
                # Tangents are improved according to formulas 8, 9, 10,
                # and 11 of paper I.
                if energies[i + 1] > energies[i] > energies[i - 1]:
                    tangent = t2.copy()
                elif energies[i + 1] < energies[i] < energies[i - 1]:
                    tangent = t1.copy()
                else:
                    deltavmax = max(abs(energies[i + 1] - energies[i]),
                                    abs(energies[i - 1] - energies[i]))
                    deltavmin = min(abs(energies[i + 1] - energies[i]),
                                    abs(energies[i - 1] - energies[i]))
                    if energies[i + 1] > energies[i - 1]:
                        tangent = t2 * deltavmax + t1 * deltavmin
                    else:
                        tangent = t2 * deltavmin + t1 * deltavmax
                # Normalize the tangent vector
                tangent /= np.linalg.norm(tangent)
            else:
                if i < imax:
                    tangent = t2
                elif i > imax:
                    tangent = t1
                else:
                    tangent = t1 + t2
                tt = np.vdot(tangent, tangent)

            f = forces[i - 1]
            ft = np.vdot(f, tangent)

            if i == imax and self.climb:
                # imax not affected by the spring forces. The full force
                # with component along the elestic band converted
                # (formula 5 of Paper II)
                if self.method == 'aseneb':
                    f -= 2 * ft / tt * tangent
                else:
                    f -= 2 * ft * tangent
            elif self.method == 'eb':
                f -= ft * tangent
                # Spring forces
                # (formula C1, C5, C6 and C7 of Paper III)
                f1 = -(nt1 - eqlength) * t1 / nt1 * self.k[i - 1]
                f2 = (nt2 - eqlength) * t2 / nt2 * self.k[i]
                if self.climb and abs(i - imax) == 1:
                    deltavmax = max(abs(energies[i + 1] - energies[i]),
                                    abs(energies[i - 1] - energies[i]))
                    deltavmin = min(abs(energies[i + 1] - energies[i]),
                                    abs(energies[i - 1] - energies[i]))
                    f += (f1 + f2) * deltavmin / deltavmax
                else:
                    f += f1 + f2
            elif self.method == 'improvedtangent':
                f -= ft * tangent
                # Improved parallel spring force (formula 12 of paper I)
                f += (nt2 * self.k[i] - nt1 * self.k[i - 1]) * tangent
            else:
                f -= ft / tt * tangent
                f -= np.vdot(t1 * self.k[i - 1] -
                             t2 * self.k[i], tangent) / tt * tangent

            t1 = t2
            nt1 = nt2

        return forces.reshape((-1, 3))

    def get_potential_energy(self, force_consistent=False):
        """Return the maximum potential energy along the band.
        Note that the force_consistent keyword is ignored and is only
        present for compatibility with ase.Atoms.get_potential_energy."""
        return self.emax

    def __len__(self):
        # Corresponds to number of optimizable degrees of freedom, i.e.
        # virtual atom count for the optimization algorithm.
        return (self.nimages - 2) * self.natoms

    def iterimages(self):
        # Allows trajectory to convert NEB into several images
        if not self.parallel or self.world.size == 1:
            for atoms in self.images:
                yield atoms
            return

        for i, atoms in enumerate(self.images):
            if i == 0 or i == self.nimages - 1:
                yield atoms
            else:
                atoms = atoms.copy()
                atoms.calc = SinglePointCalculator(energy=self.energies[i],
                                                   forces=self.real_forces[i],
                                                   atoms=atoms)
                yield atoms


class IDPP(Calculator):
    """Image dependent pair potential.

    See:
        Improved initial guess for minimum energy path calculations.
        Søren Smidstrup, Andreas Pedersen, Kurt Stokbro and Hannes Jónsson
        Chem. Phys. 140, 214106 (2014)
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, target, mic):
        Calculator.__init__(self)
        self.target = target
        self.mic = mic

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        P = atoms.get_positions()
        d = []
        D = []
        for p in P:
            Di = P - p
            if self.mic:
                Di, di = find_mic(Di, atoms.get_cell(), atoms.get_pbc())
            else:
                di = np.sqrt((Di**2).sum(1))
            d.append(di)
            D.append(Di)
        d = np.array(d)
        D = np.array(D)

        dd = d - self.target
        d.ravel()[::len(d) + 1] = 1  # avoid dividing by zero
        d4 = d**4
        e = 0.5 * (dd**2 / d4).sum()
        f = -2 * ((dd * (1 - 2 * dd / d) / d**5)[..., np.newaxis] * D).sum(0)
        self.results = {'energy': e, 'forces': f}


class SingleCalculatorNEB(NEB):
    def __init__(self, images, k=0.1, climb=False):
        if isinstance(images, basestring):
            # this is a filename
            images = read(images)

        NEB.__init__(self, images, k, climb, False)
        self.calculators = [None] * self.nimages
        self.energies_ok = False
        self.first = True

    def interpolate(self, initial=0, final=-1, mic=False):
        """Interpolate linearly between initial and final images."""
        if final < 0:
            final = self.nimages + final
        n = final - initial
        pos1 = self.images[initial].get_positions()
        pos2 = self.images[final].get_positions()
        dist = (pos2 - pos1)
        if mic:
            cell = self.images[initial].get_cell()
            assert((cell == self.images[final].get_cell()).all())
            pbc = self.images[initial].get_pbc()
            assert((pbc == self.images[final].get_pbc()).all())
            dist, D_len = find_mic(dist, cell, pbc)
        dist /= n
        for i in range(1, n):
            self.images[initial + i].set_positions(pos1 + i * dist)

    def refine(self, steps=1, begin=0, end=-1, mic=False):
        """Refine the NEB trajectory."""
        if end < 0:
            end = self.nimages + end
        j = begin
        n = end - begin
        for i in range(n):
            for k in range(steps):
                self.images.insert(j + 1, self.images[j].copy())
                self.calculators.insert(j + 1, None)
            self.k[j:j + 1] = [self.k[j] * (steps + 1)] * (steps + 1)
            self.nimages = len(self.images)
            self.interpolate(j, j + steps + 1, mic=mic)
            j += steps + 1

    def set_positions(self, positions):
        # new positions -> new forces
        if self.energies_ok:
            # restore calculators
            self.set_calculators(self.calculators[1:-1])
        NEB.set_positions(self, positions)

    def get_calculators(self):
        """Return the original calculators."""
        calculators = []
        for i, image in enumerate(self.images):
            if self.calculators[i] is None:
                calculators.append(image.get_calculator())
            else:
                calculators.append(self.calculators[i])
        return calculators

    def set_calculators(self, calculators):
        """Set new calculators to the images."""
        self.energies_ok = False
        self.first = True

        if not isinstance(calculators, list):
            calculators = [calculators] * self.nimages

        n = len(calculators)
        if n == self.nimages:
            for i in range(self.nimages):
                self.images[i].set_calculator(calculators[i])
        elif n == self.nimages - 2:
            for i in range(1, self.nimages - 1):
                self.images[i].set_calculator(calculators[i - 1])
        else:
            raise RuntimeError(
                'len(calculators)=%d does not fit to len(images)=%d'
                % (n, self.nimages))

    def get_energies_and_forces(self):
        """Evaluate energies and forces and hide the calculators"""
        if self.energies_ok:
            return

        self.emax = -1.e32

        def calculate_and_hide(i):
            image = self.images[i]
            calc = image.get_calculator()
            if self.calculators[i] is None:
                self.calculators[i] = calc
            if calc is not None:
                if not isinstance(calc, SinglePointCalculator):
                    self.images[i].set_calculator(
                        SinglePointCalculator(
                            image,
                            energy=image.get_potential_energy(
                                apply_constraint=False),
                            forces=image.get_forces(apply_constraint=False)))
                self.emax = min(self.emax, image.get_potential_energy())

        if self.first:
            calculate_and_hide(0)

        # Do all images - one at a time:
        for i in range(1, self.nimages - 1):
            calculate_and_hide(i)

        if self.first:
            calculate_and_hide(-1)
            self.first = False

        self.energies_ok = True

    def get_forces(self):
        self.get_energies_and_forces()
        return NEB.get_forces(self)

    def n(self):
        return self.nimages

    def write(self, filename):
        from ase.io.trajectory import Trajectory
        traj = Trajectory(filename, 'w', self)
        traj.write()
        traj.close()

    def __add__(self, other):
        for image in other:
            self.images.append(image)
        return self


def fit0(E, F, R, cell=None, pbc=None):
    """Constructs curve parameters from the NEB images."""
    E = np.array(E) - E[0]
    n = len(E)
    Efit = np.empty((n - 1) * 20 + 1)
    Sfit = np.empty((n - 1) * 20 + 1)

    s = [0]
    dR = np.zeros_like(R)
    for i in range(n):
        if i < n - 1:
            dR[i] = R[i + 1] - R[i]
            if cell is not None and pbc is not None:
                dR[i], _ = find_mic(dR[i], cell, pbc)
            s.append(s[i] + sqrt((dR[i]**2).sum()))
        else:
            dR[i] = R[i] - R[i - 1]
            if cell is not None and pbc is not None:
                dR[i], _ = find_mic(dR[i], cell, pbc)

    lines = []
    dEds0 = None
    for i in range(n):
        d = dR[i]
        if i == 0:
            ds = 0.5 * s[1]
        elif i == n - 1:
            ds = 0.5 * (s[-1] - s[-2])
        else:
            ds = 0.25 * (s[i + 1] - s[i - 1])

        d = d / sqrt((d**2).sum())
        dEds = -(F[i] * d).sum()
        x = np.linspace(s[i] - ds, s[i] + ds, 3)
        y = E[i] + dEds * (x - s[i])
        lines.append((x, y))

        if i > 0:
            s0 = s[i - 1]
            s1 = s[i]
            x = np.linspace(s0, s1, 20, endpoint=False)
            c = np.linalg.solve(np.array([(1, s0, s0**2, s0**3),
                                          (1, s1, s1**2, s1**3),
                                          (0, 1, 2 * s0, 3 * s0**2),
                                          (0, 1, 2 * s1, 3 * s1**2)]),
                                np.array([E[i - 1], E[i], dEds0, dEds]))
            y = c[0] + x * (c[1] + x * (c[2] + x * c[3]))
            Sfit[(i - 1) * 20:i * 20] = x
            Efit[(i - 1) * 20:i * 20] = y

        dEds0 = dEds

    Sfit[-1] = s[-1]
    Efit[-1] = E[-1]
    return s, E, Sfit, Efit, lines


class NEBTools:
    """Class to make many of the common tools for NEB analysis available to
    the user. Useful for scripting the output of many jobs. Initialize with
    list of images which make up a single band."""

    def __init__(self, images):
        self._images = images

    def get_barrier(self, fit=True, raw=False):
        """Returns the barrier estimate from the NEB, along with the
        Delta E of the elementary reaction. If fit=True, the barrier is
        estimated based on the interpolated fit to the images; if
        fit=False, the barrier is taken as the maximum-energy image
        without interpolation. Set raw=True to get the raw energy of the
        transition state instead of the forward barrier."""
        s, E, Sfit, Efit, lines = self.get_fit()
        dE = E[-1] - E[0]
        if fit:
            barrier = max(Efit)
        else:
            barrier = max(E)
        if raw:
            barrier += self._images[0].get_potential_energy()
        return barrier, dE

    def plot_band(self, ax=None):
        """Plots the NEB band on matplotlib axes object 'ax'. If ax=None
        returns a new figure object."""
        ax = plot_band_from_fit(*self.get_fit(), ax=ax)
        return ax.figure

    def get_fmax(self, **kwargs):
        """Returns fmax, as used by optimizers with NEB."""
        neb = NEB(self._images, **kwargs)
        forces = neb.get_forces()
        return np.sqrt((forces**2).sum(axis=1).max())

    def get_fit(self):
        """Returns the parameters for fitting images to band."""
        images = self._images
        R = [atoms.positions for atoms in images]
        E = [atoms.get_potential_energy() for atoms in images]
        F = [atoms.get_forces() for atoms in images]
        A = images[0].cell
        pbc = images[0].pbc
        s, E, Sfit, Efit, lines = fit0(E, F, R, A, pbc)
        return s, E, Sfit, Efit, lines


def plot_band_from_fit(s, E, Sfit, Efit, lines, ax=None):
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()

    ax.plot(s, E, 'o')
    for x, y in lines:
        ax.plot(x, y, '-g')
    ax.plot(Sfit, Efit, 'k-')
    ax.set_xlabel(r'path [$\AA$]')
    ax.set_ylabel('energy [eV]')
    Ef = max(Efit) - E[0]
    Er = max(Efit) - E[-1]
    dE = E[-1] - E[0]
    ax.set_title('$E_\\mathrm{f} \\approx$ %.3f eV; '
                 '$E_\\mathrm{r} \\approx$ %.3f eV; '
                 '$\\Delta E$ = %.3f eV'
                 % (Ef, Er, dE))
    return ax


NEBtools = NEBTools  # backwards compatibility


def interpolate(images, mic=False):
    """Given a list of images, linearly interpolate the positions of the
    interior images."""
    pos1 = images[0].get_positions()
    pos2 = images[-1].get_positions()
    d = pos2 - pos1
    if mic:
        d = find_mic(d, images[0].get_cell(), images[0].pbc)[0]
    d /= (len(images) - 1.0)
    for i in range(1, len(images) - 1):
        images[i].set_positions(pos1 + i * d)


if __name__ == '__main__':
    # This stuff is used by ASE's GUI
    import matplotlib.pyplot as plt
    fit = pickle.load(sys.stdin)
    plot_band_from_fit(*fit)
    plt.show()
