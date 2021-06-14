from __future__ import division
from math import sqrt
from warnings import warn
from ase.geometry import find_mic
from ase.calculators.calculator import PropertyNotImplementedError

import numpy as np
from scipy.linalg import expm

__all__ = ['FixCartesian', 'FixBondLength', 'FixedMode', 'FixConstraintSingle',
           'FixAtoms', 'UnitCellFilter', 'ExpCellFilter', 'FixScaled', 'StrainFilter',
           'FixCom', 'FixedPlane', 'Filter', 'FixConstraint', 'FixedLine',
           'FixBondLengths', 'FixInternals', 'Hookean', 'ExternalForce']


def dict2constraint(dct):
    if dct['name'] not in __all__:
        raise ValueError
    return globals()[dct['name']](**dct['kwargs'])


def slice2enlist(s, n):
    """Convert a slice object into a list of (new, old) tuples."""
    if isinstance(s, slice):
        return enumerate(range(*s.indices(n)))
    return enumerate(s)


def constrained_indices(atoms, only_include=None):
    """Returns a list of indices for the atoms that are constrained
    by a constraint that is applied.  By setting only_include to a
    specific type of constraint you can make it only look for that
    given constraint.
    """
    indices = []
    for constraint in atoms.constraints:
        if only_include is not None:
            if not isinstance(constraint, only_include):
                continue
        indices.extend(np.array(constraint.get_indices()))
    return np.array(np.unique(indices))


class FixConstraint:
    """Base class for classes that fix one or more atoms in some way."""

    def index_shuffle(self, atoms, ind):
        """Change the indices.

        When the ordering of the atoms in the Atoms object changes,
        this method can be called to shuffle the indices of the
        constraints.

        ind -- List or tuple of indices.

        """
        raise NotImplementedError

    def repeat(self, m, n):
        """ basic method to multiply by m, needs to know the length
        of the underlying atoms object for the assignment of
        multiplied constraints to work.
        """
        msg = ("Repeat is not compatible with your atoms' constraints."
               ' Use atoms.set_constraint() before calling repeat to '
               'remove your constraints.')
        raise NotImplementedError(msg)

    def adjust_momenta(self, atoms, momenta):
        """Adjusts momenta in identical manner to forces."""
        self.adjust_forces(atoms, momenta)

    def copy(self):
        return dict2constraint(self.todict().copy())


class FixConstraintSingle(FixConstraint):
    """Base class for classes that fix a single atom."""

    def __init__(self, a):
        self.a = a

    def index_shuffle(self, atoms, ind):
        """The atom index must be stored as self.a."""
        newa = None   # Signal error
        if self.a < 0:
            self.a += len(atoms)
        for new, old in slice2enlist(ind, len(atoms)):
            if old == self.a:
                newa = new
                break
        if newa is None:
            raise IndexError('Constraint not part of slice')
        self.a = newa

    def get_indices(self):
        return [self.a]


class FixAtoms(FixConstraint):
    """Constraint object for fixing some chosen atoms."""

    def __init__(self, indices=None, mask=None):
        """Constrain chosen atoms.

        Parameters
        ----------
        indices : list of int
           Indices for those atoms that should be constrained.
        mask : list of bool
           One boolean per atom indicating if the atom should be
           constrained or not.

        Examples
        --------
        Fix all Copper atoms:

        >>> mask = [s == 'Cu' for s in atoms.get_chemical_symbols()]
        >>> c = FixAtoms(mask=mask)
        >>> atoms.set_constraint(c)

        Fix all atoms with z-coordinate less than 1.0 Angstrom:

        >>> c = FixAtoms(mask=atoms.positions[:, 2] < 1.0)
        >>> atoms.set_constraint(c)
        """

        if indices is None and mask is None:
            raise ValueError('Use "indices" or "mask".')
        if indices is not None and mask is not None:
            raise ValueError('Use only one of "indices" and "mask".')

        if mask is not None:
            indices = np.arange(len(mask))[np.asarray(mask, bool)]
        else:
            # Check for duplicates:
            srt = np.sort(indices)
            if (np.diff(srt) == 0).any():
                raise ValueError(
                    'FixAtoms: The indices array contained duplicates. '
                    'Perhaps you wanted to specify a mask instead, but '
                    'forgot the mask= keyword.')
        self.index = np.asarray(indices, int)

        if self.index.ndim != 1:
            raise ValueError('Wrong argument to FixAtoms class!')

        self.removed_dof = 3 * len(self.index)

    def adjust_positions(self, atoms, new):
        new[self.index] = atoms.positions[self.index]

    def adjust_forces(self, atoms, forces):
        forces[self.index] = 0.0

    def index_shuffle(self, atoms, ind):
        # See docstring of superclass
        index = []
        for new, old in slice2enlist(ind, len(atoms)):
            if old in self.index:
                index.append(new)
        if len(index) == 0:
            raise IndexError('All indices in FixAtoms not part of slice')
        self.index = np.asarray(index, int)

    def get_indices(self):
        return self.index

    def __repr__(self):
        return 'FixAtoms(indices=%s)' % ints2string(self.index)

    def todict(self):
        return {'name': 'FixAtoms',
                'kwargs': {'indices': self.index}}

    def repeat(self, m, n):
        i0 = 0
        natoms = 0
        if isinstance(m, int):
            m = (m, m, m)
        index_new = []
        for m2 in range(m[2]):
            for m1 in range(m[1]):
                for m0 in range(m[0]):
                    i1 = i0 + n
                    index_new += [i + natoms for i in self.index]
                    i0 = i1
                    natoms += n
        self.index = np.asarray(index_new, int)
        return self

    def delete_atoms(self, indices, natoms):
        """Removes atom number ind from the index array, if present.

        Required for removing atoms with existing FixAtoms constraints.
        """

        i = np.zeros(natoms, int) - 1
        new = np.delete(np.arange(natoms), indices)
        i[new] = np.arange(len(new))
        index = i[self.index]
        self.index = index[index >= 0]
        if len(self.index) == 0:
            return None
        return self


class FixCom(FixConstraint):
    """Constraint class for fixing the center of mass.

    References

    https://pubs.acs.org/doi/abs/10.1021/jp9722824

    """

    def __init__(self):

        self.removed_dof = 3

    def adjust_positions(self, atoms, new):
        masses = atoms.get_masses()
        old_cm = atoms.get_center_of_mass()
        new_cm = np.dot(masses, new) / masses.sum()
        d = old_cm - new_cm
        new += d

    def adjust_forces(self, atoms, forces):
        m = atoms.get_masses()
        mm = np.tile(m, (3, 1)).T
        lb = np.sum(mm * forces, axis=0) / sum(m**2)
        forces -= mm * lb

    def todict(self):
        return {'name': 'FixCom',
                'kwargs': {}}


def ints2string(x, threshold=None):
    """Convert ndarray of ints to string."""
    if threshold is None or len(x) <= threshold:
        return str(x.tolist())
    return str(x[:threshold].tolist())[:-1] + ', ...]'


class FixBondLengths(FixConstraint):
    maxiter = 500

    def __init__(self, pairs, tolerance=1e-13,
                 bondlengths=None, iterations=None):
        """iterations:
                Ignored"""
        self.pairs = np.asarray(pairs)
        self.tolerance = tolerance
        self.bondlengths = bondlengths

        self.removed_dof = len(pairs)

    def adjust_positions(self, atoms, new):
        old = atoms.positions
        masses = atoms.get_masses()

        if self.bondlengths is None:
            self.bondlengths = self.initialize_bond_lengths(atoms)

        for i in range(self.maxiter):
            converged = True
            for j, ab in enumerate(self.pairs):
                a = ab[0]
                b = ab[1]
                cd = self.bondlengths[j]
                r0 = old[a] - old[b]
                d0 = find_mic([r0], atoms.cell, atoms._pbc)[0][0]
                d1 = new[a] - new[b] - r0 + d0
                m = 1 / (1 / masses[a] + 1 / masses[b])
                x = 0.5 * (cd**2 - np.dot(d1, d1)) / np.dot(d0, d1)
                if abs(x) > self.tolerance:
                    new[a] += x * m / masses[a] * d0
                    new[b] -= x * m / masses[b] * d0
                    converged = False
            if converged:
                break
        else:
            raise RuntimeError('Did not converge')

    def adjust_momenta(self, atoms, p):
        old = atoms.positions
        masses = atoms.get_masses()

        if self.bondlengths is None:
            self.bondlengths = self.initialize_bond_lengths(atoms)

        for i in range(self.maxiter):
            converged = True
            for j, ab in enumerate(self.pairs):
                a = ab[0]
                b = ab[1]
                cd = self.bondlengths[j]
                d = old[a] - old[b]
                d = find_mic([d], atoms.cell, atoms._pbc)[0][0]
                dv = p[a] / masses[a] - p[b] / masses[b]
                m = 1 / (1 / masses[a] + 1 / masses[b])
                x = -np.dot(dv, d) / cd**2
                if abs(x) > self.tolerance:
                    p[a] += x * m * d
                    p[b] -= x * m * d
                    converged = False
            if converged:
                break
        else:
            raise RuntimeError('Did not converge')

    def adjust_forces(self, atoms, forces):
        self.constraint_forces = -forces
        self.adjust_momenta(atoms, forces)
        self.constraint_forces += forces

    def initialize_bond_lengths(self, atoms):
        bondlengths = np.zeros(len(self.pairs))

        for i, ab in enumerate(self.pairs):
            bondlengths[i] = atoms.get_distance(ab[0], ab[1], mic=True)

        return bondlengths

    def get_indices(self):
        return np.unique(self.pairs.ravel())

    def todict(self):
        return {'name': 'FixBondLengths',
                'kwargs': {'pairs': self.pairs,
                           'tolerance': self.tolerance}}

    def index_shuffle(self, atoms, ind):
        """Shuffle the indices of the two atoms in this constraint"""
        map = np.zeros(len(atoms), int)
        map[ind] = 1
        n = map.sum()
        map[:] = -1
        map[ind] = range(n)
        pairs = map[self.pairs]
        self.pairs = pairs[(pairs != -1).all(1)]
        if len(self.pairs) == 0:
            raise IndexError('Constraint not part of slice')


def FixBondLength(a1, a2):
    """Fix distance between atoms with indices a1 and a2."""
    return FixBondLengths([(a1, a2)])


class FixedMode(FixConstraint):
    """Constrain atoms to move along directions orthogonal to
    a given mode only."""

    def __init__(self, mode):
        self.mode = (np.asarray(mode) / np.sqrt((mode**2).sum())).reshape(-1)

    def adjust_positions(self, atoms, newpositions):
        newpositions = newpositions.ravel()
        oldpositions = atoms.positions.ravel()
        step = newpositions - oldpositions
        newpositions -= self.mode * np.dot(step, self.mode)

    def adjust_forces(self, atoms, forces):
        forces = forces.ravel()
        forces -= self.mode * np.dot(forces, self.mode)

    def index_shuffle(self, atoms, ind):
        eps = 1e-12
        mode = self.mode.reshape(-1, 3)
        excluded = np.ones(len(mode), dtype=bool)
        excluded[ind] = False
        if (abs(mode[excluded]) > eps).any():
            raise IndexError('All nonzero parts of mode not in slice')
        self.mode = mode[ind].ravel()

    def get_indices(self):
        # This function will never properly work because it works on all
        # atoms and it has no idea how to tell how many atoms it is
        # attached to.  If it is being used, surely the user knows
        # everything is being constrained.
        return []

    def todict(self):
        return {'name': 'FixedMode',
                'kwargs': {'mode': self.mode}}

    def __repr__(self):
        return 'FixedMode(%s)' % self.mode.tolist()


class FixedPlane(FixConstraintSingle):
    """Constrain an atom index *a* to move in a given plane only.

    The plane is defined by its normal vector *direction*."""

    removed_dof = 1

    def __init__(self, a, direction):
        self.a = a
        self.dir = np.asarray(direction) / sqrt(np.dot(direction, direction))

    def adjust_positions(self, atoms, newpositions):
        step = newpositions[self.a] - atoms.positions[self.a]
        newpositions[self.a] -= self.dir * np.dot(step, self.dir)

    def adjust_forces(self, atoms, forces):
        forces[self.a] -= self.dir * np.dot(forces[self.a], self.dir)

    def todict(self):
        return {'name': 'FixedPlane',
                'kwargs': {'a': self.a, 'direction': self.dir}}

    def __repr__(self):
        return 'FixedPlane(%d, %s)' % (self.a, self.dir.tolist())


class FixedLine(FixConstraintSingle):
    """Constrain an atom index *a* to move on a given line only.

    The line is defined by its vector *direction*."""

    removed_dof = 2

    def __init__(self, a, direction):
        self.a = a
        self.dir = np.asarray(direction) / sqrt(np.dot(direction, direction))

    def adjust_positions(self, atoms, newpositions):
        step = newpositions[self.a] - atoms.positions[self.a]
        x = np.dot(step, self.dir)
        newpositions[self.a] = atoms.positions[self.a] + x * self.dir

    def adjust_forces(self, atoms, forces):
        forces[self.a] = self.dir * np.dot(forces[self.a], self.dir)

    def __repr__(self):
        return 'FixedLine(%d, %s)' % (self.a, self.dir.tolist())

    def todict(self):
        return {'name': 'FixedLine',
                'kwargs': {'a': self.a, 'direction': self.dir}}


class FixCartesian(FixConstraintSingle):
    'Fix an atom index *a* in the directions of the cartesian coordinates.'

    def __init__(self, a, mask=(1, 1, 1)):
        self.a = a
        self.mask = ~np.asarray(mask, bool)
        self.removed_dof = 3 - self.mask.sum()

    def adjust_positions(self, atoms, new):
        step = new[self.a] - atoms.positions[self.a]
        step *= self.mask
        new[self.a] = atoms.positions[self.a] + step

    def adjust_forces(self, atoms, forces):
        forces[self.a] *= self.mask

    def __repr__(self):
        return 'FixCartesian(a={0}, mask={1})'.format(self.a,
                                                      list(~self.mask))

    def todict(self):
        return {'name': 'FixCartesian',
                'kwargs': {'a': self.a, 'mask': ~self.mask}}


class FixScaled(FixConstraintSingle):
    'Fix an atom index *a* in the directions of the unit vectors.'

    def __init__(self, cell, a, mask=(1, 1, 1)):
        self.cell = np.asarray(cell)
        self.a = a
        self.mask = np.array(mask, bool)
        self.removed_dof = self.mask.sum()

    def adjust_positions(self, atoms, new):
        scaled_old = np.linalg.solve(self.cell.T, atoms.positions.T).T
        scaled_new = np.linalg.solve(self.cell.T, new.T).T
        for n in range(3):
            if self.mask[n]:
                scaled_new[self.a, n] = scaled_old[self.a, n]
        new[self.a] = np.dot(scaled_new, self.cell)[self.a]

    def adjust_forces(self, atoms, forces):
        scaled_forces = np.linalg.solve(self.cell.T, forces.T).T
        scaled_forces[self.a] *= -(self.mask - 1)
        forces[self.a] = np.dot(scaled_forces, self.cell)[self.a]

    def todict(self):
        return {'name': 'FixScaled',
                'kwargs': {'a': self.a,
                           'cell': self.cell,
                           'mask': self.mask}}

    def __repr__(self):
        return 'FixScaled(%s, %d, %s)' % (repr(self.cell),
                                          self.a,
                                          repr(self.mask))


# TODO: Better interface might be to use dictionaries in place of very
# nested lists/tuples
class FixInternals(FixConstraint):
    """Constraint object for fixing multiple internal coordinates.

    Allows fixing bonds, angles, and dihedrals."""

    def __init__(self, bonds=None, angles=None, dihedrals=None,
                 epsilon=1.e-7):
        self.bonds = bonds or []
        self.angles = angles or []
        self.dihedrals = dihedrals or []

        # Initialize these at run-time:
        self.n = 0
        self.constraints = []
        self.epsilon = epsilon

        self.initialized = False
        self.removed_dof = (len(self.bonds) +
                            len(self.angles) +
                            len(self.dihedrals))

    def initialize(self, atoms):
        if self.initialized:
            return
        masses = atoms.get_masses()
        self.n = len(self.bonds) + len(self.angles) + len(self.dihedrals)
        self.constraints = []
        for bond in self.bonds:
            masses_bond = masses.take(bond[1])
            self.constraints.append(self.FixBondLengthAlt(bond[0], bond[1],
                                                          masses_bond))
        for angle in self.angles:
            masses_angle = masses.take(angle[1])
            self.constraints.append(self.FixAngle(angle[0], angle[1],
                                                  masses_angle))
        for dihedral in self.dihedrals:
            masses_dihedral = masses.take(dihedral[1])
            self.constraints.append(self.FixDihedral(dihedral[0],
                                                     dihedral[1],
                                                     masses_dihedral))
        self.initialized = True

    def get_indices(self):
        cons = self.bonds + self.dihedrals + self.angles
        return np.unique(np.ravel([constraint[1]
                                   for constraint in cons]))

    def todict(self):
        return {'name': 'FixInternals',
                'kwargs': {'bonds': self.bonds,
                           'angles': self.angles,
                           'dihedrals': self.dihedrals,
                           'epsilon': self.epsilon}}

    def adjust_positions(self, atoms, new):
        self.initialize(atoms)
        for constraint in self.constraints:
            constraint.set_h_vectors(atoms.positions)
        for j in range(50):
            maxerr = 0.0
            for constraint in self.constraints:
                constraint.adjust_positions(atoms.positions, new)
                maxerr = max(abs(constraint.sigma), maxerr)
            if maxerr < self.epsilon:
                return
        raise ValueError('Shake did not converge.')

    def adjust_forces(self, atoms, forces):
        """Project out translations and rotations and all other constraints"""
        self.initialize(atoms)
        positions = atoms.positions
        N = len(forces)
        list2_constraints = list(np.zeros((6, N, 3)))
        tx, ty, tz, rx, ry, rz = list2_constraints

        list_constraints = [r.ravel() for r in list2_constraints]

        tx[:, 0] = 1.0
        ty[:, 1] = 1.0
        tz[:, 2] = 1.0
        ff = forces.ravel()

        # Calculate the center of mass
        center = positions.sum(axis=0) / N

        rx[:, 1] = -(positions[:, 2] - center[2])
        rx[:, 2] = positions[:, 1] - center[1]
        ry[:, 0] = positions[:, 2] - center[2]
        ry[:, 2] = -(positions[:, 0] - center[0])
        rz[:, 0] = -(positions[:, 1] - center[1])
        rz[:, 1] = positions[:, 0] - center[0]

        # Normalizing transl., rotat. constraints
        for r in list2_constraints:
            r /= np.linalg.norm(r.ravel())

        # Add all angle, etc. constraint vectors
        for constraint in self.constraints:
            constraint.adjust_forces(positions, forces)
            list_constraints.insert(0, constraint.h)
        # QR DECOMPOSITION - GRAM SCHMIDT

        list_constraints = [r.ravel() for r in list_constraints]
        aa = np.column_stack(list_constraints)
        (aa, bb) = np.linalg.qr(aa)
        # Projection
        hh = []
        for i, constraint in enumerate(self.constraints):
            hh.append(aa[:, i] * np.row_stack(aa[:, i]))

        txx = aa[:, self.n] * np.row_stack(aa[:, self.n])
        tyy = aa[:, self.n + 1] * np.row_stack(aa[:, self.n + 1])
        tzz = aa[:, self.n + 2] * np.row_stack(aa[:, self.n + 2])
        rxx = aa[:, self.n + 3] * np.row_stack(aa[:, self.n + 3])
        ryy = aa[:, self.n + 4] * np.row_stack(aa[:, self.n + 4])
        rzz = aa[:, self.n + 5] * np.row_stack(aa[:, self.n + 5])
        T = txx + tyy + tzz + rxx + ryy + rzz
        for vec in hh:
            T += vec
        ff = np.dot(T, np.row_stack(ff))
        forces[:, :] -= np.dot(T, np.row_stack(ff)).reshape(-1, 3)

    def __repr__(self):
        constraints = repr(self.constraints)
        return 'FixInternals(_copy_init=%s, epsilon=%s)' % (constraints,
                                                            repr(self.epsilon))

    def __str__(self):
        return '\n'.join([repr(c) for c in self.constraints])

    # Classes for internal use in FixInternals
    class FixBondLengthAlt:
        """Constraint subobject for fixing bond length within FixInternals."""

        def __init__(self, bond, indices, masses, maxstep=0.01):
            """Fix distance between atoms with indices a1, a2."""
            self.indices = indices
            self.bond = bond
            self.h1 = None
            self.h2 = None
            self.masses = masses
            self.h = []
            self.sigma = 1.

        def set_h_vectors(self, pos):
            dist1 = pos[self.indices[0]] - pos[self.indices[1]]
            self.h1 = 2 * dist1
            self.h2 = -self.h1

        def adjust_positions(self, old, new):
            h1 = self.h1 / self.masses[0]
            h2 = self.h2 / self.masses[1]
            dist1 = new[self.indices[0]] - new[self.indices[1]]
            dist = np.dot(dist1, dist1)
            self.sigma = dist - self.bond**2
            lamda = -self.sigma / (2 * np.dot(dist1, (h1 - h2)))
            new[self.indices[0]] += lamda * h1
            new[self.indices[1]] += lamda * h2

        def adjust_forces(self, positions, forces):
            self.h1 = 2 * (positions[self.indices[0]] -
                           positions[self.indices[1]])
            self.h2 = -self.h1
            self.h = np.zeros([len(forces) * 3])
            self.h[(self.indices[0]) * 3] = self.h1[0]
            self.h[(self.indices[0]) * 3 + 1] = self.h1[1]
            self.h[(self.indices[0]) * 3 + 2] = self.h1[2]
            self.h[(self.indices[1]) * 3] = self.h2[0]
            self.h[(self.indices[1]) * 3 + 1] = self.h2[1]
            self.h[(self.indices[1]) * 3 + 2] = self.h2[2]
            self.h /= np.linalg.norm(self.h)

        def __repr__(self):
            return 'FixBondLengthAlt(%s, %d, %d)' % \
                (repr(self.bond), self.indices[0], self.indices[1])

    class FixAngle:
        """Constraint object for fixing an angle within
        FixInternals."""

        def __init__(self, angle, indices, masses):
            """Fix atom movement to construct a constant angle."""
            self.indices = indices
            self.a1m, self.a2m, self.a3m = masses
            self.angle = np.cos(angle)
            self.h1 = self.h2 = self.h3 = None
            self.h = []
            self.sigma = 1.

        def set_h_vectors(self, pos):
            r21 = pos[self.indices[0]] - pos[self.indices[1]]
            r21_len = np.linalg.norm(r21)
            e21 = r21 / r21_len
            r23 = pos[self.indices[2]] - pos[self.indices[1]]
            r23_len = np.linalg.norm(r23)
            e23 = r23 / r23_len
            angle = np.dot(e21, e23)
            self.h1 = -2 * angle * ((angle * e21 - e23) / (r21_len))
            self.h3 = -2 * angle * ((angle * e23 - e21) / (r23_len))
            self.h2 = -(self.h1 + self.h3)

        def adjust_positions(self, oldpositions, newpositions):
            r21 = newpositions[self.indices[0]] - newpositions[self.indices[1]]
            r21_len = np.linalg.norm(r21)
            e21 = r21 / r21_len
            r23 = newpositions[self.indices[2]] - newpositions[self.indices[1]]
            r23_len = np.linalg.norm(r23)
            e23 = r23 / r23_len
            angle = np.dot(e21, e23)
            self.sigma = (angle - self.angle) * (angle + self.angle)
            h1 = self.h1 / self.a1m
            h3 = self.h3 / self.a3m
            h2 = self.h2 / self.a2m
            h21 = h1 - h2
            h23 = h3 - h2
            # Calculating new positions
            deriv = (((np.dot(r21, h23) + np.dot(r23, h21)) /
                      (r21_len * r23_len)) -
                     (np.dot(r21, h21) / (r21_len * r21_len) +
                      np.dot(r23, h23) / (r23_len * r23_len)) * angle)
            deriv *= 2 * angle
            lamda = -self.sigma / deriv
            newpositions[self.indices[0]] += lamda * h1
            newpositions[self.indices[1]] += lamda * h2
            newpositions[self.indices[2]] += lamda * h3

        def adjust_forces(self, positions, forces):
            r21 = positions[self.indices[0]] - positions[self.indices[1]]
            r21_len = np.linalg.norm(r21)
            e21 = r21 / r21_len
            r23 = positions[self.indices[2]] - positions[self.indices[1]]
            r23_len = np.linalg.norm(r23)
            e23 = r23 / r23_len
            angle = np.dot(e21, e23)
            self.h1 = -2 * angle * (angle * e21 - e23) / r21_len
            self.h3 = -2 * angle * (angle * e23 - e21) / r23_len
            self.h2 = -(self.h1 + self.h3)
            self.h = np.zeros([len(positions) * 3])
            self.h[(self.indices[0]) * 3] = self.h1[0]
            self.h[(self.indices[0]) * 3 + 1] = self.h1[1]
            self.h[(self.indices[0]) * 3 + 2] = self.h1[2]
            self.h[(self.indices[1]) * 3] = self.h2[0]
            self.h[(self.indices[1]) * 3 + 1] = self.h2[1]
            self.h[(self.indices[1]) * 3 + 2] = self.h2[2]
            self.h[(self.indices[2]) * 3] = self.h3[0]
            self.h[(self.indices[2]) * 3 + 1] = self.h3[1]
            self.h[(self.indices[2]) * 3 + 2] = self.h3[2]
            self.h /= np.linalg.norm(self.h)

        def __repr__(self):
            return 'FixAngle(%s, %f)' % (tuple(self.indices),
                                         np.arccos(self.angle))

    class FixDihedral:
        """Constraint object for fixing an dihedral using
        the shake algorithm. This one allows also other constraints."""

        def __init__(self, angle, indices, masses):
            """Fix atom movement to construct a constant dihedral angle."""
            self.indices = indices
            self.a1m, self.a2m, self.a3m, self.a4m = masses
            self.angle = np.cos(angle)
            self.h1 = self.h2 = self.h3 = self.h4 = None
            self.h = []
            self.sigma = 1.

        def set_h_vectors(self, pos):
            r12 = pos[self.indices[1]] - pos[self.indices[0]]
            r23 = pos[self.indices[2]] - pos[self.indices[1]]
            r23_len = np.linalg.norm(r23)
            e23 = r23 / r23_len
            r34 = pos[self.indices[3]] - pos[self.indices[2]]
            a = -r12 - np.dot(-r12, e23) * e23
            a_len = np.linalg.norm(a)
            ea = a / a_len
            b = r34 - np.dot(r34, e23) * e23
            b_len = np.linalg.norm(b)
            eb = b / b_len
            angle = np.dot(ea, eb).clip(-1.0, 1.0)
            self.h1 = (eb - angle * ea) / a_len
            self.h4 = (ea - angle * eb) / b_len
            self.h2 = self.h1 * (np.dot(-r12, e23) / r23_len - 1)
            self.h2 += np.dot(r34, e23) / r23_len * self.h4
            self.h3 = -self.h4 * (np.dot(r34, e23) / r23_len + 1)
            self.h3 += np.dot(r12, e23) / r23_len * self.h1

        def adjust_positions(self, oldpositions, newpositions):
            r12 = newpositions[self.indices[1]] - newpositions[self.indices[0]]
            r23 = newpositions[self.indices[2]] - newpositions[self.indices[1]]
            r34 = newpositions[self.indices[3]] - newpositions[self.indices[2]]
            n1 = np.cross(r12, r23)
            n1_len = np.linalg.norm(n1)
            n1e = n1 / n1_len
            n2 = np.cross(r23, r34)
            n2_len = np.linalg.norm(n2)
            n2e = n2 / n2_len
            angle = np.dot(n1e, n2e).clip(-1.0, 1.0)
            self.sigma = (angle - self.angle) * (angle + self.angle)
            h1 = self.h1 / self.a1m
            h2 = self.h2 / self.a2m
            h3 = self.h3 / self.a3m
            h4 = self.h4 / self.a4m
            h12 = h2 - h1
            h23 = h3 - h2
            h34 = h4 - h3
            deriv = ((np.dot(n1, np.cross(r34, h23) + np.cross(h34, r23)) +
                      np.dot(n2, np.cross(r23, h12) + np.cross(h23, r12))) /
                     (n1_len * n2_len))
            deriv -= (((np.dot(n1, np.cross(r23, h12) + np.cross(h23, r12)) /
                        n1_len**2) +
                       (np.dot(n2, np.cross(r34, h23) + np.cross(h34, r23)) /
                        n2_len**2)) * angle)
            deriv *= -2 * angle
            lamda = -self.sigma / deriv
            newpositions[self.indices[0]] += lamda * h1
            newpositions[self.indices[1]] += lamda * h2
            newpositions[self.indices[2]] += lamda * h3
            newpositions[self.indices[3]] += lamda * h4

        def adjust_forces(self, positions, forces):
            r12 = positions[self.indices[1]] - positions[self.indices[0]]
            r23 = positions[self.indices[2]] - positions[self.indices[1]]
            r23_len = np.linalg.norm(r23)
            e23 = r23 / r23_len
            r34 = positions[self.indices[3]] - positions[self.indices[2]]
            a = -r12 - np.dot(-r12, e23) * e23
            a_len = np.linalg.norm(a)
            ea = a / a_len
            b = r34 - np.dot(r34, e23) * e23
            b_len = np.linalg.norm(b)
            eb = b / b_len
            angle = np.dot(ea, eb).clip(-1.0, 1.0)
            self.h1 = (eb - angle * ea) / a_len
            self.h4 = (ea - angle * eb) / b_len
            self.h2 = self.h1 * (np.dot(-r12, e23) / r23_len - 1)
            self.h2 += np.dot(r34, e23) / r23_len * self.h4
            self.h3 = -self.h4 * (np.dot(r34, e23) / r23_len + 1)
            self.h3 -= np.dot(-r12, e23) / r23_len * self.h1

            self.h = np.zeros([len(positions) * 3])
            self.h[(self.indices[0]) * 3] = self.h1[0]
            self.h[(self.indices[0]) * 3 + 1] = self.h1[1]
            self.h[(self.indices[0]) * 3 + 2] = self.h1[2]
            self.h[(self.indices[1]) * 3] = self.h2[0]
            self.h[(self.indices[1]) * 3 + 1] = self.h2[1]
            self.h[(self.indices[1]) * 3 + 2] = self.h2[2]
            self.h[(self.indices[2]) * 3] = self.h3[0]
            self.h[(self.indices[2]) * 3 + 1] = self.h3[1]
            self.h[(self.indices[2]) * 3 + 2] = self.h3[2]
            self.h[(self.indices[3]) * 3] = self.h4[0]
            self.h[(self.indices[3]) * 3 + 1] = self.h4[1]
            self.h[(self.indices[3]) * 3 + 2] = self.h4[2]
            self.h /= np.linalg.norm(self.h)

        def __repr__(self):
            return 'FixDihedral(%s, %f)' % (tuple(self.indices), self.angle)


class Hookean(FixConstraint):
    """Applies a Hookean restorative force between a pair of atoms, an atom
    and a point, or an atom and a plane."""

    def __init__(self, a1, a2, k, rt=None):
        """Forces two atoms to stay close together by applying no force if
        they are below a threshold length, rt, and applying a Hookean
        restorative force when the distance between them exceeds rt. Can
        also be used to tether an atom to a fixed point in space or to a
        distance above a plane.

        a1 : int
           Index of atom 1
        a2 : one of three options
           1) index of atom 2
           2) a fixed point in cartesian space to which to tether a1
           3) a plane given as (A, B, C, D) in A x + B y + C z + D = 0.
        k : float
           Hooke's law (spring) constant to apply when distance
           exceeds threshold_length. Units of eV A^-2.
        rt : float
           The threshold length below which there is no force. The
           length is 1) between two atoms, 2) between atom and point.
           This argument is not supplied in case 3. Units of A.

        If a plane is specified, the Hooke's law force is applied if the atom
        is on the normal side of the plane. For instance, the plane with
        (A, B, C, D) = (0, 0, 1, -7) defines a plane in the xy plane with a z
        intercept of +7 and a normal vector pointing in the +z direction.
        If the atom has z > 7, then a downward force would be applied of
        k * (atom.z - 7). The same plane with the normal vector pointing in
        the -z direction would be given by (A, B, C, D) = (0, 0, -1, 7).
        """

        if isinstance(a2, int):
            self._type = 'two atoms'
            self.indices = [a1, a2]
        elif len(a2) == 3:
            self._type = 'point'
            self.index = a1
            self.origin = np.array(a2)
        elif len(a2) == 4:
            self._type = 'plane'
            self.index = a1
            self.plane = a2
        else:
            raise RuntimeError('Unknown type for a2')
        self.threshold = rt
        self.spring = k

    def todict(self):
        dct = {'name': 'Hookean'}
        dct['kwargs'] = {'rt': self.threshold,
                         'k': self.spring}
        if self._type == 'two atoms':
            dct['kwargs']['a1'] = self.indices[0]
            dct['kwargs']['a2'] = self.indices[1]
        elif self._type == 'point':
            dct['kwargs']['a1'] = self.index
            dct['kwargs']['a2'] = self.origin
        elif self._type == 'plane':
            dct['kwargs']['a1'] = self.index
            dct['kwargs']['a2'] = self.plane
        else:
            raise NotImplementedError('Bad type: %s' % self._type)
        return dct

    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_momenta(self, atoms, momenta):
        pass

    def adjust_forces(self, atoms, forces):
        positions = atoms.positions
        if self._type == 'plane':
            A, B, C, D = self.plane
            x, y, z = positions[self.index]
            d = ((A * x + B * y + C * z + D) /
                 np.sqrt(A**2 + B**2 + C**2))
            if d < 0:
                return
            magnitude = self.spring * d
            direction = - np.array((A, B, C)) / np.linalg.norm((A, B, C))
            forces[self.index] += direction * magnitude
            return
        if self._type == 'two atoms':
            p1, p2 = positions[self.indices]
        elif self._type == 'point':
            p1 = positions[self.index]
            p2 = self.origin
        displace = find_mic([p2 - p1], atoms.cell, atoms._pbc)[0][0]
        bondlength = np.linalg.norm(displace)
        if bondlength > self.threshold:
            magnitude = self.spring * (bondlength - self.threshold)
            direction = displace / np.linalg.norm(displace)
            if self._type == 'two atoms':
                forces[self.indices[0]] += direction * magnitude
                forces[self.indices[1]] -= direction * magnitude
            else:
                forces[self.index] += direction * magnitude

    def adjust_potential_energy(self, atoms):
        """Returns the difference to the potential energy due to an active
        constraint. (That is, the quantity returned is to be added to the
        potential energy.)"""
        positions = atoms.positions
        if self._type == 'plane':
            A, B, C, D = self.plane
            x, y, z = positions[self.index]
            d = ((A * x + B * y + C * z + D) /
                 np.sqrt(A**2 + B**2 + C**2))
            if d > 0:
                return 0.5 * self.spring * d**2
            else:
                return 0.
        if self._type == 'two atoms':
            p1, p2 = positions[self.indices]
        elif self._type == 'point':
            p1 = positions[self.index]
            p2 = self.origin
        displace = find_mic([p2 - p1], atoms.cell, atoms._pbc)[0][0]
        bondlength = np.linalg.norm(displace)
        if bondlength > self.threshold:
            return 0.5 * self.spring * (bondlength - self.threshold)**2
        else:
            return 0.

    def get_indices(self):
        if self._type == 'two atoms':
            return self.indices
        elif self._type == 'point':
            return self.index
        elif self._type == 'plane':
            return self.index

    def index_shuffle(self, atoms, ind):
        # See docstring of superclass
        if self._type == 'two atoms':
            newa = [-1, -1]  # Signal error
            for new, old in slice2enlist(ind, len(atoms)):
                for i, a in enumerate(self.indices):
                    if old == a:
                        newa[i] = new
            if newa[0] == -1 or newa[1] == -1:
                raise IndexError('Constraint not part of slice')
            self.indices = newa
        elif (self._type == 'point') or (self._type == 'plane'):
            newa = -1   # Signal error
            for new, old in slice2enlist(ind, len(atoms)):
                if old == self.index:
                    newa = new
                    break
            if newa == -1:
                raise IndexError('Constraint not part of slice')
            self.index = newa

    def __repr__(self):
        if self._type == 'two atoms':
            return 'Hookean(%d, %d)' % tuple(self.indices)
        elif self._type == 'point':
            return 'Hookean(%d) to cartesian' % self.index
        else:
            return 'Hookean(%d) to plane' % self.index


class ExternalForce(FixConstraint):
    """Constraint object for pulling two atoms apart by an external force.

    You can combine this constraint for example with FixBondLength but make
    sure that the ExternalForce-constraint comes first in the list:

    >>> con1 = ExternalForce(atom1, atom2, f_ext)
    >>> con2 = FixBondLength(atom3, atom4)
    >>> atoms.set_constraint([con1, con2])

    see ase/test/external_force.py"""

    def __init__(self, a1, a2, f_ext):
        self.indices = [a1, a2]
        self.external_force = f_ext

    def adjust_positions(self, atoms, new):
        pass

    def adjust_forces(self, atoms, forces):
        dist = np.subtract.reduce(atoms.positions[self.indices])
        force = self.external_force * dist / np.linalg.norm(dist)
        forces[self.indices] += (force, -force)

    def adjust_potential_energy(self, atoms):
        dist = np.subtract.reduce(atoms.positions[self.indices])
        return -np.linalg.norm(dist) * self.external_force

    def index_shuffle(self, atoms, ind):
        """Shuffle the indices of the two atoms in this constraint"""
        newa = [-1, -1]  # Signal error
        for new, old in slice2enlist(ind, len(atoms)):
            for i, a in enumerate(self.indices):
                if old == a:
                    newa[i] = new
        if newa[0] == -1 or newa[1] == -1:
            raise IndexError('Constraint not part of slice')
        self.indices = newa

    def __repr__(self):
        return 'ExternalForce(%d, %d, %f)' % (self.indices[0],
                                              self.indices[1],
                                              self.external_force)

    def todict(self):
        return {'name': 'ExternalForce',
                'kwargs': {'a1': self.indices[0], 'a2': self.indices[1],
                           'f_ext': self.external_force}}


class Filter:
    """Subset filter class."""

    def __init__(self, atoms, indices=None, mask=None):
        """Filter atoms.

        This filter can be used to hide degrees of freedom in an Atoms
        object.

        Parameters
        ----------
        indices : list of int
           Indices for those atoms that should remain visible.
        mask : list of bool
           One boolean per atom indicating if the atom should remain
           visible or not.

        If a Trajectory tries to save this object, it will instead
        save the underlying Atoms object.  To prevent this, override
        the iterimages method.
        """

        self.atoms = atoms
        self.constraints = []
        # Make self.info a reference to the underlying atoms' info dictionary.
        self.info = self.atoms.info

        if indices is None and mask is None:
            raise ValueError('Use "indices" or "mask".')
        if indices is not None and mask is not None:
            raise ValueError('Use only one of "indices" and "mask".')

        if mask is not None:
            self.index = np.asarray(mask, bool)
            self.n = self.index.sum()
        else:
            self.index = np.asarray(indices, int)
            self.n = len(self.index)

    def iterimages(self):
        # Present the real atoms object to Trajectory and friends
        return self.atoms.iterimages()

    def get_cell(self):
        """Returns the computational cell.

        The computational cell is the same as for the original system.
        """
        return self.atoms.get_cell()

    def get_pbc(self):
        """Returns the periodic boundary conditions.

        The boundary conditions are the same as for the original system.
        """
        return self.atoms.get_pbc()

    def get_positions(self):
        'Return the positions of the visible atoms.'
        return self.atoms.get_positions()[self.index]

    def set_positions(self, positions, **kwargs):
        'Set the positions of the visible atoms.'
        pos = self.atoms.get_positions()
        pos[self.index] = positions
        self.atoms.set_positions(pos, **kwargs)

    positions = property(get_positions, set_positions,
                         doc='Positions of the atoms')

    def get_momenta(self):
        'Return the momenta of the visible atoms.'
        return self.atoms.get_momenta()[self.index]

    def set_momenta(self, momenta, **kwargs):
        'Set the momenta of the visible atoms.'
        mom = self.atoms.get_momenta()
        mom[self.index] = momenta
        self.atoms.set_momenta(mom, **kwargs)

    def get_atomic_numbers(self):
        'Return the atomic numbers of the visible atoms.'
        return self.atoms.get_atomic_numbers()[self.index]

    def set_atomic_numbers(self, atomic_numbers):
        'Set the atomic numbers of the visible atoms.'
        z = self.atoms.get_atomic_numbers()
        z[self.index] = atomic_numbers
        self.atoms.set_atomic_numbers(z)

    def get_tags(self):
        'Return the tags of the visible atoms.'
        return self.atoms.get_tags()[self.index]

    def set_tags(self, tags):
        'Set the tags of the visible atoms.'
        tg = self.atoms.get_tags()
        tg[self.index] = tags
        self.atoms.set_tags(tg)

    def get_forces(self, *args, **kwargs):
        return self.atoms.get_forces(*args, **kwargs)[self.index]

    def get_stress(self):
        return self.atoms.get_stress()

    def get_stresses(self):
        return self.atoms.get_stresses()[self.index]

    def get_masses(self):
        return self.atoms.get_masses()[self.index]

    def get_potential_energy(self, **kwargs):
        """Calculate potential energy.

        Returns the potential energy of the full system.
        """
        return self.atoms.get_potential_energy(**kwargs)

    def get_chemical_symbols(self):
        return self.atoms.get_chemical_symbols()

    def get_initial_magnetic_moments(self):
        return self.atoms.get_initial_magnetic_moments()

    def get_calculator(self):
        """Returns the calculator.

        WARNING: The calculator is unaware of this filter, and sees a
        different number of atoms.
        """
        return self.atoms.get_calculator()

    def get_celldisp(self):
        return self.atoms.get_celldisp()

    def has(self, name):
        'Check for existence of array.'
        return self.atoms.has(name)

    def __len__(self):
        'Return the number of movable atoms.'
        return self.n

    def __getitem__(self, i):
        'Return an atom.'
        return self.atoms[self.index[i]]


class StrainFilter(Filter):
    """Modify the supercell while keeping the scaled positions fixed.

    Presents the strain of the supercell as the generalized positions,
    and the global stress tensor (times the volume) as the generalized
    force.

    This filter can be used to relax the unit cell until the stress is
    zero.  If MDMin is used for this, the timestep (dt) to be used
    depends on the system size. 0.01/x where x is a typical dimension
    seems like a good choice.

    The stress and strain are presented as 6-vectors, the order of the
    components follow the standard engingeering practice: xx, yy, zz,
    yz, xz, xy.

    """

    def __init__(self, atoms, mask=None):
        """Create a filter applying a homogeneous strain to a list of atoms.

        The first argument, atoms, is the atoms object.

        The optional second argument, mask, is a list of six booleans,
        indicating which of the six independent components of the
        strain that are allowed to become non-zero.  It defaults to
        [1,1,1,1,1,1].

        """

        self.strain = np.zeros(6)

        if mask is None:
            mask = np.ones(6)
        else:
            mask = np.array(mask)

        Filter.__init__(self, atoms, mask=mask)
        self.mask = mask
        self.origcell = atoms.get_cell()

    def get_positions(self):
        return self.strain.reshape((2, 3)).copy()

    def set_positions(self, new):
        new = new.ravel() * self.mask
        eps = np.array([[1.0 + new[0], 0.5 * new[5], 0.5 * new[4]],
                        [0.5 * new[5], 1.0 + new[1], 0.5 * new[3]],
                        [0.5 * new[4], 0.5 * new[3], 1.0 + new[2]]])

        self.atoms.set_cell(np.dot(self.origcell, eps), scale_atoms=True)
        self.strain[:] = new

    def get_forces(self):
        stress = self.atoms.get_stress()
        return -self.atoms.get_volume() * (stress * self.mask).reshape((2, 3))

    def has(self, x):
        return self.atoms.has(x)

    def __len__(self):
        return 2


# The indices of the full stiffness matrix of (orthorhombic) interest
voigt_notation = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def full_3x3_to_voigt_6_index(i, j):
    if i == j:
        return i
    return 6 - i - j


def voigt_6_to_full_3x3_strain(strain_vector):
    """
    Form a 3x3 strain matrix from a 6 component vector in Voigt notation
    """
    e1, e2, e3, e4, e5, e6 = np.transpose(strain_vector)
    return np.transpose([[1.0 + e1, 0.5 * e6, 0.5 * e5],
                         [0.5 * e6, 1.0 + e2, 0.5 * e4],
                         [0.5 * e5, 0.5 * e4, 1.0 + e3]])


def voigt_6_to_full_3x3_stress(stress_vector):
    """
    Form a 3x3 stress matrix from a 6 component vector in Voigt notation
    """
    s1, s2, s3, s4, s5, s6 = np.transpose(stress_vector)
    return np.transpose([[s1, s6, s5],
                         [s6, s2, s4],
                         [s5, s4, s3]])


def full_3x3_to_voigt_6_strain(strain_matrix):
    """
    Form a 6 component strain vector in Voigt notation from a 3x3 matrix
    """
    strain_matrix = np.asarray(strain_matrix)
    return np.transpose([strain_matrix[..., 0, 0] - 1.0,
                         strain_matrix[..., 1, 1] - 1.0,
                         strain_matrix[..., 2, 2] - 1.0,
                         strain_matrix[..., 1, 2] + strain_matrix[..., 2, 1],
                         strain_matrix[..., 0, 2] + strain_matrix[..., 2, 0],
                         strain_matrix[..., 0, 1] + strain_matrix[..., 1, 0]])


def full_3x3_to_voigt_6_stress(stress_matrix):
    """
    Form a 6 component stress vector in Voigt notation from a 3x3 matrix
    """
    stress_matrix = np.asarray(stress_matrix)
    return np.transpose([stress_matrix[..., 0, 0],
                         stress_matrix[..., 1, 1],
                         stress_matrix[..., 2, 2],
                         (stress_matrix[..., 1, 2] +
                          stress_matrix[..., 1, 2]) / 2,
                         (stress_matrix[..., 0, 2] +
                          stress_matrix[..., 0, 2]) / 2,
                         (stress_matrix[..., 0, 1] +
                          stress_matrix[..., 0, 1]) / 2])


class UnitCellFilter(Filter):
    """Modify the supercell and the atom positions. """
    def __init__(self, atoms, mask=None,
                 cell_factor=None,
                 hydrostatic_strain=False,
                 constant_volume=False,
                 scalar_pressure=0.0):
        """Create a filter that returns the atomic forces and unit cell
        stresses together, so they can simultaneously be minimized.

        The first argument, atoms, is the atoms object. The optional second
        argument, mask, is a list of booleans, indicating which of the six
        independent components of the strain are relaxed.

        - True = relax to zero
        - False = fixed, ignore this component

        Degrees of freedom are the positions in the original undeformed cell,
        plus the deformation tensor (extra 3 "atoms"). This gives forces
        consistent with numerical derivatives of the potential energy
        with respect to the cell degreees of freedom.

        For full details see:
            E. B. Tadmor, G. S. Smith, N. Bernstein, and E. Kaxiras,
            Phys. Rev. B 59, 235 (1999)

        You can still use constraints on the atoms, e.g. FixAtoms, to control
        the relaxation of the atoms.

        >>> # this should be equivalent to the StrainFilter
        >>> atoms = Atoms(...)
        >>> atoms.set_constraint(FixAtoms(mask=[True for atom in atoms]))
        >>> ucf = UnitCellFilter(atoms)

        You should not attach this UnitCellFilter object to a
        trajectory. Instead, create a trajectory for the atoms, and
        attach it to an optimizer like this:

        >>> atoms = Atoms(...)
        >>> ucf = UnitCellFilter(atoms)
        >>> qn = QuasiNewton(ucf)
        >>> traj = Trajectory('TiO2.traj', 'w', atoms)
        >>> qn.attach(traj)
        >>> qn.run(fmax=0.05)

        Helpful conversion table:

        - 0.05 eV/A^3   = 8 GPA
        - 0.003 eV/A^3  = 0.48 GPa
        - 0.0006 eV/A^3 = 0.096 GPa
        - 0.0003 eV/A^3 = 0.048 GPa
        - 0.0001 eV/A^3 = 0.02 GPa

        Additional optional arguments:

        cell_factor: float (default float(len(atoms)))
            Factor by which deformation gradient is multiplied to put
            it on the same scale as the positions when assembling
            the combined position/cell vector. The stress contribution to
            the forces is scaled down by the same factor. This can be thought
            of as a very simple preconditioners. Default is number of atoms
            which gives approximately the correct scaling.

        hydrostatic_strain: bool (default False)
            Constrain the cell by only allowing hydrostatic deformation.
            The virial tensor is replaced by np.diag([np.trace(virial)]*3).

        constant_volume: bool (default False)
            Project out the diagonal elements of the virial tensor to allow
            relaxations at constant volume, e.g. for mapping out an
            energy-volume curve. Note: this only approximately conserves
            the volume and breaks energy/force consistency so can only be
            used with optimizers that do require do a line minimisation
            (e.g. FIRE).

        scalar_pressure: float (default 0.0)
            Applied pressure to use for enthalpy pV term. As above, this
            breaks energy/force consistency.
        """

        Filter.__init__(self, atoms, indices=range(len(atoms)))
        self.atoms = atoms
        self.deform_grad = np.eye(3)
        self.atom_positions = atoms.get_positions()
        self.orig_cell = atoms.get_cell()
        self.stress = None

        if mask is None:
            mask = np.ones(6)
        mask = np.asarray(mask)
        if mask.shape == (6,):
            self.mask = voigt_6_to_full_3x3_stress(mask)
        elif mask.shape == (3, 3):
            self.mask = mask
        else:
            raise ValueError('shape of mask should be (3,3) or (6,)')

        if cell_factor is None:
            cell_factor = float(len(atoms))
        self.hydrostatic_strain = hydrostatic_strain
        self.constant_volume = constant_volume
        self.scalar_pressure = scalar_pressure
        self.cell_factor = cell_factor
        self.copy = self.atoms.copy
        self.arrays = self.atoms.arrays

    def get_positions(self):
        '''
        this returns an array with shape (natoms + 3,3).

        the first natoms rows are the positions of the atoms, the last
        three rows are the deformation tensor associated with the unit cell,
        scaled by self.cell_factor.
        '''

        natoms = len(self.atoms)
        pos = np.zeros((natoms + 3, 3))
        pos[:natoms] = self.atom_positions
        pos[natoms:] = self.cell_factor * self.deform_grad
        return pos

    def set_positions(self, new, **kwargs):
        '''
        new is an array with shape (natoms+3,3).

        the first natoms rows are the positions of the atoms, the last
        three rows are the deformation tensor used to change the cell shape.

        the positions are first set with respect to the original
        undeformed cell, and then the cell is transformed by the
        current deformation gradient.
        '''

        natoms = len(self.atoms)
        self.atom_positions[:] = new[:natoms]
        self.deform_grad = new[natoms:] / self.cell_factor
        self.atoms.set_positions(self.atom_positions, **kwargs)
        self.atoms.set_cell(self.orig_cell, scale_atoms=False)
        self.atoms.set_cell(np.dot(self.orig_cell, self.deform_grad.T),
                            scale_atoms=True)

    def get_potential_energy(self, force_consistent=True):
        '''
        returns potential energy including enthalpy PV term.
        '''
        atoms_energy = self.atoms.get_potential_energy(
            force_consistent=force_consistent)
        return atoms_energy + self.scalar_pressure * self.atoms.get_volume()

    def get_forces(self, apply_constraint=False):
        '''
        returns an array with shape (natoms+3,3) of the atomic forces
        and unit cell stresses.

        the first natoms rows are the forces on the atoms, the last
        three rows are the forces on the unit cell, which are
        computed from the stress tensor.
        '''

        atoms_forces = self.atoms.get_forces()
        stress = self.atoms.get_stress()

        volume = self.atoms.get_volume()
        virial = -volume * (voigt_6_to_full_3x3_stress(stress) +
                            np.diag([self.scalar_pressure] * 3))
        atoms_forces = np.dot(atoms_forces, self.deform_grad)
        dg_inv = np.linalg.inv(self.deform_grad)
        virial = np.dot(virial, dg_inv.T)

        if self.hydrostatic_strain:
            vtr = virial.trace()
            virial = np.diag([vtr / 3.0, vtr / 3.0, vtr / 3.0])

        # Zero out components corresponding to fixed lattice elements
        if (self.mask != 1.0).any():
            virial *= self.mask

        if self.constant_volume:
            vtr = virial.trace()
            np.fill_diagonal(virial, np.diag(virial) - vtr / 3.0)

        natoms = len(self.atoms)
        forces = np.zeros((natoms + 3, 3))
        forces[:natoms] = atoms_forces
        forces[natoms:] = virial / self.cell_factor

        self.stress = -full_3x3_to_voigt_6_stress(virial)/volume
        return forces

    def get_stress(self):
        raise PropertyNotImplementedError

    def has(self, x):
        return self.atoms.has(x)

    def __len__(self):
        return (len(self.atoms) + 3)


class ExpCellFilter(UnitCellFilter):
    """Modify the supercell and the atom positions."""
    def __init__(self, atoms, mask=None,
                 cell_factor=None,
                 hydrostatic_strain=False,
                 constant_volume=False,
                 scalar_pressure=0.0):
        r"""Create a filter that returns the atomic forces and unit cell
        stresses together, so they can simultaneously be minimized.

        The first argument, atoms, is the atoms object. The optional second
        argument, mask, is a list of booleans, indicating which of the six
        independent components of the strain are relaxed.

        - True = relax to zero
        - False = fixed, ignore this component

        Degrees of freedom are the positions in the original undeformed cell,
        plus the log of the deformation tensor (extra 3 "atoms"). This gives forces
        consistent with numerical derivatives of the potential energy
        with respect to the cell degrees of freedom.

        For full details see:
            E. B. Tadmor, G. S. Smith, N. Bernstein, and E. Kaxiras,
            Phys. Rev. B 59, 235 (1999)

        You can still use constraints on the atoms, e.g. FixAtoms, to control
        the relaxation of the atoms.

        >>> # this should be equivalent to the StrainFilter
        >>> atoms = Atoms(...)
        >>> atoms.set_constraint(FixAtoms(mask=[True for atom in atoms]))
        >>> ucf = UnitCellFilter(atoms)

        You should not attach this UnitCellFilter object to a
        trajectory. Instead, create a trajectory for the atoms, and
        attach it to an optimizer like this:

        >>> atoms = Atoms(...)
        >>> ucf = UnitCellFilter(atoms)
        >>> qn = QuasiNewton(ucf)
        >>> traj = Trajectory('TiO2.traj', 'w', atoms)
        >>> qn.attach(traj)
        >>> qn.run(fmax=0.05)

        Helpful conversion table:

        - 0.05 eV/A^3   = 8 GPA
        - 0.003 eV/A^3  = 0.48 GPa
        - 0.0006 eV/A^3 = 0.096 GPa
        - 0.0003 eV/A^3 = 0.048 GPa
        - 0.0001 eV/A^3 = 0.02 GPa

        Additional optional arguments:

        cell_factor: (DEPRECATED)
            Retained for backwards compatibility, but no longer used.

        hydrostatic_strain: bool (default False)
            Constrain the cell by only allowing hydrostatic deformation.
            The virial tensor is replaced by np.diag([np.trace(virial)]*3).

        constant_volume: bool (default False)
            Project out the diagonal elements of the virial tensor to allow
            relaxations at constant volume, e.g. for mapping out an
            energy-volume curve.

        scalar_pressure: float (default 0.0)
            Applied pressure to use for enthalpy pV term. As above, this
            breaks energy/force consistency.

        Implementation details:

        The implementation is based on that of Christoph Ortner in JuLIP.jl:
        https://github.com/libAtoms/JuLIP.jl/blob/expcell/src/Constraints.jl#L244

        We decompose the deformation gradient as

            F = exp(U) F0
            x =  F * F0^{-1} z  = exp(U) z

        If we write the energy as a function of U we can transform the
        stress associated with a perturbation V into a derivative using a linear map
        V -> L(U, V).

        \phi( exp(U+tV) (z+tv) ) ~ \phi'(x) . (exp(U) v) + \phi'(x) . ( L(U, V) exp(-U) exp(U) z )
           >>> \nabla E(U) : V  =  [S exp(-U)'] : L(U,V)
                                =  L'(U, S exp(-U)') : V
                                =  L(U', S exp(-U)') : V
                                =  L(U, S exp(-U)) : V     (provided U = U')

        where the : operator represents double contraction, i.e. A:B = trace(A'B), and

          F = deformation tensor - 3x3 matrix
          F0 = reference deformation tensor - 3x3 matrix, np.eye(3) here
          U = cell degrees of freedom used here - 3x3 matrix
          V = perturbation to cell DoFs - 3x3 matrix
          v = perturbation to position DoFs
          x = atomic positions in deformed cell
          z = atomic positions in original cell
          \phi = potential energy
          S = stress tensor [3x3 matrix]
          L(U, V) = directional derivative of exp at U in direction V, i.e
          d/dt exp(U + t V)|_{t=0} = L(U, V)

        This means we can write

          d/dt E(U + t V)|_{t=0} = L(U, S exp (-U)) : V

        and therefore the contribution to the gradient of the energy is

          \nabla E(U) / \nabla U_ij =  [L(U, S exp(-U))]_ij

        """

        Filter.__init__(self, atoms, indices=range(len(atoms)))
        self.atoms = atoms
        self.deform_grad = np.eye(3)
        self.deform_grad_log = np.zeros((3,3))
        self.atom_positions = atoms.get_positions()
        self.orig_cell = atoms.get_cell()
        self.stress = None

        if mask is None:
            mask = np.ones(6)
        mask = np.asarray(mask)
        if mask.shape == (6,):
            self.mask = voigt_6_to_full_3x3_stress(mask)
        elif mask.shape == (3, 3):
            self.mask = mask
        else:
            raise ValueError('shape of mask should be (3,3) or (6,)')

        if cell_factor is not None:
            warn("cell_factor is no longer used")
        self.hydrostatic_strain = hydrostatic_strain
        self.constant_volume = constant_volume
        self.scalar_pressure = scalar_pressure
        self.copy = self.atoms.copy
        self.arrays = self.atoms.arrays

    def get_positions(self):
        '''
        this returns an array with shape (natoms + 3,3).

        the first natoms rows are the positions of the atoms, the last
        three rows are the log of the deformation tensor associated with
        the unit cell.
        '''

        natoms = len(self.atoms)
        pos = np.zeros((natoms + 3, 3))
        pos[:natoms] = self.atom_positions
        pos[natoms:] = self.deform_grad_log
        return pos

    def set_positions(self, new, **kwargs):
        '''
        new is an array with shape (natoms+3,3).

        the first natoms rows are the positions of the atoms, the last
        three rows are the deformation tensor used to change the cell shape.

        the positions are first set with respect to the original
        undeformed cell, and then the cell is transformed by the
        current deformation gradient.
        '''

        natoms = len(self.atoms)
        self.atom_positions[:] = new[:natoms]
        self.deform_grad_log = new[natoms:]
        self.deform_grad = expm(self.deform_grad_log)
        self.atoms.set_positions(self.atom_positions, **kwargs)
        self.atoms.set_cell(self.orig_cell, scale_atoms=False)
        self.atoms.set_cell(np.dot(self.orig_cell, self.deform_grad.T),
                            scale_atoms=True)

    def get_potential_energy(self, force_consistent=True):
        '''
        returns potential energy including enthalpy PV term.
        '''
        atoms_energy = self.atoms.get_potential_energy(force_consistent=force_consistent)
        return atoms_energy + self.scalar_pressure*self.atoms.get_volume()

    def get_forces(self, apply_constraint=False):
        '''
        returns an array with shape (natoms+2,3) of the atomic forces
        and unit cell stresses.

        the first natoms rows are the forces on the atoms, the last
        three rows are the forces on the unit cell, which are
        computed from the stress tensor.
        '''

        atoms_forces = self.atoms.get_forces()
        stress = self.atoms.get_stress()

        volume = self.atoms.get_volume()
        virial = -volume * voigt_6_to_full_3x3_stress(stress) - np.diag([self.scalar_pressure]*3)*volume
        atoms_forces = np.dot(atoms_forces, self.deform_grad)

        if self.hydrostatic_strain:
            vtr = virial.trace()
            virial = np.diag([vtr / 3.0, vtr / 3.0, vtr / 3.0])

        # Zero out components corresponding to fixed lattice elements
        if (self.mask != 1.0).any():
            virial *= self.mask

        deform_grad_log_force_naive = virial.copy()
        Y = np.zeros((6,6))
        Y[0:3,0:3] = self.deform_grad_log
        Y[3:6,3:6] = self.deform_grad_log
        Y[0:3,3:6] = -np.dot(virial,expm(-self.deform_grad_log))
        deform_grad_log_force = -expm(Y)[0:3,3:6]
        for (i1,i2) in [(0,1),(0,2),(1,2)]:
            ff = 0.5*(deform_grad_log_force[i1,i2] + deform_grad_log_force[i2,i1])
            deform_grad_log_force[i1,i2] = ff
            deform_grad_log_force[i2,i1] = ff

        # check for reasonable alignment between naive and exact search directions
        if (np.sum(deform_grad_log_force*deform_grad_log_force_naive) /
            np.sqrt(np.sum(deform_grad_log_force**2) * np.sum(deform_grad_log_force_naive**2)) > 0.8):
            deform_grad_log_force = deform_grad_log_force_naive

        # Cauchy stress used for convergence testing
        convergence_crit_stress = -(virial/volume)
        if self.constant_volume:
            # apply constraint to force
            dglf_trace = deform_grad_log_force.trace()
            np.fill_diagonal(deform_grad_log_force, np.diag(deform_grad_log_force) - dglf_trace / 3.0)
            # apply constraint to Cauchy stress used for convergence testing
            ccs_trace = convergence_crit_stress.trace()
            np.fill_diagonal(convergence_crit_stress, np.diag(convergence_crit_stress) - ccs_trace / 3.0)

        # pack gradients into vector
        natoms = len(self.atoms)
        forces = np.zeros((natoms + 3, 3))
        forces[:natoms] = atoms_forces
        forces[natoms:] = deform_grad_log_force

        self.stress = full_3x3_to_voigt_6_stress(convergence_crit_stress)

        return forces

    def get_stress(self):
        raise PropertyNotImplementedError

    def has(self, x):
        return self.atoms.has(x)

    def __len__(self):
        return (len(self.atoms) + 3)
