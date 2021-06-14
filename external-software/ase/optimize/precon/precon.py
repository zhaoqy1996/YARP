"""
Implementation of the Precon abstract base class and subclasses
"""
from __future__ import print_function

#import time
import warnings

import numpy as np
from scipy import sparse, rand
from scipy.sparse.linalg import spsolve

from ase.constraints import Filter, FixAtoms
from ase.utils import longsum
from ase.geometry import wrap_positions
import ase.utils.ff as ff
import ase.units as units
from ase.optimize.precon.neighbors import (get_neighbours,
                                           estimate_nearest_neighbour_distance)
try:
    from pyamg import smoothed_aggregation_solver
    have_pyamg = True
except ImportError:
    have_pyamg = False

THz = 1e12 * 1. / units.s


class Precon(object):

    def __init__(self, r_cut=None, r_NN=None,
                 mu=None, mu_c=None,
                 dim=3, c_stab=0.1, force_stab=False,
                 recalc_mu=False, array_convention='C',
                 solver="auto", solve_tol=1e-8,
                 apply_positions=True, apply_cell=True,
                 estimate_mu_eigmode=False):
        """Initialise a preconditioner object based on passed parameters.

        Args:
            r_cut: float. This is a cut-off radius. The preconditioner matrix
                will be created by considering pairs of atoms that are within a
                distance r_cut of each other. For a regular lattice, this is
                usually taken somewhere between the first- and second-nearest
                neighbour distance. If r_cut is not provided, default is
                2 * r_NN (see below)
            r_NN: nearest neighbour distance. If not provided, this is
                  calculated
                from input structure.
            mu: float
                energy scale for position degreees of freedom. If `None`, mu
                is precomputed using finite difference derivatives.
            mu_c: float
                energy scale for cell degreees of freedom. Also precomputed
                if None.
            estimate_mu_eigmode:
                If True, estimates mu based on the lowest eigenmodes of
                unstabilised preconditioner. If False it uses the sine based
                approach.
            dim: int; dimensions of the problem
            c_stab: float. The diagonal of the preconditioner matrix will have
                a stabilisation constant added, which will be the value of
                c_stab times mu.
            force_stab:
                If True, always add the stabilisation to diagnonal, regardless
                of the presence of fixed atoms.
            recalc_mu: if True, the value of mu will be recalculated every time
                self.make_precon is called. This can be overridden in specific
                cases with recalc_mu argument in self.make_precon. If recalc_mu
                is set to True here, the value passed for mu will be
                irrelevant unless recalc_mu is set False the first time
                make_precon is called.
            array_convention: Either 'C' or 'F' for Fortran; this will change
                the preconditioner to reflect the ordering of the indices in
                the vector it will operate on. The C convention assumes the
                vector will be arranged atom-by-atom (ie [x1, y1, z1, x2, ...])
                while the F convention assumes it will be arranged component
                by component (ie [x1, x2, ..., y1, y2, ...]).
            solver: One of "auto", "direct" or "pyamg", specifying whether to use
               a direct sparse solver or PyAMG to solve P x = y. Default is "auto" which
               uses PyAMG if available, falling back to sparse solver if not.
            solve_tol: tolerance used for PyAMG sparse linear solver,
            if available.
            apply_positions: if True, apply preconditioner to position DoF
            apply_cell: if True, apply preconditioner to cell DoF

        Raises:
            ValueError for problem with arguments

        """

        self.r_NN = r_NN
        self.r_cut = r_cut
        self.mu = mu
        self.mu_c = mu_c
        self.estimate_mu_eigmode = estimate_mu_eigmode
        self.c_stab = c_stab
        self.force_stab = force_stab
        self.array_convention = array_convention
        self.recalc_mu = recalc_mu
        self.P = None
        self.old_positions = None

        use_pyamg = False
        if solver == "auto":
            use_pyamg = have_pyamg
        elif solver == "direct":
            use_pyamg = False
        elif solver == "pyamg":
            if not have_pyamg:
                raise RuntimeError('solver="pyamg" but PyAMG cannot be imported!')
            use_pyamg = True
        else:
            raise ValueError('unknown solver - should be "auto", "direct" or "pyamg"')

        self.use_pyamg = use_pyamg
        self.solve_tol = solve_tol
        self.apply_positions = apply_positions
        self.apply_cell = apply_cell

        if dim < 1:
            raise ValueError('Dimension must be at least 1')
        self.dim = dim

    def make_precon(self, atoms, recalc_mu=None):
        """Create a preconditioner matrix based on the passed set of atoms.

        Creates a general-purpose preconditioner for use with optimization
        algorithms, based on examining distances between pairs of atoms in the
        lattice. The matrix will be stored in the attribute self.P and
        returned.

        Args:
            atoms: the Atoms object used to create the preconditioner.
                Can also
            recalc_mu: if True, self.mu (and self.mu_c for variable cell)
                will be recalculated by calling self.estimate_mu(atoms)
                before the preconditioner matrix is created. If False, self.mu
                will be calculated only if it does not currently have a value
                (ie, the first time this function is called).

        Returns:
            A two-element tuple:
                P: A sparse scipy csr_matrix. BE AWARE that using
                    numpy.dot() with sparse matrices will result in
                    errors/incorrect results - use the .dot method directly
                    on the matrix instead.
        """

        if self.r_NN is None:
            self.r_NN = estimate_nearest_neighbour_distance(atoms)

        if self.r_cut is None:
            # This is the first time this function has been called, and no
            # cutoff radius has been specified, so calculate it automatically.
            self.r_cut = 2.0 * self.r_NN
        elif self.r_cut < self.r_NN:
            warning = ('WARNING: r_cut (%.2f) < r_NN (%.2f), '
                       'increasing to 1.1*r_NN = %.2f' % (self.r_cut,
                                                          self.r_NN,
                                                          1.1 * self.r_NN))
            warnings.warn(warning)
            self.r_cut = 1.1 * self.r_NN

        if recalc_mu is None:
            # The caller has not specified whether or not to recalculate mu,
            # so the Precon's setting is used.
            recalc_mu = self.recalc_mu

        if self.mu is None:
            # Regardless of what the caller has specified, if we don't
            # currently have a value of mu, then we need one.
            recalc_mu = True

        if recalc_mu:
            self.estimate_mu(atoms)

        if self.P is not None:
            real_atoms = atoms
            if isinstance(atoms, Filter):
                real_atoms = atoms.atoms
            if self.old_positions is None:
                self.old_positions = wrap_positions(real_atoms.positions,
                                                    real_atoms.cell)
            displacement = wrap_positions(real_atoms.positions,
                                          real_atoms.cell) - self.old_positions
            self.old_positions = real_atoms.get_positions()
            max_abs_displacement = abs(displacement).max()
            #print('max(abs(displacements)) = %.2f A (%.2f r_NN)' %
            #            (max_abs_displacement, max_abs_displacement / self.r_NN))
            if max_abs_displacement < 0.5 * self.r_NN:
                return self.P

        #start_time = time.time()

        # Create the preconditioner:
        self._make_sparse_precon(atoms, force_stab=self.force_stab)

        #print('--- Precon created in %s seconds ---' %
        #            (time.time() - start_time))
        return self.P

    def _make_sparse_precon(self, atoms, initial_assembly=False,
                            force_stab=False):
        """Create a sparse preconditioner matrix based on the passed atoms.

        Creates a general-purpose preconditioner for use with optimization
        algorithms, based on examining distances between pairs of atoms in the
        lattice. The matrix will be stored in the attribute self.P and
        returned. Note that this function will use self.mu, whatever it is.

        Args:
            atoms: the Atoms object used to create the preconditioner.

        Returns:
            A scipy.sparse.csr_matrix object, representing a d*N by d*N matrix
            (where N is the number of atoms, and d is the value of self.dim).
            BE AWARE that using numpy.dot() with this object will result in
            errors/incorrect results - use the .dot method directly on the
            sparse matrix instead.

        """
#         print('creating sparse precon: initial_assembly=%r, '
#               'force_stab=%r, apply_positions=%r, apply_cell=%r' %
#                (initial_assembly, force_stab, self.apply_positions,
#                self.apply_cell))

        N = len(atoms)
        diag_i = np.arange(N, dtype=int)
        #start_time = time.time()
        if self.apply_positions:
            # compute neighbour list
            i, j, rij, fixed_atoms = get_neighbours(atoms, self.r_cut)
            #print('--- neighbour list created in %s s ---' %
            #            ((time.time() - start_time)))

            # compute entries in triplet format: without the constraints
            #start_time = time.time()
            coeff = self.get_coeff(rij)
            diag_coeff = np.bincount(i, -coeff, minlength=N).astype(np.float64)
            if force_stab or len(fixed_atoms) == 0:
                #print('adding stabilisation to preconditioner')
                diag_coeff += self.mu * self.c_stab
        else:
            diag_coeff = np.ones(N)

        # precon is mu_c*identity for cell DoF
        if isinstance(atoms, Filter):
            if self.apply_cell:
                diag_coeff[-3] = self.mu_c
                diag_coeff[-2] = self.mu_c
                diag_coeff[-1] = self.mu_c
            else:
                diag_coeff[-3] = 1.0
                diag_coeff[-2] = 1.0
                diag_coeff[-1] = 1.0
        #print('--- computed triplet format in %s s ---' %
        #            (time.time() - start_time))

        if self.apply_positions and not initial_assembly:
            # apply the constraints
            #start_time = time.time()
            mask = np.ones(N)
            mask[fixed_atoms] = 0.0
            coeff *= mask[i] * mask[j]
            diag_coeff[fixed_atoms] = 1.0
            #print('--- applied fixed_atoms in %s s ---' %
            #            (time.time() - start_time))

        if self.apply_positions:
            # remove zeros
            #start_time = time.time()
            inz = np.nonzero(coeff)
            i = np.hstack((i[inz], diag_i))
            j = np.hstack((j[inz], diag_i))
            coeff = np.hstack((coeff[inz], diag_coeff))
            #print('--- remove zeros in %s s ---' %
            #            (time.time() - start_time))
        else:
            i = diag_i
            j = diag_i
            coeff = diag_coeff

        # create the matrix
        #start_time = time.time()
        csc_P = sparse.csc_matrix((coeff, (i, j)), shape=(N, N))
        #print('--- created CSC matrix in %s s ---' %
        #            (time.time() - start_time))

        self.csc_P = csc_P

        #start_time = time.time()
        if self.dim == 1:
            self.P = csc_P
        elif self.array_convention == 'F':
            csc_P = csc_P.tocsr()
            self.P = csc_P
            for i in range(self.dim - 1):
                self.P = sparse.block_diag((self.P, csc_P)).tocsr()
        else:
            # convert back to triplet and read the arrays
            csc_P = csc_P.tocoo()
            i = csc_P.row * self.dim
            j = csc_P.col * self.dim
            z = csc_P.data

            # N-dimensionalise, interlaced coordinates
            I = np.hstack([i + d for d in range(self.dim)])
            J = np.hstack([j + d for d in range(self.dim)])
            Z = np.hstack([z for d in range(self.dim)])
            self.P = sparse.csc_matrix((Z, (I, J)),
                                       shape=(self.dim * N, self.dim * N))
            self.P = self.P.tocsr()
        #print('--- N-dim precon created in %s s ---' %
        #            (time.time() - start_time))

        # Create solver
        if self.use_pyamg and have_pyamg:
            #start_time = time.time()
            self.ml = smoothed_aggregation_solver(
                self.P, B=None,
                strength=('symmetric', {'theta': 0.0}),
                smooth=(
                    'jacobi', {'filter': True, 'weighting': 'local'}),
                improve_candidates=[('block_gauss_seidel',
                                     {'sweep': 'symmetric', 'iterations': 4}),
                                    None, None, None, None, None, None, None,
                                    None, None, None, None, None, None, None],
                aggregate='standard',
                presmoother=('block_gauss_seidel',
                             {'sweep': 'symmetric', 'iterations': 1}),
                postsmoother=('block_gauss_seidel',
                              {'sweep': 'symmetric', 'iterations': 1}),
                max_levels=15,
                max_coarse=300,
                coarse_solver='pinv')
            #print('--- multi grid solver created in %s s ---' %
            #            (time.time() - start_time))

        return self.P

    def dot(self, x, y):
        """
        Return the preconditioned dot product <P x, y>

        Uses 128-bit floating point math for vector dot products
        """
        return longsum(self.P.dot(x) * y)

    def solve(self, x):
        """
        Solve the (sparse) linear system P x = y and return y
        """
        #start_time = time.time()
        if self.use_pyamg and have_pyamg:
            y = self.ml.solve(x, x0=rand(self.P.shape[0]),
                              tol=self.solve_tol,
                              accel='cg',
                              maxiter=300,
                              cycle='W')
        else:
            y = spsolve(self.P, x)
        #print('--- Precon applied in %s seconds ---' %
        #            (time.time() - start_time))
        return y

    def get_coeff(self, r):
        raise NotImplementedError('Must be overridden by subclasses')

    def estimate_mu(self, atoms, H=None):
        r"""Estimate optimal preconditioner coefficient \mu

        \mu is estimated from a numerical solution of

            [dE(p+v) -  dE(p)] \cdot v = \mu < P1 v, v >

        with perturbation

            v(x,y,z) = H P_lowest_nonzero_eigvec(x, y, z)

            or

            v(x,y,z) = H (sin(x / Lx), sin(y / Ly), sin(z / Lz))

        After the optimal \mu is found, self.mu will be set to its value.

        If `atoms` is an instance of Filter an additional \mu_c
        will be computed for the cell degrees of freedom .

        Args:
            atoms: Atoms object for initial system

            H: 3x3 array or None
                Magnitude of deformation to apply.
                Default is 1e-2*rNN*np.eye(3)

        Returns:
            mu   : float
            mu_c : float or None
        """

        if self.dim != 3:
            raise ValueError('Automatic calculation of mu only possible for '
                             'three-dimensional preconditioners. Try setting '
                             'mu manually instead.')

        if self.r_NN is None:
            self.r_NN = estimate_nearest_neighbour_distance(atoms)

        # deformation matrix, default is diagonal
        if H is None:
            H = 1e-2 * self.r_NN * np.eye(3)

        # compute perturbation
        p = atoms.get_positions()

        if self.estimate_mu_eigmode:
            self.mu = 1.0
            self.mu_c = 1.0
            c_stab = self.c_stab
            self.c_stab = 0.0

            if isinstance(atoms, Filter):
                n = len(atoms.atoms)
            else:
                n = len(atoms)
            P0 = self._make_sparse_precon(atoms,
                                          initial_assembly=True)[:3 * n,
                                                                 :3 * n]
            eigvals, eigvecs = sparse.linalg.eigsh(P0, k=4, which='SM')

            #print('estimate_mu(): lowest 4 eigvals = %f %f %f %f'
            #             % (eigvals[0], eigvals[1], eigvals[2], eigvals[3]))
            # check eigenvalues
            if any(eigvals[0:3] > 1e-6):
                raise ValueError('First 3 eigenvalues of preconditioner matrix'
                                 'do not correspond to translational modes.')
            elif eigvals[3] < 1e-6:
                raise ValueError('Fourth smallest eigenvalue of '
                                 'preconditioner matrix '
                                 'is too small, increase r_cut.')

            x = np.zeros(n)
            for i in range(n):
                x[i] = eigvecs[:, 3][3 * i]
            x = x / np.linalg.norm(x)
            if x[0] < 0:
                x = -x

            v = np.zeros(3 * len(atoms))
            for i in range(n):
                v[3 * i] = x[i]
                v[3 * i + 1] = x[i]
                v[3 * i + 2] = x[i]
            v = v / np.linalg.norm(v)
            v = v.reshape((-1, 3))

            self.c_stab = c_stab
        else:
            Lx, Ly, Lz = [p[:, i].max() - p[:, i].min() for i in range(3)]
            #print('estimate_mu(): Lx=%.1f Ly=%.1f Lz=%.1f' % (Lx, Ly, Lz))

            x, y, z = p.T
            # sine_vr = [np.sin(x/Lx), np.sin(y/Ly), np.sin(z/Lz)], but we need
            # to take into account the possibility that one of Lx/Ly/Lz is
            # zero.
            sine_vr = [x, y, z]

            for i, L in enumerate([Lx, Ly, Lz]):
                if L == 0:
                    warnings.warn(
                        'Cell length L[%d] == 0. Setting H[%d,%d] = 0.' %
                        (i, i, i))
                    H[i, i] = 0.0
                else:
                    sine_vr[i] = np.sin(sine_vr[i] / L)

            v = np.dot(H, sine_vr).T

        natoms = len(atoms)
        if isinstance(atoms, Filter):
            natoms = len(atoms.atoms)
            eps = H / self.r_NN
            v[natoms:, :] = eps

        v1 = v.reshape(-1)

        # compute LHS
        dE_p = -atoms.get_forces().reshape(-1)
        atoms_v = atoms.copy()
        atoms_v.set_calculator(atoms.get_calculator())
        if isinstance(atoms, Filter):
            atoms_v = atoms.__class__(atoms_v)
            if hasattr(atoms, 'constant_volume'):
                atoms_v.constant_volume = atoms.constant_volume
        atoms_v.set_positions(p + v)
        dE_p_plus_v = -atoms_v.get_forces().reshape(-1)

        # compute left hand side
        LHS = (dE_p_plus_v - dE_p) * v1

        # assemble P with \mu = 1
        self.mu = 1.0
        self.mu_c = 1.0

        P1 = self._make_sparse_precon(atoms, initial_assembly=True)

        # compute right hand side
        RHS = P1.dot(v1) * v1

        # use partial sums to compute separate mu for positions and cell DoFs
        self.mu = longsum(LHS[:3 * natoms]) / longsum(RHS[:3 * natoms])
        if self.mu < 1.0:
            warnings.warn('mu (%.3f) < 1.0, capping at mu=1.0' % self.mu)
            self.mu = 1.0

        if isinstance(atoms, Filter):
            self.mu_c = longsum(LHS[3 * natoms:]) / longsum(RHS[3 * natoms:])
            if self.mu_c < 1.0:
                print(
                    'mu_c (%.3f) < 1.0, capping at mu_c=1.0' % self.mu_c)
                self.mu_c = 1.0

        print('estimate_mu(): mu=%r, mu_c=%r' % (self.mu, self.mu_c))

        self.P = None  # force a rebuild with new mu (there may be fixed atoms)
        return (self.mu, self.mu_c)


class Pfrommer(object):
    """Use initial guess for inverse Hessian from Pfrommer et al. as a
    simple preconditioner

    J. Comput. Phys. vol 131 p233-240 (1997)
    """

    def __init__(self, bulk_modulus=500 * units.GPa, phonon_frequency=50 * THz,
                 apply_positions=True, apply_cell=True):
        """
        Default bulk modulus is 500 GPa and default phonon frequency is 50 THz
        """

        self.bulk_modulus = bulk_modulus
        self.phonon_frequency = phonon_frequency
        self.apply_positions = apply_positions
        self.apply_cell = apply_cell
        self.H0 = None

    def make_precon(self, atoms):
        if self.H0 is not None:
            # only build H0 on first call
            return NotImplemented

        variable_cell = False
        if isinstance(atoms, Filter):
            variable_cell = True
            atoms = atoms.atoms

        # position DoF
        omega = self.phonon_frequency
        mass = atoms.get_masses().mean()
        block = np.eye(3) / (mass * omega**2)
        blocks = [block] * len(atoms)

        # cell DoF
        if variable_cell:
            coeff = 1.0
            if self.apply_cell:
                coeff = 1.0 / (3 * self.bulk_modulus)
            blocks.append(np.diag([coeff] * 9))

        self.H0 = sparse.block_diag(blocks, format='csr')
        return NotImplemented

    def dot(self, x, y):
        """
        Return the preconditioned dot product <P x, y>

        Uses 128-bit floating point math for vector dot products
        """
        raise NotImplementedError

    def solve(self, x):
        """
        Solve the (sparse) linear system P x = y and return y
        """
        y = self.H0.dot(x)
        return y


class C1(Precon):
    """Creates matrix by inserting a constant whenever r_ij is less than r_cut.
    """

    def __init__(self, r_cut=None, mu=None, mu_c=None, dim=3, c_stab=0.1,
                 force_stab=False,
                 recalc_mu=False, array_convention='C',
                 solver="auto", solve_tol=1e-9,
                 apply_positions=True, apply_cell=True):
        Precon.__init__(self, r_cut=r_cut, mu=mu, mu_c=mu_c,
                        dim=dim, c_stab=c_stab,
                        force_stab=force_stab,
                        recalc_mu=recalc_mu,
                        array_convention=array_convention,
                        solver=solver, solve_tol=solve_tol,
                        apply_positions=apply_positions,
                        apply_cell=apply_cell)

    def get_coeff(self, r):
        return -self.mu * np.ones_like(r)


class Exp(Precon):
    """Creates matrix with values decreasing exponentially with distance.
    """

    def __init__(self, A=3.0, r_cut=None, r_NN=None, mu=None, mu_c=None,
                 dim=3, c_stab=0.1,
                 force_stab=False, recalc_mu=False, array_convention='C',
                 solver="auto", solve_tol=1e-9,
                 apply_positions=True, apply_cell=True,
                 estimate_mu_eigmode=False):
        """Initialise an Exp preconditioner with given parameters.

        Args:
            r_cut, mu, c_stab, dim, sparse, recalc_mu, array_convention: see
                precon.__init__()
            A: coefficient in exp(-A*r/r_NN). Default is A=3.0.
        """
        Precon.__init__(self, r_cut=r_cut, r_NN=r_NN,
                        mu=mu, mu_c=mu_c, dim=dim, c_stab=c_stab,
                        force_stab=force_stab,
                        recalc_mu=recalc_mu,
                        array_convention=array_convention,
                        solver=solver,
                        solve_tol=solve_tol,
                        apply_positions=apply_positions,
                        apply_cell=apply_cell,
                        estimate_mu_eigmode=estimate_mu_eigmode)

        self.A = A

    def get_coeff(self, r):
        return -self.mu * np.exp(-self.A * (r / self.r_NN - 1))


class FF(Precon):
    """Creates matrix using morse/bond/angle/dihedral force field parameters.
    """

    def __init__(self, dim=3, c_stab=0.1, force_stab=False,
                 array_convention='C', solver="auto", solve_tol=1e-9,
                 apply_positions=True, apply_cell=True,
                 hessian='spectral', morses=None, bonds=None, angles=None,
                 dihedrals=None):
        """Initialise an FF preconditioner with given parameters.

        Args:
             dim, c_stab, force_stab, array_convention: see
             precon.__init__(), use_pyamg, solve_tol
             morses: class Morse
             bonds: class Bond
             angles: class Angle
             dihedrals: class Dihedral
        """

        if (morses is None and bonds is None and angles is None and
            dihedrals is None):
            raise ImportError(
                'At least one of morses, bonds, angles or dihedrals must be '
                'defined!')

        Precon.__init__(self,
                        dim=dim, c_stab=c_stab,
                        force_stab=force_stab,
                        array_convention=array_convention,
                        solver=solver,
                        solve_tol=solve_tol,
                        apply_positions=apply_positions,
                        apply_cell=apply_cell)

        self.hessian = hessian
        self.morses = morses
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals

    def make_precon(self, atoms):

        #start_time = time.time()
        # Create the preconditioner:
        self._make_sparse_precon(atoms, force_stab=self.force_stab)
        #print('--- Precon created in %s seconds ---' % time.time() - start_time)
        return self.P

    def _make_sparse_precon(self, atoms, initial_assembly=False,
                            force_stab=False):
        """ """

        #start_time = time.time()

        N = len(atoms)

        row = []
        col = []
        data = []

        if self.morses is not None:

            for n in range(len(self.morses)):
                if self.hessian == 'reduced':
                    i, j, Hx = ff.get_morse_potential_reduced_hessian(
                        atoms, self.morses[n])
                elif self.hessian == 'spectral':
                    i, j, Hx = ff.get_morse_potential_hessian(
                        atoms, self.morses[n], spectral=True)
                else:
                    raise NotImplementedError('Not implemented hessian')
                x = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
                row.extend(np.repeat(x, 6))
                col.extend(np.tile(x, 6))
                data.extend(Hx.flatten())

        if self.bonds is not None:

            for n in range(len(self.bonds)):
                if self.hessian == 'reduced':
                    i, j, Hx = ff.get_bond_potential_reduced_hessian(
                        atoms, self.bonds[n], self.morses)
                elif self.hessian == 'spectral':
                    i, j, Hx = ff.get_bond_potential_hessian(
                        atoms, self.bonds[n], self.morses, spectral=True)
                else:
                    raise NotImplementedError('Not implemented hessian')
                x = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
                row.extend(np.repeat(x, 6))
                col.extend(np.tile(x, 6))
                data.extend(Hx.flatten())

        if self.angles is not None:

            for n in range(len(self.angles)):
                if self.hessian == 'reduced':
                    i, j, k, Hx = ff.get_angle_potential_reduced_hessian(
                        atoms, self.angles[n], self.morses)
                elif self.hessian == 'spectral':
                    i, j, k, Hx = ff.get_angle_potential_hessian(
                        atoms, self.angles[n], self.morses, spectral=True)
                else:
                    raise NotImplementedError('Not implemented hessian')
                x = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 *
                     j + 1, 3 * j + 2, 3 * k, 3 * k + 1, 3 * k + 2]
                row.extend(np.repeat(x, 9))
                col.extend(np.tile(x, 9))
                data.extend(Hx.flatten())

        if self.dihedrals is not None:

            for n in range(len(self.dihedrals)):
                if self.hessian == 'reduced':
                    i, j, k, l, Hx = \
                        ff.get_dihedral_potential_reduced_hessian(
                            atoms, self.dihedrals[n], self.morses)
                elif self.hessian == 'spectral':
                    i, j, k, l, Hx = ff.get_dihedral_potential_hessian(
                        atoms, self.dihedrals[n], self.morses, spectral=True)
                else:
                    raise NotImplementedError('Not implemented hessian')
                x = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j +
                     2, 3 * k, 3 * k + 1, 3 * k + 2, 3 * l, 3 * l + 1,
                     3 * l + 2]
                row.extend(np.repeat(x, 12))
                col.extend(np.tile(x, 12))
                data.extend(Hx.flatten())

        row.extend(range(self.dim * N))
        col.extend(range(self.dim * N))
        data.extend([self.c_stab] * self.dim * N)

        # create the matrix
        #start_time = time.time()
        self.P = sparse.csc_matrix(
            (data, (row, col)), shape=(self.dim * N, self.dim * N))
        #print('--- created CSC matrix in %s s ---' %
        #            (time.time() - start_time))

        fixed_atoms = []
        for constraint in atoms.constraints:
            if isinstance(constraint, FixAtoms):
                fixed_atoms.extend(list(constraint.index))
            else:
                raise TypeError(
                    'only FixAtoms constraints are supported by Precon class')
        if len(fixed_atoms) != 0:
            self.P.tolil()
        for i in fixed_atoms:
            self.P[i, :] = 0.0
            self.P[:, i] = 0.0
            self.P[i, i] = 1.0

        self.P = self.P.tocsr()

        #print('--- N-dim precon created in %s s ---' %
        #            (time.time() - start_time))

        # Create solver
        if self.use_pyamg:
            #start_time = time.time()
            self.ml = smoothed_aggregation_solver(
                self.P, B=None,
                strength=('symmetric', {'theta': 0.0}),
                smooth=(
                    'jacobi', {'filter': True, 'weighting': 'local'}),
                improve_candidates=[('block_gauss_seidel',
                                     {'sweep': 'symmetric', 'iterations': 4}),
                                    None, None, None, None, None, None, None,
                                    None, None, None, None, None, None, None],
                aggregate='standard',
                presmoother=('block_gauss_seidel',
                             {'sweep': 'symmetric', 'iterations': 1}),
                postsmoother=('block_gauss_seidel',
                              {'sweep': 'symmetric', 'iterations': 1}),
                max_levels=15,
                max_coarse=300,
                coarse_solver='pinv')
            #print('--- multi grid solver created in %s s ---' %
            #            (time.time() - start_time))

        return self.P


class Exp_FF(Exp, FF):
    """Creates matrix with values decreasing exponentially with distance.
    """

    def __init__(self, A=3.0, r_cut=None, r_NN=None, mu=None, mu_c=None,
                 dim=3, c_stab=0.1,
                 force_stab=False, recalc_mu=False, array_convention='C',
                 solver="auto", solve_tol=1e-9,
                 apply_positions=True, apply_cell=True,
                 estimate_mu_eigmode=False,
                 hessian='spectral', morses=None, bonds=None, angles=None,
                 dihedrals=None):
        """Initialise an Exp+FF preconditioner with given parameters.

        Args:
            r_cut, mu, c_stab, dim, recalc_mu, array_convention: see
                precon.__init__()
            A: coefficient in exp(-A*r/r_NN). Default is A=3.0.
        """
        if (morses is None and bonds is None and angles is None and
            dihedrals is None):
            raise ImportError(
                'At least one of morses, bonds, angles or dihedrals must '
                'be defined!')

        Precon.__init__(self, r_cut=r_cut, r_NN=r_NN,
                        mu=mu, mu_c=mu_c, dim=dim, c_stab=c_stab,
                        force_stab=force_stab,
                        recalc_mu=recalc_mu,
                        array_convention=array_convention,
                        solver=solver,
                        solve_tol=solve_tol,
                        apply_positions=apply_positions,
                        apply_cell=apply_cell,
                        estimate_mu_eigmode=estimate_mu_eigmode)

        self.A = A
        self.hessian = hessian
        self.morses = morses
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals

    def make_precon(self, atoms, recalc_mu=None):

        if self.r_NN is None:
            self.r_NN = estimate_nearest_neighbour_distance(atoms)

        if self.r_cut is None:
            # This is the first time this function has been called, and no
            # cutoff radius has been specified, so calculate it automatically.
            self.r_cut = 2.0 * self.r_NN
        elif self.r_cut < self.r_NN:
            warning = ('WARNING: r_cut (%.2f) < r_NN (%.2f), '
                       'increasing to 1.1*r_NN = %.2f' % (self.r_cut,
                                                          self.r_NN,
                                                          1.1 * self.r_NN))
            warnings.warn(warning)
            self.r_cut = 1.1 * self.r_NN

        if recalc_mu is None:
            # The caller has not specified whether or not to recalculate mu,
            # so the Precon's setting is used.
            recalc_mu = self.recalc_mu

        if self.mu is None:
            # Regardless of what the caller has specified, if we don't
            # currently have a value of mu, then we need one.
            recalc_mu = True

        if recalc_mu:
            self.estimate_mu(atoms)

        if self.P is not None:
            real_atoms = atoms
            if isinstance(atoms, Filter):
                real_atoms = atoms.atoms
            if self.old_positions is None:
                self.old_positions = wrap_positions(real_atoms.positions,
                                                    real_atoms.cell)
            displacement = wrap_positions(real_atoms.positions,
                                          real_atoms.cell) - self.old_positions
            self.old_positions = real_atoms.get_positions()
            max_abs_displacement = abs(displacement).max()
            print('max(abs(displacements)) = %.2f A (%.2f r_NN)' %
                        (max_abs_displacement,
                         max_abs_displacement / self.r_NN))
            if max_abs_displacement < 0.5 * self.r_NN:
                return self.P

        #start_time = time.time()

        # Create the preconditioner:
        self._make_sparse_precon(atoms, force_stab=self.force_stab)

        #print('--- Precon created in %s seconds ---' % (time.time() - start_time))
        return self.P

    def _make_sparse_precon(self, atoms, initial_assembly=False,
                            force_stab=False):
        """Create a sparse preconditioner matrix based on the passed atoms.

        Args:
            atoms: the Atoms object used to create the preconditioner.

        Returns:
            A scipy.sparse.csr_matrix object, representing a d*N by d*N matrix
            (where N is the number of atoms, and d is the value of self.dim).
            BE AWARE that using numpy.dot() with this object will result in
            errors/incorrect results - use the .dot method directly on the
            sparse matrix instead.

        """
        #print('creating sparse precon: initial_assembly=%r, '
        #       'force_stab=%r, apply_positions=%r, apply_cell=%r' %
        #       (initial_assembly, force_stab, self.apply_positions,
        #       self.apply_cell))

        N = len(atoms)
        #start_time = time.time()
        if self.apply_positions:
            # compute neighbour list
            i_list, j_list, rij_list, fixed_atoms = get_neighbours(
                atoms, self.r_cut)
            #print('--- neighbour list created in %s s ---' %
            #            (time.time() - start_time))

        row = []
        col = []
        data = []

        # precon is mu_c*identity for cell DoF
        if isinstance(atoms, Filter):
            i = N - 3
            j = N - 2
            k = N - 1
            x = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 *
                 j + 1, 3 * j + 2, 3 * k, 3 * k + 1, 3 * k + 2]
            row.extend(x)
            col.extend(x)
            if self.apply_cell:
                data.extend(np.repeat(self.mu_c, 9))
            else:
                data.extend(np.repeat(self.mu_c, 9))
        #print('--- computed triplet format in %s s ---' %
        #            (time.time() - start_time))

        conn = sparse.lil_matrix((N, N), dtype=bool)

        if self.apply_positions and not initial_assembly:

            if self.morses is not None:

                for n in range(len(self.morses)):
                    if self.hessian == 'reduced':
                        i, j, Hx = ff.get_morse_potential_reduced_hessian(
                            atoms, self.morses[n])
                    elif self.hessian == 'spectral':
                        i, j, Hx = ff.get_morse_potential_hessian(
                            atoms, self.morses[n], spectral=True)
                    else:
                        raise NotImplementedError('Not implemented hessian')
                    x = [3 * i, 3 * i + 1, 3 * i + 2,
                         3 * j, 3 * j + 1, 3 * j + 2]
                    row.extend(np.repeat(x, 6))
                    col.extend(np.tile(x, 6))
                    data.extend(Hx.flatten())
                    conn[i, j] = True
                    conn[j, i] = True

            if self.bonds is not None:

                for n in range(len(self.bonds)):
                    if self.hessian == 'reduced':
                        i, j, Hx = ff.get_bond_potential_reduced_hessian(
                            atoms, self.bonds[n], self.morses)
                    elif self.hessian == 'spectral':
                        i, j, Hx = ff.get_bond_potential_hessian(
                            atoms, self.bonds[n], self.morses, spectral=True)
                    else:
                        raise NotImplementedError('Not implemented hessian')
                    x = [3 * i, 3 * i + 1, 3 * i + 2,
                         3 * j, 3 * j + 1, 3 * j + 2]
                    row.extend(np.repeat(x, 6))
                    col.extend(np.tile(x, 6))
                    data.extend(Hx.flatten())
                    conn[i, j] = True
                    conn[j, i] = True

            if self.angles is not None:

                for n in range(len(self.angles)):
                    if self.hessian == 'reduced':
                        i, j, k, Hx = ff.get_angle_potential_reduced_hessian(
                            atoms, self.angles[n], self.morses)
                    elif self.hessian == 'spectral':
                        i, j, k, Hx = ff.get_angle_potential_hessian(
                            atoms, self.angles[n], self.morses, spectral=True)
                    else:
                        raise NotImplementedError('Not implemented hessian')
                    x = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 *
                         j + 1, 3 * j + 2, 3 * k, 3 * k + 1, 3 * k + 2]
                    row.extend(np.repeat(x, 9))
                    col.extend(np.tile(x, 9))
                    data.extend(Hx.flatten())
                    conn[i, j] = conn[i, k] = conn[j, k] = True
                    conn[j, i] = conn[k, i] = conn[k, j] = True

            if self.dihedrals is not None:

                for n in range(len(self.dihedrals)):
                    if self.hessian == 'reduced':
                        i, j, k, l, Hx = \
                            ff.get_dihedral_potential_reduced_hessian(
                                atoms, self.dihedrals[n], self.morses)
                    elif self.hessian == 'spectral':
                        i, j, k, l, Hx = ff.get_dihedral_potential_hessian(
                            atoms, self.dihedrals[n], self.morses,
                            spectral=True)
                    else:
                        raise NotImplementedError('Not implemented hessian')
                    x = [3 * i, 3 * i + 1, 3 * i + 2,
                         3 * j, 3 * j + 1, 3 * j + 2,
                         3 * k, 3 * k + 1, 3 * k + 2,
                         3 * l, 3 * l + 1, 3 * l + 2]
                    row.extend(np.repeat(x, 12))
                    col.extend(np.tile(x, 12))
                    data.extend(Hx.flatten())
                    conn[i, j] = conn[i, k] = conn[i, l] = conn[
                        j, k] = conn[j, l] = conn[k, l] = True
                    conn[j, i] = conn[k, i] = conn[l, i] = conn[
                        k, j] = conn[l, j] = conn[l, k] = True

        if self.apply_positions:
            for i, j, rij in zip(i_list, j_list, rij_list):
                if not conn[i, j]:
                    coeff = self.get_coeff(rij)
                    x = [3 * i, 3 * i + 1, 3 * i + 2]
                    y = [3 * j, 3 * j + 1, 3 * j + 2]
                    row.extend(x + x)
                    col.extend(x + y)
                    data.extend(3 * [-coeff] + 3 * [coeff])

        row.extend(range(self.dim * N))
        col.extend(range(self.dim * N))
        if initial_assembly:
            data.extend([self.mu * self.c_stab] * self.dim * N)
        else:
            data.extend([self.c_stab] * self.dim * N)

        # create the matrix
        #start_time = time.time()
        self.P = sparse.csc_matrix(
            (data, (row, col)), shape=(self.dim * N, self.dim * N))
        #print('--- created CSC matrix in %s s ---' %
        #            (time.time() - start_time))

        if not initial_assembly:
            if len(fixed_atoms) != 0:
                self.P.tolil()
            for i in fixed_atoms:
                self.P[i, :] = 0.0
                self.P[:, i] = 0.0
                self.P[i, i] = 1.0

        self.P = self.P.tocsr()

        # Create solver
        if self.use_pyamg:
            #start_time = time.time()
            self.ml = smoothed_aggregation_solver(
                self.P, B=None,
                strength=('symmetric', {'theta': 0.0}),
                smooth=(
                    'jacobi', {'filter': True, 'weighting': 'local'}),
                improve_candidates=[('block_gauss_seidel',
                                     {'sweep': 'symmetric', 'iterations': 4}),
                                    None, None, None, None, None, None, None,
                                    None, None, None, None, None, None, None],
                aggregate='standard',
                presmoother=('block_gauss_seidel',
                             {'sweep': 'symmetric', 'iterations': 1}),
                postsmoother=('block_gauss_seidel',
                              {'sweep': 'symmetric', 'iterations': 1}),
                max_levels=15,
                max_coarse=300,
                coarse_solver='pinv')
            #print('--- multi grid solver created in %s s ---' %
            #            (time.time() - start_time))

        return self.P
