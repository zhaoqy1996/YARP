import numpy as np
from numpy.linalg import norm, solve

from ase.utils import gcd, basestring
from ase.build import bulk
from copy import deepcopy


def surface(lattice, indices, layers, vacuum=None, tol=1e-10, termination=0):
    """Create surface from a given lattice and Miller indices.

    lattice: Atoms object or str
        Bulk lattice structure of alloy or pure metal.  Note that the
        unit-cell must be the conventional cell - not the primitive cell.
        One can also give the chemical symbol as a string, in which case the
        correct bulk lattice will be generated automatically.
    indices: sequence of three int
        Surface normal in Miller indices (h,k,l).
    layers: int
        Number of equivalent layers of the slab.
    vacuum: float
        Amount of vacuum added on both sides of the slab.
    termination: int
        The termination "number" for your crystal. The same value will not
        produce the same termination for different symetrically identical
        bulk structures, but changing this value allows your to explore all
        the possible terminations for the bulk structure you provide it.
        note: this code is not well tested
    """

    indices = np.asarray(indices)

    if indices.shape != (3,) or not indices.any() or indices.dtype != int:
        raise ValueError('%s is an invalid surface type' % indices)

    if isinstance(lattice, basestring):
        lattice = bulk(lattice, cubic=True)

    h, k, l = indices
    h0, k0, l0 = (indices == 0)
    
    if termination != 0:  #changing termination
        import warnings
        warnings.warn('Work on changing terminations is currently in '
                      'progress.  Code may not behave as expected.')
        lattice1 = deepcopy(lattice)
        cell = lattice1.get_cell()
        pt = [0,0,0]
        millers = list(indices)
        for index,item in enumerate(millers):
            if item == 0:
                millers[index] = 10**9 #make zeros large numbers
            elif pt == [0,0,0]:        #for numerical stability
                pt = list(cell[index]/float(item)/np.linalg.norm(cell[index]))
        h1,k1,l1 = millers
        N = np.array(cell[0]/h1+cell[1]/k1+cell[2]/l1)
        n = N/np.linalg.norm(N)# making a unit vector normal to cut plane
        d = [np.round(np.dot(n,(a-pt)),4) for a in lattice.get_scaled_positions()]
        d = set(d)
        d = sorted(list(d))
        d = [0]+d #distances of atoms from cut plane
        displacement = (h*cell[0]+k*cell[1]+l*cell[2])*d[termination]
        lattice1.positions += displacement
        lattice = lattice1
       
    if h0 and k0 or h0 and l0 or k0 and l0:  # if two indices are zero
        if not h0:
            c1, c2, c3 = [(0, 1, 0), (0, 0, 1), (1, 0, 0)]
        if not k0:
            c1, c2, c3 = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]
        if not l0:
            c1, c2, c3 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    else:
        p, q = ext_gcd(k, l)
        a1, a2, a3 = lattice.cell

        # constants describing the dot product of basis c1 and c2:
        # dot(c1,c2) = k1+i*k2, i in Z
        k1 = np.dot(p * (k * a1 - h * a2) + q * (l * a1 - h * a3),
                    l * a2 - k * a3)
        k2 = np.dot(l * (k * a1 - h * a2) - k * (l * a1 - h * a3),
                    l * a2 - k * a3)

        if abs(k2) > tol:
            i = -int(round(k1 / k2))  # i corresponding to the optimal basis
            p, q = p + i * l, q - i * k

        a, b = ext_gcd(p * k + q * l, h)

        c1 = (p * k + q * l, -p * h, -q * h)
        c2 = np.array((0, l, -k)) // abs(gcd(l, k))
        c3 = (b, a * p, a * q)

    surf = build(lattice, np.array([c1, c2, c3]), layers, tol)
    if vacuum is not None:
        surf.center(vacuum=vacuum, axis=2)
    return surf


def build(lattice, basis, layers, tol):
    surf = lattice.copy()
    scaled = solve(basis.T, surf.get_scaled_positions().T).T
    scaled -= np.floor(scaled + tol)
    surf.set_scaled_positions(scaled)
    surf.set_cell(np.dot(basis, surf.cell), scale_atoms=True)
    surf *= (1, 1, layers)

    a1, a2, a3 = surf.cell
    surf.set_cell([a1, a2,
                   np.cross(a1, a2) * np.dot(a3, np.cross(a1, a2)) /
                   norm(np.cross(a1, a2))**2])

    # Change unit cell to have the x-axis parallel with a surface vector
    # and z perpendicular to the surface:
    a1, a2, a3 = surf.cell
    surf.set_cell([(norm(a1), 0, 0),
                   (np.dot(a1, a2) / norm(a1),
                    np.sqrt(norm(a2)**2 - (np.dot(a1, a2) / norm(a1))**2), 0),
                   (0, 0, norm(a3))],
                  scale_atoms=True)

    surf.pbc = (True, True, False)

    # Move atoms into the unit cell:
    scaled = surf.get_scaled_positions()
    scaled[:, :2] %= 1
    surf.set_scaled_positions(scaled)

    surf.cell[2] = 0.0

    return surf


def ext_gcd(a, b):
    if b == 0:
        return 1, 0
    elif a % b == 0:
        return 0, 1
    else:
        x, y = ext_gcd(b, a % b)
        return y, x - y * (a // b)
