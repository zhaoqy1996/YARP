import numpy as np


def rotation_matrix_from_points(m0, m1):
    """Returns a rigid transformation/rotation matrix that minimizes the
    RMSD between two set of points.
    
    m0 and m1 should be (3, npoints) numpy arrays with
    coordinates as columns::

        (x1  x2   x3   ... xN
         y1  y2   y3   ... yN
         z1  z2   z3   ... zN)

    The centeroids should be set to origin prior to
    computing the rotation matrix.

    The rotation matrix is computed using quaternion
    algebra as detailed in::
        
        Melander et al. J. Chem. Theory Comput., 2015, 11,1055
    """

    v0 = np.copy(m0)
    v1 = np.copy(m1)

    # compute the rotation quaternion

    R11, R22, R33 = np.sum(v0 * v1, axis=1)
    R12, R23, R31 = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
    R13, R21, R32 = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)

    f = [[R11 + R22 + R33, R23 - R32, R31 - R13, R12 - R21],
         [R23 - R32, R11 - R22 - R33, R12 + R21, R13 + R31],
         [R31 - R13, R12 + R21, -R11 + R22 - R33, R23 + R32],
         [R12 - R21, R13 + R31, R23 + R32, -R11 - R22 + R33]]

    F = np.array(f)

    w, V = np.linalg.eigh(F)
    # eigenvector corresponding to the most
    # positive eigenvalue
    q = V[:, np.argmax(w)]

    # Rotation matrix from the quaternion q

    R = quaternion_to_matrix(q)

    return R

    
def quaternion_to_matrix(q):
    """Returns a rotation matrix.
    
    Computed from a unit quaternion Input as (4,) numpy array.
    """

    q0, q1, q2, q3 = q
    R_q = [[q0**2 + q1**2 - q2**2 - q3**2,
            2 * (q1 * q2 - q0 * q3),
            2 * (q1 * q3 + q0 * q2)],
           [2 * (q1 * q2 + q0 * q3),
            q0**2 - q1**2 + q2**2 - q3**2,
            2 * (q2 * q3 - q0 * q1)],
           [2 * (q1 * q3 - q0 * q2),
            2 * (q2 * q3 + q0 * q1),
            q0**2 - q1**2 - q2**2 + q3**2]]
    return np.array(R_q)


def minimize_rotation_and_translation(target, atoms):
    """Minimize RMSD between atoms and target.
    
    Rotate and translate atoms to best match target.  For more details, see::
        
        Melander et al. J. Chem. Theory Comput., 2015, 11,1055
    """

    p = atoms.get_positions()
    p0 = target.get_positions()

    # centeroids to origin
    c = np.mean(p, axis=0)
    p -= c
    c0 = np.mean(p0, axis=0)
    p0 -= c0

    # Compute rotation matrix
    R = rotation_matrix_from_points(p.T, p0.T)

    atoms.set_positions(np.dot(p, R.T) + c0)
