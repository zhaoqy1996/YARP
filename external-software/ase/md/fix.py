import numpy as np

class FixRotation:
    """Remove rotation from an atoms object.
    
    This class is intended as an observer on an atoms class during
    a molecular dynamics simulation.  When it is called, it removes
    any rotation around the center of mass.
    
    It assumes that the system is a (nano)particle with free boundary
    conditions.
    
    Bugs:
    Should check that the boundary conditions make sense.
    """
    def __init__(self, atoms):
        self.atoms = atoms

    def __call__(self):
        atoms = self.atoms

        r = atoms.get_positions() - atoms.get_center_of_mass()
        v = atoms.get_velocities()
        p = atoms.get_momenta()
        m = atoms.get_masses()

        x = r[:,0]
        y = r[:,1]
        z = r[:,2]

        I11 = np.sum(m * (y**2 + z**2))
        I22 = np.sum(m * (x**2 + z**2))
        I33 = np.sum(m * (x**2 + y**2))
        I12 = np.sum(-m * x * y)
        I13 = np.sum(-m * x * z)
        I23 = np.sum(-m * y * z)

        I = np.array([[I11, I12, I13],
                      [I12, I22, I23],
                      [I13, I23, I33]])

        w = np.dot(np.linalg.inv(I), np.sum(np.cross(r, p), axis=0))

        self.atoms.set_velocities(v - np.cross(w, r))

