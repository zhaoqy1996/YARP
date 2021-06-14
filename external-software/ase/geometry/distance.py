import numpy as np


def distance(s1, s2, permute=True):
    """Get the distance between two structures s1 and s2.
    
    The distance is defined by the Frobenius norm of
    the spatial distance between all coordinates (see
    numpy.linalg.norm for the definition).

    permute: minimise the distance by 'permuting' same elements
    """

    s1 = s1.copy()
    s2 = s2.copy()
    for s in [s1, s2]:
        s.translate(-s.get_center_of_mass())
    s2pos = 1. * s2.get_positions()
    
    def align(struct, xaxis='x', yaxis='y'):
        """Align moments of inertia with the coordinate system."""
        Is, Vs = struct.get_moments_of_inertia(True)
        IV = list(zip(Is, Vs))
        IV.sort(key=lambda x: x[0])
        struct.rotate(IV[0][1], xaxis)
        
        Is, Vs = struct.get_moments_of_inertia(True)
        IV = list(zip(Is, Vs))
        IV.sort(key=lambda x: x[0])
        struct.rotate(IV[1][1], yaxis)

    align(s1)

    def dd(s1, s2, permute):
        if permute:
            s2 = s2.copy()
            dist = 0
            for a in s1:
                imin = None
                dmin = np.Inf
                for i, b in enumerate(s2):
                    if a.symbol == b.symbol:
                        d = np.sum((a.position - b.position)**2)
                        if d < dmin:
                            dmin = d
                            imin = i
                dist += dmin
                s2.pop(imin)
            return np.sqrt(dist)
        else:
            return np.linalg.norm(s1.get_positions() - s2.get_positions())

    dists = []
    # principles
    for x, y in zip(['x', '-x', 'x', '-x'], ['y', 'y', '-y', '-y']):
        s2.set_positions(s2pos)
        align(s2, x, y)
        dists.append(dd(s1, s2, permute))
   
    return min(dists)
