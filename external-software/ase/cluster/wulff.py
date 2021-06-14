from __future__ import print_function
import numpy as np
from ase.utils import basestring


delta = 1e-10
_debug = False


def wulff_construction(symbol, surfaces, energies, size, structure,
                       rounding='closest', latticeconstant=None,
                       debug=False, maxiter=100):
    """Create a cluster using the Wulff construction.

    A cluster is created with approximately the number of atoms
    specified, following the Wulff construction, i.e. minimizing the
    surface energy of the cluster.

    Parameters:

    symbol: The chemical symbol (or atomic number) of the desired element.

    surfaces: A list of surfaces. Each surface is an (h, k, l) tuple or
    list of integers.

    energies: A list of surface energies for the surfaces.

    size: The desired number of atoms.

    structure: The desired crystal structure.  One of the strings
    "fcc", "bcc", or "sc".

    rounding (optional): Specifies what should be done if no Wulff
    construction corresponds to exactly the requested number of atoms.
    Should be a string, either "above", "below" or "closest" (the
    default), meaning that the nearest cluster above or below - or the
    closest one - is created instead.

    latticeconstant (optional): The lattice constant.  If not given,
    extracted from ase.data.

    debug (optional): If non-zero, information about the iteration towards
    the right cluster size is printed.
    """

    global _debug
    _debug = debug

    if debug:
        print('Wulff: Aiming for cluster with %i atoms (%s)' %
              (size, rounding))

        if rounding not in ['above', 'below', 'closest']:
            raise ValueError('Invalid rounding: %s' % rounding)

    # Interpret structure, if it is a string.
    if isinstance(structure, basestring):
        if structure == 'fcc':
            from ase.cluster.cubic import FaceCenteredCubic as structure
        elif structure == 'bcc':
            from ase.cluster.cubic import BodyCenteredCubic as structure
        elif structure == 'sc':
            from ase.cluster.cubic import SimpleCubic as structure
        elif structure == 'hcp':
            from ase.cluster.hexagonal import \
                HexagonalClosedPacked as structure
        elif structure == 'graphite':
            from ase.cluster.hexagonal import Graphite as structure
        else:
            error = 'Crystal structure %s is not supported.' % structure
            raise NotImplementedError(error)

    # Check number of surfaces
    nsurf = len(surfaces)
    if len(energies) != nsurf:
        raise ValueError('The energies array should contain %d values.'
                         % (nsurf,))

    # Copy energies array so it is safe to modify it
    energies = np.array(energies)

    # We should check that for each direction, the surface energy plus
    # the energy in the opposite direction is positive.  But this is
    # very difficult in the general case!

    # Before starting, make a fake cluster just to extract the
    # interlayer distances in the relevant directions, and use these
    # to "renormalize" the surface energies such that they can be used
    # to convert to number of layers instead of to distances.
    atoms = structure(symbol, surfaces, 5 * np.ones(len(surfaces), int),
                      latticeconstant=latticeconstant)
    for i, s in enumerate(surfaces):
        d = atoms.get_layer_distance(s)
        energies[i] /= d
        
    # First guess a size that is not too large.
    wanted_size = size ** (1.0 / 3.0)
    max_e = max(energies)
    factor = wanted_size / max_e
    atoms, layers = make_atoms(symbol, surfaces, energies, factor, structure,
                               latticeconstant)
    if len(atoms) == 0:
        # Probably the cluster is very flat
        if debug:
            print('First try made an empty cluster, trying again.')
        factor = 1 / energies.min()
        atoms, layers = make_atoms(symbol, surfaces, energies, factor,
                                   structure, latticeconstant)
        if len(atoms) == 0:
            raise RuntimeError('Failed to create a finite cluster.')

    # Second guess: scale to get closer.
    old_factor = factor
    old_layers = layers
    old_atoms = atoms
    factor *= (size / len(atoms))**(1.0 / 3.0)
    atoms, layers = make_atoms(symbol, surfaces, energies, factor,
                               structure, latticeconstant)
    if len(atoms) == 0:
        print('Second guess gave an empty cluster, discarding it.')
        atoms = old_atoms
        factor = old_factor
        layers = old_layers
    else:
        del old_atoms

    # Find if the cluster is too small or too large (both means perfect!)
    below = above = None
    if len(atoms) <= size:
        below = atoms
    if len(atoms) >= size:
        above = atoms

    # Now iterate towards the right cluster
    iter = 0
    while (below is None or above is None):
        if len(atoms) < size:
            # Find a larger cluster
            if debug:
                print('Making a larger cluster.')
            factor = ((layers + 0.5 + delta) / energies).min()
            atoms, new_layers = make_atoms(symbol, surfaces, energies, factor,
                                           structure, latticeconstant)
            assert (new_layers - layers).max() == 1
            assert (new_layers - layers).min() >= 0
            layers = new_layers
        else:
            # Find a smaller cluster
            if debug:
                print('Making a smaller cluster.')
            factor = ((layers - 0.5 - delta) / energies).max()
            atoms, new_layers = make_atoms(symbol, surfaces, energies, factor,
                                           structure, latticeconstant)
            assert (new_layers - layers).max() <= 0
            assert (new_layers - layers).min() == -1
            layers = new_layers
        if len(atoms) <= size:
            below = atoms
        if len(atoms) >= size:
            above = atoms
        iter += 1
        if iter == maxiter:
            raise RuntimeError('Runaway iteration.')
    if rounding == 'below':
        if debug:
            print('Choosing smaller cluster with %i atoms' % len(below))
        return below
    elif rounding == 'above':
        if debug:
            print('Choosing larger cluster with %i atoms' % len(above))
        return above
    else:
        assert rounding == 'closest'
        if (len(above) - size) < (size - len(below)):
            atoms = above
        else:
            atoms = below
        if debug:
            print('Choosing closest cluster with %i atoms' % len(atoms))
        return atoms


def make_atoms(symbol, surfaces, energies, factor, structure, latticeconstant):
    layers1 = factor * np.array(energies)
    layers = np.round(layers1).astype(int)
    atoms = structure(symbol, surfaces, layers,
                      latticeconstant=latticeconstant)
    if _debug:
        print('Created a cluster with %i atoms: %s' % (len(atoms),
                                                       str(layers)))
    return (atoms, layers)
