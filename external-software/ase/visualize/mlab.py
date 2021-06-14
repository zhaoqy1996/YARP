from __future__ import print_function
import optparse

import numpy as np

from ase.data import covalent_radii
from ase.io.cube import read_cube_data
from ase.data.colors import cpk_colors
from ase.calculators.calculator import get_calculator


def plot(atoms, data, contours):
    """Plot atoms, unit-cell and iso-surfaces using Mayavi.

    Parameters:

    atoms: Atoms object
        Positions, atomiz numbers and unit-cell.
    data: 3-d ndarray of float
        Data for iso-surfaces.
    countours: list of float
        Contour values.
    """

    # Delay slow imports:
    from mayavi import mlab

    mlab.figure(1, bgcolor=(1, 1, 1))  # make a white figure

    # Plot the atoms as spheres:
    for pos, Z in zip(atoms.positions, atoms.numbers):
        mlab.points3d(*pos,
                      scale_factor=covalent_radii[Z],
                      resolution=20,
                      color=tuple(cpk_colors[Z]))

    # Draw the unit cell:
    A = atoms.cell
    for i1, a in enumerate(A):
        i2 = (i1 + 1) % 3
        i3 = (i1 + 2) % 3
        for b in [np.zeros(3), A[i2]]:
            for c in [np.zeros(3), A[i3]]:
                p1 = b + c
                p2 = p1 + a
                mlab.plot3d([p1[0], p2[0]],
                            [p1[1], p2[1]],
                            [p1[2], p2[2]],
                            tube_radius=0.1)

    cp = mlab.contour3d(data, contours=contours, transparent=True,
                        opacity=0.5, colormap='hot')
    # Do some tvtk magic in order to allow for non-orthogonal unit cells:
    polydata = cp.actor.actors[0].mapper.input
    pts = np.array(polydata.points) - 1
    # Transform the points to the unit cell:
    polydata.points = np.dot(pts, A / np.array(data.shape)[:, np.newaxis])

    # Apparently we need this to redraw the figure, maybe it can be done in
    # another way?
    mlab.view(azimuth=155, elevation=70, distance='auto')
    # Show the 3d plot:
    mlab.show()


description = """\
Plot iso-surfaces from a cube-file or a wave function or an electron
density from a calculator-restart file."""


def main(args=None):
    parser = optparse.OptionParser(usage='%prog [options] filename',
                                   description=description)
    add = parser.add_option
    add('-n', '--band-index', type=int, metavar='INDEX',
        help='Band index counting from zero.')
    add('-s', '--spin-index', type=int, metavar='SPIN',
        help='Spin index: zero or one.')
    add('-e', '--electrostatic-potential', action='store_true',
        help='Plot the electrostatic potential.')
    add('-c', '--contours', default='4',
        help='Use "-c 3" for 3 contours or "-c -0.5,0.5" for specific ' +
        'values.  Default is four contours.')
    add('-r', '--repeat', help='Example: "-r 2,2,2".')
    add('-C', '--calculator-name', metavar='NAME', help='Name of calculator.')

    opts, args = parser.parse_args(args)
    if len(args) != 1:
        parser.error('Incorrect number of arguments')

    arg = args[0]
    if arg.endswith('.cube'):
        data, atoms = read_cube_data(arg)
    else:
        calc = get_calculator(opts.calculator_name)(arg, txt=None)
        atoms = calc.get_atoms()
        if opts.band_index is None:
            if opts.electrostatic_potential:
                data = calc.get_electrostatic_potential()
            else:
                data = calc.get_pseudo_density(opts.spin_index)
        else:
            data = calc.get_pseudo_wave_function(opts.band_index,
                                                 opts.spin_index or 0)
            if data.dtype == complex:
                data = abs(data)

    mn = data.min()
    mx = data.max()
    print('Min: %16.6f' % mn)
    print('Max: %16.6f' % mx)

    if opts.contours.isdigit():
        n = int(opts.contours)
        d = (mx - mn) / n
        contours = np.linspace(mn + d / 2, mx - d / 2, n).tolist()
    else:
        contours = [float(x) for x in opts.contours.rstrip(',').split(',')]

    if len(contours) == 1:
        print('1 contour:', contours[0])
    else:
        print('%d contours: %.6f, ..., %.6f' %
              (len(contours), contours[0], contours[-1]))

    if opts.repeat:
        repeat = [int(r) for r in opts.repeat.split(',')]
        data = np.tile(data, repeat)
        atoms *= repeat

    plot(atoms, data, contours)


if __name__ == '__main__':
    main()
