from __future__ import print_function
import numpy as np

from ase.io import read
from ase.geometry import crystal_structure_from_cell
from ase.dft.kpoints import (get_special_points, special_paths,
                             parse_path_string, labels_from_kpts,
                             get_monkhorst_pack_size_and_offset)
from ase.dft.bz import bz1d_plot, bz2d_plot, bz3d_plot


def plot_reciprocal_cell(atoms, path='default',
                         k_points=False,
                         ibz_k_points=False,
                         plot_vectors=True, dimension=3, output=None,
                         verbose=False):
    import matplotlib.pyplot as plt

    cell = atoms.get_cell()
    icell = atoms.get_reciprocal_cell()

    try:
        cs = crystal_structure_from_cell(cell)
    except ValueError:
        cs = None

    if verbose:
        if cs:
            print('Crystal:', cs)
            print('Special points:', special_paths[cs])
        print('Lattice vectors:')
        for i, v in enumerate(cell):
            print('{}: ({:16.9f},{:16.9f},{:16.9f})'.format(i + 1, *v))
        print('Reciprocal vectors:')
        for i, v in enumerate(icell):
            print('{}: ({:16.9f},{:16.9f},{:16.9f})'.format(i + 1, *v))

    # band path
    if path:
        if path == 'default':
            path = special_paths[cs]
        paths = []
        special_points = get_special_points(cell)
        for names in parse_path_string(path):
            points = []
            for name in names:
                points.append(np.dot(icell.T, special_points[name]))
            paths.append((names, points))
    else:
        paths = None

    # k points
    points = None
    if atoms.calc is not None and hasattr(atoms.calc, 'get_bz_k_points'):
        bzk = atoms.calc.get_bz_k_points()
        if path is None:
            try:
                size, offset = get_monkhorst_pack_size_and_offset(bzk)
            except ValueError:
                # This was not a MP-grid.  Must be a path in the BZ:
                path = ''.join(labels_from_kpts(bzk, cell)[2])

        if k_points:
            points = bzk
        elif ibz_k_points:
            points = atoms.calc.get_ibz_k_points()
        if points is not None:
            for i in range(len(points)):
                points[i] = np.dot(icell.T, points[i])

    kwargs = {'cell': cell,
              'vectors': plot_vectors,
              'paths': paths,
              'points': points}

    if dimension == 1:
        bz1d_plot(**kwargs)
    elif dimension == 2:
        bz2d_plot(**kwargs)
    else:
        bz3d_plot(interactive=True, **kwargs)

    if output:
        plt.savefig(output)
    else:
        plt.show()


class CLICommand:
    """Show the reciprocal space.

    Read unit cell from a file and show a plot of the 1. Brillouin zone.  If
    the file contains information about k-points, then those can be plotted
    too.

    Examples:

        $ # Show GXWLG path in FCC-BZ:
        $ ase build -x fcc Al al.traj
        $ ase reciprocal al.traj -p GXWLG

        $ # And now with k-points:
        $ ase run gpaw al.traj -p kpts=6,6,6,mode=pw \
        >   --after "atoms.calc.write('al.gpw')" > al.txt
        $ ase reciprocal al.gpw -i -p GXWLG
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('name', metavar='input-file',
            help='Input file containing unit cell.')
        add('output', nargs='?', help='Write plot to file (.png, .svg, ...).')
        add('-v', '--verbose', action='store_true', help='More output.')
        add('-p', '--path', nargs='?', type=str, const='default',
            help='Add a band path.  Example: "GXL".')
        add('-d', '--dimension', type=int, default=3,
            help='Dimension of the cell.')
        add('--no-vectors', action='store_true',
            help="Don't show reciprocal vectors.")
        kp = parser.add_mutually_exclusive_group(required=False)
        kp.add_argument('-k', '--k-points', action='store_true',
                        help='Add k-points of the calculator.')
        kp.add_argument('-i', '--ibz-k-points', action='store_true',
                        help='Add irreducible k-points of the calculator.')

    @staticmethod
    def run(args, parser):
        atoms = read(args.name)

        plot_reciprocal_cell(atoms,
                             output=args.output,
                             verbose=args.verbose,
                             path=args.path,
                             dimension=args.dimension,
                             plot_vectors=not args.no_vectors,
                             k_points=args.k_points,
                             ibz_k_points=args.ibz_k_points)
