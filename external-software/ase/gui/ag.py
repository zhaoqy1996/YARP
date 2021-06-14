# Copyright 2008, 2009
# CAMd (see accompanying license files for details).
from __future__ import print_function, unicode_literals
import warnings


class CLICommand:
    """ASE's graphical user interface.

    ASE-GUI.  See the online manual
    (https://wiki.fysik.dtu.dk/ase/ase/gui/gui.html)
    for more information.
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('filenames', nargs='*',
            help='Files to open.  Append @SLICE to a filename to pick '
            'a subset of images from that file.  See --image-number '
            'for SLICE syntax.')
        add('-n', '--image-number', metavar='SLICE', default=':',
            help='Pick individual image or slice from each of the files.  '
            'SLICE can be a number or a Python slice-like expression '
            'such as :STOP, START:STOP, or START:STOP:STEP, '
            'where START, STOP, and STEP are integers.  '
            'Indexing counts from 0.  '
            'Negative numbers count backwards from last image.  '
            'Using @SLICE syntax for a filename overrides this option '
            'for that file.')
        add('-r', '--repeat',
            default='1',
            help='Repeat unit cell.  Use "-r 2" or "-r 2,3,1".')
        add('-R', '--rotations', default='',
            help='Examples: "-R -90x", "-R 90z,-30x".')
        add('-o', '--output', metavar='FILE',
            help='Write configurations to FILE.')
        add('-g', '--graph',
            # TRANSLATORS: EXPR abbreviates 'expression'
            metavar='EXPR',
            help='Plot x,y1,y2,... graph from configurations or '
            'write data to sdtout in terminal mode.  Use the '
            'symbols: i, s, d, fmax, e, ekin, A, R, E and F.  See '
            'https://wiki.fysik.dtu.dk/ase/ase/gui/gui.html'
            '#plotting-data for more details.')
        add('-t', '--terminal',
            action='store_true',
            default=False,
            help='Run in terminal window - no GUI.')
        add('--interpolate',
            type=int, metavar='N',
            help='Interpolate N images between 2 given images.')
        add('-b', '--bonds',
            action='store_true',
            default=False,
            help='Draw bonds between atoms.')
        add('-s', '--scale', dest='radii_scale', metavar='FLOAT',
            default=None, type=float,
            help='Scale covalent radii.')

    @staticmethod
    def run(args):
        from ase.gui.images import Images
        from ase.atoms import Atoms

        images = Images()

        if args.filenames:
            images.read(args.filenames, args.image_number)
        else:
            images.initialize([Atoms()])

        if args.interpolate:
            images.interpolate(args.interpolate)

        if args.repeat != '1':
            r = args.repeat.split(',')
            if len(r) == 1:
                r = 3 * r
            images.repeat_images([int(c) for c in r])

        if args.radii_scale:
            images.scale_radii(args.radii_scale)

        if args.output is not None:
            warnings.warn('You should be using "ase convert ..." instead!')
            images.write(args.output, rotations=args.rotations)
            args.terminal = True

        if args.terminal:
            if args.graph is not None:
                data = images.graph(args.graph)
                for line in data.T:
                    for x in line:
                        print(x, end=' ')
                    print()
        else:
            import os
            from ase.gui.gui import GUI

            backend = os.environ.get('MPLBACKEND', '')
            if backend == 'module://ipykernel.pylab.backend_inline':
                # Jupyter should not steal our windows
                del os.environ['MPLBACKEND']

            gui = GUI(images, args.rotations, args.bonds, args.graph)
            gui.run()
