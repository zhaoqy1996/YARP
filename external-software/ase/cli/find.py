from __future__ import print_function
import os
import os.path as op
import sys

from ase.io import read
from ase.io.formats import filetype, UnknownFileTypeError
from ase.db import connect
from ase.db.core import parse_selection
from ase.db.jsondb import JSONDatabase
from ase.db.row import atoms2dict


class CLICommand:
    """Find files with atoms in them.

    Search through files known to ASE applying a query to filter the results.

    See https://wiki.fysik.dtu.dk/ase/ase/db/db.html#querying for more
    informations on how to construct the query string.
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('folder', help='Folder to look in.')
        parser.add_argument(
            'query', nargs='?',
            help='Examples: More than 2 hydrogens and no silver: "H>2,Ag=0". '
            'More than 1000 atoms: "natoms>1000". '
            'Slab geometry containing Cu and Ni: "pbc=TTF,Cu,Ni".')
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='More output.')
        parser.add_argument('-l', '--long', action='store_true',
                            help='Show also periodic boundary conditions, '
                            'chemical formula and filetype.')
        parser.add_argument('-i', '--include', help='Include only filenames '
                            'ending with given strings.  Example: '
                            '"-i .xyz,.traj".')
        parser.add_argument('-x', '--exclude', help='Exclude filenames '
                            'ending with given strings.  Example: '
                            '"-x .cif".')

    @staticmethod
    def run(args):
        main(args)


def main(args):
    query = parse_selection(args.query)
    include = args.include.split(',') if args.include else []
    exclude = args.exclude.split(',') if args.exclude else []

    if args.long:
        print('pbc {:10} {:15} path'.format('formula', 'filetype'))

    for path in allpaths(args.folder, include, exclude):
        format, row = check(path, query, args.verbose)
        if format:
            if args.long:
                print('{} {:10} {:15} {}'
                      .format(''.join(str(p) for p in row.pbc.astype(int)),
                              row.formula,
                              format,
                              path))
            else:
                print(path)


def allpaths(folder, include, exclude):
    """Generate paths."""
    exclude += ['.py', '.pyc']
    for dirpath, dirnames, filenames in os.walk(folder):
        for name in filenames:
            if any(name.endswith(ext) for ext in exclude):
                continue
            if include:
                for ext in include:
                    if name.endswith(ext):
                        break
                else:
                    continue
            path = op.join(dirpath, name)
            yield path

        # Skip .git, __pycache__ and friends:
        dirnames[:] = (name for name in dirnames if name[0] not in '._')


def check(path, query, verbose):
    """Check a path.

    Returns a (filetype, AtomsRow object) tuple.
    """

    try:
        format = filetype(path, guess=False)
    except (OSError, UnknownFileTypeError):
        return '', None

    if format in ['db', 'json']:
        db = connect(path)
    else:
        try:
            atoms = read(path, format=format)
        except Exception as x:
            if verbose:
                print(path + ':', x, file=sys.stderr)
            return '', None
        db = FakeDB(atoms)

    try:
        for row in db._select(*query):
            return format, row
    except Exception as x:
        if verbose:
            print(path + ':', x, file=sys.stderr)

    return '', None


class FakeDB(JSONDatabase):
    def __init__(self, atoms):
        self.bigdct = {1: atoms2dict(atoms)}

    def _read_json(self):
        return self.bigdct, [1], 2
