from __future__ import print_function
import json


class CLICommand:
    """Get calculations from NOMAD and write to JSON files.

    ...
    """

    @staticmethod
    def add_arguments(p):
        p.add_argument('uri', nargs='+', metavar='nmd://<hash>',
                       help='URIs to get')

    @staticmethod
    def run(args):
        from ase.nomad import download
        for uri in args.uri:
            calculation = download(uri)
            identifier = calculation.hash.replace('/', '.')
            fname = 'nmd.{}.nomad.json'.format(identifier)
            with open(fname, 'w') as fd:
                json.dump(calculation, fd)
            print(uri)
