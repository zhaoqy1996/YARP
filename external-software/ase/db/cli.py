from __future__ import print_function
import json
import sys
from collections import defaultdict
from random import randint

import ase.io
from ase.db import connect
from ase.db.core import convert_str_to_int_float_or_str
from ase.db.summary import Summary
from ase.db.table import Table, all_columns
from ase.db.web import process_metadata
from ase.utils import plural, basestring

try:
    input = raw_input  # Python 2+3 compatibility
except NameError:
    pass


class CLICommand:
    """Manipulate and query ASE database.

    Query is a comma-separated list of
    selections where each selection is of the type "ID", "key" or
    "key=value".  Instead of "=", one can also use "<", "<=", ">=", ">"
    and  "!=" (these must be protected from the shell by using quotes).
    Special keys:

    * id
    * user
    * calculator
    * age
    * natoms
    * energy
    * magmom
    * charge

    Chemical symbols can also be used to select number of
    specific atomic species (H, He, Li, ...).  Selection examples:

        calculator=nwchem
        age<1d
        natoms=1
        user=alice
        2.2<bandgap<4.1
        Cu>=10

    See also: https://wiki.fysik.dtu.dk/ase/ase/db/db.html.
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('database', help='SQLite3 file, JSON file or postgres URL.')
        add('query', nargs='*', help='Query string.')
        add('-v', '--verbose', action='store_true', help='More output.')
        add('-q', '--quiet', action='store_true', help='Less output.')
        add('-n', '--count', action='store_true',
            help='Count number of selected rows.')
        add('-l', '--long', action='store_true',
            help='Long description of selected row')
        add('-i', '--insert-into', metavar='db-name',
            help='Insert selected rows into another database.')
        add('-a', '--add-from-file', metavar='filename',
            help='Add configuration(s) from file.  '
            'If the file contains more than one configuration then you can '
            'use the syntax filename@: to add all of them.  Default is to '
            'only add the last.')
        add('-k', '--add-key-value-pairs', metavar='key1=val1,key2=val2,...',
            help='Add key-value pairs to selected rows.  Values must '
            'be numbers or strings and keys must follow the same rules as '
            'keywords.')
        add('-L', '--limit', type=int, default=20, metavar='N',
            help='Show only first N rows (default is 20 rows).  Use --limit=0 '
            'to show all.')
        add('--offset', type=int, default=0, metavar='N',
            help='Skip first N rows.  By default, no rows are skipped')
        add('--delete', action='store_true',
            help='Delete selected rows.')
        add('--delete-keys', metavar='key1,key2,...',
            help='Delete keys for selected rows.')
        add('-y', '--yes', action='store_true',
            help='Say yes.')
        add('--explain', action='store_true',
            help='Explain query plan.')
        add('-c', '--columns', metavar='col1,col2,...',
            help='Specify columns to show.  Precede the column specification '
            'with a "+" in order to add columns to the default set of '
            'columns.  Precede by a "-" to remove columns.  Use "++" for all.')
        add('-s', '--sort', metavar='column', default='id',
            help='Sort rows using "column".  Use "column-" for a descending '
            'sort.  Default is to sort after id.')
        add('--cut', type=int, default=35, help='Cut keywords and key-value '
            'columns after CUT characters.  Use --cut=0 to disable cutting. '
            'Default is 35 characters')
        add('-p', '--plot', metavar='x,y1,y2,...',
            help='Example: "-p x,y": plot y row against x row. Use '
            '"-p a:x,y" to make a plot for each value of a.')
        add('-P', '--plot-data', metavar='name',
            help="Show plot from data['name'] from the selected row.")
        add('--csv', action='store_true',
            help='Write comma-separated-values file.')
        add('-w', '--open-web-browser', action='store_true',
            help='Open results in web-browser.')
        add('--no-lock-file', action='store_true', help="Don't use lock-files")
        add('--analyse', action='store_true',
            help='Gathers statistics about tables and indices to help make '
            'better query planning choices.')
        add('-j', '--json', action='store_true',
            help='Write json representation of selected row.')
        add('-m', '--show-metadata', action='store_true',
            help='Show metadata as json.')
        add('--set-metadata', metavar='something.json',
            help='Set metadata from a json file.')
        add('-M', '--metadata-from-python-script', metavar='something.py',
            help='Use metadata from a Python file.')
        add('--unique', action='store_true',
            help='Give rows a new unique id when using --insert-into.')
        add('--strip-data', action='store_true',
            help='Strip data when using --insert-into.')
        add('--show-keys', action='store_true',
            help='Show all keys.')
        add('--show-values', metavar='key1,key2,...',
            help='Show values for key(s).')
        add('--write-summary-files', metavar='prefix',
            help='Write summary-files with a "<prefix>-<uid>-" prefix.')

    @staticmethod
    def run(args):
        main(args)


def main(args):
    verbosity = 1 - args.quiet + args.verbose
    query = ','.join(args.query)

    if args.sort.endswith('-'):
        # Allow using "key-" instead of "-key" for reverse sorting
        args.sort = '-' + args.sort[:-1]

    if query.isdigit():
        query = int(query)

    add_key_value_pairs = {}
    if args.add_key_value_pairs:
        for pair in args.add_key_value_pairs.split(','):
            key, value = pair.split('=')
            add_key_value_pairs[key] = convert_str_to_int_float_or_str(value)

    if args.delete_keys:
        delete_keys = args.delete_keys.split(',')
    else:
        delete_keys = []

    db = connect(args.database, use_lock_file=not args.no_lock_file)

    def out(*args):
        if verbosity > 0:
            print(*args)

    if args.analyse:
        db.analyse()
        return

    if args.show_keys:
        keys = defaultdict(int)
        for row in db.select(query):
            for key in row._keys:
                keys[key] += 1

        n = max(len(key) for key in keys) + 1
        for key, number in keys.items():
            print('{:{}} {}'.format(key + ':', n, number))
        return

    if args.show_values:
        keys = args.show_values.split(',')
        values = {key: defaultdict(int) for key in keys}
        numbers = set()
        for row in db.select(query):
            kvp = row.key_value_pairs
            for key in keys:
                value = kvp.get(key)
                if value is not None:
                    values[key][value] += 1
                    if not isinstance(value, str):
                        numbers.add(key)

        n = max(len(key) for key in keys) + 1
        for key in keys:
            vals = values[key]
            if key in numbers:
                print('{:{}} [{}..{}]'
                      .format(key + ':', n, min(vals), max(vals)))
            else:
                print('{:{}} {}'
                      .format(key + ':', n,
                              ', '.join('{}({})'.format(v, n)
                                        for v, n in vals.items())))
        return

    if args.add_from_file:
        filename = args.add_from_file
        configs = ase.io.read(filename)
        if not isinstance(configs, list):
            configs = [configs]
        for atoms in configs:
            db.write(atoms, key_value_pairs=add_key_value_pairs)
        out('Added ' + plural(len(configs), 'row'))
        return

    if args.count:
        n = db.count(query)
        print('%s' % plural(n, 'row'))
        return

    if args.explain:
        for row in db.select(query, explain=True,
                             verbosity=verbosity,
                             limit=args.limit, offset=args.offset):
            print(row['explain'])
        return

    if args.show_metadata:
        print(json.dumps(db.metadata, sort_keys=True, indent=4))
        return

    if args.set_metadata:
        with open(args.set_metadata) as fd:
            db.metadata = json.load(fd)
        return

    if args.insert_into:
        nkvp = 0
        nrows = 0
        with connect(args.insert_into,
                     use_lock_file=not args.no_lock_file) as db2:
            for row in db.select(query, sort=args.sort):
                kvp = row.get('key_value_pairs', {})
                nkvp -= len(kvp)
                kvp.update(add_key_value_pairs)
                nkvp += len(kvp)
                if args.unique:
                    row['unique_id'] = '%x' % randint(16**31, 16**32 - 1)
                if args.strip_data:
                    db2.write(row.toatoms(), **kvp)
                else:
                    db2.write(row, data=row.get('data'), **kvp)
                nrows += 1

        out('Added %s (%s updated)' %
            (plural(nkvp, 'key-value pair'),
             plural(len(add_key_value_pairs) * nrows - nkvp, 'pair')))
        out('Inserted %s' % plural(nrows, 'row'))
        return

    if add_key_value_pairs or delete_keys:
        ids = [row['id'] for row in db.select(query)]
        M = 0
        N = 0
        with db:
            for id in ids:
                m, n = db.update(id, delete_keys=delete_keys,
                                 **add_key_value_pairs)
                M += m
                N += n
        out('Added %s (%s updated)' %
            (plural(M, 'key-value pair'),
             plural(len(add_key_value_pairs) * len(ids) - M, 'pair')))
        out('Removed', plural(N, 'key-value pair'))

        return

    if args.delete:
        ids = [row['id'] for row in db.select(query)]
        if ids and not args.yes:
            msg = 'Delete %s? (yes/No): ' % plural(len(ids), 'row')
            if input(msg).lower() != 'yes':
                return
        db.delete(ids)
        out('Deleted %s' % plural(len(ids), 'row'))
        return

    if args.plot_data:
        from ase.db.plot import dct2plot
        dct2plot(db.get(query).data, args.plot_data)
        return

    if args.plot:
        if ':' in args.plot:
            tags, keys = args.plot.split(':')
            tags = tags.split(',')
        else:
            tags = []
            keys = args.plot
        keys = keys.split(',')
        plots = defaultdict(list)
        X = {}
        labels = []
        for row in db.select(query, sort=args.sort, include_data=False):
            name = ','.join(str(row[tag]) for tag in tags)
            x = row.get(keys[0])
            if x is not None:
                if isinstance(x, basestring):
                    if x not in X:
                        X[x] = len(X)
                        labels.append(x)
                    x = X[x]
                plots[name].append([x] + [row.get(key) for key in keys[1:]])
        import matplotlib.pyplot as plt
        for name, plot in plots.items():
            xyy = zip(*plot)
            x = xyy[0]
            for y, key in zip(xyy[1:], keys[1:]):
                plt.plot(x, y, label=name + ':' + key)
        if X:
            plt.xticks(range(len(labels)), labels, rotation=90)
        plt.legend()
        plt.show()
        return

    if args.json:
        row = db.get(query)
        db2 = connect(sys.stdout, 'json', use_lock_file=False)
        kvp = row.get('key_value_pairs', {})
        db2.write(row, data=row.get('data'), **kvp)
        return

    db.python = args.metadata_from_python_script

    if args.long:
        db.meta = process_metadata(db, html=args.open_web_browser)
        row = db.get(query)
        summary = Summary(row, db.meta)
        summary.write()
        return

    if args.open_web_browser:
        import ase.db.app as app
        app.databases['default'] = db
        app.app.run(host='0.0.0.0', debug=True)
        return

    if args.write_summary_files:
        prefix = args.write_summary_files
        db.meta = process_metadata(db, html=args.open_web_browser)
        ukey = db.meta.get('unique_key', 'id')
        for row in db.select(query):
            uid = row.get(ukey)
            summary = Summary(row,
                              db.meta,
                              prefix='{}-{}-'.format(prefix, uid))
        return

    columns = list(all_columns)
    c = args.columns
    if c and c.startswith('++'):
        keys = set()
        for row in db.select(query,
                             limit=args.limit, offset=args.offset,
                             include_data=False):
            keys.update(row._keys)
        columns.extend(keys)
        if c[2:3] == ',':
            c = c[3:]
        else:
            c = ''
    if c:
        if c[0] == '+':
            c = c[1:]
        elif c[0] != '-':
            columns = []
        for col in c.split(','):
            if col[0] == '-':
                columns.remove(col[1:])
            else:
                columns.append(col.lstrip('+'))

    table = Table(db, verbosity=verbosity, cut=args.cut)
    table.select(query, columns, args.sort, args.limit, args.offset)
    if args.csv:
        table.write_csv()
    else:
        table.write(query)
