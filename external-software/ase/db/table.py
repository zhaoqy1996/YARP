from __future__ import print_function

import numpy as np

from ase.db.core import float_to_time_string, now


all_columns = ['id', 'age', 'user', 'formula', 'calculator',
               'energy', 'fmax', 'pbc', 'volume',
               'charge', 'mass', 'smax', 'magmom']


def get_sql_columns(columns):
    """ Map the names of table columns to names of columns in
    the SQL tables"""
    sql_columns = columns[:]
    if 'age' in columns:
        sql_columns.remove('age')
        sql_columns += ['mtime', 'ctime']
    if 'user' in columns:
        sql_columns[sql_columns.index('user')] = 'username'
    if 'formula' in columns:
        sql_columns[sql_columns.index('formula')] = 'numbers'
    if 'fmax' in columns:
        sql_columns[sql_columns.index('fmax')] = 'forces'
    if 'smax' in columns:
        sql_columns[sql_columns.index('smax')] = 'stress'
    if 'volume' in columns:
        sql_columns[sql_columns.index('volume')] = 'cell'
    if 'mass' in columns:
        sql_columns[sql_columns.index('mass')] = 'masses'
    if 'charge' in columns:
        sql_columns[sql_columns.index('charge')] = 'charges'

    sql_columns.append('key_value_pairs')
    sql_columns.append('constraints')
    if 'id' not in sql_columns:
        sql_columns.append('id')

    return sql_columns


def plural(n, word):
    if n == 1:
        return '1 ' + word
    return '%d %ss' % (n, word)


def cut(txt, length):
    if len(txt) <= length or length == 0:
        return txt
    return txt[:length - 3] + '...'


def cutlist(lst, length):
    if len(lst) <= length or length == 0:
        return lst
    return lst[:9] + ['... ({} more)'.format(len(lst) - 9)]


class Table:
    def __init__(self, connection, unique_key='id', verbosity=1, cut=35):
        self.connection = connection
        self.verbosity = verbosity
        self.cut = cut
        self.rows = []
        self.columns = None
        self.id = None
        self.right = None
        self.keys = None
        self.unique_key = unique_key

    def select(self, query, columns, sort, limit, offset):
        sql_columns = get_sql_columns(columns)
        self.limit = limit
        self.offset = offset
        self.rows = [Row(row, columns, self.unique_key)
                     for row in self.connection.select(
                         query, verbosity=self.verbosity,
                         limit=limit, offset=offset, sort=sort,
                         include_data=False, columns=sql_columns)]

        delete = set(range(len(columns)))
        for row in self.rows:
            for n in delete.copy():
                if row.values[n] is not None:
                    delete.remove(n)
        delete = sorted(delete, reverse=True)
        for row in self.rows:
            for n in delete:
                del row.values[n]

        self.columns = list(columns)
        for n in delete:
            del self.columns[n]

    def format(self, subscript=None):
        right = set()  # right-adjust numbers
        allkeys = set()
        for row in self.rows:
            numbers = row.format(self.columns, subscript)
            right.update(numbers)
            allkeys.update(row.dct.get('key_value_pairs', {}))

        right.add('age')
        self.right = [column in right for column in self.columns]

        self.keys = sorted(allkeys)

    def write(self, query=None):
        self.format()
        L = [[len(s) for s in row.strings]
             for row in self.rows]
        L.append([len(c) for c in self.columns])
        N = np.max(L, axis=0)

        fmt = '{:{align}{width}}'
        if self.verbosity > 0:
            print('|'.join(fmt.format(c, align='<>'[a], width=w)
                           for c, a, w in zip(self.columns, self.right, N)))
        for row in self.rows:
            print('|'.join(fmt.format(c, align='<>'[a], width=w)
                           for c, a, w in
                           zip(row.strings, self.right, N)))

        if self.verbosity == 0:
            return

        nrows = len(self.rows)

        if self.limit and nrows == self.limit:
            n = self.connection.count(query)
            print('Rows:', n, '(showing first {})'.format(self.limit))
        else:
            print('Rows:', nrows)

        if self.keys:
            print('Keys:', ', '.join(cutlist(self.keys, self.cut)))

    def write_csv(self):
        if self.verbosity > 0:
            print(', '.join(self.columns))
        for row in self.rows:
            print(', '.join(str(val) for val in row.values))


class Row:
    def __init__(self, dct, columns, unique_key='id'):
        self.dct = dct
        self.values = None
        self.strings = None
        self.more = False
        self.set_columns(columns)
        self.uid = dct[unique_key]

    def set_columns(self, columns):
        self.values = []
        for c in columns:
            if c == 'age':
                value = float_to_time_string(now() - self.dct.ctime)
            elif c == 'pbc':
                value = ''.join('FT'[p] for p in self.dct.pbc)
            else:
                value = getattr(self.dct, c, None)
            self.values.append(value)

    def toggle(self):
        self.more = not self.more

    def format(self, columns, subscript=None):
        self.strings = []
        numbers = set()
        for value, column in zip(self.values, columns):
            if column == 'formula' and subscript:
                value = subscript.sub(r'<sub>\1</sub>', value)
            elif isinstance(value, dict):
                value = str(value)
            elif isinstance(value, list):
                value = str(value)
            elif isinstance(value, np.ndarray):
                value = str(value.tolist())
            elif isinstance(value, int):
                value = str(value)
                numbers.add(column)
            elif isinstance(value, float):
                numbers.add(column)
                value = '{:.3f}'.format(value)
            elif value is None:
                value = ''
            self.strings.append(value)

        return numbers
