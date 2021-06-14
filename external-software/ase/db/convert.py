import optparse
import os

import numpy as np

from ase.db import connect
from ase.db.sqlite import index_statements
from ase.utils import basestring


def convert(name, opts):
    con1 = connect(name, use_lock_file=False)
    con1._allow_reading_old_format = True
    newname = name[:-2] + 'new.db'
    with connect(newname, create_indices=False, use_lock_file=False) as con2:
        row = None
        for row in con1.select():
            kvp = row.get('key_value_pairs', {})
            if opts.convert_strings_to_numbers:
                for key, value in kvp.items():
                    if isinstance(value, basestring):
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                        else:
                            kvp[key] = value
            if opts.convert_minus_to_not_a_number:
                for key, value in kvp.items():
                    if value == '-':
                        kvp[key] = np.nan

            atoms = row.toatoms()
            if opts.remove_constrints:
                atoms.constraints = []
            con2.write(atoms, data=row.get('data'), **kvp)

        assert row is not None, 'Your database is empty!'

    c = con2._connect()
    for statement in index_statements:
        c.execute(statement)
    c.commit()

    os.rename(name, name[:-2] + 'old.db')
    os.rename(newname, name)


def main():
    parser = optparse.OptionParser()
    parser.add_option('-S', '--convert-strings-to-numbers',
                      action='store_true')
    parser.add_option('-N', '--convert-minus-to-not-a-number',
                      action='store_true')
    parser.add_option('-C', '--remove-constraints',
                      action='store_true')
    opts, args = parser.parse_args()
    for name in args:
        convert(name, opts)


if __name__ == '__main__':
    main()
