from __future__ import absolute_import, print_function
import os
import sys

import numpy as np

from ase.db.core import Database, ops, lock, now
from ase.db.row import AtomsRow
from ase.io.jsonio import encode, decode
from ase.parallel import world, parallel_function
from ase.utils import basestring


class JSONDatabase(Database, object):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        pass

    def _write(self, atoms, key_value_pairs, data, id):
        Database._write(self, atoms, key_value_pairs, data)

        bigdct = {}
        ids = []
        nextid = 1

        if (isinstance(self.filename, basestring) and
            os.path.isfile(self.filename)):
            try:
                bigdct, ids, nextid = self._read_json()
            except (SyntaxError, ValueError):
                pass

        mtime = now()

        if isinstance(atoms, AtomsRow):
            row = atoms
        else:
            row = AtomsRow(atoms)
            row.ctime = mtime
            row.user = os.getenv('USER')

        dct = {}
        for key in row.__dict__:
            if key[0] == '_' or key in row._keys or key == 'id':
                continue
            dct[key] = row[key]

        dct['mtime'] = mtime

        if key_value_pairs:
            dct['key_value_pairs'] = key_value_pairs

        if data:
            dct['data'] = data

        constraints = row.get('constraints')
        if constraints:
            dct['constraints'] = constraints

        if id is None:
            id = nextid
            ids.append(id)
            nextid += 1
        else:
            assert id in bigdct

        bigdct[id] = dct
        self._write_json(bigdct, ids, nextid)
        return id

    def _read_json(self):
        if isinstance(self.filename, basestring):
            with open(self.filename) as fd:
                bigdct = decode(fd.read())
        else:
            bigdct = decode(self.filename.read())
            if self.filename is not sys.stdin:
                self.filename.seek(0)
        ids = bigdct.get('ids')
        if ids is None:
            # Allow for missing "ids" and "nextid":
            assert 1 in bigdct
            return bigdct, [1], 2
        if not isinstance(ids, list):
            ids = ids.tolist()
        return bigdct, ids, bigdct['nextid']

    def _write_json(self, bigdct, ids, nextid):
        if world.rank > 0:
            return

        if isinstance(self.filename, basestring):
            fd = open(self.filename, 'w')
        else:
            fd = self.filename
        print('{', end='', file=fd)
        for id in ids:
            dct = bigdct[id]
            txt = ',\n '.join('"{0}": {1}'.format(key, encode(dct[key]))
                              for key in sorted(dct.keys()))
            print('"{0}": {{\n {1}}},'.format(id, txt), file=fd)
        if self._metadata is not None:
            print('"metadata": {0},'.format(encode(self.metadata)), file=fd)
        print('"ids": {0},'.format(ids), file=fd)
        print('"nextid": {0}}}'.format(nextid), file=fd)

        if fd is not self.filename:
            fd.close()

    @parallel_function
    @lock
    def delete(self, ids):
        bigdct, myids, nextid = self._read_json()
        for id in ids:
            del bigdct[id]
            myids.remove(id)
        self._write_json(bigdct, myids, nextid)

    def _get_row(self, id):
        bigdct, ids, nextid = self._read_json()
        if id is None:
            assert len(ids) == 1
            id = ids[0]
        dct = bigdct[id]
        dct['id'] = id
        return AtomsRow(dct)

    def _select(self, keys, cmps, explain=False, verbosity=0,
                limit=None, offset=0, sort=None, include_data=True,
                columns='all'):
        if explain:
            yield {'explain': (0, 0, 0, 'scan table')}
            return

        if sort:
            if sort[0] == '-':
                reverse = True
                sort = sort[1:]
            else:
                reverse = False

            def f(row):
                return row.get(sort, missing)

            rows = []
            missing = []
            for row in self._select(keys, cmps):
                key = row.get(sort)
                if key is None:
                    missing.append((0, row))
                else:
                    rows.append((key, row))

            rows.sort(reverse=reverse, key=lambda x: x[0])
            rows += missing

            if limit:
                rows = rows[offset:offset + limit]
            for key, row in rows:
                yield row
            return

        try:
            bigdct, ids, nextid = self._read_json()
        except IOError:
            return

        if not limit:
            limit = -offset - 1

        cmps = [(key, ops[op], val) for key, op, val in cmps]
        n = 0
        for id in ids:
            if n - offset == limit:
                return
            dct = bigdct[id]
            if not include_data:
                dct.pop('data', None)
            row = AtomsRow(dct)
            row.id = id
            for key in keys:
                if key not in row:
                    break
            else:
                for key, op, val in cmps:
                    if isinstance(key, int):
                        value = np.equal(row.numbers, key).sum()
                    else:
                        value = row.get(key)
                        if key == 'pbc':
                            assert op in [ops['='], ops['!=']]
                            value = ''.join('FT'[x] for x in value)
                    if value is None or not op(value, val):
                        break
                else:
                    if n >= offset:
                        yield row
                    n += 1

    @property
    def metadata(self):
        if self._metadata is None:
            bigdct, myids, nextid = self._read_json()
            self._metadata = bigdct.get('metadata', {})
        return self._metadata.copy()

    @metadata.setter
    def metadata(self, dct):
        bigdct, ids, nextid = self._read_json()
        self._metadata = dct
        self._write_json(bigdct, ids, nextid)
