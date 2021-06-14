import ase.db
from ase.utils import basestring
from ase.io.formats import string2index


def read_db(filename, index, **kwargs):
    db = ase.db.connect(filename, serial=True, **kwargs)

    if isinstance(index, basestring):
        try:
            index = string2index(index)
        except ValueError:
            pass

    if isinstance(index, int):
        index = slice(index, index + 1 or None)

    if isinstance(index, basestring):
        # index is a database query string:
        for row in db.select(index):
            yield row.toatoms()
    else:
        start, stop, step = index.indices(db.count())
        if start == stop:
            return
        assert step == 1
        for row in db.select(offset=start, limit=stop - start):
            yield row.toatoms()


def write_db(filename, images, **kwargs):
    con = ase.db.connect(filename, serial=True, **kwargs)
    for atoms in images:
        con.write(atoms)


read_json = read_db
write_json = write_db
read_postgresql = read_db
write_postgresql = write_db
