from time import time
import ase.db
from ase import Atoms

for name in ['x.json', 'x.db']:
    print(name)
    db = ase.db.connect(name, append=False)
    db.write(Atoms(), x=1, data={'a': 1})
    db.update(1, y=2, data={'b': 2})
    db.update(1, delete_keys=['x'])
    row = db.get(1)
    print(row.y, row.data)
    assert 'x' not in row
    db.update(1, atoms=Atoms('H'))
    row = db.get(1)
    print(row.y, row.data, row.numbers)
    assert (row.numbers == [1]).all()
    assert sorted(row.data) == ['a', 'b']

    db.write(Atoms(), id=1)
    row = db.get(1)
    assert len(row.data) == 0
    assert len(row.key_value_pairs) == 0
    assert len(row.numbers) == 0

    # N = 100
    N = 5
    for i in range(N):
        db.write(Atoms('H10'), i=i, data={'c': 3})

    t0 = time()
    for id in range(2, 2 + N):
        db.update(id, z=3)
    print(time() - t0)

    # This should be faster for large N:
    t0 = time()
    with db:
        for id in range(2, 2 + N):
            db.update(id, z=3)
    print(time() - t0)
