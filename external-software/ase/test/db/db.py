import os
import time
from ase.test import cli
from ase.db import connect

cmd = """
ase build H | ase run emt -d testase.json &&
ase build H2O | ase run emt -d testase.json &&
ase build O2 | ase run emt -d testase.json &&
ase build H2 | ase run emt -f 0.02 -d testase.json &&
ase build O2 | ase run emt -f 0.02 -d testase.json &&
ase build -x fcc Cu | ase run emt -E 5,1 -d testase.json &&
ase db -v testase.json natoms=1,Cu=1 --delete --yes &&
ase db -v testase.json "H>0" -k hydro=1,abc=42,foo=bar &&
ase db -v testase.json "H>0" --delete-keys foo"""


def count(n, *args, **kwargs):
    m = len(list(con.select(columns=['id'], *args, **kwargs)))
    assert m == n, (m, n)


t0 = time.time()
for name in ['testase.json', 'testase.db', 'postgresql']:
    if name == 'postgresql':
        if os.environ.get('POSTGRES_DB'):  # gitlab-ci
            name = 'postgresql://ase:ase@postgres:5432/testase'
        else:
            name = os.environ.get('ASE_TEST_POSTGRES_URL')
            if name is None:
                continue

    con = connect(name)
    t1 = time.time()
    if 'postgres' in name:
        con.delete([row.id for row in con.select()])

    cli(cmd.replace('testase.json', name))
    assert con.get_atoms(H=1)[0].magmom == 1
    count(5)
    count(3, 'hydro')
    count(0, 'foo')
    count(3, abc=42)
    count(3, 'abc')
    count(0, 'abc,foo')
    count(3, 'abc,hydro')
    count(0, foo='bar')
    count(1, formula='H2')
    count(1, formula='H2O')
    count(3, 'fmax<0.1')
    count(1, '0.5<mass<1.5')
    count(5, 'energy')

    id = con.reserve(abc=7)
    assert con[id].abc == 7

    for key in ['calculator', 'energy', 'abc', 'name', 'fmax']:
        count(6, sort=key)
        count(6, sort='-' + key)

    cli('ase -T gui --terminal {}@3'.format(name))

    con.delete([id])

    if name != 'testase.json':  # transfer between db formats
        if name == 'testase.db':
            from_db = 'testase.json'
            factor = 2
        else:
            from_db = 'testase.db'
            factor = 3

        cli('ase db {} --insert-into {}'.format(from_db, name))

        count(5 * factor)
        count(3 * factor, 'hydro')
        count(0, 'foo')
        count(3 * factor, abc=42)
        count(3 * factor, 'abc')
        count(0, 'abc,foo')
        count(3 * factor, 'abc,hydro')
        count(0, foo='bar')
        count(factor, formula='H2')
        count(factor, formula='H2O')
        count(3 * factor, 'fmax<0.1')
        count(factor, '0.5<mass<1.5')
        count(5 * factor, 'energy')

    t2 = time.time()

    print('----------------------------------')
    print('Finnished test for {}'.format(name))
    print('runtime = {} sec'.format(t2 - t1))
    print('----------------------------------')


print('Total runtime = {} sec'.format(t2 - t0))
