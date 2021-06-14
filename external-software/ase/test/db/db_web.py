from ase import Atoms
from ase.db import connect
import ase.db.app as app

c = connect('test.db', append=False)
x = [0, 1, 2]
t1 = [1, 2, 0]
t2 = [[2, 3], [1, 1], [1, 0]]

atoms = Atoms('H2O')
atoms.center(vacuum=5)
atoms.set_pbc(True)

c.write(atoms,
        foo=42.0,
        bar='abc',
        data={'x': x,
              't1': t1,
              't2': t2})

c.python = {'title': 'Test title',
            'key_descriptions':
                {'foo': ('FOO', 'FOO ...', '`m_e`')},
            'default_columns': ['foo', 'formula', 'bar']}
app.databases['default'] = c
app.app.testing = True
c = app.app.test_client()
page = c.get('/').data.decode()
assert 'Test title' in page
assert 'FOO' in page
c.get('/default/row/1')
c.get('/default/json/1').data
c.get('/default/sqlite/1').data
c.get('/default/sqlite?x=1').data
c.get('/default/json?x=1').data
