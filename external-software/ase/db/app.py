"""WSGI Flask-app for browsing a database.

You can launch Flask's local webserver like this::

    $ ase db abc.db -w

For a real webserver, you need to set the $ASE_DB_APP_CONFIG environment
variable to point to a configuration file like this::

    ASE_DB_NAMES = ['/path/to/db-file/project1.db',
                    'postgresql://user:pw@localhost:5432/project2']
    ASE_DB_HOMEPAGE = '<a href="https://home.page.dk">HOME</a> ::'

Start with something like::

    twistd web --wsgi=ase.db.app.app --port=8000

"""

from __future__ import print_function
import collections
import functools
import io
import os
import os.path as op
import re
import sys
import tempfile

from flask import Flask, render_template, request, send_from_directory, flash

try:
    import matplotlib
    matplotlib.use('Agg', warn=False)
except ImportError:
    pass

import ase.db
import ase.db.web
from ase.db.core import convert_str_to_int_float_or_str
from ase.db.plot import atoms2png
from ase.db.summary import Summary
from ase.db.table import Table, all_columns
from ase.visualize import view
from ase import Atoms
from ase.calculators.calculator import kptdensity2monkhorstpack


# Every client-connetions gets one of these tuples:
Connection = collections.namedtuple(
    'Connection',
    ['query',  # query string
     'nrows',  # number of rows matched
     'page',  # page number
     'columns',  # what columns to show
     'sort',  # what column to sort after
     'limit'])  # number of rows per page

app = Flask(__name__)

app.secret_key = 'asdf'

databases = {}
home = ''  # link to homepage
ase_db_footer = ''  # footer (for a license)
open_ase_gui = True  # click image to open ASE's GUI
download_button = True

# List of (project-name, title, nrows) tuples (will be filled in at run-time):
projects = []  # type: List[str, str, int]


def connect_databases(uris):
    python_configs = []
    dbs = []
    for uri in uris:
        if uri.endswith('.py'):
            python_configs.append(uri)
            continue
        if uri.startswith('postgresql://'):
            project = uri.rsplit('/', 1)[1]
        else:
            project = uri.rsplit('/', 1)[-1].split('.')[0]
        db = ase.db.connect(uri)
        db.python = None
        databases[project] = db
        dbs.append(db)

    for py, db in zip(python_configs, dbs):
        db.python = py


next_con_id = 1
connections = {}

if 'ASE_DB_APP_CONFIG' in os.environ:
    app.config.from_envvar('ASE_DB_APP_CONFIG')
    connect_databases(str(name) for name in app.config['ASE_DB_NAMES'])
    home = app.config['ASE_DB_HOMEPAGE']
    ase_db_footer = app.config['ASE_DB_FOOTER']
    tmpdir = str(app.config['ASE_DB_TMPDIR'])
    download_button = app.config['ASE_DB_DOWNLOAD']
    open_ase_gui = False
else:
    tmpdir = tempfile.mkdtemp(prefix='ase-db-app-')  # used to cache png-files

# Find numbers in formulas so that we can convert H2O to H<sub>2</sub>O:
SUBSCRIPT = re.compile(r'(\d+)')


errors = 0


def error(e):
    """Write traceback and other stuff to 00-99.error files."""
    global errors
    import traceback
    x = request.args.get('x', '0')
    try:
        cid = int(x)
    except ValueError:
        cid = 0
    con = connections.get(cid)
    with open(op.join(tmpdir, '{:02}.err'.format(errors % 100)), 'w') as fd:
        print(repr((errors, con, e, request)), file=fd)
        if hasattr(e, '__traceback__'):
            traceback.print_tb(e.__traceback__, file=fd)
    errors += 1
    raise e


# app.register_error_handler(Exception, error)


@app.route('/', defaults={'project': None})
@app.route('/<project>/')
@app.route('/<project>')
def index(project):
    global next_con_id

    # Backwards compatibility:
    project = request.args.get('project') or project

    if not projects:
        # First time: initialize list of projects
        for proj, db in sorted(databases.items()):
            meta = ase.db.web.process_metadata(db)
            db.meta = meta
            nrows = len(db)
            projects.append((proj, db.meta.get('title', proj), nrows))
            print('Initialized {proj}: {nrows} rows'
                  .format(proj=proj, nrows=nrows))

    if project is None and len(projects) > 1:
        return render_template('projects.html',
                               projects=projects,
                               home=home,
                               md=None,
                               ase_db_footer=ase_db_footer)

    if project is None:
        project = list(databases)[0]

    con_id = int(request.args.get('x', '0'))
    if con_id in connections:
        query, nrows, page, columns, sort, limit = connections[con_id]

    if con_id not in connections:
        # Give this connetion a new id:
        con_id = next_con_id
        next_con_id += 1
        query = ['', {}, '']
        nrows = None
        page = 0
        columns = None
        sort = 'id'
        limit = 25

    db = databases.get(project)
    if db is None:
        return 'No such project: ' + project

    meta = db.meta

    if columns is None:
        columns = meta.get('default_columns')[:] or list(all_columns)

    if 'sort' in request.args:
        column = request.args['sort']
        if column == sort:
            sort = '-' + column
        elif '-' + column == sort:
            sort = 'id'
        else:
            sort = column
        page = 0
    elif 'query' in request.args:
        dct = {}
        query = [request.args['query']]
        q = query[0]
        for special in meta['special_keys']:
            kind, key = special[:2]
            if kind == 'SELECT':
                value = request.args['select_' + key]
                dct[key] = convert_str_to_int_float_or_str(value)
                if value:
                    q += ',{}={}'.format(key, value)
            elif kind == 'BOOL':
                value = request.args['bool_' + key]
                dct[key] = convert_str_to_int_float_or_str(value)
                if value:
                    q += ',{}={}'.format(key, value)
            else:
                v1 = request.args['from_' + key]
                v2 = request.args['to_' + key]
                var = request.args['range_' + key]
                dct[key] = (v1, v2, var)
                if v1 or v2:
                    var = request.args['range_' + key]
                    if v1:
                        q += ',{}>={}'.format(var, v1)
                    if v2:
                        q += ',{}<={}'.format(var, v2)
        q = q.lstrip(',')
        query += [dct, q]
        sort = 'id'
        page = 0
        nrows = None
    elif 'limit' in request.args:
        limit = int(request.args['limit'])
        page = 0
    elif 'page' in request.args:
        page = int(request.args['page'])

    if 'toggle' in request.args:
        column = request.args['toggle']
        if column == 'reset':
            columns = meta.get('default_columns')[:] or list(all_columns)
        else:
            if column in columns:
                columns.remove(column)
                if column == sort.lstrip('-'):
                    sort = 'id'
                    page = 0
            else:
                columns.append(column)

    okquery = query

    if nrows is None:
        try:
            nrows = db.count(query[2])
        except (ValueError, KeyError) as e:
            flash(', '.join(['Bad query'] + list(e.args)))
            okquery = ('', {}, 'id=0')  # this will return no rows
            nrows = 0

    table = Table(db, meta.get('unique_key', 'id'))
    table.select(okquery[2], columns, sort, limit, offset=page * limit)

    con = Connection(query, nrows, page, columns, sort, limit)
    connections[con_id] = con

    if len(connections) > 1000:
        # Forget old connections:
        for cid in sorted(connections)[:200]:
            del connections[cid]

    table.format(SUBSCRIPT)

    addcolumns = [column for column in all_columns + table.keys
                  if column not in table.columns]

    return render_template('table.html',
                           project=project,
                           t=table,
                           md=meta,
                           con=con,
                           x=con_id,
                           home=home,
                           ase_db_footer=ase_db_footer,
                           pages=pages(page, nrows, limit),
                           nrows=nrows,
                           addcolumns=addcolumns,
                           row1=page * limit + 1,
                           row2=min((page + 1) * limit, nrows),
                           download_button=download_button)


@app.route('/<project>/image/<name>')
def image(project, name):
    id = int(name[:-4])
    name = project + '-' + name
    path = op.join(tmpdir, name)
    if not op.isfile(path):
        db = databases[project]
        atoms = db.get_atoms(id)
        atoms2png(atoms, path)

    return send_from_directory(tmpdir, name)


@app.route('/<project>/cif/<name>')
def cif(project, name):
    id = int(name[:-4])
    name = project + '-' + name
    path = op.join(tmpdir, name)
    if not op.isfile(path):
        db = databases[project]
        atoms = db.get_atoms(id)
        atoms.write(path)
    return send_from_directory(tmpdir, name)


@app.route('/<project>/plot/<uid>/<png>')
def plot(project, uid, png):
    png = project + '-' + uid + '-' + png
    return send_from_directory(tmpdir, png)


@app.route('/<project>/gui/<int:id>')
def gui(project, id):
    if open_ase_gui:
        db = databases[project]
        atoms = db.get_atoms(id)
        view(atoms)
    return '', 204, []


@app.route('/<project>/row/<uid>')
def row(project, uid):
    db = databases[project]
    if not hasattr(db, 'meta'):
        db.meta = ase.db.web.process_metadata(db)
    prefix = '{}/{}-{}-'.format(tmpdir, project, uid)
    key = db.meta.get('unique_key', 'id')
    try:
        uid = int(uid)
    except ValueError:
        pass
    row = db.get(**{key: uid})
    s = Summary(row, db.meta, SUBSCRIPT, prefix)
    atoms = Atoms(cell=row.cell, pbc=row.pbc)
    n1, n2, n3 = kptdensity2monkhorstpack(atoms,
                                          kptdensity=1.8,
                                          even=False)
    return render_template('summary.html',
                           project=project,
                           s=s,
                           uid=uid,
                           n1=n1,
                           n2=n2,
                           n3=n3,
                           home=home,
                           back=True,
                           ase_db_footer=ase_db_footer,
                           md=db.meta,
                           open_ase_gui=open_ase_gui)


def tofile(project, query, type, limit=0):
    fd, name = tempfile.mkstemp(suffix='.' + type)
    con = ase.db.connect(name, use_lock_file=False)
    db = databases[project]
    for row in db.select(query, limit=limit):
        con.write(row,
                  data=row.get('data', {}),
                  **row.get('key_value_pairs', {}))
    os.close(fd)
    data = open(name, 'rb').read()
    os.unlink(name)
    return data


def download(f):
    @functools.wraps(f)
    def ff(*args, **kwargs):
        text, name = f(*args, **kwargs)
        if name is None:
            return text
        headers = [('Content-Disposition',
                    'attachment; filename="{}"'.format(name)),
                   ]  # ('Content-type', 'application/sqlite3')]
        return text, 200, headers
    return ff


@app.route('/<project>/xyz/<int:id>')
@download
def xyz(project, id):
    fd = io.StringIO()
    from ase.io.xyz import write_xyz
    db = databases[project]
    write_xyz(fd, db.get_atoms(id))
    data = fd.getvalue()
    return data, '{}.xyz'.format(id)


if download_button:
    @app.route('/<project>/json')
    @download
    def jsonall(project):
        con_id = int(request.args['x'])
        con = connections[con_id]
        data = tofile(project, con.query[2], 'json', con.limit)
        return data, 'selection.json'


@app.route('/<project>/json/<int:id>')
@download
def json1(project, id):
    if project not in databases:
        return 'No such project: ' + project, None
    data = tofile(project, id, 'json')
    return data, '{}.json'.format(id)


if download_button:
    @app.route('/<project>/sqlite')
    @download
    def sqliteall(project):
        con_id = int(request.args['x'])
        con = connections[con_id]
        data = tofile(project, con.query[2], 'db', con.limit)
        return data, 'selection.db'


@app.route('/<project>/sqlite/<int:id>')
@download
def sqlite1(project, id):
    if project not in databases:
        return 'No such project: ' + project, None
    data = tofile(project, id, 'db')
    return data, '{}.db'.format(id)


@app.route('/robots.txt')
def robots():
    return ('User-agent: *\n'
            'Disallow: /\n'
            '\n'
            'User-agent: Baiduspider\n'
            'Disallow: /\n'
            '\n'
            'User-agent: SiteCheck-sitecrawl by Siteimprove.com\n'
            'Disallow: /\n',
            200)


@app.route('/cif/<stuff>')
def oldcif(stuff):
    return 'Bad URL'


def pages(page, nrows, limit):
    """Helper function for pagination stuff."""
    npages = (nrows + limit - 1) // limit
    p1 = min(5, npages)
    p2 = max(page - 4, p1)
    p3 = min(page + 5, npages)
    p4 = max(npages - 4, p3)
    pgs = list(range(p1))
    if p1 < p2:
        pgs.append(-1)
    pgs += list(range(p2, p3))
    if p3 < p4:
        pgs.append(-1)
    pgs += list(range(p4, npages))
    pages = [(page - 1, 'previous')]
    for p in pgs:
        if p == -1:
            pages.append((-1, '...'))
        elif p == page:
            pages.append((-1, str(p + 1)))
        else:
            pages.append((p, str(p + 1)))
    nxt = min(page + 1, npages - 1)
    if nxt == page:
        nxt = -1
    pages.append((nxt, 'next'))
    return pages


if __name__ == '__main__':
    if len(sys.argv) > 1:
        connect_databases(sys.argv[1:])
    open_ase_gui = False
    app.run(host='0.0.0.0', debug=True)
