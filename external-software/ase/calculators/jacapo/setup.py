from distutils.core import setup
import sys

long_description = """\
Jacapo is a python package providing an interface to Dacapo that
is compatible with the open source Atomic Simulation
Environment in the python scripting language."""

if sys.version_info < (2, 3, 0, 'final', 0):
    raise SystemExit('Python 2.3 or later is required!')

packages = ['Jacapo']

tools = ['tools/ncsum',
         'tools/plotnetcdf',
         'tools/pysub',
         'tools/qn_relax',
         'tools/stripnetcdf']

# Get the current version number:
exec(compile(open('version.py').read(), 'version.py', 'exec'))

setup(name = 'python-Jacapo',
      # version=version,
      description='Jacapo - ase + Dacapo',
      url='http://www.fysik.dtu.dk/Campos/ase',
      maintainer='John Kitchin',
      maintainer_email='jkitchin@andrew.cmu.edu',
      license='LGPL',
      platforms=['linux'],
      packages=packages,
      scripts=tools,
      long_description=long_description)
