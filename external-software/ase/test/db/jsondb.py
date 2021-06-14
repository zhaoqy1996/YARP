"""Read and write json from/to file descriptor."""
import sys
from io import StringIO, BytesIO

from ase.io import read, write

s = u"""
{"1":
     {"numbers": [1, 1],
      "positions": [[0.0, 0.0, 0.35],
                    [0.0, 0.0, -0.35]]}}
"""

fd = StringIO(s)
a = read(fd, format='json')
assert a.get_chemical_formula() == 'H2'

if sys.version_info[0] == 2:
    fd = BytesIO()
else:
    fd = StringIO()
write(fd, a, format='json')

fd.seek(0)
a = read(fd, format='json')
assert a.get_chemical_formula() == 'H2'
