from __future__ import print_function
import sys

import numpy as np

msg = "\n'TypeError: array cannot be safely cast to required type'\n"
msg += "means you are probably using a broken ScientficPython, \n"
msg += "see: https://bugs.launchpad.net/ubuntu/+source/python-scientific/+bug/1041302\n"

import Scientific.IO.NetCDF as netcdf
import Scientific
version = Scientific.__version__.split(".")
print('Found ScientificPython version: ',Scientific.__version__)
if list(map(int,version)) < [2,8]:
    print('ScientificPython 2.8 or greater required for numpy support in NetCDF')
    raise RuntimeError('ScientificPython version 2.8 or greater is requied')

handle = netcdf.NetCDFFile("test.nc", "w")
try:
    handle.test = np.array([1.0])
except TypeError:
    print(msg, file=sys.stderr)
    raise
handle.close()
