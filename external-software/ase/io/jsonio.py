import datetime
import json

import numpy as np
from ase.utils import basestring


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype == complex:
                return {'__complex_ndarray__': (obj.real.tolist(),
                                                obj.imag.tolist())}
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, datetime.datetime):
            return {'__datetime__': obj.isoformat()}
        if hasattr(obj, 'todict'):
            return obj.todict()
        return json.JSONEncoder.default(self, obj)


encode = MyEncoder().encode


def object_hook(dct):
    if '__datetime__' in dct:
        return datetime.datetime.strptime(dct['__datetime__'],
                                          '%Y-%m-%dT%H:%M:%S.%f')
    if '__complex_ndarray__' in dct:
        r, i = (np.array(x) for x in dct['__complex_ndarray__'])
        return r + i * 1j
    return dct


mydecode = json.JSONDecoder(object_hook=object_hook).decode


def intkey(key):
    try:
        return int(key)
    except ValueError:
        return key


def numpyfy(obj):
    if isinstance(obj, dict):
        if '__complex_ndarray__' in obj:
            r, i = (np.array(x) for x in obj['__complex_ndarray__'])
            return r + i * 1j
        return dict((intkey(key), numpyfy(value))
                    for key, value in obj.items())
    if isinstance(obj, list) and len(obj) > 0:
        try:
            a = np.array(obj)
        except ValueError:
            pass
        else:
            if a.dtype in [bool, int, float]:
                return a
        obj = [numpyfy(value) for value in obj]
    return obj


def decode(txt):
    return numpyfy(mydecode(txt))


def read_json(name):
    if isinstance(name, basestring):
        fd = open(name, 'r')
    else:
        fd = name
    dct = decode(fd.read())
    fd.close()
    return dct
