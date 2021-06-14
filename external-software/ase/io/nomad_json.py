from ase.nomad import read as _read_nomad_json
from ase.utils import basestring

def read_nomad_json(fd, index):
    # wth, we should not be passing index like this!
    from ase.io.formats import string2index
    if isinstance(index, basestring):
        index = string2index(index)

    d = _read_nomad_json(fd)
    images = list(d.iterimages())
    return images[index]
