import os
from ase.calculators.octopus import parse_input_file, kwargs2atoms
from ase.utils import basestring


def read_octopus(fileobj, get_kwargs=False):
    if isinstance(fileobj, basestring):  # This could be solved with decorators...
        fileobj = open(fileobj)

    kwargs = parse_input_file(fileobj)

    # input files may contain internal references to other files such
    # as xyz or xsf.  We need to know the directory where the file
    # resides in order to locate those.  If fileobj is a real file
    # object, it contains the path and we can use it.  Else assume
    # pwd.
    #
    # Maybe this is ugly; maybe it can lead to strange bugs if someone
    # wants a non-standard file-like type.  But it's probably better than
    # failing 'ase gui somedir/inp'
    try:
        fname = fileobj.name
    except AttributeError:
        directory = None
    else:
        directory = os.path.split(fname)[0]

    atoms, remaining_kwargs = kwargs2atoms(kwargs, directory=directory)
    if get_kwargs:
        return atoms, remaining_kwargs
    else:
        return atoms
