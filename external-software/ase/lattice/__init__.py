import functools
import warnings
from ase.build.bulk import bulk as newbulk
__all__ = ['bulk']


@functools.wraps(newbulk)
def bulk(*args, **kwargs):
    warnings.warn('Use ase.build.bulk() instead', stacklevel=2)
    return newbulk(*args, **kwargs)
