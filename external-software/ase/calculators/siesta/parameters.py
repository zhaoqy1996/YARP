from ase.calculators.calculator import Parameters
from ase.utils import basestring

"""
2017.04 - Pedro Brandimarte: changes for python 2-3 compatible
"""

class PAOBasisBlock(Parameters):
    """
    Representing a block in PAO.Basis for one species.
    """
    def __init__(self, block):
        """
        Parameters:
            -block : String. A block defining the basis set of a single
                     species using the format of a PAO.Basis block.
                     The initial label should be left out since it is
                     determined programatically.
                     Example1: 2 nodes 1.0
                               n=2 0 2 E 50.0 2.5
                               3.50 3.50
                               0.95 1.00
                               1 1 P 2
                               3.50
                     Example2: 1
                               0 2 S 0.2
                               5.00 0.00
                     See siesta manual for details.
        """
        assert isinstance(block, basestring)
        Parameters.__init__(self, block=block)

    def script(self, label):
        """
        Write the fdf script for the block.

        Parameters:
            -label : The label to insert in front of the block.
        """
        return label + ' ' + self['block']


class Species(Parameters):
    """
    Parameters for specifying the behaviour for a single species in the
    calculation. If the tag argument is set to an integer then atoms with
    the specified element and tag will be a separate species.

    Pseudopotential and basis set can be specified. Additionally the species
    can be set be a ghost species, meaning that they will not be considered
    atoms, but the corresponding basis set will be used.
    """
    def __init__(self,
                 symbol,
                 basis_set='DZP',
                 pseudopotential=None,
                 tag=None,
                 ghost=False,
                 excess_charge=None):
        kwargs = locals()
        kwargs.pop('self')
        Parameters.__init__(self, **kwargs)


def format_fdf(key, value):
    """
    Write an fdf key-word value pair.

    Parameters:
        - key   : The fdf-key
        - value : The fdf value.
    """
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return ''

    key = format_key(key)
    new_value = format_value(value)

    if isinstance(value, (list, tuple)):
        string = '%block ' + key + '\n' +\
            new_value + '\n' + \
            '%endblock ' + key + '\n'
    else:
        string = '%s  %s\n' % (key, new_value)

    return string


def format_value(value):
    """
    Format python values to fdf-format.

    Parameters:
        - value : The value to format.
    """
    if isinstance(value, tuple):
        sub_values = [format_value(v) for v in value]
        value = '\t'.join(sub_values)
    elif isinstance(value, list):
        sub_values = [format_value(v) for v in value]
        value = '\n'.join(sub_values)
    else:
        value = str(value)

    return value


def format_key(key):
    """ Fix the fdf-key replacing '_' with '.' and '__' with '_' """
    key = key.replace('__', '#')
    key = key.replace('_', '.')
    key = key.replace('#', '_')

    return key
