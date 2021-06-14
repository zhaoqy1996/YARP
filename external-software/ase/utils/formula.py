from ase.data import chemical_symbols
from collections import Counter

import sys
# should use math.gcd from python >= 3.5
if sys.version_info.major > 2 and sys.version_info.minor > 4:
    from math import gcd
else:
    from fractions import gcd


# no need to re-create this list at each function call
# non metals, half-metals/metalloid, halogen, noble gas
non_metals = ['H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne',
              'Si', 'P', 'S', 'Cl', 'Ar',
              'Ge', 'As', 'Se', 'Br', 'Kr',
              'Sb', 'Te', 'I', 'Xe',
              'Po', 'At', 'Rn']

def _count_symbols(numbers):
    """Take a list of atomic numbers and return a ditionary with elemt symbosl
    as keys and occurences as values"""
    if isinstance(numbers, dict):
        count = dict(numbers)
    else:
        count = Counter([chemical_symbols[Z] for Z in numbers])
    return count

def _empirical_symbols(count):
    """Find the least common multiple of all symbols"""
    counts = [c for c in count.values()]
    i = counts[0]
    for j in counts[1:]:
        _gcd = gcd(i,j)
        i=_gcd
    return {k : v//_gcd for k, v in count.items()}


def formula_hill(numbers, empirical=False):
    """Convert list of atomic numbers to a chemical formula as a string.

    Elements are alphabetically ordered with C and H first.

    If argument `empirical`, element counts will be divided by greatest common
    divisor to yield an empirical formula"""
    count = _count_symbols(numbers)
    if empirical:
        count = _empirical_symbols(count)
    result = [(s, count.pop(s)) for s in 'CH' if s in count]
    result += [(s, count[s]) for s in sorted(count)]
    return ''.join('{0}{1}'.format(symbol, n) if n > 1 else symbol
                   for symbol, n in result)


def formula_metal(numbers, empirical=False):
    """Convert list of atomic numbers to a chemical formula as a string.

    Elements are alphabetically ordered with metals first.

    If argument `empirical`, element counts will be divided by greatest common
    divisor to yield an empirical formula"""
    count = _count_symbols(numbers)
    if empirical:
        count = _empirical_symbols(count)
    result2 = [(s, count.pop(s)) for s in non_metals if s in count]
    result = [(s, count[s]) for s in sorted(count)]
    result += sorted(result2)
    return ''.join('{0}{1}'.format(symbol, n) if n > 1 else symbol
                   for symbol, n in result)

