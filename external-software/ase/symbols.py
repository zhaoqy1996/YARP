import warnings

import numpy as np

from ase.data import atomic_numbers, chemical_symbols
from ase.utils import basestring, formula_hill, formula_metal


def string2symbols(s):
    """Convert string to list of chemical symbols."""
    n = len(s)

    if n == 0:
        return []

    c = s[0]

    if c.isdigit():
        i = 1
        while i < n and s[i].isdigit():
            i += 1
        return int(s[:i]) * string2symbols(s[i:])

    if c == '(':
        p = 0
        for i, c in enumerate(s):
            if c == '(':
                p += 1
            elif c == ')':
                p -= 1
                if p == 0:
                    break
        j = i + 1
        while j < n and s[j].isdigit():
            j += 1
        if j > i + 1:
            m = int(s[i + 1:j])
        else:
            m = 1
        return m * string2symbols(s[1:i]) + string2symbols(s[j:])

    if c.isupper():
        i = 1
        if 1 < n and s[1].islower():
            i += 1
        j = i
        while j < n and s[j].isdigit():
            j += 1
        if j > i:
            m = int(s[i:j])
        else:
            m = 1
        symbol = s[:i]
        if symbol not in atomic_numbers:
            raise ValueError
        return m * [symbol] + string2symbols(s[j:])
    else:
        raise ValueError


def symbols2numbers(symbols):
    if isinstance(symbols, basestring):
        symbols = string2symbols(symbols)
    numbers = []
    for s in symbols:
        if isinstance(s, basestring):
            numbers.append(atomic_numbers[s])
        else:
            numbers.append(s)
    return numbers


class Symbols:
    def __init__(self, numbers):
        self.numbers = numbers

    @classmethod
    def fromsymbols(cls, symbols):
        numbers = symbols2numbers(symbols)
        return cls(np.array(numbers))

    def __getitem__(self, key):
        num = self.numbers[key]
        if np.isscalar(num):
            return chemical_symbols[num]
        return Symbols(num)

    def __setitem__(self, key, value):
        numbers = symbols2numbers(value)
        if len(numbers) == 1:
            numbers = numbers[0]
        self.numbers[key] = numbers

    def __len__(self):
        return len(self.numbers)

    def __str__(self):
        return self.get_chemical_formula('reduce')

    def __repr__(self):
        return 'Symbols(\'{}\')'.format(self)

    def __eq__(self, obj):
        if not hasattr(obj, '__len__'):
            return False

        try:
            symbols = Symbols.fromsymbols(obj)
        except Exception:
            # Typically this would happen if obj cannot be converged to
            # atomic numbers.
            return False
        return self.numbers == symbols.numbers

    def get_chemical_formula(self, mode='hill', empirical=False):
        """Get chemical formula.

        See documentation of ase.atoms.Atoms.get_chemical_formula()."""
        if mode in ('reduce', 'all') and empirical:
            warnings.warn("Empirical chemical formula not available "
                          "for mode '{}'".format(mode))

        if len(self) == 0:
            return ''

        numbers = self.numbers

        if mode == 'reduce':
            n = len(numbers)
            changes = np.concatenate(([0], np.arange(1, n)[numbers[1:] !=
                                                           numbers[:-1]]))
            symbols = [chemical_symbols[e] for e in numbers[changes]]
            counts = np.append(changes[1:], n) - changes

            tokens = []
            for s, c in zip(symbols, counts):
                tokens.append(s)
                if c > 1:
                    tokens.append(str(c))
            formula = ''.join(tokens)
        elif mode == 'hill':
            formula = formula_hill(numbers, empirical=empirical)
        elif mode == 'all':
            formula = ''.join([chemical_symbols[n] for n in numbers])
        elif mode == 'metal':
            formula = formula_metal(numbers, empirical=empirical)
        else:
            raise ValueError("Use mode = 'all', 'reduce', 'hill' or 'metal'.")

        return formula
