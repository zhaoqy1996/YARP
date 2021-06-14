"""This module defines the Atom object."""

import numpy as np

from ase.data import atomic_numbers, chemical_symbols, atomic_masses
from ase.utils import basestring

# Singular, plural, default value:
names = {'position': ('positions', np.zeros(3)),
         'number': ('numbers', 0),
         'tag': ('tags', 0),
         'momentum': ('momenta', np.zeros(3)),
         'mass': ('masses', None),
         'magmom': ('initial_magmoms', 0.0),
         'charge': ('initial_charges', 0.0)}


def atomproperty(name, doc):
    """Helper function to easily create Atom attribute property."""

    def getter(self):
        return self.get(name)

    def setter(self, value):
        self.set(name, value)

    def deleter(self):
        self.delete(name)

    return property(getter, setter, deleter, doc)


def abcproperty(index):
    """Helper function to easily create Atom ABC-property."""

    def getter(self):
        spos = self.atoms.get_scaled_positions()
        return spos[self.index][index]

    def setter(self, value):
        spos = self.atoms.get_scaled_positions()
        spos[self.index][index] = value
        self.atoms.set_scaled_positions(spos)

    return property(getter, setter, doc='ABC'[index] + '-coordinate')


def xyzproperty(index):
    """Helper function to easily create Atom XYZ-property."""

    def getter(self):
        return self.position[index]

    def setter(self, value):
        self.position[index] = value

    return property(getter, setter, doc='XYZ'[index] + '-coordinate')


class Atom(object):
    """Class for representing a single atom.

    Parameters:

    symbol: str or int
        Can be a chemical symbol (str) or an atomic number (int).
    position: sequence of 3 floats
        Atomic position.
    tag: int
        Special purpose tag.
    momentum: sequence of 3 floats
        Momentum for atom.
    mass: float
        Atomic mass in atomic units.
    magmom: float or 3 floats
        Magnetic moment.
    charge: float
        Atomic charge.
    """
    __slots__ = ['data', 'atoms', 'index']

    def __init__(self, symbol='X', position=(0, 0, 0),
                 tag=None, momentum=None, mass=None,
                 magmom=None, charge=None,
                 atoms=None, index=None):

        self.data = d = {}

        if atoms is None:
            # This atom is not part of any Atoms object:
            if isinstance(symbol, basestring):
                d['number'] = atomic_numbers[symbol]
            else:
                d['number'] = symbol
            d['position'] = np.array(position, float)
            d['tag'] = tag
            if momentum is not None:
                momentum = np.array(momentum, float)
            d['momentum'] = momentum
            d['mass'] = mass
            if magmom is not None:
                magmom = np.array(magmom, float)
            d['magmom'] = magmom
            d['charge'] = charge

        self.index = index
        self.atoms = atoms

    def __repr__(self):
        s = "Atom('%s', %s" % (self.symbol, list(self.position))
        for name in ['tag', 'momentum', 'mass', 'magmom', 'charge']:
            value = self.get_raw(name)
            if value is not None:
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                s += ', %s=%s' % (name, value)
        if self.atoms is None:
            s += ')'
        else:
            s += ', index=%d)' % self.index
        return s

    def cut_reference_to_atoms(self):
        """Cut reference to atoms object."""
        for name in names:
            self.data[name] = self.get_raw(name)
        self.index = None
        self.atoms = None

    def get_raw(self, name):
        """Get name attribute, return None if not explicitely set."""
        if name == 'symbol':
            return chemical_symbols[self.get_raw('number')]

        if self.atoms is None:
            return self.data[name]

        plural = names[name][0]
        if plural in self.atoms.arrays:
            return self.atoms.arrays[plural][self.index]
        else:
            return None

    def get(self, name):
        """Get name attribute, return default if not explicitely set."""
        value = self.get_raw(name)
        if value is None:
            if name == 'mass':
                value = atomic_masses[self.number]
            else:
                value = names[name][1]
        return value

    def set(self, name, value):
        """Set name attribute to value."""
        if name == 'symbol':
            name = 'number'
            value = atomic_numbers[value]

        if self.atoms is None:
            assert name in names
            self.data[name] = value
        else:
            plural, default = names[name]
            if plural in self.atoms.arrays:
                array = self.atoms.arrays[plural]
                if name == 'magmom' and array.ndim == 2:
                    assert len(value) == 3
                array[self.index] = value
            else:
                if name == 'magmom' and np.asarray(value).ndim == 1:
                    array = np.zeros((len(self.atoms), 3))
                elif name == 'mass':
                    array = self.atoms.get_masses()
                else:
                    default = np.asarray(default)
                    array = np.zeros((len(self.atoms),) + default.shape,
                                     default.dtype)
                array[self.index] = value
                self.atoms.new_array(plural, array)

    def delete(self, name):
        """Delete name attribute."""
        assert self.atoms is None
        assert name not in ['number', 'symbol', 'position']
        self.data[name] = None

    symbol = atomproperty('symbol', 'Chemical symbol')
    number = atomproperty('number', 'Atomic number')
    position = atomproperty('position', 'XYZ-coordinates')
    tag = atomproperty('tag', 'Integer tag')
    momentum = atomproperty('momentum', 'XYZ-momentum')
    mass = atomproperty('mass', 'Atomic mass')
    magmom = atomproperty('magmom', 'Initial magnetic moment')
    charge = atomproperty('charge', 'Initial atomic charge')
    x = xyzproperty(0)
    y = xyzproperty(1)
    z = xyzproperty(2)

    scaled_position = atomproperty('scaled_position', 'ABC-coordinates')
    a = abcproperty(0)
    b = abcproperty(1)
    c = abcproperty(2)
