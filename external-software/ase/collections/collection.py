import os.path as op

from ase.db.row import AtomsRow
from ase.io.jsonio import read_json


class Collection:
    """Collection of atomic configurations and associated data.

    Example of use:

    >>> from ase.collections import s22
    >>> len(s22)
    22
    >>> s22.names[:3]
    ['Ammonia_dimer', 'Water_dimer', 'Formic_acid_dimer']
    >>> dimer = s22['Water_dimer']
    >>> dimer.get_chemical_symbols()
    ['O', 'H', 'H', 'O', 'H', 'H']
    >>> s22.data['Ammonia_dimer']
    {'cc_energy': -0.1375}
    >>> sum(len(atoms) for atoms in s22)
    414
    """
    def __init__(self, name):
        """Create a collection lazily.

        Will read data from json file when needed.

        A collection can be iterated over to get the Atoms objects and indexed
        with names to get individual members.

        Attributes:

        name: str
            Name of collection.
        data: dict
            Data dictionary.
        filename: str
            Location of json file.
        names: list
            Names of configurations in the collection.
        """

        self.name = name
        self._names = []
        self._systems = {}
        self._data = {}
        self.filename = op.join(op.dirname(__file__), name + '.json')

    def __getitem__(self, name):
        self._read()
        return self._systems[name].copy()

    def has(self, name):
        # Not __contains__() because __iter__ yields the systems.
        self._read()
        return name in self._systems

    def __iter__(self):
        for name in self.names:
            yield self[name]

    def __len__(self):
        return len(self.names)

    def __str__(self):
        return '<{0}-collection, {1} systems: {2}, {3}, ...>'.format(
            self.name, len(self), *self.names[:2])

    def __repr__(self):
        return 'Collection({0!r})'.format(self.name)

    @property
    def names(self):
        self._read()
        return list(self._names)

    @property
    def data(self):
        self._read()
        return self._data

    def _read(self):
        if self._names:
            return
        bigdct = read_json(self.filename)
        for id in bigdct['ids']:
            dct = bigdct[id]
            kvp = dct['key_value_pairs']
            name = str(kvp['name'])
            self._names.append(name)
            self._systems[name] = AtomsRow(dct).toatoms()
            del kvp['name']
            self._data[name] = dict((str(k), v) for k, v in kvp.items())
