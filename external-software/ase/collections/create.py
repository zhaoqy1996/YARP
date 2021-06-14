import os

import ase.db
from ase import Atoms
from ase.build import niggli_reduce
from ase.io import read


def dcdft():
    """Create delta-codes-DFT collection.

    Data from: https://github.com/molmod/DeltaCodesDFT
    """
    os.environ['USER'] = 'ase'
    con = ase.db.connect('dcdft.json')
    with open('history/exp.txt') as fd:
        lines = fd.readlines()
    experiment = {}
    for line in lines[2:-1]:
        words = line.split()
        print(words)
        experiment[words[0]] = [float(word) for word in words[1:]]
    with open('WIEN2k.txt') as fd:
        lines = fd.readlines()
    for line in lines[2:73]:
        words = line.split()
        symbol = words.pop(0)
        vol, B, Bp = (float(x) for x in words)
        filename = 'primCIFs/' + symbol + '.cif'
        atoms = read(filename)
        if symbol in ['Li', 'Na']:
            niggli_reduce(atoms)
        M = {'Fe': 2.3,
             'Co': 1.2,
             'Ni': 0.6,
             'Cr': 1.5,
             'O': 1.5,
             'Mn': 2.0}.get(symbol)
        if M is not None:
            magmoms = [M] * len(atoms)
            if symbol in ['Cr', 'O', 'Mn']:
                magmoms[len(atoms) // 2:] = [-M] * (len(atoms) // 2)
            atoms.set_initial_magnetic_moments(magmoms)

        extra = {}
        exp = experiment.get(symbol, [])
        for key, val in zip(['exp_volume', 'exp_B', 'exp_Bp'], exp):
            extra[key] = val
        con.write(atoms, name=symbol,
                  wien2k_B=B, wien2k_Bp=Bp, wien2k_volume=vol,
                  **extra)


def g2():
    from ase.data.g2 import data
    os.environ['USER'] = 'ase'
    con = ase.db.connect('g2.json')
    for name, d in data.items():
        kwargs = {}
        if d['magmoms']:
            kwargs['magmoms'] = d['magmoms']
        atoms = Atoms(d['symbols'], d['positions'], **kwargs)
        con.write(atoms, name=name)
