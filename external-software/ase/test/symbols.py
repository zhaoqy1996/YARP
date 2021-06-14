from ase.build import molecule
from ase.symbols import Symbols

atoms = molecule('CH3CH2OH')
print(atoms.symbols)
atoms.symbols[0] = 'X'
atoms.symbols[2:4] = 'Pu'
atoms.numbers[6:8] = 79

assert atoms.numbers[0] == 0
assert (atoms.numbers[2:4] == 94).all()
assert sum(atoms.symbols == 'Au') == 2
assert (atoms.symbols[6:8] == 'Au').all()
assert (atoms.symbols[:3] == 'XCPu').all()

print(atoms)
print(atoms.numbers)

assert atoms.get_chemical_symbols()
string = str(atoms.symbols)
symbols = Symbols.fromsymbols(string)
assert (symbols == atoms.symbols).all()
