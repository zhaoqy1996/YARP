from ase import Atoms

assert Atoms('MoS2').get_chemical_formula() == 'MoS2'
assert Atoms('SnO2').get_chemical_formula(mode='metal') == 'SnO2'
