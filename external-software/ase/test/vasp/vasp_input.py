"""

Check VASP input handling

"""

from ase.calculators.vasp.create_input import _args_without_comment
from ase.calculators.vasp.create_input import _to_vasp_bool, _from_vasp_bool

from ase.calculators.vasp import Vasp
from ase.build import molecule
from ase.test import must_raise

# Molecules come with no unit cell

atoms = molecule('CH4')
calc = Vasp()

with must_raise(RuntimeError):
    atoms.write('POSCAR')

with must_raise(ValueError):
    atoms.set_calculator(calc)
    atoms.get_total_energy()

# Comment splitting logic

clean_args = _args_without_comment(['a', 'b', '#', 'c'])
assert len(clean_args) == 2
clean_args = _args_without_comment(['a', 'b', '!', 'c', '#', 'd'])
assert len(clean_args) == 2
clean_args = _args_without_comment(['#', 'a', 'b', '!', 'c', '#', 'd'])
assert len(clean_args) == 0

# Boolean handling: input

for s in ('T', '.true.'):
    assert(_from_vasp_bool(s) is True)
for s in ('f', '.False.'):
    assert(_from_vasp_bool(s) is False)
with must_raise(ValueError):
    _from_vasp_bool('yes')
with must_raise(AssertionError):
    _from_vasp_bool(True)

# Boolean handling: output

for x in ('T', '.true.', True):
    assert(_to_vasp_bool(x) == '.TRUE.')
for x in ('f', '.FALSE.', False):
    assert(_to_vasp_bool(x) == '.FALSE.')

with must_raise(ValueError):
    _to_vasp_bool('yes')
with must_raise(AssertionError):
    _from_vasp_bool(1)
