import Scientific
assert [int(x) for x in Scientific.__version__.split('.')] >= [2, 8]
from ase.calculators.jacapo.jacapo import Jacapo, read
__all__ = ['Jacapo', 'read']
