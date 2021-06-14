"""
Run some tests to ensure that the xc setting in the VASP calculator
works.

"""

from ase.test.vasp import installed2 as installed
from ase.calculators.vasp import Vasp2 as Vasp
assert installed()

def dict_is_subset(d1, d2):
    """True if all the key-value pairs in dict 1 are in dict 2"""
    for key, value in d1.items():
        if key not in d2:
            return False
        elif d2[key] != value:
            return False
    else:
        return True

calc_vdw = Vasp(xc='optb86b-vdw')

assert dict_is_subset({'param1': 0.1234, 'param2': 1.0},
                      calc_vdw.float_params)

calc_hse = Vasp(xc='hse06', hfscreen=0.1, gga='RE',
                encut=400, sigma=0.5)

assert dict_is_subset({'hfscreen': 0.1, 'encut': 400, 'sigma': 0.5},
                      calc_hse.float_params)
assert dict_is_subset({'gga': 'RE'}, calc_hse.string_params)
