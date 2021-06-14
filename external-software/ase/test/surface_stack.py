from ase.build.surface import _all_surface_functions
from ase.build import stack
from ase.calculators.calculator import compare_atoms


# The purpose of this test is to test the stack() function and verify
# that the various surface builder functions produce configurations
# consistent with stacking.

d = _all_surface_functions()
exclude = {'mx2'}  # mx2 is not like the others

for name in sorted(d):
    if name in exclude:
        continue

    func = d[name]

    def has(var):
        c = func.__code__
        return var in c.co_varnames[:c.co_argcount]

    for nlayers in range(1, 7):
        atoms = func('Au', size=(2, 2, nlayers), periodic=True, a=4.0)
        big_atoms = func('Au', size=(2, 2, 2 * nlayers), periodic=True, a=4.0)
        stacked_atoms = stack(atoms, atoms)

        changes = compare_atoms(stacked_atoms, big_atoms, tol=1e-11)
        if not changes:
            print('OK', name, nlayers)
            break
    else:
        assert 0, 'Unstackable surface {}'.format(name)
