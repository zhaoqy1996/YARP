from ase.build import fcc111, add_adsorbate
import warnings

# some random system
slab = fcc111('Al', size=(2,2,3))
add_adsorbate(slab, 'C', 2.5, 'bridge')
add_adsorbate(slab, 'C', 3.5, 'bridge')
add_adsorbate(slab, 'H', 1.5, 'ontop')
add_adsorbate(slab, 'H', 1.5, 'fcc')
add_adsorbate(slab, 'C', 0.5, 'bridge')
add_adsorbate(slab, 'C', 1.5, 'bridge')

assert slab.get_chemical_formula(mode='hill') == u'C4H2Al12'
assert slab.get_chemical_formula(mode='metal') == u'Al12C4H2'
all_str = u'Al'*12 + u'C'*2 + u'H'*2 + u'C'*2
assert slab.get_chemical_formula(mode='all') == all_str
reduce_str = u'Al12C2H2C2'
assert slab.get_chemical_formula(mode='reduce') == reduce_str

assert slab.get_chemical_formula(mode='hill', empirical=True) == u'C2HAl6'
assert slab.get_chemical_formula(mode='metal', empirical=True) == u'Al6C2H'


# check for warning if empirical formula is not available
for mode in ('all', 'reduce'):
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        slab.get_chemical_formula(mode=mode, empirical=True)
        # Verify some things
        assert len(w) == 1
        assert issubclass(w[-1].category, Warning)
