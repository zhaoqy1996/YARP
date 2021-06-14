"""Check if ase issues a warning if level shift breaks symmetry."""

from warnings import catch_warnings, simplefilter
from ase import Atoms
from ase.calculators.nwchem import NWChem


def main():
    """The main routine for the prove of the warning."""
    cr_atom = Atoms('Cr', positions=[(0, 0, 0)], pbc=False, magmoms=[5.0])
    calculator = NWChem(task='energy',
        geometry='nocenter noautosym noautoz',
        convergence={'energy': 1e-3,
                'density': 1e-2,
                'gradient': 5e-2},
        basis='Wachters+f',
        charge=1)
    cr_atom.set_calculator(calculator)
    with catch_warnings(record=True) as thrown_warning:
        simplefilter('always', RuntimeWarning)
        cr_atom.get_potential_energy()
        assert len(thrown_warning) == 1
        assert 'levelshift' in str(thrown_warning[-1].message)

if __name__ == '__main__':
    main()
