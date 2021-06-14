"""Check if we can deal with spin-broken symmetries."""
from numpy import array
from ase import Atoms
from ase.calculators.nwchem import NWChem


def main():
    """Perform C_{\\inf v} calculation on Cr_2."""
    # PBE from
    # J. Chem. Phys. 112 , 5576 (2000)
    # http://dx.doi.org/10.1063/1.481183
    e_literature = -1.34

    names = ['Cr2_sp_up.mos',
             'Cr2_sp_down.mos']
    fragment_energies = array([.0] * 2)
    cr_atom = Atoms('Cr', positions=[(0, 0, 0)], pbc=False)
    for orientation in range(2):    # create two fragments
        imm = 6 * (-1)**orientation
        cr_atom.set_initial_magnetic_moments([imm])
        calculator = NWChem(task='energy', xc='PBE',
                            geometry='nocenter noautosym noautoz',
                            convergence={'energy': 1e-3,
                                         'density': 1e-2,
                                         'gradient': 5e-2},
                            basis='"DZVP2 (DFT Orbital)"',
                            vectors='input atomic output {0}'
                            .format(names[orientation]),
                            charge=0)
        cr_atom.set_calculator(calculator)
        fragment_energies[orientation] = cr_atom.get_potential_energy()
    cr_dimer = Atoms('Cr2', positions=[(0, 0, 0), (0, 0, 1.93)], pbc=False)
    cr_dimer.set_initial_magnetic_moments([0, 0])
    calculator = NWChem(task='energy',
                        xc='PBE',
                        geometry='nocenter noautosym noautoz',
                        convergence={'energy': 1e-3,
                                     'density': 1e-2,
                                     'gradient': 5e-2},
                        odft=True,
                        basis='"DZVP2 (DFT Orbital)"',
                        vectors='input fragment {0} output Cr2_AF.mos'
                        .format(' '.join(names)),
                        charge=0)
    cr_dimer.set_calculator(calculator)
    e_dimer = cr_dimer.get_potential_energy()
    e_tot = e_dimer - fragment_energies.sum()
    assert abs(e_tot - e_literature) < 0.01


main()
