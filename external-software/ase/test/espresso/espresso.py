"""Check that QE calculation can run."""

from ase.build import bulk
from ase.calculators.espresso import Espresso

# Default pseudos can go in ~/espresso/pseudo
# Get these from SSSP http://materialscloud.org/sssp/
PSEUDO = {'Si': 'Si.pbe-n-rrkjus_psl.1.0.0.UPF'}

# Don't forget to
# export ASE_ESPRESSO_COMMAND="mpirun -n 4 $HOME/Compile/q-e/bin/pw.x -in PREFIX.pwi > PREFIX.pwo"
# export ESPRESSO_PSEUDO="/path/to/pseudos"

def main():
    silicon = bulk('Si')
    calc = Espresso(pseudopotentials=PSEUDO)
    silicon.set_calculator(calc)
    silicon.get_potential_energy()

main()
