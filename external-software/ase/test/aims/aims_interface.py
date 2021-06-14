import tempfile
import os

from ase.calculators.aims import Aims
from ase import Atoms

# test the new command handling + legacy behavior
aims_command = 'aims.x'
aims_command_alternative = 'mpirun -np 4 fhiaims.x'
outfilename = 'alternative_aims.out'
outfilename_default = 'aims.out'
command = '{0:s} > {1:s}'.format(aims_command, outfilename)
command_alternative = '{0:s} > {1:s}'.format(aims_command_alternative, outfilename)
command_default = '{0:s} > {1:s}'.format(aims_command, outfilename_default)
legacy_command = 'aims.version.serial.x > aims.out'
legacy_aims_command = legacy_command.split('>')[0].strip()
legacy_outfilename = legacy_command.split('>')[-1].strip()

# legacy behavior of empty init
calc = Aims()
assert calc.command == legacy_command
assert calc.outfilename == legacy_outfilename
assert calc.aims_command == legacy_aims_command

# behavior of empty init with env variable
os.environ['ASE_AIMS_COMMAND'] = aims_command_alternative
calc = Aims()
assert calc.command == '{0} > {1}'.format(aims_command_alternative, outfilename_default)
assert calc.outfilename == outfilename_default
assert calc.aims_command == aims_command_alternative

# legacy behavior of "proper" command
calc = Aims(run_command=command)
assert calc.command == command
assert calc.outfilename == outfilename
assert calc.aims_command == aims_command

# legacy behavior of an "improper" command
calc = Aims(run_command=aims_command)
assert calc.command == command_default
assert calc.aims_command == aims_command
assert calc.outfilename == outfilename_default

# fixed "command"  behavior
calc = Aims(command=command)
assert calc.command == command
assert calc.outfilename == outfilename
assert calc.aims_command == aims_command

# novel way to use aims_command, no specific outfile
calc = Aims(aims_command=aims_command)
assert calc.command == command_default
assert calc.outfilename == outfilename_default
assert calc.aims_command == aims_command

calc = Aims(aims_command=aims_command,
            outfilename=outfilename)
assert calc.command == command
assert calc.outfilename == outfilename
assert calc.aims_command == aims_command

# # testing the setters
calc.command = command_default
assert calc.outfilename == outfilename_default
assert calc.aims_command == aims_command
assert calc.command == command_default

#calc.set_aims_command(aims_command_alternative)
calc.aims_command = aims_command_alternative
assert calc.aims_command == aims_command_alternative
assert calc.outfilename == outfilename_default
assert calc.command == '{} > {}'.format(aims_command_alternative, outfilename_default)

calc.outfilename = outfilename
assert calc.command == '{} > {}'.format(aims_command_alternative, outfilename)
assert calc.aims_command == aims_command_alternative
assert calc.outfilename == outfilename


# test writing files
tmp_dir = tempfile.mkdtemp()
water = Atoms('HOH', [(1, 0, 0), (0, 0, 0), (0, 1, 0)])
calc = Aims(xc='PBE',
            output=['dipole'],
            sc_accuracy_etot=1e-6,
            sc_accuracy_eev=1e-3,
            sc_accuracy_rho=1e-6,
            species_dir="/data/rittmeyer/FHIaims/species_defaults/light/",
            sc_accuracy_forces=1e-4,
            label=tmp_dir,
            )
try:
    calc.prepare_input_files()
    raise AssertionError
except ValueError:
    pass

calc.atoms = water
calc.prepare_input_files()
for f in ['control.in', 'geometry.in']:
    assert os.path.isfile(os.path.join(tmp_dir,f))
