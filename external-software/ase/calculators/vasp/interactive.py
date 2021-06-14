from __future__ import print_function

from subprocess import Popen, PIPE

from ase.calculators.calculator import Calculator
from ase.io import read

from .create_input import GenerateVaspInput

import time
import os
import sys


class VaspInteractive(GenerateVaspInput, Calculator):
    name = "VaspInteractive"
    implemented_properties = ['energy', 'forces', 'stress']

    mandatory_input = {'potim': 0.0,
                       'ibrion': -1,
                       'interactive': True,
                       }

    default_input = {'nsw': 2000,
                     }

    def __init__(self, txt="interactive.log", print_log=False, process=None,
                 command=None, path="./", **kwargs):
        
        GenerateVaspInput.__init__(self)

        for kw, val in self.mandatory_input.items():
            if kw in kwargs and val != kwargs[kw]:
                raise ValueError('Keyword {} cannot be overridden! '
                                 'It must have have value {}, but {} '
                                 'was provided instead.'.format(kw, val,
                                                                kwargs[kw]))
        kwargs.update(self.mandatory_input)

        for kw, val in self.default_input.items():
            if kw not in kwargs:
                kwargs[kw] = val

        self.set(**kwargs)

        self.process = process
        self.path = path

        if txt is not None:
            self.txt = open(txt, "a")
        else:
            self.txt = None
        self.print_log = print_log

        if command is not None:
            self.command = command
        elif 'VASP_COMMAND' in os.environ:
            self.command = os.environ['VASP_COMMAND']
        elif 'VASP_SCRIPT' in os.environ:
            self.command = os.environ['VASP_SCRIPT']
        else:
            raise RuntimeError('Please set either command in calculator'
                               ' or VASP_COMMAND environment variable')

        if isinstance(self.command, str):
            self.command = self.command.split()

        self.atoms = None

    def _stdin(self, text, ending="\n"):
        if self.txt is not None:
            self.txt.write(text + ending)
        if self.print_log:
            print(text, end=ending)
        self.process.stdin.write(text + ending)
        if sys.version_info[0] >= 3:
            self.process.stdin.flush()

    def _stdout(self, text):
        if self.txt is not None:
            self.txt.write(text)
        if self.print_log:
            print(text, end="")

    def _run_vasp(self, atoms):
        if self.process is None:
            stopcar = os.path.join(self.path, 'STOPCAR')
            if os.path.isfile(stopcar):
                os.remove(stopcar)
            self._stdout("Writing VASP input files\n")
            self.initialize(atoms)
            self.write_input(atoms, directory=self.path)
            self._stdout("Starting VASP for initial step...\n")
            if sys.version_info[0] >= 3:
                self.process = Popen(self.command, stdout=PIPE,
                                     stdin=PIPE, stderr=PIPE, cwd=self.path,
                                     universal_newlines=True)
            else:
                self.process = Popen(self.command, stdout=PIPE,
                                     stdin=PIPE, stderr=PIPE, cwd=self.path)
        else:
            self._stdout("Inputting positions...\n")
            for atom in atoms.get_scaled_positions():
                self._stdin(' '.join(map('{:19.16f}'.format, atom)))

        while self.process.poll() is None:
            text = self.process.stdout.readline()
            self._stdout(text)
            if "POSITIONS: reading from stdin" in text:
                return

        # If we've reached this point, then VASP has exited without asking for
        # new positions, meaning it either exited without error unexpectedly,
        # or it exited with an error. Either way, we need to raise an error.

        raise RuntimeError("VASP exited unexpectedly with exit code {}"
                           "".format(self.subprocess.poll()))

    def close(self):
        if self.process is None:
            return

        self._stdout('Attemping to close VASP cleanly\n')
        with open(os.path.join(self.path, 'STOPCAR'), 'w') as stopcar:
            stopcar.write('LABORT = .TRUE.')

        self._run_vasp(self.atoms)
        self._run_vasp(self.atoms)
        while self.process.poll() is None:
            time.sleep(1)
        self._stdout("VASP has been closed\n")
        self.process = None

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['positions', 'numbers', 'cell']):
        Calculator.calculate(self, atoms, properties, system_changes)

        if not system_changes:
            return

        if 'numbers' in system_changes:
            self.close()

        self._run_vasp(atoms)

        new = read(os.path.join(self.path, 'vasprun.xml'), index=-1)

        self.results = {'free_energy': new.get_potential_energy(force_consistent=True),
                        'energy': new.get_potential_energy(),
                        'forces': new.get_forces()[self.resort],
                        'stress': new.get_stress()}

    def __del__(self):
        self.close()
