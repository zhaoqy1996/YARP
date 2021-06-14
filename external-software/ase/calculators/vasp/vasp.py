from __future__ import print_function
# Copyright (C) 2008 CSC - Scientific Computing Ltd.
"""This module defines an ASE interface to VASP.

Developed on the basis of modules by Jussi Enkovaara and John
Kitchin.  The path of the directory containing the pseudopotential
directories (potpaw,potpaw_GGA, potpaw_PBE, ...) should be set
by the environmental flag $VASP_PP_PATH.

The user should also set the environmental flag $VASP_SCRIPT pointing
to a python script looking something like::

   import os
   exitcode = os.system('vasp')

Alternatively, user can set the environmental flag $VASP_COMMAND pointing
to the command use the launch vasp e.g. 'vasp' or 'mpirun -n 16 vasp'

http://cms.mpi.univie.ac.at/vasp/
"""

import os
import sys
import re
from ase.calculators.general import Calculator

import numpy as np

import ase.io
from ase.utils import devnull, basestring

from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.calculator import PropertyNotImplementedError
from .create_input import GenerateVaspInput


class Vasp(GenerateVaspInput, Calculator):
    name = 'Vasp'

    implemented_properties = ['energy', 'forces', 'dipole', 'fermi', 'stress',
                              'magmom', 'magmoms']

    def __init__(self, restart=None,
                 output_template='vasp',
                 track_output=False,
                 **kwargs):
        GenerateVaspInput.__init__(self)

        self.restart = restart
        self.track_output = track_output
        self.output_template = output_template
        if restart:
            self.restart_load()
            return

        self.nbands = self.int_params['nbands']
        self.atoms = None
        self.positions = None
        self.run_counts = 0

        # If no XC combination, GGA functional or POTCAR type is specified,
        # default to PW91. This is mostly chosen for backwards compatiblity.
        if kwargs.get('xc', None):
            pass
        elif not (kwargs.get('gga', None) or kwargs.get('pp', None)):
            self.input_params.update({'xc': 'PW91'})
        # A null value of xc is permitted; custom recipes can be
        # used by explicitly setting the pseudopotential set and
        # INCAR keys
        else:
            self.input_params.update({'xc': None})

        self.set(**kwargs)

    def update(self, atoms):
        if self.calculation_required(atoms, ['energy']):
            if (((self.atoms is None) or
                 (self.atoms.positions.shape != atoms.positions.shape)
                 )):
                # Completely new calculation just reusing the same
                # calculator, so delete any old VASP files found.
                self.clean()
            self.calculate(atoms)

    def calculate(self, atoms):
        """Generate necessary files in the working directory and run VASP.

        The method first write VASP input files, then calls the method
        which executes VASP. When the VASP run is finished energy, forces,
        etc. are read from the VASP output.
        """

        # Check if there is only a zero unit cell
        if not atoms.cell.any():
            raise ValueError("The lattice vectors are zero! "
                    "This is the default value - please specify a "
                    "unit cell.")

        # Initialize calculations
        self.initialize(atoms)

        # Write input
        self.write_input(atoms)

        # Execute VASP
        self.run()
        # Read output
        atoms_sorted = ase.io.read('CONTCAR', format='vasp')

        if (self.int_params['ibrion'] is not None and
                self.int_params['nsw'] is not None):
            if self.int_params['ibrion'] > -1 and self.int_params['nsw'] > 0:
                # Update atomic positions and unit cell with the ones read
                # from CONTCAR.
                atoms.positions = atoms_sorted[self.resort].positions
                atoms.cell = atoms_sorted.cell

        self.converged = self.read_convergence()
        self.set_results(atoms)

    def set_results(self, atoms):
        self.read(atoms)
        if self.spinpol:
            self.magnetic_moment = self.read_magnetic_moment()
            if (self.int_params['lorbit'] is not None and
                (self.int_params['lorbit'] >= 10 or
                 self.list_float_params['rwigs'])):
                self.magnetic_moments = self.read_magnetic_moments(atoms)
            else:
                self.magnetic_moments = None
        self.old_float_params = self.float_params.copy()
        self.old_exp_params = self.exp_params.copy()
        self.old_string_params = self.string_params.copy()
        self.old_int_params = self.int_params.copy()
        self.old_input_params = self.input_params.copy()
        self.old_bool_params = self.bool_params.copy()
        self.old_list_bool_params = self.list_bool_params.copy()
        self.old_list_int_params = self.list_int_params.copy()
        self.old_list_float_params = self.list_float_params.copy()
        self.old_dict_params = self.dict_params.copy()
        self.atoms = atoms.copy()
        self.name = 'vasp'
        self.version = self.read_version()
        self.niter = self.read_number_of_iterations()
        self.sigma = self.read_electronic_temperature()
        self.nelect = self.read_number_of_electrons()

    def run(self):
        """Method which explicitely runs VASP."""

        if self.track_output:
            self.out = self.output_template + str(self.run_counts) + '.out'
            self.run_counts += 1
        else:
            self.out = self.output_template + '.out'
        stderr = sys.stderr
        p = self.input_params
        if p['txt'] is None:
            sys.stderr = devnull
        elif p['txt'] == '-':
            pass
        elif isinstance(p['txt'], basestring):
            sys.stderr = open(p['txt'], 'w')
        if 'VASP_COMMAND' in os.environ:
            vasp = os.environ['VASP_COMMAND']
            exitcode = os.system('%s > %s' % (vasp, self.out))
        elif 'VASP_SCRIPT' in os.environ:
            vasp = os.environ['VASP_SCRIPT']
            locals = {}
            exec(compile(open(vasp).read(), vasp, 'exec'), {}, locals)
            exitcode = locals['exitcode']
        else:
            raise RuntimeError('Please set either VASP_COMMAND'
                               ' or VASP_SCRIPT environment variable')
        sys.stderr = stderr
        if exitcode != 0:
            raise RuntimeError('Vasp exited with exit code: %d.  ' % exitcode)

    def restart_load(self):
        """Method which is called upon restart."""
        # Try to read sorting file
        if os.path.isfile('ase-sort.dat'):
            self.sort = []
            self.resort = []
            file = open('ase-sort.dat', 'r')
            lines = file.readlines()
            file.close()
            for line in lines:
                data = line.split()
                self.sort.append(int(data[0]))
                self.resort.append(int(data[1]))
            atoms = ase.io.read('CONTCAR', format='vasp')[self.resort]
        else:
            atoms = ase.io.read('CONTCAR', format='vasp')
            self.sort = list(range(len(atoms)))
            self.resort = list(range(len(atoms)))
        self.atoms = atoms.copy()
        self.read_incar()
        self.read_outcar()
        self.set_results(atoms)
        if not self.float_params['kspacing']:
            self.read_kpoints()
        self.read_potcar()

        self.old_input_params = self.input_params.copy()
        self.converged = self.read_convergence()

    def set_atoms(self, atoms):
        if (atoms != self.atoms):
            self.converged = None
        self.atoms = atoms.copy()

    def get_atoms(self):
        atoms = self.atoms.copy()
        atoms.set_calculator(self)
        return atoms

    def get_version(self):
        self.update(self.atoms)
        return self.version

    def read_version(self):
        version = None
        for line in open('OUTCAR'):
            if line.find(' vasp.') != -1:  # find the first occurrence
                version = line[len(' vasp.'):].split()[0]
                break
        return version

    def get_potential_energy(self, atoms, force_consistent=False):
        self.update(atoms)
        if force_consistent:
            return self.energy_free
        else:
            return self.energy_zero

    def get_number_of_iterations(self):
        self.update(self.atoms)
        return self.niter

    def read_number_of_iterations(self):
        niter = None
        for line in open('OUTCAR'):
            # find the last iteration number
            if line.find('- Iteration') != -1:
                niter = int(line.split(')')[0].split('(')[-1].strip())
        return niter

    def get_electronic_temperature(self):
        self.update(self.atoms)
        return self.sigma

    def read_electronic_temperature(self):
        sigma = None
        for line in open('OUTCAR'):
            if line.find('Fermi-smearing in eV        SIGMA') != -1:
                sigma = float(line.split('=')[1].strip())
        return sigma

    def get_default_number_of_electrons(self, filename='POTCAR'):
        """Get list of tuples (atomic symbol, number of valence electrons)
        for each atomtype from a POTCAR file.  """
        return self.read_default_number_of_electrons(filename)

    def read_default_number_of_electrons(self, filename='POTCAR'):
        nelect = []
        lines = open(filename).readlines()
        for n, line in enumerate(lines):
            if line.find('TITEL') != -1:
                symbol = line.split('=')[1].split()[1].split('_')[0].strip()
                valence = float(lines[n + 4].split(';')[1]
                                .split('=')[1].split()[0].strip())
                nelect.append((symbol, valence))
        return nelect

    def get_number_of_electrons(self):
        self.update(self.atoms)
        return self.nelect

    def read_number_of_electrons(self):
        nelect = None
        for line in open('OUTCAR'):
            if line.find('total number of electrons') != -1:
                nelect = float(line.split('=')[1].split()[0].strip())
        return nelect

    def get_forces(self, atoms):
        self.update(atoms)
        return self.forces

    def get_stress(self, atoms):
        self.update(atoms)
        if self.stress is None:
            raise PropertyNotImplementedError
        return self.stress

    def read_stress(self):
        stress = None
        for line in open('OUTCAR'):
            if line.find(' in kB  ') != -1:
                stress = -np.array([float(a) for a in line.split()[2:]])
                stress = stress[[0, 1, 2, 4, 5, 3]] * 1e-1 * ase.units.GPa
        return stress

    def read_ldau(self):
        ldau_luj = None
        ldauprint = None
        ldau = None
        ldautype = None
        atomtypes = []
        # read ldau parameters from outcar
        for line in open('OUTCAR'):
            if line.find('TITEL') != -1:    # What atoms are present
                atomtypes.append(line.split()[3].split('_')[0].split('.')[0])
            if line.find('LDAUTYPE') != -1:  # Is this a DFT+U calculation
                ldautype = int(line.split('=')[-1])
                ldau = True
                ldau_luj = {}
            if line.find('LDAUL') != -1:
                L = line.split('=')[-1].split()
            if line.find('LDAUU') != -1:
                U = line.split('=')[-1].split()
            if line.find('LDAUJ') != -1:
                J = line.split('=')[-1].split()
        # create dictionary
        if ldau:
            for i, symbol in enumerate(atomtypes):
                ldau_luj[symbol] = {'L': int(L[i]),
                                    'U': float(U[i]),
                                    'J': float(J[i])}
            self.dict_params['ldau_luj'] = ldau_luj
        return ldau, ldauprint, ldautype, ldau_luj

    def calculation_required(self, atoms, quantities):
        if (((self.positions is None) or
             (self.atoms != atoms) or
             (self.float_params != self.old_float_params) or
             (self.exp_params != self.old_exp_params) or
             (self.string_params != self.old_string_params) or
             (self.int_params != self.old_int_params) or
             (self.bool_params != self.old_bool_params) or
             (self.list_bool_params != self.old_list_bool_params) or
             (self.list_int_params != self.old_list_int_params) or
             (self.list_float_params != self.old_list_float_params) or
             (self.input_params != self.old_input_params) or
             (self.dict_params != self.old_dict_params) or
             not self.converged)):
            return True
        if 'magmom' in quantities:
            return not hasattr(self, 'magnetic_moment')
        return False

    def get_number_of_bands(self):
        return self.nbands

    def get_k_point_weights(self):
        self.update(self.atoms)
        return self.read_k_point_weights()

    def get_number_of_spins(self):
        if self.spinpol is None:
            return 1
        else:
            return 1 + int(self.spinpol)

    def get_eigenvalues(self, kpt=0, spin=0):
        self.update(self.atoms)
        return self.read_eigenvalues(kpt, spin)

    def get_occupation_numbers(self, kpt=0, spin=0):
        self.update(self.atoms)
        return self.read_occupation_numbers(kpt, spin)

    def get_fermi_level(self):
        return self.fermi

    def get_number_of_grid_points(self):
        raise NotImplementedError

    def get_pseudo_density(self):
        raise NotImplementedError

    def get_pseudo_wavefunction(self, n=0, k=0, s=0, pad=True):
        raise NotImplementedError

    def get_bz_k_points(self):
        raise NotImplementedError

    def get_ibz_kpoints(self):
        self.update(self.atoms)
        return self.read_ibz_kpoints()

    def get_ibz_k_points(self):
        return self.get_ibz_kpoints()

    def get_spin_polarized(self):
        if not hasattr(self, 'spinpol'):
            self.spinpol = self.atoms.get_initial_magnetic_moments().any()
        return self.spinpol

    def get_magnetic_moment(self, atoms):
        self.update(atoms)
        return self.magnetic_moment

    def get_magnetic_moments(self, atoms):
        if ((self.int_params['lorbit'] is not None and
             self.int_params['lorbit'] >= 10) or
                self.list_float_params['rwigs']):
            self.update(atoms)
            return self.magnetic_moments
        else:
            return None

    def get_dipole_moment(self, atoms):
        """Returns total dipole moment of the system."""
        self.update(atoms)
        return self.dipole

    def get_xc_functional(self):
        """Returns the XC functional or the pseudopotential type

        If a XC recipe is set explicitly with 'xc', this is returned.
        Otherwise, the XC functional associated with the
        pseudopotentials (LDA, PW91 or PBE) is returned.
        The string is always cast to uppercase for consistency
        in checks."""
        if self.input_params.get('xc', None):
            return self.input_params['xc'].upper()
        elif self.input_params.get('pp', None):
            return self.input_params['pp'].upper()
        else:
            raise ValueError('No xc or pp found.')

    # Methods for reading information from OUTCAR files:
    def read_energy(self, all=None):
        [energy_free, energy_zero] = [0, 0]
        if all:
            energy_free = []
            energy_zero = []
        for line in open('OUTCAR', 'r'):
            # Free energy
            if line.lower().startswith('  free  energy   toten'):
                if all:
                    energy_free.append(float(line.split()[-2]))
                else:
                    energy_free = float(line.split()[-2])
            # Extrapolated zero point energy
            if line.startswith('  energy  without entropy'):
                if all:
                    energy_zero.append(float(line.split()[-1]))
                else:
                    energy_zero = float(line.split()[-1])
        return [energy_free, energy_zero]

    def read_forces(self, atoms, all=False):
        """Method that reads forces from OUTCAR file.

        If 'all' is switched on, the forces for all ionic steps
        in the OUTCAR file be returned, in other case only the
        forces for the last ionic configuration is returned."""

        file = open('OUTCAR', 'r')
        lines = file.readlines()
        file.close()
        n = 0
        if all:
            all_forces = []
        for line in lines:
            if line.rfind('TOTAL-FORCE') > -1:
                forces = []
                for i in range(len(atoms)):
                    forces.append(np.array([float(f) for f in
                                            lines[n + 2 + i].split()[3:6]]))
                if all:
                    all_forces.append(np.array(forces)[self.resort])
            n += 1
        if all:
            return np.array(all_forces)
        else:
            return np.array(forces)[self.resort]

    def read_fermi(self):
        """Method that reads Fermi energy from OUTCAR file"""
        E_f = None
        for line in open('OUTCAR', 'r'):
            if line.rfind('E-fermi') > -1:
                E_f = float(line.split()[2])
        return E_f

    def read_dipole(self):
        dipolemoment = np.zeros([1, 3])
        for line in open('OUTCAR', 'r'):
            if line.rfind('dipolmoment') > -1:
                dipolemoment = np.array([float(f) for f in line.split()[1:4]])
        return dipolemoment

    def read_magnetic_moments(self, atoms):
        magnetic_moments = np.zeros(len(atoms))
        n = 0
        lines = open('OUTCAR', 'r').readlines()
        for line in lines:
            if line.rfind('magnetization (x)') > -1:
                for m in range(len(atoms)):
                    magnetic_moments[m] = float(lines[n + m + 4].split()[4])
            n += 1
        return np.array(magnetic_moments)[self.resort]

    def read_magnetic_moment(self):
        n = 0
        for line in open('OUTCAR', 'r'):
            if line.rfind('number of electron  ') > -1:
                magnetic_moment = float(line.split()[-1])
            n += 1
        return magnetic_moment

    def read_nbands(self):
        for line in open('OUTCAR', 'r'):
            line = self.strip_warnings(line)
            if line.rfind('NBANDS') > -1:
                return int(line.split()[-1])

    def strip_warnings(self, line):
        """Returns empty string instead of line from warnings in OUTCAR."""
        if line[0] == "|":
            return ""
        else:
            return line

    def read_convergence(self):
        """Method that checks whether a calculation has converged."""
        converged = None
        # First check electronic convergence
        for line in open('OUTCAR', 'r'):
            if 0:  # vasp always prints that!
                if line.rfind('aborting loop') > -1:  # scf failed
                    raise RuntimeError(line.strip())
                    break
            if line.rfind('EDIFF  ') > -1:
                ediff = float(line.split()[2])
            if line.rfind('total energy-change') > -1:
                # I saw this in an atomic oxygen calculation. it
                # breaks this code, so I am checking for it here.
                if 'MIXING' in line:
                    continue
                split = line.split(':')
                a = float(split[1].split('(')[0])
                b = split[1].split('(')[1][0:-2]
                # sometimes this line looks like (second number wrong format!):
                # energy-change (2. order) :-0.2141803E-08  ( 0.2737684-111)
                # we are checking still the first number so
                # let's "fix" the format for the second one
                if 'e' not in b.lower():
                    # replace last occurrence of - (assumed exponent) with -e
                    bsplit = b.split('-')
                    bsplit[-1] = 'e' + bsplit[-1]
                    b = '-'.join(bsplit).replace('-e', 'e-')
                b = float(b)
                if [abs(a), abs(b)] < [ediff, ediff]:
                    converged = True
                else:
                    converged = False
                    continue
        # Then if ibrion in [1,2,3] check whether ionic relaxation
        # condition been fulfilled
        if ((self.int_params['ibrion'] in [1, 2, 3] and
             self.int_params['nsw'] not in [0])):
            if not self.read_relaxed():
                converged = False
            else:
                converged = True
        return converged

    def read_ibz_kpoints(self):
        lines = open('OUTCAR', 'r').readlines()
        ibz_kpts = []
        n = 0
        i = 0
        for line in lines:
            if line.rfind('Following cartesian coordinates') > -1:
                m = n + 2
                while i == 0:
                    ibz_kpts.append([float(lines[m].split()[p])
                                     for p in range(3)])
                    m += 1
                    if lines[m] == ' \n':
                        i = 1
            if i == 1:
                continue
            n += 1
        ibz_kpts = np.array(ibz_kpts)
        return np.array(ibz_kpts)

    def read_k_point_weights(self):
        file = open('IBZKPT')
        lines = file.readlines()
        file.close()
        if 'Tetrahedra\n' in lines:
            N = lines.index('Tetrahedra\n')
        else:
            N = len(lines)
        kpt_weights = []
        for n in range(3, N):
            kpt_weights.append(float(lines[n].split()[3]))
        kpt_weights = np.array(kpt_weights)
        kpt_weights /= np.sum(kpt_weights)
        return kpt_weights

    def read_eigenvalues(self, kpt=0, spin=0):
        file = open('EIGENVAL', 'r')
        lines = file.readlines()
        file.close()
        eigs = []
        for n in range(8 + kpt * (self.nbands + 2),
                       8 + kpt * (self.nbands + 2) + self.nbands):
            eigs.append(float(lines[n].split()[spin + 1]))
        return np.array(eigs)

    def read_occupation_numbers(self, kpt=0, spin=0):
        lines = open('OUTCAR').readlines()
        nspins = self.get_number_of_spins()
        start = 0
        if nspins == 1:
            for n, line in enumerate(lines):  # find it in the last iteration
                m = re.search(' k-point *' + str(kpt + 1) + ' *:', line)
                if m is not None:
                    start = n
        else:
            for n, line in enumerate(lines):
                # find it in the last iteration
                if line.find(' spin component ' + str(spin + 1)) != -1:
                    start = n
            for n2, line2 in enumerate(lines[start:]):
                m = re.search(' k-point *' + str(kpt + 1) + ' *:', line2)
                if m is not None:
                    start = start + n2
                    break
        for n2, line2 in enumerate(lines[start + 2:]):
            if not line2.strip():
                break
        occ = []
        for line in lines[start + 2:start + 2 + n2]:
            occ.append(float(line.split()[2]))
        return np.array(occ)

    def read_relaxed(self):
        for line in open('OUTCAR', 'r'):
            if line.rfind('reached required accuracy') > -1:
                return True
        return False

    def read_outcar(self):
        # Spin polarized calculation?
        file = open('OUTCAR', 'r')
        lines = file.readlines()
        file.close()
        for line in lines:
            if line.rfind('ISPIN') > -1:
                if int(line.split()[2]) == 2:
                    self.spinpol = True
                else:
                    self.spinpol = None
        self.energy_free, self.energy_zero = self.read_energy()
        self.forces = self.read_forces(self.atoms)
        self.dipole = self.read_dipole()
        self.fermi = self.read_fermi()
        self.stress = self.read_stress()
        self.nbands = self.read_nbands()
        self.read_ldau()
        p = self.int_params
        q = self.list_float_params
        if self.spinpol:
            self.magnetic_moment = self.read_magnetic_moment()
            if ((p['lorbit'] is not None and p['lorbit'] >= 10)
                    or (p['lorbit'] is None and q['rwigs'])):
                self.magnetic_moments = self.read_magnetic_moments(self.atoms)
            else:
                self.magnetic_moments = None
        self.set(nbands=self.nbands)

    def read_vib_freq(self):
        """Read vibrational frequencies.

        Returns list of real and list of imaginary frequencies."""
        freq = []
        i_freq = []
        with open('OUTCAR', 'r') as fd:
            lines = fd.readlines()
        for line in lines:
            data = line.split()
            if 'THz' in data:
                if 'f/i=' not in data:
                    freq.append(float(data[-2]))
                else:
                    i_freq.append(float(data[-2]))
        return freq, i_freq

    def get_nonselfconsistent_energies(self, bee_type):
        """ Method that reads and returns BEE energy contributions
            written in OUTCAR file.
        """
        assert bee_type == 'beefvdw'
        cmd = 'grep -32 "BEEF xc energy contributions" OUTCAR | tail -32'
        p = os.popen(cmd,
                     'r')
        s = p.readlines()
        p.close()
        xc = np.array([])
        for i, l in enumerate(s):
            l_ = float(l.split(":")[-1])
            xc = np.append(xc, l_)
        assert len(xc) == 32
        return xc

    def check_state(self, atoms, tol=1e-15):
        """Check for system changes since last calculation."""
        from ase.calculators.calculator import all_changes, equal
        if self.atoms is None:
            system_changes = all_changes[:]
        else:
            system_changes = []
            if not equal(self.atoms.positions, atoms.positions, tol):
                system_changes.append('positions')
            if not equal(self.atoms.numbers, atoms.numbers):
                system_changes.append('numbers')
            if not equal(self.atoms.cell, atoms.cell, tol):
                system_changes.append('cell')
            if not equal(self.atoms.pbc, atoms.pbc):
                system_changes.append('pbc')
            if not equal(self.atoms.get_initial_magnetic_moments(),
                         atoms.get_initial_magnetic_moments(), tol):
                system_changes.append('initial_magmoms')
            if not equal(self.atoms.get_initial_charges(),
                         atoms.get_initial_charges(), tol):
                system_changes.append('initial_charges')

        return system_changes

    def get_property(self, name, atoms=None, allow_calculation=True):
        """Returns the value of a property"""

        if name not in Vasp.implemented_properties:
            raise PropertyNotImplementedError

        if atoms is None:
            atoms = self.atoms

        saved_property = {
            'energy': 'energy_zero',
            'forces': 'forces',
            'dipole': 'dipole',
            'fermi': 'fermi',
            'stress': 'stress',
            'magmom': 'magnetic_moment',
            'magmoms': 'magnetic_moments'
        }
        property_getter = {
            'energy':  {'function': 'get_potential_energy', 'args': [atoms]}, 
            'forces':  {'function': 'get_forces',           'args': [atoms]},
            'dipole':  {'function': 'get_dipole_moment',    'args': [atoms]},
            'fermi':   {'function': 'get_fermi_level',      'args': []},
            'stress':  {'function': 'get_stress',           'args': [atoms]},
            'magmom':  {'function': 'get_magnetic_moment',  'args': [atoms]},
            'magmoms': {'function': 'get_magnetic_moments', 'args': [atoms]}
        }

        if allow_calculation:
            function = property_getter[name]['function']
            args = property_getter[name]['args']
            result = getattr(self, function)(*args)
        else:
            if hasattr(self, saved_property[name]):
                result = getattr(self, saved_property[name])
            else:
                result = None

        if isinstance(result, np.ndarray):
            result = result.copy()
        return result


class VaspChargeDensity(object):
    """Class for representing VASP charge density"""

    def __init__(self, filename='CHG'):
        # Instance variables
        self.atoms = []   # List of Atoms objects
        self.chg = []     # Charge density
        self.chgdiff = []  # Charge density difference, if spin polarized
        self.aug = ''     # Augmentation charges, not parsed just a big string
        self.augdiff = ''  # Augmentation charge differece, is spin polarized

        # Note that the augmentation charge is not a list, since they
        # are needed only for CHGCAR files which store only a single
        # image.
        if filename is not None:
            self.read(filename)

    def is_spin_polarized(self):
        if len(self.chgdiff) > 0:
            return True
        return False

    def _read_chg(self, fobj, chg, volume):
        """Read charge from file object

        Utility method for reading the actual charge density (or
        charge density difference) from a file object. On input, the
        file object must be at the beginning of the charge block, on
        output the file position will be left at the end of the
        block. The chg array must be of the correct dimensions.

        """
        # VASP writes charge density as
        # WRITE(IU,FORM) (((C(NX,NY,NZ),NX=1,NGXC),NY=1,NGYZ),NZ=1,NGZC)
        # Fortran nested implied do loops; innermost index fastest
        # First, just read it in
        for zz in range(chg.shape[2]):
            for yy in range(chg.shape[1]):
                chg[:, yy, zz] = np.fromfile(fobj, count=chg.shape[0],
                                             sep=' ')
        chg /= volume

    def read(self, filename='CHG'):
        """Read CHG or CHGCAR file.

        If CHG contains charge density from multiple steps all the
        steps are read and stored in the object. By default VASP
        writes out the charge density every 10 steps.

        chgdiff is the difference between the spin up charge density
        and the spin down charge density and is thus only read for a
        spin-polarized calculation.

        aug is the PAW augmentation charges found in CHGCAR. These are
        not parsed, they are just stored as a string so that they can
        be written again to a CHGCAR format file.

        """
        import ase.io.vasp as aiv
        f = open(filename)
        self.atoms = []
        self.chg = []
        self.chgdiff = []
        self.aug = ''
        self.augdiff = ''
        while True:
            try:
                atoms = aiv.read_vasp(f)
            except (IOError, ValueError, IndexError):
                # Probably an empty line, or we tried to read the
                # augmentation occupancies in CHGCAR
                break
            f.readline()
            ngr = f.readline().split()
            ng = (int(ngr[0]), int(ngr[1]), int(ngr[2]))
            chg = np.empty(ng)
            self._read_chg(f, chg, atoms.get_volume())
            self.chg.append(chg)
            self.atoms.append(atoms)
            # Check if the file has a spin-polarized charge density part, and
            # if so, read it in.
            fl = f.tell()
            # First check if the file has an augmentation charge part (CHGCAR
            # file.)
            line1 = f.readline()
            if line1 == '':
                break
            elif line1.find('augmentation') != -1:
                augs = [line1]
                while True:
                    line2 = f.readline()
                    if line2.split() == ngr:
                        self.aug = ''.join(augs)
                        augs = []
                        chgdiff = np.empty(ng)
                        self._read_chg(f, chgdiff, atoms.get_volume())
                        self.chgdiff.append(chgdiff)
                    elif line2 == '':
                        break
                    else:
                        augs.append(line2)
                if len(self.aug) == 0:
                    self.aug = ''.join(augs)
                    augs = []
                else:
                    self.augdiff = ''.join(augs)
                    augs = []
            elif line1.split() == ngr:
                chgdiff = np.empty(ng)
                self._read_chg(f, chgdiff, atoms.get_volume())
                self.chgdiff.append(chgdiff)
            else:
                f.seek(fl)
        f.close()

    def _write_chg(self, fobj, chg, volume, format='chg'):
        """Write charge density

        Utility function similar to _read_chg but for writing.

        """
        # Make a 1D copy of chg, must take transpose to get ordering right
        chgtmp = chg.T.ravel()
        # Multiply by volume
        chgtmp = chgtmp * volume
        # Must be a tuple to pass to string conversion
        chgtmp = tuple(chgtmp)
        # CHG format - 10 columns
        if format.lower() == 'chg':
            # Write all but the last row
            for ii in range((len(chgtmp) - 1) // 10):
                fobj.write(' %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G\
 %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G\n' % chgtmp[ii * 10:(ii + 1) * 10]
                           )
            # If the last row contains 10 values then write them without a
            # newline
            if len(chgtmp) % 10 == 0:
                fobj.write(' %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G'
                           ' %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G' %
                           chgtmp[len(chgtmp) - 10:len(chgtmp)])
            # Otherwise write fewer columns without a newline
            else:
                for ii in range(len(chgtmp) % 10):
                    fobj.write((' %#11.5G')
                               % chgtmp[len(chgtmp) - len(chgtmp) % 10 + ii])
        # Other formats - 5 columns
        else:
            # Write all but the last row
            for ii in range((len(chgtmp) - 1) // 5):
                fobj.write(' %17.10E %17.10E %17.10E %17.10E %17.10E\n'
                           % chgtmp[ii * 5:(ii + 1) * 5])
            # If the last row contains 5 values then write them without a
            # newline
            if len(chgtmp) % 5 == 0:
                fobj.write(' %17.10E %17.10E %17.10E %17.10E %17.10E'
                           % chgtmp[len(chgtmp) - 5:len(chgtmp)])
            # Otherwise write fewer columns without a newline
            else:
                for ii in range(len(chgtmp) % 5):
                    fobj.write((' %17.10E')
                               % chgtmp[len(chgtmp) - len(chgtmp) % 5 + ii])
        # Write a newline whatever format it is
        fobj.write('\n')
        # Clean up
        del chgtmp

    def write(self, filename='CHG', format=None):
        """Write VASP charge density in CHG format.

        filename: str
            Name of file to write to.
        format: str
            String specifying whether to write in CHGCAR or CHG
            format.

        """
        import ase.io.vasp as aiv
        if format is None:
            if filename.lower().find('chgcar') != -1:
                format = 'chgcar'
            elif filename.lower().find('chg') != -1:
                format = 'chg'
            elif len(self.chg) == 1:
                format = 'chgcar'
            else:
                format = 'chg'
        f = open(filename, 'w')
        for ii, chg in enumerate(self.chg):
            if format == 'chgcar' and ii != len(self.chg) - 1:
                continue  # Write only the last image for CHGCAR
            aiv.write_vasp(f, self.atoms[ii], direct=True, long_format=False)
            f.write('\n')
            for dim in chg.shape:
                f.write(' %4i' % dim)
            f.write('\n')
            vol = self.atoms[ii].get_volume()
            self._write_chg(f, chg, vol, format)
            if format == 'chgcar':
                f.write(self.aug)
            if self.is_spin_polarized():
                if format == 'chg':
                    f.write('\n')
                for dim in chg.shape:
                    f.write(' %4i' % dim)
                self._write_chg(f, self.chgdiff[ii], vol, format)
                if format == 'chgcar':
                    f.write('\n')
                    f.write(self.augdiff)
            if format == 'chg' and len(self.chg) > 1:
                f.write('\n')
        f.close()


class VaspDos(object):
    """Class for representing density-of-states produced by VASP

    The energies are in property self.energy

    Site-projected DOS is accesible via the self.site_dos method.

    Total and integrated DOS is accessible as numpy.ndarray's in the
    properties self.dos and self.integrated_dos. If the calculation is
    spin polarized, the arrays will be of shape (2, NDOS), else (1,
    NDOS).

    The self.efermi property contains the currently set Fermi
    level. Changing this value shifts the energies.

    """

    def __init__(self, doscar='DOSCAR', efermi=0.0):
        """Initialize"""
        self._efermi = 0.0
        self.read_doscar(doscar)
        self.efermi = efermi

        # we have determine the resort to correctly map ase atom index to the
        # POSCAR.
        self.sort = []
        self.resort = []
        if os.path.isfile('ase-sort.dat'):
            file = open('ase-sort.dat', 'r')
            lines = file.readlines()
            file.close()
            for line in lines:
                data = line.split()
                self.sort.append(int(data[0]))
                self.resort.append(int(data[1]))

    def _set_efermi(self, efermi):
        """Set the Fermi level."""
        ef = efermi - self._efermi
        self._efermi = efermi
        self._total_dos[0, :] = self._total_dos[0, :] - ef
        try:
            self._site_dos[:, 0, :] = self._site_dos[:, 0, :] - ef
        except IndexError:
            pass

    def _get_efermi(self):
        return self._efermi

    efermi = property(_get_efermi, _set_efermi, None, "Fermi energy.")

    def _get_energy(self):
        """Return the array with the energies."""
        return self._total_dos[0, :]
    energy = property(_get_energy, None, None, "Array of energies")

    def site_dos(self, atom, orbital):
        """Return an NDOSx1 array with dos for the chosen atom and orbital.

        atom: int
            Atom index
        orbital: int or str
            Which orbital to plot

        If the orbital is given as an integer:
        If spin-unpolarized calculation, no phase factors:
        s = 0, p = 1, d = 2
        Spin-polarized, no phase factors:
        s-up = 0, s-down = 1, p-up = 2, p-down = 3, d-up = 4, d-down = 5
        If phase factors have been calculated, orbitals are
        s, py, pz, px, dxy, dyz, dz2, dxz, dx2
        double in the above fashion if spin polarized.

        """
        # Correct atom index for resorting if we need to. This happens when the
        # ase-sort.dat file exists, and self.resort is not empty.
        if self.resort:
            atom = self.resort[atom]

        # Integer indexing for orbitals starts from 1 in the _site_dos array
        # since the 0th column contains the energies
        if isinstance(orbital, int):
            return self._site_dos[atom, orbital + 1, :]
        n = self._site_dos.shape[1]
        if n == 4:
            norb = {'s': 1, 'p': 2, 'd': 3}
        elif n == 7:
            norb = {'s+': 1, 's-up': 1, 's-': 2, 's-down': 2,
                    'p+': 3, 'p-up': 3, 'p-': 4, 'p-down': 4,
                    'd+': 5, 'd-up': 5, 'd-': 6, 'd-down': 6}
        elif n == 10:
            norb = {'s': 1, 'py': 2, 'pz': 3, 'px': 4,
                    'dxy': 5, 'dyz': 6, 'dz2': 7, 'dxz': 8,
                    'dx2': 9}
        elif n == 19:
            norb = {'s+': 1, 's-up': 1, 's-': 2, 's-down': 2,
                    'py+': 3, 'py-up': 3, 'py-': 4, 'py-down': 4,
                    'pz+': 5, 'pz-up': 5, 'pz-': 6, 'pz-down': 6,
                    'px+': 7, 'px-up': 7, 'px-': 8, 'px-down': 8,
                    'dxy+': 9, 'dxy-up': 9, 'dxy-': 10, 'dxy-down': 10,
                    'dyz+': 11, 'dyz-up': 11, 'dyz-': 12, 'dyz-down': 12,
                    'dz2+': 13, 'dz2-up': 13, 'dz2-': 14, 'dz2-down': 14,
                    'dxz+': 15, 'dxz-up': 15, 'dxz-': 16, 'dxz-down': 16,
                    'dx2+': 17, 'dx2-up': 17, 'dx2-': 18, 'dx2-down': 18}
        return self._site_dos[atom, norb[orbital.lower()], :]

    def _get_dos(self):
        if self._total_dos.shape[0] == 3:
            return self._total_dos[1, :]
        elif self._total_dos.shape[0] == 5:
            return self._total_dos[1:3, :]
    dos = property(_get_dos, None, None, 'Average DOS in cell')

    def _get_integrated_dos(self):
        if self._total_dos.shape[0] == 3:
            return self._total_dos[2, :]
        elif self._total_dos.shape[0] == 5:
            return self._total_dos[3:5, :]
    integrated_dos = property(_get_integrated_dos, None, None,
                              'Integrated average DOS in cell')

    def read_doscar(self, fname="DOSCAR"):
        """Read a VASP DOSCAR file"""
        f = open(fname)
        natoms = int(f.readline().split()[0])
        [f.readline() for nn in range(4)]  # Skip next 4 lines.
        # First we have a block with total and total integrated DOS
        ndos = int(f.readline().split()[2])
        dos = []
        for nd in range(ndos):
            dos.append(np.array([float(x) for x in f.readline().split()]))
        self._total_dos = np.array(dos).T
        # Next we have one block per atom, if INCAR contains the stuff
        # necessary for generating site-projected DOS
        dos = []
        for na in range(natoms):
            line = f.readline()
            if line == '':
                # No site-projected DOS
                break
            ndos = int(line.split()[2])
            line = f.readline().split()
            cdos = np.empty((ndos, len(line)))
            cdos[0] = np.array(line)
            for nd in range(1, ndos):
                line = f.readline().split()
                cdos[nd] = np.array([float(x) for x in line])
            dos.append(cdos.T)
        self._site_dos = np.array(dos)


class xdat2traj:
    def __init__(self, trajectory=None, atoms=None, poscar=None,
                 xdatcar=None, sort=None, calc=None):
        """
        trajectory is the name of the file to write the trajectory to
        poscar is the name of the poscar file to read. Default: POSCAR
        """
        if not poscar:
            self.poscar = 'POSCAR'
        else:
            self.poscar = poscar

        if not atoms:
            # This reads the atoms sorted the way VASP wants
            self.atoms = ase.io.read(self.poscar, format='vasp')
            resort_reqd = True
        else:
            # Assume if we pass atoms that it is sorted the way we want
            self.atoms = atoms
            resort_reqd = False

        if not calc:
            self.calc = Vasp()
        else:
            self.calc = calc
        if not sort:
            if not hasattr(self.calc, 'sort'):
                self.calc.sort = list(range(len(self.atoms)))
        else:
            self.calc.sort = sort
        self.calc.resort = list(range(len(self.calc.sort)))
        for n in range(len(self.calc.resort)):
            self.calc.resort[self.calc.sort[n]] = n

        if not xdatcar:
            self.xdatcar = 'XDATCAR'
        else:
            self.xdatcar = xdatcar

        if not trajectory:
            self.trajectory = 'out.traj'
        else:
            self.trajectory = trajectory

        self.out = ase.io.trajectory.Trajectory(self.trajectory,
                                                mode='w')

        if resort_reqd:
            self.atoms = self.atoms[self.calc.resort]
        self.energies = self.calc.read_energy(all=True)[1]
        # Forces are read with the atoms sorted using resort
        self.forces = self.calc.read_forces(self.atoms, all=True)

    def convert(self):
        lines = open(self.xdatcar).readlines()
        if len(lines[7].split()) == 0:
            del(lines[0:8])
        elif len(lines[5].split()) == 0:
            del(lines[0:6])
        elif len(lines[4].split()) == 0:
            del(lines[0:5])
        elif lines[7].split()[0] == 'Direct':
            del(lines[0:8])
        step = 0
        iatom = 0
        scaled_pos = []
        for line in lines:
            if iatom == len(self.atoms):
                if step == 0:
                    self.out.write_header(self.atoms[self.calc.resort])
                scaled_pos = np.array(scaled_pos)
                # Now resort the positions to match self.atoms
                self.atoms.set_scaled_positions(scaled_pos[self.calc.resort])

                calc = SinglePointCalculator(self.atoms,
                                             energy=self.energies[step],
                                             forces=self.forces[step])
                self.atoms.set_calculator(calc)
                self.out.write(self.atoms)
                scaled_pos = []
                iatom = 0
                step += 1
            else:
                if not line.split()[0] == 'Direct':
                    iatom += 1
                    scaled_pos.append([float(line.split()[n])
                                       for n in range(3)])

        # Write also the last image
        # I'm sure there is also more clever fix...
        if step == 0:
            self.out.write_header(self.atoms[self.calc.resort])
        scaled_pos = np.array(scaled_pos)[self.calc.resort]
        self.atoms.set_scaled_positions(scaled_pos)
        calc = SinglePointCalculator(self.atoms,
                                     energy=self.energies[step],
                                     forces=self.forces[step])
        self.atoms.set_calculator(calc)
        self.out.write(self.atoms)

        self.out.close()
