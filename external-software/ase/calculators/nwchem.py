"""This module defines an ASE interface to NWchem

http://www.nwchem-sw.org/
"""
import os
import re
import numpy as np

from warnings import warn
from ase.atoms import Atoms
from ase.units import Hartree, Bohr
from ase.io.nwchem import write_nwchem
from ase.calculators.calculator import (FileIOCalculator, Parameters,
                                        ReadError, EigenvalOccupationMixin)


class KPoint:
    def __init__(self, s):
        self.s = s
        self.eps_n = []
        self.f_n = []


class NWChem(FileIOCalculator, EigenvalOccupationMixin):
    implemented_properties = ['energy', 'forces', 'dipole', 'magmom',
                              'orbitals']
    command = 'nwchem PREFIX.nw > PREFIX.out'

    default_parameters = dict(
        xc='LDA',
        smearing=None,
        charge=None,
        task='gradient',
        # Warning: nwchem centers atoms by default
        # see ase-developers/2012-March/001356.html
        geometry='nocenter noautosym',
        convergence={'energy': None,
                     'density': None,
                     'gradient': None,
                     'lshift': None,
                     # set lshift to 0.0 for nolevelshifting
                     'damp': None,
                     },
        basis='3-21G',
        basispar=None,
        ecp=None,
        so=None,
        spinorbit=False,
        odft=False,
        raw='')  # additional outside of dft block control string

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='nwchem', atoms=None, **kwargs):
        """Construct NWchem-calculator object."""
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def check_state(self, atoms):
        system_changes = FileIOCalculator.check_state(self, atoms)
        # Ignore unit cell and boundary conditions:
        if 'cell' in system_changes:
            system_changes.remove('cell')
        if 'pbc' in system_changes:
            system_changes.remove('pbc')
        return system_changes

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        p = self.parameters
        p.initial_magmoms = atoms.get_initial_magnetic_moments().tolist()
        p.write(self.label + '.ase')
        del p['initial_magmoms']
        f = open(self.label + '.nw', 'w')
        if p.charge is not None:
            f.write('charge %s\n' % p.charge)
        write_nwchem(f, atoms, p.geometry)

        f.write('start\n')

        if p.basispar is not None:
            basispar = 'basis ' + p.basispar
        else:
            basispar = 'basis'

        def format_basis_set(string, tag=basispar):
            formatted = tag + '\n'
            lines = string.split('\n')
            if len(lines) > 1:
                formatted += string
            else:
                formatted += '  * library ' + string
            return formatted + '\nend\n'

        basis = format_basis_set(p.basis)
        if p.ecp is not None:
            basis += format_basis_set(p.ecp, 'ecp')
        if p.so is not None:
            basis += format_basis_set(p.so, 'so')
        f.write(basis)

        if p.xc == 'RHF':
            task = 'scf'
        elif p.xc == 'MP2':
            task = 'mp2'
        else:
            if p.spinorbit:
                task = 'sodft'
            else:
                task = 'dft'
            xc = {'LDA': 'slater pw91lda',
                  'PBE': 'xpbe96 cpbe96',
                  'revPBE': 'revpbe cpbe96',
                  'RPBE': 'rpbe cpbe96'}.get(p.xc, p.xc)
            f.write('\n' + task + '\n')
            f.write('  xc ' + xc + '\n')
            for key in p.convergence:
                if p.convergence[key] is not None:
                    if key == 'lshift':
                        if p.convergence[key] <= 0.0:
                            f.write('  convergence nolevelshifting\n')
                        else:
                            f.write('  convergence %s %s\n' %
                                    (key, p.convergence[key] / Hartree))
                    else:
                        f.write('  convergence %s %s\n' %
                                (key, p.convergence[key]))
            if p.smearing is not None:
                assert p.smearing[0].lower() == 'gaussian', p.smearing
                f.write('  smear %s\n' % (p.smearing[1] / Hartree))
            if 'mult' not in p:
                # Obtain multiplicity from magnetic momenta:
                tot_magmom = atoms.get_initial_magnetic_moments().sum()
                if tot_magmom < 0:
                    mult = tot_magmom - 1  # fill minority bands
                else:
                    mult = tot_magmom + 1
            else:
                mult = p.mult
            if mult != int(mult):
                raise RuntimeError('Noninteger multiplicity not possible. ' +
                                   'Check initial magnetic moments.')
            f.write('  mult %d\n' % mult)
            if p.odft:
                f.write('  odft\n')  # open shell aka spin polarized dft
            for key in sorted(p.keys()):
                if key in ['charge', 'geometry', 'basis', 'basispar', 'ecp',
                           'so', 'xc', 'spinorbit', 'convergence', 'smearing',
                           'raw', 'mult', 'task', 'odft']:
                    continue
                f.write(u"  {0} {1}\n".format(key, p[key]))
            f.write('end\n')

        if p.raw:
            f.write(p.raw + '\n')
        f.write('\ntask ' + task + ' ' + p.task + '\n')
        f.close()

    def read(self, label):
        FileIOCalculator.read(self, label)
        if not os.path.isfile(self.label + '.out'):
            raise ReadError

        f = open(self.label + '.nw')
        for line in f:
            if line.startswith('geometry'):
                break
        symbols = []
        positions = []
        for line in f:
            if line.startswith('end'):
                break
            words = line.split()
            symbols.append(words[0])
            positions.append([float(word) for word in words[1:]])

        self.parameters = Parameters.read(self.label + '.ase')
        self.atoms = Atoms(symbols, positions,
                           magmoms=self.parameters.pop('initial_magmoms'))
        self.read_results()

    def read_results(self):
        self.read_energy()
        if self.parameters.task.find('gradient') > -1:
            self.read_forces()
        if self.parameters.task.find('optimize') > -1:
            self.read_coordinates()
            self.read_forces()
        self.niter = self.read_number_of_iterations()
        self.nelect = self.read_number_of_electrons()
        self.nvector = self.read_number_of_bands()
        self.results['magmom'] = self.read_magnetic_moment()
        dipole = self.read_dipole_moment()
        if dipole is not None:
            self.results['dipole'] = dipole
        self.read_mos()

    def get_ibz_k_points(self):
        return np.array([0., 0., 0.])

    def get_number_of_bands(self):
        return self.nvector

    def read_number_of_bands(self):
        nvector = 0
        for line in open(self.label + '.out'):
            if line.find('Vector ') != -1:  # count all printed vectors
                nvector += 1
        if not nvector:
            nvector = None
        return nvector

    def get_number_of_electrons(self):
        return self.nelect

    def read_number_of_electrons(self):
        nelect = None
        for line in open(self.label + '.out'):  # find last one
            if line.find('of electrons') != -1:
                nelect = float(line.split(':')[1].strip())
        return nelect

    def get_number_of_iterations(self):
        return self.niter

    def read_number_of_iterations(self):
        niter = 0
        for line in open(self.label + '.out'):
            if line.find('d= ') != -1:  # count all iterations
                niter += 1
        if not niter:
            niter = None
        return niter

    def read_magnetic_moment(self):
        magmom = None
        for line in open(self.label + '.out'):
            if line.find('Spin multiplicity') != -1:  # last one
                magmom = float(line.split(':')[-1].strip())
                if magmom < 0:
                    magmom += 1
                else:
                    magmom -= 1
        return magmom

    def read_dipole_moment(self):
        dipolemoment = []
        for line in open(self.label + '.out'):
            for component in ['1   1 0 0',
                              '1   0 1 0',
                              '1   0 0 1']:
                if line.find(component) != -1:
                    value = float(line.split(component)[1].split()[0])
                    value = value * Bohr
                    dipolemoment.append(value)
        if len(dipolemoment) == 0:
            if len(self.atoms) == 1:
                dipolemoment = [0.0, 0.0, 0.0]
            else:
                return None
        return np.array(dipolemoment)

    def read_energy(self):
        """Read Energy from nwchem output file."""
        text = open(self.label + '.out', 'r').read()
        lines = iter(text.split('\n'))

        # Energy:
        estring = 'Total '
        if self.parameters.xc == 'RHF':
            estring += 'SCF'
        elif self.parameters.xc == 'MP2':
            estring += 'MP2'
        else:
            estring += 'DFT'
        estring += ' energy'
        for line in lines:
            if line.find(estring) >= 0:
                energy = float(line.split()[-1])
        self.results['energy'] = energy * Hartree

        # All lines have been 'eaten' while iterating; loop from scratch.
        # (We could insert a break, but I (askhl) don't know if multiple
        # energies might be listed; then we would not get the last one.
        lines = text.split('\n')

        # Eigenstates
        spin = -1
        kpts = []
        for line in lines:
            if line.find('Molecular Orbital Analysis') >= 0:
                last_eps = -99999.0
                spin += 1
                kpts.append(KPoint(spin))
            if spin >= 0:
                if line.find('Vector') >= 0:
                    line = line.lower().replace('d', 'e')
                    line = line.replace('=', ' ')
                    word = line.split()
                    this_occ = float(word[3])
                    this_eps = float(word[5])
                    kpts[spin].f_n.append(this_occ)
                    kpts[spin].eps_n.append(this_eps)
                    if this_occ < 0.1 and this_eps < last_eps:
                        warn('HOMO above LUMO - if this is not an exicted ' +
                             'state - this might be introduced by levelshift.',
                             RuntimeWarning)
                    last_eps = this_eps
        self.kpts = kpts

    def read_forces(self):
        """Read Forces from nwchem output file."""
        file = open(self.label + '.out', 'r')
        lines = file.readlines()
        file.close()

        for i, line in enumerate(lines):
            if line.find('ENERGY GRADIENTS') >= 0:
                gradients = []
                for j in range(i + 4, i + 4 + len(self.atoms)):
                    word = lines[j].split()
                    gradients.append([float(word[k]) for k in range(5, 8)])

        self.results['forces'] = -np.array(gradients) * Hartree / Bohr

    def read_coordinates(self):
        """Read updated coordinates from nwchem output file."""
        file = open(self.label + '.out', 'r')
        lines = file.readlines()
        file.close()

        for i, line in enumerate(lines):
            if line.find('ENERGY GRADIENTS') >= 0:
                positions = []
                for j in range(i + 4, i + 4 + len(self.atoms)):
                    word = lines[j].split()
                    positions.append([float(word[k]) for k in range(2, 5)])

        self.atoms.set_positions(np.array(positions) * Bohr)

    def read_mos(self):
        """
        Read the molecular orbitals and molecular orbital energies
        The output can be requested by specification of the print keyword
            {'print': '\"final vectors analysis\" \"final vectors\"'}
        when constructing the calculator object.
        """
        orbitals = []
        orbital_coefficients = []
        orbital_coefficients_line = []
        filename = self.label + '.out'
        orb_number = None
        mo_read = False
        mo_init = False
        ev_number = 0
        for line in open(filename, 'r'):
            if 'Final MO vectors' in line:
                mo_read = True
                mo_init = False
                orb_number = None
                orbital_coefficients = []
                orbital_coefficients_line = []
                continue
            if mo_read:
                regex = (
                    r'^\s*(\d+)(\s+[+-]?\d+\.\d+)(\s+[+-]?\d+\.\d+)?'
                    r'(\s+[+-]?\d+\.\d+)?(\s+[+-]?\d+\.\d+)?'
                    r'(\s+[+-]?\d+\.\d+)?(\s+[+-]?\d+\.\d+)?'
                )
                match = re.search(regex, line)
                if match:
                    orb_number = int(match.group(1))
                    for i in range(2, 8):
                        if match.group(i):
                            coeff = float(match.group(i))
                            orbital_coefficients_line.append(coeff)
                    if mo_init:
                        orbital_coefficients[orb_number-1].extend(
                            orbital_coefficients_line
                        )
                    else:
                        orbital_coefficients.append(orbital_coefficients_line)
                    orbital_coefficients_line = []
                elif orb_number:
                    if orb_number == len(orbital_coefficients):
                        mo_init = True
                    if orb_number == len(orbital_coefficients[0]):
                        mo_read = False
                        continue
            if 'Vector' in line:
                regex = (
                    r'Vector\s+(\d+)\s+Occ=\s*([\w\.+-]+)\s+E=\s*([+-\.\w]+)'
                )
                match = re.search(regex, line)
                if match:
                    ev_number += 1
                    # for the case of geometry optimization output
                    if int(match.group(1)) < ev_number:
                        orbitals = []
                        ev_number = int(match.group(1))
                    orbitals.append({
                        'number': int(match.group(1)),
                        'occupation': float(re.sub('[dD]', 'E', match.group(2))),
                        'energy': float(re.sub('[dD]', 'E', match.group(3)))
                        })
        if len(orbital_coefficients) > 0:
            # this is to transpose the nested list
            mos = [[x[i] for x in orbital_coefficients]
                   for i in range(len(orbital_coefficients[0]))]
            for orbital in orbitals:
                orbital['coefficients'] = mos.pop(0)
            self.results['orbitals'] = orbitals

    def get_eigenvalues(self, kpt=0, spin=0):
        """Return eigenvalue array."""
        return np.array(self.kpts[spin].eps_n) * Hartree

    def get_occupation_numbers(self, kpt=0, spin=0):
        """Return occupation number array."""
        return self.kpts[spin].f_n

    def get_number_of_spins(self):
        """Return the number of spins in the calculation.

        Spin-paired calculations: 1, spin-polarized calculation: 2."""
        return len(self.kpts)

    def get_spin_polarized(self):
        """Is it a spin-polarized calculation?"""
        return len(self.kpts) == 2

    def get_mos(self, atoms):
        """ return eigenvectors and eigenvalues as matrices (numpy arrays) """
        orbital_coefficients = []
        eigen_energies = []
        for orbital in self.get_property('orbitals'):
            orbital_coefficients.append(orbital['coefficients'])
            eigen_energies.append(orbital['energy'])
        return [
            np.array(orbital_coefficients),
            np.diag(np.array(eigen_energies)*Hartree)
        ]
