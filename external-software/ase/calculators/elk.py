import os

import numpy as np

from ase.units import Bohr, Hartree
from ase.io.elk import read_elk
from ase.calculators.calculator import (FileIOCalculator, Parameters, kpts2mp,
                                        ReadError, PropertyNotImplementedError,
                                        EigenvalOccupationMixin)

elk_parameters = {'swidth': Hartree}


class ELK(FileIOCalculator, EigenvalOccupationMixin):
    command = 'elk > elk.out'
    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label=os.curdir, atoms=None, **kwargs):
        """Construct ELK calculator.

        The keyword arguments (kwargs) can be one of the ASE standard
        keywords: 'xc', 'kpts' and 'smearing' or any of ELK'
        native keywords.
        """

        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

    def set_label(self, label):
        self.label = label
        self.directory = label
        self.prefix = ''
        self.out = os.path.join(label, 'INFO.OUT')

    def check_state(self, atoms):
        system_changes = FileIOCalculator.check_state(self, atoms)
        # Ignore boundary conditions (ELK always uses them):
        if 'pbc' in system_changes:
            system_changes.remove('pbc')
        return system_changes

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        self.initialize(atoms)

        self.parameters.write(os.path.join(self.directory, 'parameters.ase'))

        if 'xctype' in self.parameters:
            if 'xc' in self.parameters:
                raise RuntimeError("You can't use both 'xctype' and 'xc'!")

        if self.parameters.get('autokpt'):
            if 'kpts' in self.parameters:
                raise RuntimeError("You can't use both 'autokpt' and 'kpts'!")
            if 'ngridk' in self.parameters:
                raise RuntimeError(
                    "You can't use both 'autokpt' and 'ngridk'!")
        if 'ngridk' in self.parameters:
            if 'kpts' in self.parameters:
                raise RuntimeError("You can't use both 'ngridk' and 'kpts'!")

        if self.parameters.get('autoswidth'):
            if 'smearing' in self.parameters:
                raise RuntimeError(
                    "You can't use both 'autoswidth' and 'smearing'!")
            if 'swidth' in self.parameters:
                raise RuntimeError(
                    "You can't use both 'autoswidth' and 'swidth'!")

        fd = open(os.path.join(self.directory, 'elk.in'), 'w')

        # handle custom specifications of rmt
        # (absolute or relative to default) in Bohr
        # rmt = {'H': 0.7, 'O': -0.2, ...}

        if self.parameters.get('rmt', None) is not None:
            self.rmt = self.parameters['rmt'].copy()
            assert len(self.rmt.keys()) == len(list(set(self.rmt.keys()))), \
                'redundant rmt definitions'
            self.parameters.pop('rmt')  # this is not an elk keyword!
        else:
            self.rmt = None

        inp = {}
        inp.update(self.parameters)

        if 'xc' in self.parameters:
            xctype = {'LDA': 3,  # PW92
                      'PBE': 20,
                      'REVPBE': 21,
                      'PBESOL': 22,
                      'WC06': 26,
                      'AM05': 30,
                      'mBJLDA': (100, 208, 12)}[self.parameters.xc]
            inp['xctype'] = xctype
            del inp['xc']

        if 'kpts' in self.parameters:
            mp = kpts2mp(atoms, self.parameters.kpts)
            inp['ngridk'] = tuple(mp)
            vkloff = []  # is this below correct?
            for nk in mp:
                if nk % 2 == 0:  # shift kpoint away from gamma point
                    vkloff.append(0.5)
                else:
                    vkloff.append(0)
            inp['vkloff'] = vkloff
            del inp['kpts']

        if 'smearing' in self.parameters:
            name = self.parameters.smearing[0].lower()
            if name == 'methfessel-paxton':
                stype = self.parameters.smearing[2]
            else:
                stype = {'gaussian': 0,
                         'fermi-dirac': 3,
                         }[name]
            inp['stype'] = stype
            inp['swidth'] = self.parameters.smearing[1]
            del inp['smearing']

        # convert keys to ELK units
        for key, value in inp.items():
            if key in elk_parameters:
                inp[key] /= elk_parameters[key]

        # write all keys
        for key, value in inp.items():
            fd.write('%s\n' % key)
            if isinstance(value, bool):
                fd.write('.%s.\n\n' % ('false', 'true')[value])
            elif isinstance(value, (int, float)):
                fd.write('%s\n\n' % value)
            else:
                fd.write('%s\n\n' % ' '.join([str(x) for x in value]))

        # cell
        fd.write('avec\n')
        for vec in atoms.cell:
            fd.write('%.14f %.14f %.14f\n' % tuple(vec / Bohr))
        fd.write('\n')

        # atoms
        species = {}
        symbols = []
        for a, (symbol, m) in enumerate(
            zip(atoms.get_chemical_symbols(),
                atoms.get_initial_magnetic_moments())):
            if symbol in species:
                species[symbol].append((a, m))
            else:
                species[symbol] = [(a, m)]
                symbols.append(symbol)
        fd.write('atoms\n%d\n' % len(species))
        # scaled = atoms.get_scaled_positions(wrap=False)
        scaled = np.linalg.solve(atoms.cell.T, atoms.positions.T).T
        for symbol in symbols:
            fd.write("'%s.in' : spfname\n" % symbol)
            fd.write('%d\n' % len(species[symbol]))
            for a, m in species[symbol]:
                fd.write('%.14f %.14f %.14f 0.0 0.0 %.14f\n' %
                         (tuple(scaled[a]) + (m,)))
        # species
        species_path = self.parameters.get('species_dir')
        if species_path is None:
            species_path = os.environ.get('ELK_SPECIES_PATH')
        if species_path is None:
            raise RuntimeError(
                'Missing species directory!  Use species_dir ' +
                'parameter or set $ELK_SPECIES_PATH environment variable.')
        # custom species definitions
        if self.rmt is not None:
            fd.write("\n")
            sfile = os.path.join(os.environ['ELK_SPECIES_PATH'], 'elk.in')
            assert os.path.exists(sfile)
            slines = open(sfile, 'r').readlines()
            # remove unused species
            for s in self.rmt.keys():
                if s not in species.keys():
                    self.rmt.pop(s)
            # add undefined species with defaults
            for s in species.keys():
                if s not in self.rmt.keys():
                    # use default rmt for undefined species
                    self.rmt.update({s: 0.0})
            # write custom species into elk.in
            skeys = list(set(self.rmt.keys()))  # unique
            skeys.sort()
            for s in skeys:
                found = False
                for n, line in enumerate(slines):
                    if line.find("'" + s + "'") > -1:
                        begline = n - 1
                for n, line in enumerate(slines[begline:]):
                    if not line.strip():  # first empty line
                        endline = n
                        found = True
                        break
                assert found
                fd.write("species\n")
                # set rmt on third line
                rmt = self.rmt[s]
                assert isinstance(rmt, (float, int))
                if rmt <= 0.0:  # relative
                    # split needed because H is defined with comments
                    newrmt = (float(slines[begline + 3].split()[0].strip()) +
                              rmt)
                else:
                    newrmt = rmt
                slines[begline + 3] = '%6s\n' % str(newrmt)
                for l in slines[begline: begline + endline]:
                    fd.write('%s' % l)
                fd.write('\n')
        else:
            # use default species
            # if sppath is present in elk.in it overwrites species blocks!
            fd.write("sppath\n'%s'\n\n" % os.environ['ELK_SPECIES_PATH'])

    def read(self, label):
        FileIOCalculator.read(self, label)
        totenergy = os.path.join(self.directory, 'TOTENERGY.OUT')
        eigval = os.path.join(self.directory, 'EIGVAL.OUT')
        kpoints = os.path.join(self.directory, 'KPOINTS.OUT')

        for filename in [totenergy, eigval, kpoints, self.out]:
            if not os.path.isfile(filename):
                raise ReadError('ELK output file ' + filename + ' is missing.')

        # read state from elk.in because *.OUT do not provide enough digits!
        self.atoms = read_elk(os.path.join(self.directory, 'elk.in'))
        self.parameters = Parameters.read(os.path.join(self.directory,
                                                       'parameters.ase'))
        self.initialize(self.atoms)
        self.read_results()

    def read_results(self):
        converged = self.read_convergence()
        if not converged:
            raise RuntimeError('ELK did not converge! Check ' + self.out)
        self.read_energy()
        if self.parameters.get('tforce'):
            self.read_forces()
        self.width = self.read_electronic_temperature()
        self.nbands = self.read_number_of_bands()
        self.nelect = self.read_number_of_electrons()
        self.niter = self.read_number_of_iterations()
        self.magnetic_moment = self.read_magnetic_moment()

    def initialize(self, atoms):
        if 'spinpol' not in self.parameters:  # honor elk.in settings
            self.spinpol = atoms.get_initial_magnetic_moments().any()
        else:
            self.spinpol = self.parameters['spinpol']

    def get_forces(self, atoms):
        if not self.parameters.get('tforce'):
            raise PropertyNotImplementedError
        return FileIOCalculator.get_forces(self, atoms)

    def read_energy(self):
        fd = open(os.path.join(self.directory, 'TOTENERGY.OUT'), 'r')
        e = float(fd.readlines()[-1]) * Hartree
        self.results['free_energy'] = e
        self.results['energy'] = e

    def read_forces(self):
        lines = open(self.out, 'r').readlines()
        forces = []
        for line in lines:
            if line.rfind('total force') > -1:
                forces.append(np.array([float(f)
                                        for f in line.split(':')[1].split()]))
        self.results['forces'] = np.array(forces) * Hartree / Bohr

    def read_convergence(self):
        converged = False
        text = open(self.out).read().lower()
        if ('convergence targets achieved' in text and
            'reached self-consistent loops maximum' not in text):
            converged = True
        return converged

    # more methods
    def get_electronic_temperature(self):
        return self.width * Hartree

    def get_number_of_bands(self):
        return self.nbands

    def get_number_of_electrons(self):
        return self.nelect

    def get_number_of_iterations(self):
        return self.niter

    def get_number_of_spins(self):
        return 1 + int(self.spinpol)

    def get_magnetic_moment(self, atoms=None):
        return self.magnetic_moment

    def get_magnetic_moments(self, atoms):
        # not implemented yet, so
        # so set the total magnetic moment on the atom no. 0 and fill with 0.0
        magmoms = [0.0 for a in range(len(atoms))]
        magmoms[0] = self.get_magnetic_moment(atoms)
        return np.array(magmoms)

    def get_spin_polarized(self):
        return self.spinpol

    def get_eigenvalues(self, kpt=0, spin=0):
        return self.read_eigenvalues(kpt, spin, 'eigenvalues')

    def get_occupation_numbers(self, kpt=0, spin=0):
        return self.read_eigenvalues(kpt, spin, 'occupations')

    def get_ibz_k_points(self):
        return self.read_kpts(mode='ibz_k_points')

    def get_k_point_weights(self):
        return self.read_kpts(mode='k_point_weights')

    def get_fermi_level(self):
        return self.read_fermi()

    def read_kpts(self, mode='ibz_k_points'):
        """ Returns list of kpts weights or kpts coordinates.  """
        values = []
        assert mode in ['ibz_k_points', 'k_point_weights']
        kpoints = os.path.join(self.directory, 'KPOINTS.OUT')
        lines = open(kpoints).readlines()
        kpts = None
        for line in lines:
            if line.rfind(': nkpt') > -1:
                kpts = int(line.split(':')[0].strip())
                break
        assert kpts is not None
        text = lines[1:]  # remove first line
        values = []
        for line in text:
            if mode == 'ibz_k_points':
                b = [float(c.strip()) for c in line.split()[1:4]]
            else:
                b = float(line.split()[-2])
            values.append(b)
        if len(values) == 0:
            values = None
        return np.array(values)

    def read_number_of_bands(self):
        nbands = None
        eigval = os.path.join(self.directory, 'EIGVAL.OUT')
        lines = open(eigval).readlines()
        for line in lines:
            if line.rfind(': nstsv') > -1:
                nbands = int(line.split(':')[0].strip())
                break
        if self.get_spin_polarized():
            nbands = nbands // 2
        return nbands

    def read_number_of_electrons(self):
        nelec = None
        text = open(self.out).read().lower()
        # Total electronic charge
        for line in iter(text.split('\n')):
            if line.rfind('total electronic charge :') > -1:
                nelec = float(line.split(':')[1].strip())
                break
        return nelec

    def read_number_of_iterations(self):
        niter = None
        lines = open(self.out).readlines()
        for line in lines:
            if line.rfind(' Loop number : ') > -1:
                niter = int(line.split(':')[1].split()[0].strip())  # last iter
        return niter

    def read_magnetic_moment(self):
        magmom = None
        lines = open(self.out).readlines()
        for line in lines:
            if line.rfind('total moment                :') > -1:
                magmom = float(line.split(':')[1].strip())  # last iter
        return magmom

    def read_electronic_temperature(self):
        width = None
        text = open(self.out).read().lower()
        for line in iter(text.split('\n')):
            if line.rfind('smearing width :') > -1:
                width = float(line.split(':')[1].strip())
                break
        return width

    def read_eigenvalues(self, kpt=0, spin=0, mode='eigenvalues'):
        """ Returns list of last eigenvalues, occupations
        for given kpt and spin.  """
        values = []
        assert mode in ['eigenvalues', 'occupations']
        eigval = os.path.join(self.directory, 'EIGVAL.OUT')
        lines = open(eigval).readlines()
        nstsv = None
        for line in lines:
            if line.rfind(': nstsv') > -1:
                nstsv = int(line.split(':')[0].strip())
                break
        assert nstsv is not None
        kpts = None
        for line in lines:
            if line.rfind(': nkpt') > -1:
                kpts = int(line.split(':')[0].strip())
                break
        assert kpts is not None
        text = lines[3:]  # remove first 3 lines
        # find the requested k-point
        beg = 2 + (nstsv + 4) * kpt
        end = beg + nstsv
        if self.get_spin_polarized():
            # elk prints spin-up and spin-down together
            if spin == 0:
                beg = beg
                end = beg + nstsv // 2
            else:
                beg = beg + nstsv // 2
                end = end
        values = []
        for line in text[beg:end]:
            b = [float(c.strip()) for c in line.split()[1:]]
            values.append(b)
        if mode == 'eigenvalues':
            values = [Hartree * v[0] for v in values]
        else:
            values = [v[1] for v in values]
        if len(values) == 0:
            values = None
        return np.array(values)

    def read_fermi(self):
        """Method that reads Fermi energy in Hartree from the output file
        and returns it in eV"""
        E_f = None
        text = open(self.out).read().lower()
        for line in iter(text.split('\n')):
            if line.rfind('fermi                       :') > -1:
                E_f = float(line.split(':')[1].strip())
        E_f = E_f * Hartree
        return E_f
