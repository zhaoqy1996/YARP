"""This module defines an ASE interface to Amber16.

Usage: (Tested only with Amber16, http://ambermd.org/)

Before usage, input files (infile, topologyfile, incoordfile)

"""

import os
import subprocess
import numpy as np

from ase.calculators.calculator import Calculator, FileIOCalculator
import ase.units as units
from scipy.io import netcdf


class Amber(FileIOCalculator):
    """Class for doing Amber classical MM calculations.

    Example:

    mm.in::

        Minimization with Cartesian restraints
        &cntrl
        imin=1, maxcyc=200, (invoke minimization)
        ntpr=5, (print frequency)
        &end
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='amber', atoms=None, command=None,
                 amber_exe='sander -O ',
                 infile='mm.in', outfile='mm.out',
                 topologyfile='mm.top', incoordfile='mm.crd',
                 outcoordfile='mm_dummy.crd',
                 **kwargs):
        """Construct Amber-calculator object.

        Parameters
        ==========
        label: str
            Name used for all files.  May contain a directory.
        atoms: Atoms object
            Optional Atoms object to which the calculator will be
            attached.  When restarting, atoms will get its positions and
            unit-cell updated from file.
        label: str
            Prefix to use for filenames (label.in, label.txt, ...).
        amber_exe: str
            Name of the amber executable, one can add options like -O
            and other paramaters here
        infile: str
            Input filename for amber, contains instuctions about the run
        outfile: str
            Logfilename for amber
        topologyfile: str
            Name of the amber topology file
        incoordfile: str
            Name of the file containing the input coordinates of atoms
        outcoordfile: str
            Name of the file containing the output coordinates of atoms
            this file is not used in case minisation/dynamics is done by ase.
            It is only relevant
            if you run MD/optimisation many steps with amber.

        """

        self.out = 'mm.log'

        self.positions = None
        self.atoms = None

        self.set(**kwargs)

        self.amber_exe = amber_exe
        self.infile = infile
        self.outfile = outfile
        self.topologyfile = topologyfile
        self.incoordfile = incoordfile
        self.outcoordfile = outcoordfile
        if command is not None:
            self.command = command
        else:
            self.command = (self.amber_exe +
                            ' -i ' + self.infile +
                            ' -o ' + self.outfile +
                            ' -p ' + self.topologyfile +
                            ' -c ' + self.incoordfile +
                            ' -r ' + self.outcoordfile)

        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def write_input(self, atoms=None, properties=None, system_changes=None):
        """Write updated coordinates to a file."""

        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        self.write_coordinates(atoms)

    def read_results(self):
        """ read energy and forces """
        self.read_energy()
        self.read_forces()

    def write_coordinates(self, atoms, filename=''):
        """ write amber coordinates in netCDF format,
            only rectangular unit cells are allowed"""
        if filename == '':
            filename = self.incoordfile
        fout = netcdf.netcdf_file(filename, 'w')
        # dimension
        fout.Conventions = 'AMBERRESTART'
        fout.ConventionVersion = "1.0"
        fout.title = 'Ase-generated-amber-restart-file'
        fout.application = "AMBER"
        fout.program = "ASE"
        fout.programVersion = "1.0"
        fout.createDimension('cell_spatial', 3)
        fout.createDimension('label', 5)
        fout.createDimension('cell_angular', 3)
        fout.createDimension('time', 1)
        time = fout.createVariable('time', 'd', ('time',))
        time.units = 'picosecond'
        fout.createDimension('spatial', 3)
        spatial = fout.createVariable('spatial', 'c', ('spatial',))
        spatial[:] = np.asarray(list('xyz'))
        # spatial = 'xyz'

        natom = len(atoms)
        fout.createDimension('atom', natom)
        coordinates = fout.createVariable('coordinates', 'd',
                                          ('atom', 'spatial'))
        coordinates.units = 'angstrom'
        coordinates[:] = atoms.get_positions()[:]

        if atoms.get_velocities() is not None:
            velocities = fout.createVariable('velocities', 'd',
                                             ('atom', 'spatial'))
            velocities.units = 'angstrom/picosecond'
            velocities[:] = atoms.get_velocities()[:]

        # title
        cell_angular = fout.createVariable('cell_angular', 'c',
                                           ('cell_angular', 'label'))
        cell_angular[0] = np.asarray(list('alpha'))
        cell_angular[1] = np.asarray(list('beta '))
        cell_angular[2] = np.asarray(list('gamma'))

        # title
        cell_spatial = fout.createVariable('cell_spatial', 'c',
                                           ('cell_spatial',))
        cell_spatial[0], cell_spatial[1], cell_spatial[2] = 'a', 'b', 'c'

        # data
        cell_lengths = fout.createVariable('cell_lengths', 'd',
                                           ('cell_spatial',))
        cell_lengths.units = 'angstrom'
        cell_lengths[0] = atoms.get_cell()[0, 0]
        cell_lengths[1] = atoms.get_cell()[1, 1]
        cell_lengths[2] = atoms.get_cell()[2, 2]

        cell_angles = fout.createVariable('cell_angles', 'd',
                                          ('cell_angular',))
        box_alpha, box_beta, box_gamma = 90.0, 90.0, 90.0
        cell_angles[0] = box_alpha
        cell_angles[1] = box_beta
        cell_angles[2] = box_gamma

        cell_angles.units = 'degree'
        fout.close()

    def read_coordinates(self, atoms, filename=''):
        """Import AMBER16 netCDF restart files.

        Reads atom positions and
        velocities (if available),
        and unit cell (if available)

        This may be usefull if you have run amber many steps and
        want to read new positions and velocities
        """

        if filename == '':
            filename = self.outcoordfile

        from scipy.io import netcdf
        import numpy as np
        import ase.units as units

        fin = netcdf.netcdf_file(filename, 'r')
        atoms.set_positions(fin.variables['coordinates'][:])
        if 'velocities' in fin.variables:
            atoms.set_velocities(
                fin.variables['velocities'][:] / (1000 * units.fs))

        if 'cell_lengths' in fin.variables:
            a = fin.variables['cell_lengths'][0]
            b = fin.variables['cell_lengths'][1]
            c = fin.variables['cell_lengths'][2]

            alpha = fin.variables['cell_angles'][0]
            beta = fin.variables['cell_angles'][1]
            gamma = fin.variables['cell_angles'][2]

            if (all(angle > 89.99 for angle in [alpha, beta, gamma]) and
                    all(angle < 90.01 for angle in [alpha, beta, gamma])):
                atoms.set_cell(
                    np.array([[a, 0, 0],
                              [0, b, 0],
                              [0, 0, c]]))
                atoms.set_pbc(True)
            else:
                raise NotImplementedError('only rectangular cells are'
                                          ' implemented in ASE-AMBER')

        else:
            atoms.set_pbc(False)

    def read_energy(self, filename='mden'):
        """ read total energy from amber file """
        lines = open(filename, 'r').readlines()
        self.results['energy'] = \
            float(lines[16].split()[2]) * units.kcal / units.mol

    def read_forces(self, filename='mdfrc'):
        """ read forces from amber file """
        f = netcdf.netcdf_file(filename, 'r')
        forces = f.variables['forces']
        self.results['forces'] = forces[-1, :, :] \
            / units.Ang * units.kcal / units.mol
        f.close()

    def set_charges(self, selection, charges, parmed_filename=None):
        """ Modify amber topology charges to contain the updated
            QM charges, needed in QM/MM.
            Using amber's parmed program to change charges.
        """
        qm_list = list(selection)
        fout = open(parmed_filename, 'w')
        fout.write('# update the following QM charges \n')
        for i, charge in zip(qm_list, charges):
            fout.write('change charge @' + str(i + 1) + ' ' +
                       str(charge) + ' \n')
        fout.write('# Output the topology file \n')
        fout.write('outparm ' + self.topologyfile + ' \n')
        fout.close()
        parmed_command = ('parmed -O -i ' + parmed_filename +
                          ' -p ' + self.topologyfile +
                          ' > ' + self.topologyfile + '.log 2>&1')
        olddir = os.getcwd()
        try:
            os.chdir(self.directory)
            errorcode = subprocess.call(parmed_command, shell=True)
        finally:
            os.chdir(olddir)
        if errorcode:
            raise RuntimeError('%s returned an error: %d' %
                               (self.label, errorcode))

    def get_virtual_charges(self, atoms):
        topology = open(self.topologyfile, 'r').readlines()
        for n, line in enumerate(topology):
            if '%FLAG CHARGE' in line:
                chargestart = n + 2
        lines1 = topology[chargestart:(chargestart
                                       + (len(atoms)-1)//5 + 1)]
        mm_charges = []
        for line in lines1:
            for el in line.split():
                mm_charges.append(float(el)/18.2223)
        charges = np.array(mm_charges)
        return charges

    def add_virtual_sites(self, positions):
        return positions  # no virtual sites

    def redistribute_forces(self, forces):
        return forces


def map(atoms, top):
    p = np.zeros((2, len(atoms)), dtype="int")

    elements = atoms.get_chemical_symbols()
    unique_elements = np.unique(atoms.get_chemical_symbols())

    for i in range(len(unique_elements)):
        idx = 0
        for j in range(len(atoms)):
            if elements[j] == unique_elements[i]:
                idx += 1
                symbol = unique_elements[i] + np.str(idx)
                for k in range(len(atoms)):
                    if top.atoms[k].name == symbol:
                        p[0, k] = j
                        p[1, j] = k
                        break
    return p

try:
    import sander
    have_sander = True
except ImportError:
    have_sander = False


class SANDER(Calculator):
    """
    Interface to SANDER using Python interface

    Requires sander Python bindings from http://ambermd.org/
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, atoms=None, label=None, top=None, crd=None,
                 mm_options=None, qm_options=None, permutation=None, **kwargs):
        if not have_sander:
            raise RuntimeError("sander Python module could not be imported!")
        Calculator.__init__(self, label, atoms)
        self.permutation = permutation
        if qm_options is not None:
            sander.setup(top, crd.coordinates, crd.box, mm_options, qm_options)
        else:
            sander.setup(top, crd.coordinates, crd.box, mm_options)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if system_changes:
            if 'energy' in self.results:
                del self.results['energy']
            if 'forces' in self.results:
                del self.results['forces']
        if 'energy' not in self.results:
            if self.permutation is None:
                crd = np.reshape(atoms.get_positions(), (1, len(atoms), 3))
            else:
                crd = np.reshape(atoms.get_positions()
                                 [self.permutation[0, :]], (1, len(atoms), 3))
            sander.set_positions(crd)
            e, f = sander.energy_forces()
            self.results['energy'] = e.tot * units.kcal / units.mol
            if self.permutation is None:
                self.results['forces'] = (np.reshape(np.array(f),
                                                     (len(atoms), 3)) *
                                          units.kcal / units.mol)
            else:
                ff = np.reshape(np.array(f), (len(atoms), 3)) * \
                    units.kcal / units.mol
                self.results['forces'] = ff[self.permutation[1, :]]
        if 'forces' not in self.results:
            if self.permutation is None:
                crd = np.reshape(atoms.get_positions(), (1, len(atoms), 3))
            else:
                crd = np.reshape(atoms.get_positions()[self.permutation[0, :]],
                                 (1, len(atoms), 3))
            sander.set_positions(crd)
            e, f = sander.energy_forces()
            self.results['energy'] = e.tot * units.kcal / units.mol
            if self.permutation is None:
                self.results['forces'] = (np.reshape(np.array(f),
                                                     (len(atoms), 3)) *
                                          units.kcal / units.mol)
            else:
                ff = np.reshape(np.array(f), (len(atoms), 3)) * \
                    units.kcal / units.mol
                self.results['forces'] = ff[self.permutation[1, :]]
