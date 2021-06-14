"""This module defines an ASE interface to DMol3.

Contacts
--------
Adam Arvidsson <adam.arvidsson@chalmers.se>
Erik Fransson  <erikfr@chalmers.se>
Anders Hellman <anders.hellman@chalmers.se>


DMol3 environment variables
----------------------------
DMOL_COMMAND should point to the RunDmol script and specify the number of cores
to prallelize over

export DMOL_COMMAND="./RunDmol.sh -np 16"


Example
--------
>>> from ase.build import bulk
>>> from ase.calculators import DMol3

>>> atoms = bulk('Al','fcc')
>>> calc = DMol3()
>>> atoms.set_calculator(calc)
>>> print 'Potential energy %5.5f eV' % atoms.get_potential_energy()


DMol3 calculator functionality
-------------------------------
This calculator does support all the functionality in DMol3.

Firstly this calculator is limited to only handling either fully
periodic structures (pbc = [1,1,1]) or non periodic structures (pbc=[0,0,0]).

Internal relaxations are not supported by the calculator,
only support for energy and forces is implemented.

Reading eigenvalues and kpts are supported.
Be careful with kpts and their directions (see internal coordinates below).

Outputting the full electron density or specific bands to .grd files can be
acheived with the plot command. The .grd files can be converted to the cube
format using grd_to_cube().


DMol3 internal coordinates
---------------------------
DMol3 may change the atomic positions / cell vectors in order to satisfy
certain criterion ( e.g. molecule symmetry axis along z ). Specifically this
happens when using Symmetry on/auto. This means the forces read from .grad
will be in a different coordinates system compared to the atoms object used.
To solve this the rotation matrix that converts the dmol coordinate system
to the ase coordinate system is found and applied to the forces.

For non periodic structures (pbc=[0,0,0]) the rotation matrix can be directly
parsed from the .rot file.
For fully periodic structures the rotation matrix is found by reading the
cell vectors and positions used by dmol and then solving the matrix problem
DMol_atoms * rot_mat = ase_atoms


DMol3 files
------------
The supported DMol3 file formats are:

car    structure file - Angstrom and cellpar description of cell.
incoor structure file - Bohr and cellvector describption of cell.
                        Note: incoor file not used if car file present.
outmol outfile from DMol3 - atomic units (Bohr and Hartree)
grad   outfile for forces from DMol3 - forces in Hartree/Bohr
grd    outfile for orbitals from DMol3 - cellpar in Angstrom

"""

from __future__ import print_function
import os
import re
import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.dmol import write_dmol_car, write_dmol_incoor
from ase.units import Hartree, Bohr
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError


class DMol3(FileIOCalculator):
    """ DMol3 calculator object. """

    implemented_properties = ['energy', 'forces']
    default_parameters = {'functional': 'pbe',
                          'symmetry': 'on'}

    if 'DMOL_COMMAND' in os.environ:
        command = os.environ['DMOL_COMMAND'] + ' PREFIX > PREFIX.out'
    else:
        command = None

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='dmol_calc/tmp', atoms=None, **kwargs):
        """ Construct DMol3 calculator. """
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

        # tracks if DMol transformed coordinate system
        self.internal_transformation = False

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def write_input(self, atoms, properties=None, system_changes=None):

        if not (np.all(atoms.pbc) or not np.any(atoms.pbc)):
            raise RuntimeError('PBC must be all true or all false')

        self.clean()   # Remove files from old run
        self.internal_transformation = False
        self.ase_positions = atoms.positions.copy()
        self.ase_cell = atoms.cell.copy()

        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        if np.all(atoms.pbc):
            write_dmol_incoor(self.label + '.incoor', atoms)
        elif not np.any(atoms.pbc):
            write_dmol_car(self.label + '.car', atoms)

        self.write_input_file()
        self.parameters.write(self.label + '.parameters.ase')

    def write_input_file(self):
        """ Writes the input file. """

        f = open(self.label + '.input', 'w')
        f.write('%-32s %s\n' % ('calculate', 'gradient'))

        # if no key about eigs
        f.write('%-32s %s\n' % ('print', 'eigval_last_it'))

        for key, value in self.parameters.items():
            if isinstance(value, str):
                f.write('%-32s %s\n' % (key, value))
            elif isinstance(value, (list, tuple)):
                for val in value:
                    f.write('%-32s %s\n' % (key, val))
            else:
                f.write('%-32s %r\n' % (key, value))

    def read(self, label):
        FileIOCalculator.read(self, label)
        geometry = self.label + '.car'
        output = self.label + '.outmol'
        force = self.label + '.grad'

        for filename in [force, output, geometry]:
            if not os.path.isfile(filename):
                raise ReadError

        self.atoms = read(geometry)
        self.parameters = Parameters.read(self.label + 'parameters.ase')
        self.read_results()

    def read_results(self):
        finished, message = self.finished_sucessfully()
        if not finished:
            raise RuntimeError('DMol3 run failed, see outmol file for'
                               ' more info\n\n%s' % message)

        self.find_dmol_transformation()
        self.read_energy()
        self.read_forces()

    def finished_sucessfully(self):
        """ Reads outmol file and checks if job completed or failed.

        Returns
        -------
        finished (bool): True if job completed, False if something went wrong
        message (str): If job failed message contains parsed errors, else empty

        """
        finished = False
        message = ""
        for line in open(self.label + '.outmol', 'r'):
            if line.rfind('Message: DMol3 job finished successfully') > -1:
                finished = True
            if line.startswith('Error'):
                message += line
        return finished, message

    def find_dmol_transformation(self, tol=1e-4):
        """Finds rotation matrix that takes us from DMol internal
        coordinates to ase coordinates.

        For pbc = [False, False, False]  the rotation matrix is parsed from
        the .rot file, if this file doesnt exist no rotation is needed.

        For pbc = [True, True, True] the Dmol internal cell vectors and
        positions are parsed and compared to self.ase_cell self.ase_positions.
        The rotation matrix can then be found by a call to the helper
        function find_transformation(atoms1, atoms2)

        If a rotation matrix is needed then self.internal_transformation is
        set to True and the rotation matrix is stored in self.rotation_matrix

        Parameters
        ----------
        tol (float): tolerance for check if positions and cell are the same
        """

        if np.all(self.atoms.pbc):  # [True, True, True]
            dmol_atoms = self.read_atoms_from_outmol()
            if (np.linalg.norm(self.atoms.positions - dmol_atoms.positions) <
                    tol) and (np.linalg.norm(self.atoms.cell - dmol_atoms.cell) < tol):
                self.internal_transformation = False
            else:
                R, err = find_transformation(dmol_atoms, self.atoms)
                if abs(np.linalg.det(R) - 1.0) > tol:
                    raise RuntimeError('Error: transformation matrix does'
                                       ' not have determinant 1.0')
                if err < tol:
                    self.internal_transformation = True
                    self.rotation_matrix = R
                else:
                    raise RuntimeError('Error: Could not find dmol'
                                       ' coordinate transformation')
        elif not np.any(self.atoms.pbc):  # [False,False,False]
            try:
                data = np.loadtxt(self.label + '.rot')
            except IOError:
                self.internal_transformation = False
            else:
                self.internal_transformation = True
                self.rotation_matrix = data[1:].transpose()

    def read_atoms_from_outmol(self):
        """ Reads atomic positions and cell from outmol file and returns atoms
        object.

        If no cell vectors are found in outmol the cell is set to np.eye(3) and
        pbc 000.

        Formatting for cell in outmol :
         translation vector [a0]    1    5.1    0.0    5.1
         translation vector [a0]    2    5.1    5.1    0.0
         translation vector [a0]    3    0.0    5.1    5.1

        Formatting for positions in outmol:
        df              ATOMIC  COORDINATES (au)
        df            x          y          z
        df   Si     0.0   0.0   0.0
        df   Si     1.3   3.5   2.2
        df  binding energy      -0.2309046Ha

        Returns
        -------
        atoms (Atoms object): read atoms object
        """

        lines = open(self.label + '.outmol', 'r').readlines()
        found_cell = False
        cell = np.zeros((3, 3))
        symbols = []
        positions = []
        pattern_translation_vectors = re.compile(r'\s+translation\s+vector')
        pattern_atomic_coordinates = re.compile(r'df\s+ATOMIC\s+COORDINATES')

        for i, line in enumerate(lines):
            if pattern_translation_vectors.match(line):
                cell[int(line.split()[3]) - 1, :] = \
                    np.array([float(x) for x in line.split()[-3:]])
                found_cell = True
            if pattern_atomic_coordinates.match(line):
                for ind, j in enumerate(range(i + 2, i + 2 + len(self.atoms))):
                    flds = lines[j].split()
                    symbols.append(flds[1])
                    positions.append(flds[2:5])
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell)
        atoms.positions *= Bohr
        atoms.cell *= Bohr

        if found_cell:
            atoms.pbc = [True, True, True]
            atoms.wrap()
        else:
            atoms.pbc = [False, False, False]
        return atoms

    def read_energy(self):
        """ Find and return last occurrence of Ef in outmole file. """
        energy_regex = re.compile(r'^Ef\s+(\S+)Ha')
        found = False
        for line in open(self.label + '.outmol', 'r'):
            match = energy_regex.match(line)
            if match:
                energy = float(match.group(1))
                found = True
        if not found:
            raise RuntimeError('Could not read energy from outmol')
        self.results['energy'] = energy * Hartree

    def read_forces(self):
        """ Read forces from .grad file. Applies self.rotation_matrix if
        self.internal_transformation is True. """
        lines = open(self.label + '.grad', 'r').readlines()
        forces = []
        for i, line in enumerate(lines):
            if line.startswith('$gradients'):
                for j in range(i + 1, i + 1 + len(self.atoms)):
                    # force = - grad(Epot)
                    forces.append(np.array(
                        [-float(x) for x in lines[j].split()[1:4]]))

        forces = np.array(forces) * Hartree / Bohr
        if self.internal_transformation:
            forces = np.dot(forces, self.rotation_matrix)
        self.results['forces'] = forces

    def get_eigenvalues(self, kpt=0, spin=0):
        return self.read_eigenvalues(kpt, spin, 'eigenvalues')

    def get_occupations(self, kpt=0, spin=0):
        return self.read_eigenvalues(kpt, spin, 'occupations')

    def get_k_point_weights(self):
        return self.read_kpts(mode='k_point_weights')

    def get_bz_k_points(self):
        raise NotImplementedError

    def get_ibz_k_points(self):
        return self.read_kpts(mode='ibz_k_points')

    def get_spin_polarized(self):
        return self.read_spin_polarized()

    def get_fermi_level(self):
        return self.read_fermi()

    def get_energy_contributions(self):
        return self.read_energy_contributions()

    def get_xc_functional(self):
        return self.parameters['functional']

    def read_eigenvalues(self, kpt=0, spin=0, mode='eigenvalues'):
        """Reads eigenvalues from .outmol file.

        This function splits into two situations:
        1. We have no kpts just the raw eigenvalues ( Gamma point )
        2. We have eigenvalues for each k-point

        If calculation is spin_restricted then all eigenvalues
        will be returned no matter what spin parameter is set to.

        If calculation has no kpts then all eigenvalues
        will be returned no matter what kpts parameter is set to.

        Note DMol does usually NOT print all unoccupied eigenvalues.
        Meaning number of eigenvalues for different kpts can vary.
        """

        assert mode in ['eigenvalues', 'occupations']
        lines = open(self.label + '.outmol', 'r').readlines()
        pattern_kpts = re.compile(r'Eigenvalues for kvector\s+%d' % (kpt + 1))
        for n, line in enumerate(lines):

            # 1. We have no kpts
            if line.split() == ['state', 'eigenvalue', 'occupation']:
                spin_key = '+'
                if self.get_spin_polarized():
                    if spin == 1:
                        spin_key = '-'
                val_index = -2
                if mode == 'occupations':
                    val_index = -1
                values = []
                m = n + 3
                while True:
                    if lines[m].strip() == '':
                        break
                    flds = lines[m].split()
                    if flds[1] == spin_key:
                        values.append(float(flds[val_index]))
                    m += 1
                return np.array(values)

            # 2. We have kpts
            if pattern_kpts.match(line):
                val_index = 3
                if self.get_spin_polarized():
                    if spin == 1:
                        val_index = 6
                if mode == 'occupations':
                    val_index += 1
                values = []
                m = n + 2
                while True:
                    if lines[m].strip() == '':
                        break
                    values.append(float(lines[m].split()[val_index]))
                    m += 1
                return np.array(values)
        return None

    def read_kpts(self, mode='ibz_k_points'):
        """ Returns list of kpts coordinates or kpts weights.  """

        assert mode in ['ibz_k_points', 'k_point_weights']
        lines = open(self.label + '.outmol', 'r').readlines()

        values = []
        for n, line in enumerate(lines):
            if line.startswith('Eigenvalues for kvector'):
                if mode == 'ibz_k_points':
                    values.append([float(k_i)
                                   for k_i in lines[n].split()[4:7]])
                if mode == 'k_point_weights':
                    values.append(float(lines[n].split()[8]))
        if values == []:
            return None
        return values

    def read_spin_polarized(self):
        """Reads, from outmol file, if calculation is spin polarized."""

        lines = open(self.label + '.outmol', 'r').readlines()
        for n, line in enumerate(lines):
            if line.rfind('Calculation is Spin_restricted') > -1:
                return False
            if line.rfind('Calculation is Spin_unrestricted') > -1:
                return True
        raise IOError('Could not read spin restriction from outmol')

    def read_fermi(self):
        """Reads the Fermi level.

        Example line in outmol:
        Fermi Energy:           -0.225556 Ha     -6.138 eV   xyz text
        """
        lines = open(self.label + '.outmol', 'r').readlines()
        pattern_fermi = re.compile(r'Fermi Energy:\s+(\S+)\s+Ha')
        for line in lines:
            m = pattern_fermi.match(line)
            if m:
                return float(m.group(1)) * Hartree
        return None

    def read_energy_contributions(self):
        """Reads the different energy contributions."""

        lines = open(self.label + '.outmol', 'r').readlines()
        energies = dict()
        for n, line in enumerate(lines):
            if line.startswith('Energy components'):
                m = n + 1
                while not lines[m].strip() == '':
                    energies[lines[m].split('=')[0].strip()] = \
                        float(re.findall(
                            r"[-+]?\d*\.\d+|\d+", lines[m])[0]) * Hartree
                    m += 1
        return energies

    def clean(self):
        """ Cleanup after dmol calculation

        Only removes dmol files in self.directory,
        does not remove the directory itself
        """
        file_extensions = ['basis', 'car', 'err', 'grad', 'input', 'inatm',
                           'incoor', 'kpoints', 'monitor', 'occup', 'outmol',
                           'outatom', 'rot', 'sdf', 'sym', 'tpotl', 'tpdensk',
                           'torder', 'out', 'parameters.ase']
        files_to_clean = ['DMol3.log', 'stdouterr.txt', 'mpd.hosts']

        files = [os.path.join(self.directory, f) for f in files_to_clean]
        files += [''.join((self.label, '.', ext)) for ext in file_extensions]

        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass


# Helper functions
# ------------------

def find_transformation(atoms1, atoms2, verbose=False, only_cell=False):
    """ Solves Ax = B where A and B are cell and positions from atoms objects.

    Uses numpys least square solver to solve the problem Ax = B where A and
    B are cell vectors and positions for atoms1 and atoms2 respectively.

    Parameters
    ----------
    atoms1 (Atoms object): First atoms object (A)
    atoms2 (Atoms object): Second atoms object (B)
    verbose (bool): If True prints for each i A[i], B[i], Ax[i]
    only_cell (bool): If True only cell in used, otherwise cell and positions.

    Returns
    -------
    x (np.array((3,3))): Least square solution to Ax = B
    error (float): The error calculated as np.linalg.norm(Ax-b)

    """

    if only_cell:
        N = 3
    elif atoms1.get_number_of_atoms() != atoms2.get_number_of_atoms():
        raise RuntimeError('Atoms object must be of same length')
    else:
        N = atoms1.get_number_of_atoms() + 3

    # Setup matrices A and B
    A = np.zeros((N, 3))
    B = np.zeros((N, 3))
    A[0:3, :] = atoms1.cell
    B[0:3, :] = atoms2.cell
    if not only_cell:
        A[3:, :] = atoms1.positions
        B[3:, :] = atoms2.positions

    # Solve least square problem Ax = B
    lstsq_fit = np.linalg.lstsq(A, B, rcond=-1)
    x = lstsq_fit[0]
    error = np.linalg.norm(np.dot(A, x) - B)

    # Print comparision between A, B and Ax
    if verbose:
        print('%17s %33s %35s %24s' % ('A', 'B', 'Ax', '|Ax-b|'))
        for a, b in zip(A, B):
            ax = np.dot(a, x)
            loss = np.linalg.norm(ax - b)
            print('(', end='')
            for a_i in a:
                print('%8.5f' % a_i, end='')
            print(')   (', end='')
            for b_i in b:
                print('%8.5f ' % b_i, end='')
            print(')   (', end='')
            for ax_i in ax:
                print('%8.5f ' % ax_i, end='')
            print(')   %8.5f' % loss)

    return x, error


def grd_to_file(atoms, grd_file, new_file):
    """ Reads grd_file and converts data to cube format and writes to
    cube_file.

    Note: content of grd_file and atoms object are assumed to match with the
          same orientation.

    Parameters
    -----------
    atoms (Atoms object): atoms object grd_file data is for
    grd_file (str): filename of .grd file
    new_file (str): filename to write grd-data to, must be ASE format
                    that supports data argument
    """
    from ase.io import write

    atoms_copy = atoms.copy()
    data, cell, origin = read_grd(grd_file)
    atoms_copy.cell = cell
    atoms_copy.positions += origin
    write(new_file, atoms_copy, data=data)


def read_grd(filename):
    """ Reads .grd file

    Notes
    -----
    origin_xyz is offset with half a grid point in all directions to be
        compatible with the cube format
    Periodic systems is not guaranteed to be oriented correctly
    """
    from ase.geometry.cell import cellpar_to_cell

    lines = open(filename, 'r').readlines()

    cell_data = np.array([float(fld) for fld in lines[2].split()])
    cell = cellpar_to_cell(cell_data)
    grid = [int(fld) + 1 for fld in lines[3].split()]
    data = np.empty(grid)

    origin_data = [int(fld) for fld in lines[4].split()[1:]]
    origin_xyz = cell[0] * (-float(origin_data[0])-0.5) / (grid[0] - 1) + \
        cell[1] * (-float(origin_data[2])-0.5) / (grid[1] - 1) + \
        cell[2] * (-float(origin_data[4])-0.5) / (grid[2] - 1)

    # Fastest index describes which index ( x or y ) varies fastest
    # 1: x  , 3: y
    fastest_index = int(lines[4].split()[0])
    assert fastest_index in [1, 3]
    if fastest_index == 3:
        grid[0], grid[1] = grid[1], grid[0]

    dummy_counter = 5
    for i in range(grid[2]):
        for j in range(grid[1]):
            for k in range(grid[0]):  # Fastest index
                if fastest_index == 1:
                    data[k, j, i] = float(lines[dummy_counter])
                elif fastest_index == 3:
                    data[j, k, i] = float(lines[dummy_counter])
                dummy_counter += 1

    return data, cell, origin_xyz


if __name__ == '__main__':
    from ase.build import molecule

    atoms = molecule('H2')
    calc = DMol3()
    atoms.set_calculator(calc)
    # ~ 60 sec calculation
    print('Potential energy %5.5f eV' % atoms.get_potential_energy())
