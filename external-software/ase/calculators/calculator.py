import os
import copy
import subprocess
from math import pi, sqrt

import numpy as np

from ase.dft.kpoints import bandpath, monkhorst_pack


class CalculatorError(RuntimeError):
    """Base class of error types related to ASE calculators."""


class CalculatorSetupError(CalculatorError):
    """Calculation cannot be performed with the given parameters.

    Reasons to raise this errors are:
      * The calculator is not properly configured
        (missing executable, environment variables, ...)
      * The given atoms object is not supported
      * Calculator parameters are unsupported

    Typically raised before a calculation."""


class EnvironmentError(CalculatorSetupError):
    """Raised if calculator is not properly set up with ASE.

    May be missing an executable or environment variables."""


class InputError(CalculatorSetupError):
    """Raised if inputs given to the calculator were incorrect.

    Bad input keywords or values, or missing pseudopotentials.

    This may be raised before or during calculation, depending on
    when the problem is detected."""


class CalculationFailed(CalculatorError):
    """Calculation failed unexpectedly.

    Reasons to raise this error are:
      * Calculation did not converge
      * Calculation ran out of memory
      * Segmentation fault or other abnormal termination
      * Arithmetic trouble (singular matrices, NaN, ...)

    Typically raised during calculation."""


class SCFError(CalculationFailed):
    """SCF loop did not converge."""


class ReadError(CalculatorError):
    """Unexpected irrecoverable error while reading calculation results."""


class PropertyNotImplementedError(NotImplementedError):
    """Raised if a calculator does not implement the requested property."""


class PropertyNotPresent(CalculatorError):
    """Requested property is missing.

    Maybe it was never calculated, or for some reason was not extracted
    with the rest of the results, without being a fatal ReadError."""


def compare_atoms(atoms1, atoms2, tol=1e-15):
    """Check for system changes since last calculation."""
    if atoms1 is None:
        system_changes = all_changes[:]
    else:
        system_changes = []
        if not equal(atoms1.positions, atoms2.positions, tol):
            system_changes.append('positions')
        if not equal(atoms1.numbers, atoms2.numbers):
            system_changes.append('numbers')
        if not equal(atoms1.cell, atoms2.cell, tol):
            system_changes.append('cell')
        if not equal(atoms1.pbc, atoms2.pbc):
            system_changes.append('pbc')
        if not equal(atoms1.get_initial_magnetic_moments(),
                     atoms2.get_initial_magnetic_moments(), tol):
            system_changes.append('initial_magmoms')
        if not equal(atoms1.get_initial_charges(),
                     atoms2.get_initial_charges(), tol):
            system_changes.append('initial_charges')

    return system_changes


all_properties = ['energy', 'forces', 'stress', 'dipole',
                  'charges', 'magmom', 'magmoms', 'free_energy']


all_changes = ['positions', 'numbers', 'cell', 'pbc',
               'initial_charges', 'initial_magmoms']


# Recognized names of calculators sorted alphabetically:
names = ['abinit', 'aims', 'amber', 'asap', 'castep', 'cp2k', 'crystal',
         'demon', 'dftb', 'dmol', 'eam', 'elk', 'emt', 'espresso',
         'exciting', 'fleur', 'gaussian', 'gpaw', 'gromacs', 'gulp',
         'hotbit', 'jacapo', 'lammpsrun',
         'lammpslib', 'lj', 'mopac', 'morse', 'nwchem', 'octopus', 'onetep',
         'openmx', 'siesta', 'tip3p', 'turbomole', 'vasp']


special = {'cp2k': 'CP2K',
           'dmol': 'DMol3',
           'eam': 'EAM',
           'elk': 'ELK',
           'emt': 'EMT',
           'crystal': 'CRYSTAL',
           'fleur': 'FLEUR',
           'gulp': 'GULP',
           'lammpsrun': 'LAMMPS',
           'lammpslib': 'LAMMPSlib',
           'lj': 'LennardJones',
           'mopac': 'MOPAC',
           'morse': 'MorsePotential',
           'nwchem': 'NWChem',
           'openmx': 'OpenMX',
           'tip3p': 'TIP3P'}


def get_calculator(name):
    """Return calculator class."""
    if name == 'asap':
        from asap3 import EMT as Calculator
    elif name == 'gpaw':
        from gpaw import GPAW as Calculator
    elif name == 'hotbit':
        from hotbit import Calculator
    elif name == 'vasp2':
        from ase.calculators.vasp import Vasp2 as Calculator
    else:
        classname = special.get(name, name.title())
        module = __import__('ase.calculators.' + name, {}, None, [classname])
        Calculator = getattr(module, classname)
    return Calculator


def equal(a, b, tol=None):
    """ndarray-enabled comparison function."""
    if isinstance(a, np.ndarray):
        b = np.array(b)
        if a.shape != b.shape:
            return False
        if tol is None:
            return (a == b).all()
        else:
            return np.allclose(a, b, rtol=tol, atol=tol)
    if isinstance(b, np.ndarray):
        return equal(b, a, tol)
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(equal(a[key], b[key], tol) for key in a.keys())
    if tol is None:
        return a == b
    return abs(a - b) < tol * abs(b) + tol


def kptdensity2monkhorstpack(atoms, kptdensity=3.5, even=True):
    """Convert k-point density to Monkhorst-Pack grid size.

    atoms: Atoms object
        Contains unit cell and information about boundary conditions.
    kptdensity: float
        Required k-point density.  Default value is 3.5 point per Ang^-1.
    even: bool
        Round up to even numbers.
    """

    recipcell = atoms.get_reciprocal_cell()
    kpts = []
    for i in range(3):
        if atoms.pbc[i]:
            k = 2 * pi * sqrt((recipcell[i]**2).sum()) * kptdensity
            if even:
                kpts.append(2 * int(np.ceil(k / 2)))
            else:
                kpts.append(int(np.ceil(k)))
        else:
            kpts.append(1)
    return np.array(kpts)


def kpts2mp(atoms, kpts, even=False):
    if kpts is None:
        return np.array([1, 1, 1])
    if isinstance(kpts, (float, int)):
        return kptdensity2monkhorstpack(atoms, kpts, even)
    else:
        return kpts


def kpts2sizeandoffsets(size=None, density=None, gamma=None, even=None,
                        atoms=None):
    """Helper function for selecting k-points.

    Use either size or density.

    size: 3 ints
        Number of k-points.
    density: float
        K-point density in units of k-points per Ang^-1.
    gamma: None or bool
        Should the Gamma-point be included?  Yes / no / don't care:
        True / False / None.
    even: None or bool
        Should the number of k-points be even?  Yes / no / don't care:
        True / False / None.
    atoms: Atoms object
        Needed for calculating k-point density.

    """

    if size is None:
        if density is None:
            size = [1, 1, 1]
        else:
            size = kptdensity2monkhorstpack(atoms, density, even)

    offsets = [0, 0, 0]

    if gamma is not None:
        for i, s in enumerate(size):
            if atoms.pbc[i] and s % 2 != bool(gamma):
                offsets[i] = 0.5 / s

    return size, offsets


def kpts2ndarray(kpts, atoms=None):
    """Convert kpts keyword to 2-d ndarray of scaled k-points."""

    if kpts is None:
        return np.zeros((1, 3))

    if isinstance(kpts, dict):
        if 'path' in kpts:
            return bandpath(cell=atoms.cell, **kpts)[0]
        size, offsets = kpts2sizeandoffsets(atoms=atoms, **kpts)
        return monkhorst_pack(size) + offsets

    if isinstance(kpts[0], int):
        return monkhorst_pack(kpts)

    return np.array(kpts)


class EigenvalOccupationMixin:
    """Define 'eigenvalues' and 'occupations' properties on class.

    eigenvalues and occupations will be arrays of shape (spin, kpts, nbands).

    Classes must implement the old-fashioned get_eigenvalues and
    get_occupations methods."""

    @property
    def eigenvalues(self):
        return self.build_eig_occ_array(self.get_eigenvalues)

    @property
    def occupations(self):
        return self.build_eig_occ_array(self.get_occupation_numbers)

    def build_eig_occ_array(self, getter):
        nspins = self.get_number_of_spins()
        nkpts = len(self.get_ibz_k_points())
        nbands = self.get_number_of_bands()
        arr = np.zeros((nspins, nkpts, nbands))
        for s in range(nspins):
            for k in range(nkpts):
                arr[s, k, :] = getter(spin=s, kpt=k)
        return arr


class Parameters(dict):
    """Dictionary for parameters.

    Special feature: If param is a Parameters instance, then param.xc
    is a shorthand for param['xc'].
    """

    def __getattr__(self, key):
        if key not in self:
            return dict.__getattribute__(self, key)
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    @classmethod
    def read(cls, filename):
        """Read parameters from file."""
        file = open(os.path.expanduser(filename))
        parameters = cls(eval(file.read()))
        file.close()
        return parameters

    def tostring(self):
        keys = sorted(self)
        return 'dict(' + ',\n     '.join(
            '{}={!r}'.format(key, self[key]) for key in keys) + ')\n'

    def write(self, filename):
        file = open(filename, 'w')
        file.write(self.tostring())
        file.close()


class Calculator(object):
    """Base-class for all ASE calculators.

    A calculator must raise PropertyNotImplementedError if asked for a
    property that it can't calculate.  So, if calculation of the
    stress tensor has not been implemented, get_stress(atoms) should
    raise PropertyNotImplementedError.  This can be achieved simply by not
    including the string 'stress' in the list implemented_properties
    which is a class member.  These are the names of the standard
    properties: 'energy', 'forces', 'stress', 'dipole', 'charges',
    'magmom' and 'magmoms'.
    """

    implemented_properties = []
    'Properties calculator can handle (energy, forces, ...)'

    default_parameters = {}
    'Default parameters'

    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None, **kwargs):
        """Basic calculator implementation.

        restart: str
            Prefix for restart file.  May contain a directory.  Default
            is None: don't restart.
        ignore_bad_restart_file: bool
            Ignore broken or missing restart file.  By default, it is an
            error if the restart file is missing or broken.
        label: str
            Name used for all files.  May contain a directory.
        atoms: Atoms object
            Optional Atoms object to which the calculator will be
            attached.  When restarting, atoms will get its positions and
            unit-cell updated from file.
        """
        self.atoms = None  # copy of atoms object from last calculation
        self.results = {}  # calculated properties (energy, forces, ...)
        self.parameters = None  # calculational parameters

        if restart is not None:
            try:
                self.read(restart)  # read parameters, atoms and results
            except ReadError:
                if ignore_bad_restart_file:
                    self.reset()
                else:
                    raise

        self.label = None
        self.directory = None
        self.prefix = None

        self.set_label(label)

        if self.parameters is None:
            # Use default parameters if they were not read from file:
            self.parameters = self.get_default_parameters()

        if atoms is not None:
            atoms.calc = self
            if self.atoms is not None:
                # Atoms were read from file.  Update atoms:
                if not (equal(atoms.numbers, self.atoms.numbers) and
                        (atoms.pbc == self.atoms.pbc).all()):
                    raise CalculatorError('Atoms not compatible with file')
                atoms.positions = self.atoms.positions
                atoms.cell = self.atoms.cell

        self.set(**kwargs)

        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()

    def set_label(self, label):
        """Set label and convert label to directory and prefix.

        Examples:

        * label='abc': (directory='.', prefix='abc')
        * label='dir1/abc': (directory='dir1', prefix='abc')

        Calculators that must write results to files with fixed names
        can overwrite this method so that the directory is set to all
        of label."""

        self.label = label

        if label is None:
            self.directory = None
            self.prefix = None
        else:
            self.directory, self.prefix = os.path.split(label)
            if self.directory == '':
                self.directory = os.curdir

    def get_default_parameters(self):
        return Parameters(copy.deepcopy(self.default_parameters))

    def todict(self, skip_default=True):
        defaults = self.get_default_parameters()
        dct = {}
        for key, value in self.parameters.items():
            if hasattr(value, 'todict'):
                value = value.todict()
            if skip_default:
                default = defaults.get(key, '_no_default_')
                if default != '_no_default_' and equal(value, default):
                    continue
            dct[key] = value
        return dct

    def reset(self):
        """Clear all information from old calculation."""

        self.atoms = None
        self.results = {}

    def read(self, label):
        """Read atoms, parameters and calculated properties from output file.

        Read result from self.label file.  Raise ReadError if the file
        is not there.  If the file is corrupted or contains an error
        message from the calculation, a ReadError should also be
        raised.  In case of succes, these attributes must set:

        atoms: Atoms object
            The state of the atoms from last calculation.
        parameters: Parameters object
            The parameter dictionary.
        results: dict
            Calculated properties like energy and forces.

        The FileIOCalculator.read() method will typically read atoms
        and parameters and get the results dict by calling the
        read_results() method."""

        self.set_label(label)

    def get_atoms(self):
        if self.atoms is None:
            raise ValueError('Calculator has no atoms')
        atoms = self.atoms.copy()
        atoms.calc = self
        return atoms

    @classmethod
    def read_atoms(cls, restart, **kwargs):
        return cls(restart=restart, label=restart, **kwargs).get_atoms()

    def set(self, **kwargs):
        """Set parameters like set(key1=value1, key2=value2, ...).

        A dictionary containing the parameters that have been changed
        is returned.

        Subclasses must implement a set() method that will look at the
        chaneged parameters and decide if a call to reset() is needed.
        If the changed parameters are harmless, like a change in
        verbosity, then there is no need to call reset().

        The special keyword 'parameters' can be used to read
        parameters from a file."""

        if 'parameters' in kwargs:
            filename = kwargs.pop('parameters')
            parameters = Parameters.read(filename)
            parameters.update(kwargs)
            kwargs = parameters

        changed_parameters = {}

        for key, value in kwargs.items():
            oldvalue = self.parameters.get(key)
            if key not in self.parameters or not equal(value, oldvalue):
                changed_parameters[key] = value
                self.parameters[key] = value

        return changed_parameters

    def check_state(self, atoms, tol=1e-15):
        """Check for system changes since last calculation."""
        return compare_atoms(self.atoms, atoms)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        energy = self.get_property('energy', atoms)
        if force_consistent:
            if 'free_energy' not in self.results:
                name = self.__class__.__name__
                # XXX but we don't know why the energy is not there.
                # We should raise PropertyNotPresent.  Discuss
                raise PropertyNotImplementedError(
                    'Force consistent/free energy ("free_energy") '
                    'not provided by {0} calculator'.format(name))
            return self.results['free_energy']
        else:
            return energy

    def get_forces(self, atoms=None):
        return self.get_property('forces', atoms)

    def get_stress(self, atoms=None):
        return self.get_property('stress', atoms)

    def get_dipole_moment(self, atoms=None):
        return self.get_property('dipole', atoms)

    def get_charges(self, atoms=None):
        return self.get_property('charges', atoms)

    def get_magnetic_moment(self, atoms=None):
        return self.get_property('magmom', atoms)

    def get_magnetic_moments(self, atoms=None):
        """Calculate magnetic moments projected onto atoms."""
        return self.get_property('magmoms', atoms)

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError('{} property not implemented'
                                              .format(name))

        if atoms is None:
            atoms = self.atoms
            system_changes = []
        else:
            system_changes = self.check_state(atoms)
            if system_changes:
                self.reset()
        if name not in self.results:
            if not allow_calculation:
                return None
            self.calculate(atoms, [name], system_changes)

        if name == 'magmom' and 'magmom' not in self.results:
            return 0.0

        if name == 'magmoms' and 'magmoms' not in self.results:
            return np.zeros(len(atoms))

        if name not in self.results:
            # For some reason the calculator was not able to do what we want,
            # and that is OK.
            raise PropertyNotImplementedError('{} not present in this '
                                              'calculation'.format(name))

        result = self.results[name]
        if isinstance(result, np.ndarray):
            result = result.copy()
        return result

    def calculation_required(self, atoms, properties):
        assert not isinstance(properties, str)
        system_changes = self.check_state(atoms)
        if system_changes:
            return True
        for name in properties:
            if name not in self.results:
                return True
        return False

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        """Do the calculation.

        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces', 'stress', 'dipole', 'charges', 'magmom'
            and 'magmoms'.
        system_changes: list of str
            List of what has changed since last calculation.  Can be
            any combination of these six: 'positions', 'numbers', 'cell',
            'pbc', 'initial_charges' and 'initial_magmoms'.

        Subclasses need to implement this, but can ignore properties
        and system_changes if they want.  Calculated properties should
        be inserted into results dictionary like shown in this dummy
        example::

            self.results = {'energy': 0.0,
                            'forces': np.zeros((len(atoms), 3)),
                            'stress': np.zeros(6),
                            'dipole': np.zeros(3),
                            'charges': np.zeros(len(atoms)),
                            'magmom': 0.0,
                            'magmoms': np.zeros(len(atoms))}

        The subclass implementation should first call this
        implementation to set the atoms attribute.
        """

        if atoms is not None:
            self.atoms = atoms.copy()

    def calculate_numerical_forces(self, atoms, d=0.001):
        """Calculate numerical forces using finite difference.

        All atoms will be displaced by +d and -d in all directions."""

        from ase.calculators.test import numeric_force
        return np.array([[numeric_force(atoms, a, i, d)
                          for i in range(3)] for a in range(len(atoms))])

    def calculate_numerical_stress(self, atoms, d=1e-6, voigt=True):
        """Calculate numerical stress using finite difference."""

        stress = np.zeros((3, 3), dtype=float)

        cell = atoms.cell.copy()
        V = atoms.get_volume()
        for i in range(3):
            x = np.eye(3)
            x[i, i] += d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            eplus = atoms.get_potential_energy(force_consistent=True)

            x[i, i] -= 2 * d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            eminus = atoms.get_potential_energy(force_consistent=True)

            stress[i, i] = (eplus - eminus) / (2 * d * V)
            x[i, i] += d

            j = i - 2
            x[i, j] = d
            x[j, i] = d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            eplus = atoms.get_potential_energy(force_consistent=True)

            x[i, j] = -d
            x[j, i] = -d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            eminus = atoms.get_potential_energy(force_consistent=True)

            stress[i, j] = (eplus - eminus) / (4 * d * V)
            stress[j, i] = stress[i, j]
        atoms.set_cell(cell, scale_atoms=True)

        if voigt:
            return stress.flat[[0, 4, 8, 5, 2, 1]]
        else:
            return stress

    def get_spin_polarized(self):
        return False

    def band_structure(self):
        """Create band-structure object for plotting."""
        from ase.dft.band_structure import get_band_structure
        # XXX This calculator is supposed to just have done a band structure
        # calculation, but the calculator may not have the correct Fermi level
        # if it updated the Fermi level after changing k-points.
        # This will be a problem with some calculators (currently GPAW), and
        # the user would have to override this by providing the Fermi level
        # from the selfconsistent calculation.
        return get_band_structure(calc=self)


class FileIOCalculator(Calculator):
    """Base class for calculators that write/read input/output files."""

    command = None
    'Command used to start calculation'

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label=None, atoms=None, command=None, **kwargs):
        """File-IO calculator.

        command: str
            Command used to start calculation.
        """

        Calculator.__init__(self, restart, ignore_bad_restart_file, label,
                            atoms, **kwargs)

        if command is not None:
            self.command = command
        else:
            name = 'ASE_' + self.name.upper() + '_COMMAND'
            self.command = os.environ.get(name, self.command)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.write_input(self.atoms, properties, system_changes)
        if self.command is None:
            raise CalculatorSetupError(
                'Please set ${} environment variable '
                .format('ASE_' + self.name.upper() + '_COMMAND') +
                'or supply the command keyword')
        command = self.command.replace('PREFIX', self.prefix)
        errorcode = subprocess.call(command, shell=True, cwd=self.directory)

        if errorcode:
            raise CalculationFailed('{} in {} returned an error: {}'
                                    .format(self.name, self.directory,
                                            errorcode))
        self.read_results()

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write input file(s).

        Call this method first in subclasses so that directories are
        created automatically."""

        if self.directory != os.curdir and not os.path.isdir(self.directory):
            os.makedirs(self.directory)

    def read_results(self):
        """Read energy, forces, ... from output file(s)."""
        pass
