from __future__ import print_function
import warnings

import numpy as np

from ase import __version__
from ase.calculators.singlepoint import SinglePointCalculator, all_properties
from ase.constraints import dict2constraint
from ase.calculators.calculator import PropertyNotImplementedError
from ase.atoms import Atoms
from ase.io.jsonio import encode, decode
from ase.io.pickletrajectory import PickleTrajectory
from ase.parallel import world

__all__ = ['Trajectory', 'PickleTrajectory']


def Trajectory(filename, mode='r', atoms=None, properties=None, master=None):
    """A Trajectory can be created in read, write or append mode.

    Parameters:

    filename: str
        The name of the file.  Traditionally ends in .traj.
    mode: str
        The mode.  'r' is read mode, the file should already exist, and
        no atoms argument should be specified.
        'w' is write mode.  The atoms argument specifies the Atoms
        object to be written to the file, if not given it must instead
        be given as an argument to the write() method.
        'a' is append mode.  It acts as write mode, except that
        data is appended to a preexisting file.
    atoms: Atoms object
        The Atoms object to be written in write or append mode.
    properties: list of str
        If specified, these calculator properties are saved in the
        trajectory.  If not specified, all supported quantities are
        saved.  Possible values: energy, forces, stress, dipole,
        charges, magmom and magmoms.
    master: bool
        Controls which process does the actual writing. The
        default is that process number 0 does this.  If this
        argument is given, processes where it is True will write.

    The atoms, properties and master arguments are ignores in read mode.
    """
    if mode == 'r':
        return TrajectoryReader(filename)
    return TrajectoryWriter(filename, mode, atoms, properties, master=master)


class TrajectoryWriter:
    """Writes Atoms objects to a .traj file."""
    def __init__(self, filename, mode='w', atoms=None, properties=None,
                 extra=[], master=None):
        """A Trajectory writer, in write or append mode.

        Parameters:

        filename: str
            The name of the file.  Traditionally ends in .traj.
        mode: str
            The mode.  'r' is read mode, the file should already exist, and
            no atoms argument should be specified.
            'w' is write mode.  The atoms argument specifies the Atoms
            object to be written to the file, if not given it must instead
            be given as an argument to the write() method.
            'a' is append mode.  It acts as write mode, except that
            data is appended to a preexisting file.
        atoms: Atoms object
            The Atoms object to be written in write or append mode.
        properties: list of str
            If specified, these calculator properties are saved in the
            trajectory.  If not specified, all supported quantities are
            saved.  Possible values: energy, forces, stress, dipole,
            charges, magmom and magmoms.
        master: bool
            Controls which process does the actual writing. The
            default is that process number 0 does this.  If this
            argument is given, processes where it is True will write.
        """
        if master is None:
            master = (world.rank == 0)
        self.master = master
        self.atoms = atoms
        self.properties = properties

        self.description = {}
        self.header_data = None
        self.multiple_headers = False

        self._open(filename, mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def set_description(self, description):
        self.description.update(description)

    def _open(self, filename, mode):
        import ase.io.ulm as ulm
        if mode not in 'aw':
            raise ValueError('mode must be "w" or "a".')
        if self.master:
            self.backend = ulm.open(filename, mode, tag='ASE-Trajectory')
            if len(self.backend) > 0 and mode == 'a':
                atoms = Trajectory(filename)[0]
                self.header_data = get_header_data(atoms)
        else:
            self.backend = ulm.DummyWriter()

    def write(self, atoms=None, **kwargs):
        """Write the atoms to the file.

        If the atoms argument is not given, the atoms object specified
        when creating the trajectory object is used.

        Use keyword arguments to add extra properties::

            writer.write(atoms, energy=117, dipole=[0, 0, 1.0])
        """
        if atoms is None:
            atoms = self.atoms

        for image in atoms.iterimages():
            self._write_atoms(image, **kwargs)

    def _write_atoms(self, atoms, **kwargs):
        b = self.backend

        if self.header_data is None:
            b.write(version=1, ase_version=__version__)
            if self.description:
                b.write(description=self.description)
            # Atomic numbers and periodic boundary conditions are written
            # in the header in the beginning.
            #
            # If an image later on has other numbers/pbc, we write a new
            # header.  All subsequent images will then have their own header
            # whether or not their numbers/pbc change.
            self.header_data = get_header_data(atoms)
            write_header = True
        else:
            if not self.multiple_headers:
                header_data = get_header_data(atoms)
                self.multiple_headers = not headers_equal(self.header_data,
                                                          header_data)
            write_header = self.multiple_headers

        write_atoms(b, atoms, write_header=write_header)

        calc = atoms.get_calculator()

        if calc is None and len(kwargs) > 0:
            calc = SinglePointCalculator(atoms)

        if calc is not None:
            if not hasattr(calc, 'get_property'):
                calc = OldCalculatorWrapper(calc)
            c = b.child('calculator')
            c.write(name=calc.name)
            if hasattr(calc, 'todict'):
                c.write(parameters=calc.todict())
            for prop in all_properties:
                if prop in kwargs:
                    x = kwargs[prop]
                else:
                    if self.properties is not None:
                        if prop in self.properties:
                            x = calc.get_property(prop, atoms)
                        else:
                            x = None
                    else:
                        try:
                            x = calc.get_property(prop, atoms,
                                                  allow_calculation=False)
                        except (PropertyNotImplementedError, KeyError):
                            # KeyError is needed for Jacapo.
                            x = None
                if x is not None:
                    if prop in ['stress', 'dipole']:
                        x = x.tolist()
                    c.write(prop, x)

        info = {}
        for key, value in atoms.info.items():
            try:
                encode(value)
            except TypeError:
                warnings.warn('Skipping "{0}" info.'.format(key))
            else:
                info[key] = value
        if info:
            b.write(info=info)

        b.sync()

    def close(self):
        """Close the trajectory file."""
        self.backend.close()

    def __len__(self):
        return world.sum(len(self.backend))


class TrajectoryReader:
    """Reads Atoms objects from a .traj file."""
    def __init__(self, filename):
        """A Trajectory in read mode.

        The filename traditionally ends in .traj.
        """

        self.numbers = None
        self.pbc = None
        self.masses = None

        self._open(filename)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def _open(self, filename):
        import ase.io.ulm as ulm
        self.backend = ulm.open(filename, 'r')
        self._read_header()

    def _read_header(self):
        b = self.backend
        if b.get_tag() != 'ASE-Trajectory':
            raise IOError('This is not a trajectory file!')

        if len(b) > 0:
            self.pbc = b.pbc
            self.numbers = b.numbers
            self.masses = b.get('masses')
            self.constraints = b.get('constraints', '[]')
            self.description = b.get('description')
            self.version = b.version
            self.ase_version = b.get('ase_version')

    def close(self):
        """Close the trajectory file."""
        self.backend.close()

    def __getitem__(self, i=-1):
        b = self.backend[i]
        if 'numbers' in b:
            # numbers and other header info was written alongside the image:
            atoms = read_atoms(b)
        else:
            # header info was not written because they are the same:
            atoms = read_atoms(b, header=[self.pbc, self.numbers, self.masses,
                                          self.constraints])
        if 'calculator' in b:
            results = {}
            c = b.calculator
            for prop in all_properties:
                if prop in c:
                    results[prop] = c.get(prop)
            calc = SinglePointCalculator(atoms, **results)
            calc.name = b.calculator.name

            if 'parameters' in c:
                calc.parameters.update(c.parameters)
            atoms.set_calculator(calc)

        return atoms

    def __len__(self):
        return len(self.backend)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def get_header_data(atoms):
    return {'pbc': atoms.pbc.copy(),
            'numbers': atoms.get_atomic_numbers(),
            'masses': atoms.get_masses() if atoms.has('masses') else None,
            'constraints': list(atoms.constraints)}


def headers_equal(headers1, headers2):
    assert len(headers1) == len(headers2)
    eq = True
    for key in headers1:
        eq &= np.array_equal(headers1[key], headers2[key])
    return eq


def read_atoms(backend, header=None):
    b = backend
    if header:
        pbc, numbers, masses, constraints = header
    else:
        pbc = b.pbc
        numbers = b.numbers
        masses = b.get('masses')
        constraints = b.get('constraints', '[]')

    atoms = Atoms(positions=b.positions,
                  numbers=numbers,
                  cell=b.cell,
                  masses=masses,
                  pbc=pbc,
                  info=b.get('info'),
                  constraint=[dict2constraint(d)
                              for d in decode(constraints)],
                  momenta=b.get('momenta'),
                  magmoms=b.get('magmoms'),
                  charges=b.get('charges'),
                  tags=b.get('tags'))
    return atoms


def write_atoms(backend, atoms, write_header=True):
    b = backend

    if write_header:
        b.write(pbc=atoms.pbc.tolist(),
                numbers=atoms.numbers)
        if atoms.constraints:
            if all(hasattr(c, 'todict') for c in atoms.constraints):
                b.write(constraints=encode(atoms.constraints))

        if atoms.has('masses'):
            b.write(masses=atoms.get_masses())

    b.write(positions=atoms.get_positions(),
            cell=atoms.get_cell().tolist())

    if atoms.has('tags'):
        b.write(tags=atoms.get_tags())
    if atoms.has('momenta'):
        b.write(momenta=atoms.get_momenta())
    if atoms.has('initial_magmoms'):
        b.write(magmoms=atoms.get_initial_magnetic_moments())
    if atoms.has('initial_charges'):
        b.write(charges=atoms.get_initial_charges())


def read_traj(fd, index):
    trj = TrajectoryReader(fd)
    for i in range(*index.indices(len(trj))):
        yield trj[i]


def write_traj(fd, images):
    """Write image(s) to trajectory."""
    trj = TrajectoryWriter(fd)
    if isinstance(images, Atoms):
        images = [images]
    for atoms in images:
        trj.write(atoms)


class OldCalculatorWrapper:
    def __init__(self, calc):
        self.calc = calc
        try:
            self.name = calc.name
        except AttributeError:
            self.name = calc.__class__.__name__.lower()

    def get_property(self, prop, atoms, allow_calculation=True):
        try:
            if (not allow_calculation and
                self.calc.calculation_required(atoms, [prop])):
                return None
        except AttributeError:
            pass

        method = 'get_' + {'energy': 'potential_energy',
                           'magmom': 'magnetic_moment',
                           'magmoms': 'magnetic_moments',
                           'dipole': 'dipole_moment'}.get(prop, prop)
        try:
            result = getattr(self.calc, method)(atoms)
        except AttributeError:
            raise PropertyNotImplementedError
        return result


def convert(name):
    import os
    t = TrajectoryWriter(name + '.new')
    for atoms in PickleTrajectory(name, _warn=False):
        t.write(atoms)
    t.close()
    os.rename(name, name + '.old')
    os.rename(name + '.new', name)


def main():
    import optparse
    parser = optparse.OptionParser(usage='python -m ase.io.trajectory '
                                   'a1.traj [a2.traj ...]',
                                   description='Convert old trajectory '
                                   'file(s) to new format. '
                                   'The old file is kept as a1.traj.old.')
    opts, args = parser.parse_args()
    for name in args:
        convert(name)


if __name__ == '__main__':
    main()
