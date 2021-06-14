"""
netcdftrajectory - I/O trajectory files in the AMBER NetCDF convention

More information on the AMBER NetCDF conventions can be found at
http://ambermd.org/netcdf/. This module supports extensions to
these conventions, such as writing of additional fields and writing to
HDF5 (NetCDF-4) files.

A netCDF4-python is required by this module:

    netCDF4-python - https://github.com/Unidata/netcdf4-python

NetCDF files can be directly visualized using the libAtoms flavor of
AtomEye (http://www.libatoms.org/),
VMD (http://www.ks.uiuc.edu/Research/vmd/)
or Ovito (http://www.ovito.org/, starting with version 2.3).
"""

from __future__ import division

import os
import warnings

import numpy as np

import ase

from ase.data import atomic_masses
from ase.geometry import cellpar_to_cell
import collections
from functools import reduce


class NetCDFTrajectory:
    """
    Reads/writes Atoms objects into an AMBER-style .nc trajectory file.
    """

    # Default dimension names
    _frame_dim = 'frame'
    _spatial_dim = 'spatial'
    _atom_dim = 'atom'
    _cell_spatial_dim = 'cell_spatial'
    _cell_angular_dim = 'cell_angular'
    _label_dim = 'label'
    _Voigt_dim = 'Voigt' # For stress/strain tensors

    # Default field names. If it is a list, check for any of these names upon
    # opening. Upon writing, use the first name.
    _spatial_var = 'spatial'
    _cell_spatial_var = 'cell_spatial'
    _cell_angular_var = 'cell_angular'
    _time_var = 'time'
    _numbers_var = ['atom_types', 'type', 'Z']
    _positions_var = 'coordinates'
    _velocities_var = 'velocities'
    _cell_origin_var = 'cell_origin'
    _cell_lengths_var = 'cell_lengths'
    _cell_angles_var = 'cell_angles'

    _default_vars = reduce(lambda x, y: x + y,
                           [_numbers_var, [_positions_var], [_velocities_var],
                            [_cell_origin_var], [_cell_lengths_var],
                            [_cell_angles_var]])

    def __init__(self, filename, mode='r', atoms=None, types_to_numbers=None,
                 double=True, netcdf_format='NETCDF3_CLASSIC', keep_open=True,
                 index_var='id', index_offset=-1, chunk_size=1000000):
        """
        A NetCDFTrajectory can be created in read, write or append mode.

        Parameters:

        filename:
            The name of the parameter file.  Should end in .nc.

        mode='r':
            The mode.

            'r' is read mode, the file should already exist, and no atoms
            argument should be specified.

            'w' is write mode. The atoms argument specifies the Atoms object
            to be written to the file, if not given it must instead be given
            as an argument to the write() method.

            'a' is append mode.  It acts a write mode, except that data is
            appended to a preexisting file.

        atoms=None:
            The Atoms object to be written in write or append mode.

        types_to_numbers=None:
            Dictionary for conversion of atom types to atomic numbers when
            reading a trajectory file.

        double=True:
            Create new variable in double precision.

        netcdf_format='NETCDF3_CLASSIC':
            Format string for the underlying NetCDF file format. Only relevant
            if a new file is created. More information can be found at
            https://www.unidata.ucar.edu/software/netcdf/docs/netcdf/File-Format.html

            'NETCDF3_CLASSIC' is the original binary format.

            'NETCDF3_64BIT' can be used to write larger files.

            'NETCDF4_CLASSIC' is HDF5 with some NetCDF limitations.

            'NETCDF4' is HDF5.

        keep_open=True:
            Keep the file open during consecutive read/write operations.
            Set to false if you experience data corruption. This will close the
            file after each read/write operation by comes with serious
            performance penalty.

        index_var='id':
            Name of variable containing the atom indices. Atoms are reordered
            by this index upon reading if this variable is present. Default
            value is for LAMMPS output.

        index_offset=-1:
            Set to 0 if atom index is zero based, set to -1 if atom index is
            one based. Default value is for LAMMPS output.

        chunk_size=1000000:
            Maximum size of consecutive number of records (along the 'atom')
            dimension read when reading from a NetCDF file. This is used to
            reduce the memory footprint of a read operation on very large files.
        """
        self.nc = None
        self.chunk_size = chunk_size

        self.numbers = None
        self.pre_observers = []   # Callback functions before write
        self.post_observers = []  # Callback functions after write are called

        self.has_header = False
        self._set_atoms(atoms)

        self.types_to_numbers = None
        if types_to_numbers:
            self.types_to_numbers = np.array(types_to_numbers)

        self.index_var = index_var
        self.index_offset = index_offset

        self._default_vars += [self.index_var]

        # 'l' should be a valid type according to the netcdf4-python
        # documentation, but does not appear to work.
        self.dtype_conv = {'l': 'i'}
        if not double:
            self.dtype_conv.update(dict(d='f'))

        self.extra_per_frame_vars = []
        self.extra_per_file_vars = []
        # per frame atts are global quantities, not quantities stored for each
        # atom
        self.extra_per_frame_atts = []

        self.mode = mode
        self.netcdf_format = netcdf_format

        if atoms:
            self.n_atoms = len(atoms)
        else:
            self.n_atoms = None

        self.filename = filename
        if keep_open is None:
            # Only netCDF4-python supports append to files
            self.keep_open = self.mode == 'r'
        else:
            self.keep_open = keep_open

    def __del__(self):
        self.close()

    def _open(self):
        """
        Opens the file.

        For internal use only.
        """
        import netCDF4
        if self.nc is not None:
            return
        if self.mode == 'a' and not os.path.exists(self.filename):
            self.mode = 'w'
        self.nc = netCDF4.Dataset(self.filename, self.mode,
                                  format=self.netcdf_format)

        self.frame = 0
        if self.mode == 'r' or self.mode == 'a':
            self._read_header()
            self.frame = self._len()

    def _set_atoms(self, atoms=None):
        """
        Associate an Atoms object with the trajectory.

        For internal use only.
        """
        if atoms is not None and not hasattr(atoms, 'get_positions'):
            raise TypeError('"atoms" argument is not an Atoms object.')
        self.atoms = atoms

    def _read_header(self):
        if not self.n_atoms:
            self.n_atoms = len(self.nc.dimensions[self._atom_dim])

        for name, var in self.nc.variables.items():
            # This can be unicode which confuses ASE
            name = str(name)
            # _default_vars is taken care of already
            if name not in self._default_vars:
                if len(var.dimensions) >= 2:
                    if var.dimensions[0] == self._frame_dim:
                        if var.dimensions[1] == self._atom_dim:
                            self.extra_per_frame_vars += [name]
                        else:
                            self.extra_per_frame_atts += [name]

                elif len(var.dimensions) == 1:
                    if var.dimensions[0] == self._atom_dim:
                        self.extra_per_file_vars += [name]
                    elif var.dimensions[0] == self._frame_dim:
                        self.extra_per_frame_atts += [name]

        self.has_header = True

    def write(self, atoms=None, frame=None, arrays=None, time=None):
        """
        Write the atoms to the file.

        If the atoms argument is not given, the atoms object specified
        when creating the trajectory object is used.
        """
        self._open()
        self._call_observers(self.pre_observers)
        if atoms is None:
            atoms = self.atoms

        if hasattr(atoms, 'interpolate'):
            # seems to be a NEB
            neb = atoms
            assert not neb.parallel
            try:
                neb.get_energies_and_forces(all=True)
            except AttributeError:
                pass
            for image in neb.images:
                self.write(image)
            return

        if not self.has_header:
            self._define_file_structure(atoms)
        else:
            if len(atoms) != self.n_atoms:
                raise ValueError('Bad number of atoms!')

        if frame is None:
            i = self.frame
        else:
            i = frame

        # Number can be per file variable
        numbers = self._get_variable(self._numbers_var)
        if numbers.dimensions[0] == self._frame_dim:
            numbers[i] = atoms.get_atomic_numbers()
        else:
            if np.any(numbers != atoms.get_atomic_numbers()):
                raise ValueError('Atomic numbers do not match!')
        self._get_variable(self._positions_var)[i] = atoms.get_positions()
        if atoms.has('momenta'):
            self._add_velocities()
            self._get_variable(self._velocities_var)[i] = \
                atoms.get_momenta() / atoms.get_masses().reshape(-1, 1)
        a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()
        if np.any(np.logical_not(atoms.pbc)):
            warnings.warn('Atoms have nonperiodic directions. Cell lengths in '
                          'these directions are lost and will be '
                          'shrink-wrapped when reading the NetCDF file.')
        cell_lengths = np.array([a, b, c]) * atoms.pbc
        self._get_variable(self._cell_lengths_var)[i] = cell_lengths
        self._get_variable(self._cell_angles_var)[i] = [alpha, beta, gamma]
        self._get_variable(self._cell_origin_var)[i] = \
            atoms.get_celldisp().reshape(3)
        if arrays is not None:
            for array in arrays:
                data = atoms.get_array(array)
                if array in self.extra_per_file_vars:
                    # This field exists but is per file data. Check that the
                    # data remains consistent.
                    if np.any(self._get_variable(array) != data):
                        raise ValueError('Trying to write Atoms object with '
                                         'incompatible data for the {0} '
                                         'array.'.format(array))
                else:
                    self._add_array(atoms, array, data.dtype, data.shape)
                    self._get_variable(array)[i] = data
        if time is not None:
            self._add_time()
            self._get_variable(self._time_var)[i] = time

        self.sync()

        self._call_observers(self.post_observers)
        self.frame += 1
        self._close()

    def write_arrays(self, atoms, frame, arrays):
        self._open()
        self._call_observers(self.pre_observers)
        for array in arrays:
            data = atoms.get_array(array)
            if array in self.extra_per_file_vars:
                # This field exists but is per file data. Check that the
                # data remains consistent.
                if np.any(self._get_variable(array) != data):
                    raise ValueError('Trying to write Atoms object with '
                                     'incompatible data for the {0} '
                                     'array.'.format(array))
            else:
                self._add_array(atoms, array, data.dtype, data.shape)
                self._get_variable(array)[frame] = data
        self._call_observers(self.post_observers)
        self._close()

    def _define_file_structure(self, atoms):
        if not hasattr(self.nc, 'Conventions'):
            self.nc.Conventions = 'AMBER'
        if not hasattr(self.nc, 'ConventionVersion'):
            self.nc.ConventionVersion = '1.0'
        if not hasattr(self.nc, 'program'):
            self.nc.program = 'ASE'
        if not hasattr(self.nc, 'programVersion'):
            self.nc.programVersion = ase.__version__

        if self._frame_dim not in self.nc.dimensions:
            self.nc.createDimension(self._frame_dim, None)
        if self._spatial_dim not in self.nc.dimensions:
            self.nc.createDimension(self._spatial_dim, 3)
        if self._atom_dim not in self.nc.dimensions:
            self.nc.createDimension(self._atom_dim, len(atoms))
        if self._cell_spatial_dim not in self.nc.dimensions:
            self.nc.createDimension(self._cell_spatial_dim, 3)
        if self._cell_angular_dim not in self.nc.dimensions:
            self.nc.createDimension(self._cell_angular_dim, 3)
        if self._label_dim not in self.nc.dimensions:
            self.nc.createDimension(self._label_dim, 5)

        # Self-describing variables from AMBER convention
        if not self._has_variable(self._spatial_var):
            self.nc.createVariable(self._spatial_var, 'S1',
                                   (self._spatial_dim,))
            self.nc.variables[self._spatial_var][:] = ['x', 'y', 'z']
        if not self._has_variable(self._cell_spatial_var):
            self.nc.createVariable(self._cell_spatial_dim, 'S1',
                                   (self._cell_spatial_dim,))
            self.nc.variables[self._cell_spatial_var][:] = ['a', 'b', 'c']
        if not self._has_variable(self._cell_angular_var):
            self.nc.createVariable(self._cell_angular_var, 'S1',
                                   (self._cell_angular_dim, self._label_dim,))
            self.nc.variables[self._cell_angular_var][0] = [x for x in 'alpha']
            self.nc.variables[self._cell_angular_var][1] = [x for x in 'beta ']
            self.nc.variables[self._cell_angular_var][2] = [x for x in 'gamma']

        if not self._has_variable(self._numbers_var):
            self.nc.createVariable(self._numbers_var[0], 'i',
                                   (self._frame_dim, self._atom_dim,))
        if not self._has_variable(self._positions_var):
            self.nc.createVariable(self._positions_var, 'f4',
                                   (self._frame_dim, self._atom_dim,
                                    self._spatial_dim))
            self.nc.variables[self._positions_var].units = 'Angstrom'
            self.nc.variables[self._positions_var].scale_factor = 1.
        if not self._has_variable(self._cell_lengths_var):
            self.nc.createVariable(self._cell_lengths_var, 'd',
                                   (self._frame_dim, self._cell_spatial_dim))
            self.nc.variables[self._cell_lengths_var].units = 'Angstrom'
            self.nc.variables[self._cell_lengths_var].scale_factor = 1.
        if not self._has_variable(self._cell_angles_var):
            self.nc.createVariable(self._cell_angles_var, 'd',
                                   (self._frame_dim, self._cell_angular_dim))
            self.nc.variables[self._cell_angles_var].units = 'degree'
        if not self._has_variable(self._cell_origin_var):
            self.nc.createVariable(self._cell_origin_var, 'd',
                                   (self._frame_dim, self._cell_spatial_dim))
            self.nc.variables[self._cell_origin_var].units = 'Angstrom'
            self.nc.variables[self._cell_origin_var].scale_factor = 1.

    def _add_time(self):
        if not self._has_variable(self._time_var):
            self.nc.createVariable(self._time_var, 'f8', (self._frame_dim,))

    def _add_velocities(self):
        if not self._has_variable(self._velocities_var):
            self.nc.createVariable(self._velocities_var, 'f4',
                                   (self._frame_dim, self._atom_dim,
                                    self._spatial_dim))
            self.nc.variables[self._positions_var].units = \
                'Angstrom/Femtosecond'
            self.nc.variables[self._positions_var].scale_factor = 1.

    def _add_array(self, atoms, array_name, type, shape):
        if not self._has_variable(array_name):
            dims = [self._frame_dim]
            for i in shape:
                if i == len(atoms):
                    dims += [self._atom_dim]
                elif i == 3:
                    dims += [self._spatial_dim]
                elif i == 6:
                    # This can only be stress/strain tensor in Voigt notation
                    if self._Voigt_dim not in self.nc.dimensions:
                        self.nc.createDimension(self._Voigt_dim, 6)
                    dims += [self._Voigt_dim]
                else:
                    raise TypeError("Don't know how to dump array of shape {0}"
                                    " into NetCDF trajectory.".format(shape))
            try:
                t = self.dtype_conv[type.char]
            except:
                t = type
            self.nc.createVariable(array_name, t, dims)

    def _get_variable(self, name, exc=True):
        if isinstance(name, list):
            for n in name:
                if n in self.nc.variables:
                    return self.nc.variables[n]
            if exc:
                raise RuntimeError(
                    'None of the variables {0} was found in the '
                    'NetCDF trajectory.'.format(', '.join(name)))
        else:
            if name in self.nc.variables:
                return self.nc.variables[name]
            if exc:
                raise RuntimeError('Variables {0} was found in the NetCDF '
                                   'trajectory.'.format(name))
        return None

    def _has_variable(self, name):
        if isinstance(name, list):
            for n in name:
                if n in self.nc.variables:
                    return True
            return False
        else:
            return name in self.nc.variables

    def _get_data(self, name, frame, index, exc=True):
        var = self._get_variable(name, exc=exc)
        if var is None:
            return None
        if var.dimensions[0] == self._frame_dim:
            data = np.zeros(var.shape[1:], dtype=var.dtype)
            s = var.shape[1]
            if s < self.chunk_size:
                data[index] = var[frame]
            else:
                # If this is a large data set, only read chunks from it to
                # reduce memory footprint of the NetCDFTrajectory reader.
                for i in range((s - 1) // self.chunk_size + 1):
                    sl = slice(i * self.chunk_size,
                               min((i + 1) * self.chunk_size, s))
                    data[index[sl]] = var[frame, sl]
        else:
            data = np.zeros(var.shape, dtype=var.dtype)
            s = var.shape[0]
            if s < self.chunk_size:
                data[index] = var
            else:
                # If this is a large data set, only read chunks from it to
                # reduce memory footprint of the NetCDFTrajectory reader.
                for i in range((s-1)//self.chunk_size+1):
                    sl = slice(i*self.chunk_size,
                               min((i+1)*self.chunk_size, s))
                    data[index[sl]] = var[sl]
        return data

    def close(self):
        """Close the trajectory file."""
        if self.nc is not None:
            self.nc.close()
            self.nc = None

    def _close(self):
        if not self.keep_open:
            self.close()
            if self.mode == 'w':
                self.mode = 'a'

    def sync(self):
        self.nc.sync()

    def __getitem__(self, i=-1):
        self._open()

        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self._len()))]

        N = self._len()
        if 0 <= i < N:
            # Non-periodic boundaries have cell_length == 0.0
            cell_lengths = \
                np.array(self.nc.variables[self._cell_lengths_var][i][:])
            pbc = np.abs(cell_lengths > 1e-6)

            # Do we have a cell origin?
            if self._has_variable(self._cell_origin_var):
                origin = np.array(
                    self.nc.variables[self._cell_origin_var][i][:])
            else:
                origin = np.zeros([3], dtype=float)

            # Do we have an index variable?
            if self._has_variable(self.index_var):
                index = np.array(self.nc.variables[self.index_var][i][:]) + \
                    self.index_offset
            else:
                index = np.arange(self.n_atoms)

            # Read element numbers
            self.numbers = self._get_data(self._numbers_var, i, index,
                                          exc=False)
            if self.numbers is None:
                self.numbers = np.ones(self.n_atoms, dtype=int)
            if self.types_to_numbers is not None:
                self.numbers = self.types_to_numbers[self.numbers]
            self.masses = atomic_masses[self.numbers]

            # Read positions
            positions = self._get_data(self._positions_var, i, index)

            # Determine cell size for non-periodic directions from shrink
            # wrapped cell.
            for dim in np.arange(3)[np.logical_not(pbc)]:
                origin[dim] = positions[:, dim].min()
                cell_lengths[dim] = positions[:, dim].max() - origin[dim]

            # Construct cell shape from cell lengths and angles
            cell = cellpar_to_cell(
                list(cell_lengths) +
                list(self.nc.variables[self._cell_angles_var][i])
            )

            # Compute momenta from velocities (if present)
            momenta = self._get_data(self._velocities_var, i, index,
                                     exc=False)
            if momenta is not None:
                momenta *= self.masses.reshape(-1, 1)

            # Fill info dict with additional data found in the NetCDF file
            info = {}
            for name in self.extra_per_frame_atts:
                info[name] = np.array(self.nc.variables[name][i])

            # Create atoms object
            atoms = ase.Atoms(
                positions=positions,
                numbers=self.numbers,
                cell=cell,
                celldisp=origin,
                momenta=momenta,
                masses=self.masses,
                pbc=pbc,
                info=info
            )

            # Attach additional arrays found in the NetCDF file
            for name in self.extra_per_frame_vars:
                atoms.set_array(name, self._get_data(name, i, index))
            for name in self.extra_per_file_vars:
                atoms.set_array(name, self._get_data(name, i, index))
            self._close()
            return atoms

        i = N + i
        if i < 0 or i >= N:
            self._close()
            raise IndexError('Trajectory index out of range.')
        return self[i]

    def _len(self):
        if self._frame_dim in self.nc.dimensions:
            return int(self._get_variable(self._positions_var).shape[0])
        else:
            return 0

    def __len__(self):
        self._open()
        n_frames = self._len()
        self._close()
        return n_frames

    def pre_write_attach(self, function, interval=1, *args, **kwargs):
        """
        Attach a function to be called before writing begins.

        function: The function or callable object to be called.

        interval: How often the function is called.  Default: every time (1).

        All other arguments are stored, and passed to the function.
        """
        if not isinstance(function, collections.Callable):
            raise ValueError('Callback object must be callable.')
        self.pre_observers.append((function, interval, args, kwargs))

    def post_write_attach(self, function, interval=1, *args, **kwargs):
        """
        Attach a function to be called after writing ends.

        function: The function or callable object to be called.

        interval: How often the function is called.  Default: every time (1).

        All other arguments are stored, and passed to the function.
        """
        if not isinstance(function, collections.Callable):
            raise ValueError('Callback object must be callable.')
        self.post_observers.append((function, interval, args, kwargs))

    def _call_observers(self, obs):
        """Call pre/post write observers."""
        for function, interval, args, kwargs in obs:
            if self.write_counter % interval == 0:
                function(*args, **kwargs)


def read_netcdftrajectory(filename, index=-1):
    traj = NetCDFTrajectory(filename, mode='r')
    return traj[index]


def write_netcdftrajectory(filename, images):
    traj = NetCDFTrajectory(filename, mode='w')

    if hasattr(images, 'get_positions'):
        images = [images]

    for atoms in images:
        traj.write(atoms)
    traj.close()
