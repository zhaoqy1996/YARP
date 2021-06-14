import os
import numpy as np
import string

from ase.units import Bohr
from ase.io.fortranfile import FortranFile


def xv_to_atoms(filename):
    """Create atoms object from xv file.

    Parameters:
        -filename : str. The filename of the '.XV' file.

    return : An Atoms object
    """
    from ase.atoms import Atoms
    if not os.path.exists(filename):
        filename += '.gz'

    with open(filename, 'r') as f:
        # Read cell vectors (lines 1-3)
        vectors = []
        for i in range(3):
            data = string.split(f.readline())
            vectors.append([string.atof(data[j]) * Bohr for j in range(3)])

        # Read number of atoms (line 4)
        string.atoi(string.split(f.readline())[0])

        # Read remaining lines
        speciesnumber, atomnumbers, xyz, V = [], [], [], []
        for line in f.readlines():
            if len(line) > 5:  # Ignore blank lines
                data = string.split(line)
                speciesnumber.append(string.atoi(data[0]))
                atomnumbers.append(string.atoi(data[1]))
                xyz.append([string.atof(data[2 + j]) * Bohr for j in range(3)])
                V.append([string.atof(data[5 + j]) * Bohr for j in range(3)])

    vectors = np.array(vectors)
    atomnumbers = np.array(atomnumbers)
    xyz = np.array(xyz)
    atoms = Atoms(numbers=atomnumbers, positions=xyz, cell=vectors)

    return atoms


def read_rho(fname):
    "Read unformatted Siesta charge density file"

    # TODO:
    #
    # Handle formatted and NetCDF files.
    #
    # Siesta source code (at least 2.0.2) can possibly also
    # save RHO as a _formatted_ file (the source code seems
    # prepared, but there seems to be no fdf-options for it though).
    # Siesta >= 3 has support for saving RHO as a NetCDF file
    # (according to manual)

    fh = FortranFile(fname)

    # Read (but ignore) unit cell vectors
    x = fh.readReals('d')
    if len(x) != 3 * 3:
        raise IOError('Failed to read cell vectors')

    # Read number of grid points and spin components
    x = fh.readInts()
    if len(x) != 4:
        raise IOError('Failed to read grid size')
    gpts = x  # number of 'X', 'Y', 'Z', 'spin' gridpoints

    rho = np.zeros(gpts)
    for ispin in range(gpts[3]):
        for n3 in range(gpts[2]):
            for n2 in range(gpts[1]):
                x = fh.readReals('f')
                if len(x) != gpts[0]:
                    raise IOError('Failed to read RHO[:,%i,%i,%i]' %
                                  (n2, n3, ispin))
                rho[:, n2, n3, ispin] = x

    fh.close()

    return rho


def get_valence_charge(filename):
    """ Read the valence charge from '.psf'-file."""
    with open(filename, 'r') as f:
        f.readline()
        f.readline()
        f.readline()
        valence = -float(f.readline().split()[-1])

    return valence


def read_vca_synth_block(filename, species_number=None):
    """ Read the SyntheticAtoms block from the output of the
    'fractional' siesta utility.

    Parameters:
        - filename: String with '.synth' output from fractional.
        - species_number: Optional argument to replace override the
                          species number in the text block.

    Returns: A string that can be inserted into the main '.fdf-file'.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = lines[1:-1]

    if species_number is not None:
        lines[0] = '%d\n' % species_number

    block = ''.join(lines).strip()

    return block


def readHSX(fname):
    """
    Read unformatted siesta HSX file
    """
    import collections

    HSX_tuple = collections.namedtuple('HSX',
                                       ['norbitals', 'norbitals_sc', 'nspin',
                                        'nonzero', 'is_gamma', 'sc_orb2uc_orb',
                                        'row2nnzero', 'sparse_ind2column',
                                        'H_sparse', 'S_sparse',
                                        'aB2RaB_sparse', 'total_elec_charge',
                                        'temp'])

    fh = FortranFile(fname)
    norbitals, norbitals_sc, nspin, nonzero = fh.readInts('i')
    is_gamma = fh.readInts('i')[0]

    sc_orb2uc_orb = 0
    if is_gamma == 0:
        sc_orb2uc_orb = fh.readInts('i')

    row2nnzero = fh.readInts('i')

    sum_row2nnzero = np.sum(row2nnzero)
    if (sum_row2nnzero != nonzero):
        raise ValueError('sum_row2nnzero != nonzero: {0} != {1}'
                         .format(sum_row2nnzero, nonzero))

    row2displ = np.zeros((norbitals), dtype=int)

    for i in range(1, norbitals):
        row2displ[i] = row2displ[i - 1] + row2nnzero[i - 1]

    max_nonzero = np.max(row2nnzero)
    int_buff = np.zeros((max_nonzero), dtype=int)
    sparse_ind2column = np.zeros((nonzero))

    # Fill the rows for each index in *_sparse arrays
    for irow in range(norbitals):
        f = row2nnzero[irow]
        int_buff[0:f] = fh.readInts('i')
        # read set of rows where nonzero elements reside
        d = row2displ[irow]
        sparse_ind2column[d:d + f] = int_buff[0:f]
    # END of Fill the rows for each index in *_sparse arrays

    # allocate H, S and X matrices
    sp_buff = np.zeros((max_nonzero), dtype=float)

    H_sparse = np.zeros((nonzero, nspin), dtype=float)
    S_sparse = np.zeros((nonzero), dtype=float)
    aB2RaB_sparse = np.zeros((3, nonzero), dtype=float)

    # Read the data to H_sparse array
    for ispin in range(nspin):
        for irow in range(norbitals):
            d = row2displ[irow]
            f = row2nnzero[irow]
            sp_buff[0:f] = fh.readReals('f')
            H_sparse[d:d + f, ispin] = sp_buff[0:f]

    # Read the data to S_sparse array
    for irow in range(norbitals):
        f = row2nnzero[irow]
        d = row2displ[irow]
        sp_buff[0:f] = fh.readReals('f')
        S_sparse[d:d + f] = sp_buff[0:f]

    total_elec_charge, temp = fh.readReals('d')

    sp_buff = np.zeros((3 * max_nonzero), dtype=float)
    # Read the data to S_sparse array
    for irow in range(norbitals):
        f = row2nnzero[irow]
        d = row2displ[irow]
        sp_buff[0: 3 * f] = fh.readReals('f')
        aB2RaB_sparse[0, d:d + f] = sp_buff[0:f]
        aB2RaB_sparse[1, d:d + f] = sp_buff[f:2 * f]
        aB2RaB_sparse[2, d:d + f] = sp_buff[2 * f:3 * f]

    fh.close()

    return HSX_tuple(norbitals, norbitals_sc, nspin, nonzero, is_gamma,
                     sc_orb2uc_orb, row2nnzero, sparse_ind2column, H_sparse,
                     S_sparse, aB2RaB_sparse, total_elec_charge, temp)


def readDIM(fname):
    """
    Read unformatted siesta DIM file
    """
    import collections

    DIM_tuple = collections.namedtuple('DIM', ['natoms_sc', 'norbitals_sc',
                                               'norbitals', 'nspin',
                                               'nnonzero',
                                               'natoms_interacting'])

    fh = FortranFile(fname)

    natoms_sc = fh.readInts('i')[0]
    norbitals_sc = fh.readInts('i')[0]
    norbitals = fh.readInts('i')[0]
    nspin = fh.readInts('i')[0]
    nnonzero = fh.readInts('i')[0]
    natoms_interacting = fh.readInts('i')[0]
    fh.close()

    return DIM_tuple(natoms_sc, norbitals_sc, norbitals, nspin,
                     nnonzero, natoms_interacting)


def readPLD(fname, norbitals, natoms):
    """
    Read unformatted siesta PLD file
    """
    import collections
    # use struct library to read mixed data type from binary
    import struct

    PLD_tuple = collections.namedtuple('PLD', ['max_rcut', 'orb2ao',
                                               'orb2uorb', 'orb2occ',
                                               'atm2sp', 'atm2shift',
                                               'coord_sc', 'cell',
                                               'nunit_cells'])

    fh = FortranFile(fname)

    orb2ao = np.zeros((norbitals), dtype=int)
    orb2uorb = np.zeros((norbitals), dtype=int)
    orb2occ = np.zeros((norbitals), dtype=float)
    
    max_rcut = fh.readReals('d')
    for iorb in range(norbitals):
        dat = fh.readRecord()
        dat_size = struct.calcsize('iid')
        val_list = struct.unpack('iid', dat[0:dat_size])
        orb2ao[iorb] = val_list[0]
        orb2uorb[iorb] = val_list[1]
        orb2occ[iorb] = val_list[2]

    atm2sp = np.zeros((natoms), dtype=int)
    atm2shift = np.zeros((natoms + 1), dtype=int)
    for iatm in range(natoms):
        atm2sp[iatm] = fh.readInts('i')[0]
    
    for iatm in range(natoms + 1):
        atm2shift[iatm] = fh.readInts('i')[0]

    cell = np.zeros((3, 3), dtype=float)
    nunit_cells = np.zeros((3), dtype=int)
    for i in range(3):
        cell[i, :] = fh.readReals('d')
    nunit_cells = fh.readInts('i')
    
    coord_sc = np.zeros((natoms, 3), dtype=float)
    for iatm in range(natoms):
        coord_sc[iatm, :] = fh.readReals('d')

    fh.close()
    return PLD_tuple(max_rcut, orb2ao, orb2uorb, orb2occ, atm2sp, atm2shift,
                     coord_sc, cell, nunit_cells)


def readWFSX(fname):
    """
    Read unformatted siesta WFSX file
    """
    import collections
    # use struct library to read mixed data type from binary
    import struct

    WFSX_tuple = collections.namedtuple('WFSX',
                                        ['nkpoints', 'nspin', 'norbitals',
                                         'gamma', 'orb2atm', 'orb2strspecies',
                                         'orb2ao', 'orb2n', 'orb2strsym',
                                         'kpoints', 'DFT_E', 'DFT_X',
                                         'mo_spin_kpoint_2_is_read'])

    fh = FortranFile(fname)

    nkpoints, gamma = fh.readInts('i')
    nspin = fh.readInts('i')[0]
    norbitals = fh.readInts('i')[0]

    orb2atm = np.zeros((norbitals), dtype=int)
    orb2strspecies = []
    orb2ao = np.zeros((norbitals), dtype=int)
    orb2n = np.zeros((norbitals), dtype=int)
    orb2strsym = []
    # for string list are better to select all the string length

    dat_size = struct.calcsize('i20sii20s')
    dat = fh.readRecord()

    ind_st = 0
    ind_fn = dat_size
    for iorb in range(norbitals):
        val_list = struct.unpack('i20sii20s', dat[ind_st:ind_fn])
        orb2atm[iorb] = val_list[0]
        orb2strspecies.append(val_list[1])
        orb2ao[iorb] = val_list[2]
        orb2n[iorb] = val_list[3]
        orb2strsym.append(val_list[4])
        ind_st = ind_st + dat_size
        ind_fn = ind_fn + dat_size
    orb2strspecies = np.array(orb2strspecies)
    orb2strsym = np.array(orb2strsym)

    kpoints = np.zeros((3, nkpoints), dtype=np.float64)
    DFT_E = np.zeros((norbitals, nspin, nkpoints), dtype=np.float64)

    if (gamma == 1):
        DFT_X = np.zeros((1, norbitals, norbitals, nspin, nkpoints),
                         dtype=np.float64)
        eigenvector = np.zeros((1, norbitals), dtype=float)
    else:
        DFT_X = np.zeros((2, norbitals, norbitals, nspin, nkpoints),
                         dtype=np.float64)
        eigenvector = np.zeros((2, norbitals), dtype=float)

    mo_spin_kpoint_2_is_read = np.zeros((norbitals, nspin, nkpoints),
                                        dtype=bool)
    mo_spin_kpoint_2_is_read[0:norbitals, 0:nspin, 0:nkpoints] = False

    dat_size = struct.calcsize('iddd')
    for ikpoint in range(nkpoints):
        for ispin in range(nspin):
            dat = fh.readRecord()
            val_list = struct.unpack('iddd', dat[0:dat_size])
            ikpoint_in = val_list[0] - 1
            kpoints[0:3, ikpoint] = val_list[1:4]
            if (ikpoint != ikpoint_in):
                raise ValueError('siesta_get_wfsx: ikpoint != ikpoint_in')
            ispin_in = fh.readInts('i')[0] - 1
            if (ispin_in > nspin - 1):
                msg = 'siesta_get_wfsx: err: ispin_in>nspin\n \
                     siesta_get_wfsx: ikpoint, ispin, ispin_in = \
                     {0}  {1}  {2}\n siesta_get_wfsx'.format(ikpoint,
                                                             ispin, ispin_in)
                raise ValueError(msg)
            
            norbitals_in = fh.readInts('i')[0]
            if (norbitals_in > norbitals):
                msg = 'siesta_get_wfsx: err: norbitals_in>norbitals\n \
                     siesta_get_wfsx: ikpoint, norbitals, norbitals_in = \
                     {0}  {1}  {2}\n siesta_get_wfsx'.format(ikpoint,
                                                             norbitals,
                                                             norbitals_in)
                raise ValueError(msg)

            for imolecular_orb in range(norbitals_in):
                imolecular_orb_in = fh.readInts('i')[0] - 1
                if (imolecular_orb_in > norbitals - 1):
                    msg = """
                        siesta_get_wfsx: err: imolecular_orb_in>norbitals\n
                        siesta_get_wfsx: ikpoint, norbitals,
                        imolecular_orb_in = {0}  {1}  {2}\n
                        siesta_get_wfsx""".format(ikpoint, norbitals,
                                                  imolecular_orb_in)
                    raise ValueError(msg)

                real_E_eV = fh.readReals('d')[0]
                eigenvector = fh.readReals('f')
                DFT_E[imolecular_orb_in, ispin_in,
                      ikpoint] = real_E_eV / 13.60580
                DFT_X[:, :, imolecular_orb_in, ispin_in,
                      ikpoint] = eigenvector
                mo_spin_kpoint_2_is_read[imolecular_orb_in, ispin_in,
                                         ikpoint] = True

            if (not all(mo_spin_kpoint_2_is_read[:, ispin_in, ikpoint])):
                msg = 'siesta_get_wfsx: warn: .not. all(mo_spin_k_2_is_read)'
                print('mo_spin_kpoint_2_is_read = ', mo_spin_kpoint_2_is_read)
                raise ValueError(msg)
 
    fh.close()
    return WFSX_tuple(nkpoints, nspin, norbitals, gamma, orb2atm,
                      orb2strspecies, orb2ao, orb2n, orb2strsym,
                      kpoints, DFT_E, DFT_X, mo_spin_kpoint_2_is_read)
